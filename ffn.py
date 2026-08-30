import numpy as np
import pandas as pd
from typing import Tuple

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold

LABELS_TXT = "first_subset_matching_genes_phenotype_vector_mouse.txt"
EMB_NPZ    = "longest_per_gene_mouse_matching_subset_embeddings.npz"
MODEL_OUT  = "all_training_pt/run_1.pt"

BATCH_SIZE = 128
EPOCHS     = 20
LR         = 1e-5
H1, H2     = 256,128
RANDOM_SEED = 42
VAL_SIZE    = 0.2

def _read_labels(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep = None, engine = "python")
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns] #safe checks
    df = df.rename(columns={df.columns[0]: "X"})
    df["X"] = df["X"].astype(str).str.split(".").str[0].str.strip() #safe checks
    df = df.dropna(subset=["X"]).drop_duplicates(subset=["X"]) #safe checks
    for c in df.columns[1:]:
        df[c] = (pd.to_numeric(df[c], errors="coerce").fillna(0) > 0).astype("int8") #safe checks
    if df.shape[1] < 2:
        raise ValueError("Label file appears to have only one column after parsing.") #safe checks
    return df

def _load_embeddings(npz_path: str) -> Tuple[pd.DataFrame, int]:
    npz = np.load(npz_path)
    ids = npz["ids"]
    emb = npz["embeddings"]
    if ids.shape[0] != emb.shape[0]: raise ValueError("ids and embeddings row counts do not match.") #safe checks
    def parse_gene_id(s: str) -> str:
        parts = str(s).split("|")
        gene = parts[2]
        return gene.split(".")[0].strip()
    gene_ids = [parse_gene_id(s) for s in ids]
    emb_df = pd.DataFrame(emb, index=gene_ids)
    emb_df.index.name = "X"
    return emb_df, emb.shape[1]

def _align(df_labels: pd.DataFrame, emb_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Index, pd.Index]:
    common = df_labels["X"].isin(emb_df.index)
    df_sub = df_labels.loc[common].copy()
    df_sub = df_sub.set_index("X").sort_index()
    emb_sub = emb_df.loc[df_sub.index].sort_index()
    Y = df_sub.astype("int8").to_numpy() #rows = genes, columns = phenotypes
    X = emb_sub.to_numpy(dtype=np.float32) #rows = genes, columns = embeddings
    tasks = df_sub.columns
    genes = df_sub.index
    return X, Y, genes, tasks

def cross_validate(X: np.ndarray, Y: np.ndarray, n_splits=5, device='cuda'):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    fold_metrics = []
    fold_idx = 0

    for train_idx, val_idx in kf.split(X):
        fold_idx += 1
        print(f"\n=== Fold {fold_idx}/{n_splits} ===")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = Y[train_idx], Y[val_idx]

        # --- compute per-phenotype pos_weight from training fold (neg / pos)
        pos = y_train.sum(axis=0).astype(np.float32)            # array of positives per label
        neg = (y_train.shape[0] - pos).astype(np.float32)       # array of negatives per label
        # avoid division by zero: if pos==0 set weight to 1.0 (these labels should typically be filtered earlier)
        pos_weight_arr = np.where(pos > 0, neg / (pos + 1e-12), 1.0).astype(np.float32)
        pos_weight = torch.tensor(pos_weight_arr, dtype=torch.float32, device=device)

        # create criterion using pos_weight
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).float())
        val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val).float())
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

        num_tasks = Y.shape[1] #phenotypes = # columns
        model = MultiTaskBinaryFFN(
            input_dim=X.shape[1],
            hidden1=H1,
            hidden2=H2,
            num_tasks=num_tasks
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_f1 = -1.0
        best_acc = None
        for epoch in range(1, EPOCHS + 1):
            train_loss = train(model, train_loader, optimizer, criterion, device=device)
            val_acc, val_f1 = evaluate(model, val_loader, device=device) 
            print(f"Epoch {epoch:02d}/{EPOCHS} | Training loss = {train_loss:.4f} | Validation accuracy = {val_acc:.4f} | Validation F1 = {val_f1:.4f}")
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_acc = val_acc
                fold_model_path = MODEL_OUT.replace('.pt', f'_fold{fold_idx}.pt')
                torch.save(model.state_dict(), fold_model_path)

        print(f"Fold {fold_idx} complete. Best F1 = {best_f1:.4f}")
        fold_metrics.append((best_acc, best_f1))

    accs = [m[0] for m in fold_metrics]
    f1s  = [m[1] for m in fold_metrics]
    print('\n=== Cross-validation summary ===')
    print(f"Per-fold accuracy: {[f'{a:.4f}' for a in accs]}")
    print(f"Per-fold F1 : {[f'{f:.4f}' for f in f1s]}")
    print(f"Mean accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Mean F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    return fold_metrics

class MultiTaskBinaryFFN(nn.Module): #3-layer FFN
    def __init__(self, input_dim, hidden1, hidden2, num_tasks):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_tasks)  # output: one neuron per phenotype
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits  # return logits (not probabilities)

def train(model, dataloader, optimizer, criterion, device="cuda"): #training
    model.train()
    total_loss = 0.0
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_X)               # logits
        loss = criterion(logits, batch_y)    # BCE averaged across tasks
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device="cuda", threshold=0.5, average="micro"): #evaluation function
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            logits = model(batch_X)               # logits
            all_logits.append(logits.cpu())
            all_labels.append(batch_y.cpu())
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    probs = torch.sigmoid(all_logits).numpy()
    preds = (probs > threshold).astype(int)
    labels_np = all_labels.numpy()
    acc = accuracy_score(labels_np, preds)
    f1  = f1_score(labels_np, preds, average=average, zero_division=0)
    return acc, f1

def main():
    df = _read_labels(LABELS_TXT)
    print(f"Labels shape: {df.shape}  (genes x phenotypes incl. id column (X))")
    emb_df, D = _load_embeddings(EMB_NPZ)
    print(f"Embeddings shape: {emb_df.shape}  (genes x {D})")
    X, Y, genes, tasks = _align(df, emb_df)
    print(f"After alignment: X={X.shape}, Y={Y.shape}, genes={len(genes)}, tasks={len(tasks)}")
    
    torch.manual_seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    fold_metrics = cross_validate(X, Y, n_splits=5, device=device)

if __name__ == "__main__":
    main()