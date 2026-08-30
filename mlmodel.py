import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
# -----------------------------
# Model: 3-layer FFN
# -----------------------------
class MultiTaskBinaryFFN(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, num_tasks):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_tasks)  # output: one neuron per phenotype
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # sigmoid for probabilities
        return x  # shape: (batch_size, num_tasks)
# -----------------------------
# Training function
# -----------------------------
def train(model, dataloader, optimizer, device="cuda"):
    model.train()
    criterion = nn.BCELoss()   # expects sigmoid outputs in [0,1]
    total_loss = 0.0
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        probs = model(batch_X)              # forward pass → probabilities
        loss = criterion(probs, batch_y)    # BCE averaged across tasks
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)
# -----------------------------
# Evaluation function
# -----------------------------
def evaluate(model, dataloader, device="cuda", average="macro"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            probs = model(batch_X)                 # probabilities
            preds = (probs > 0.5).float().cpu()    # threshold at 0.5
            all_preds.append(preds)
            all_labels.append(batch_y.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average=average)
    return acc, f1