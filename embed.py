import os, csv #CSV output
import numpy #numeric operations
import torch
import esm
from Bio import SeqIO

FASTA = "longest_per_gene_human.fa"                      # change for each FASTA file
REPR_LAYER=6
BATCH_SIZE = 8
MAX_LEN = 1022                                      # ESM2 AA limit
CHUNK_OVERLAP = 128
NAME = os.path.splitext(os.path.basename(FASTA))[0]
OUT_CSV = f"embeddings_{NAME}.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()  # 320 dim model
model.eval().to(device)
batch_converter = alphabet.get_batch_converter()
embed_dim = model.embed_dim

def embed_batch_mean(seq_batch):
    labels, strs, toks = batch_converter(seq_batch)
    toks = toks.to(device)
    with torch.no_grad():
        out = model(toks, repr_layers=[REPR_LAYER], return_contacts = False) #each tensor is 3D table of batch size x sequence length x embedding dimension (B x L x D)
        rep = out["representations"][REPR_LAYER]
    results = []
    for j,(hdr,seq) in enumerate(seq_batch):
        L = len(seq)
        v = rep[j,1:1+L].mean(0).detach().to("cpu").numpy()
        results.append(v)
    return results

def embed_long_with_overlap(seq):
    Lbig = len(seq)
    stride = MAX_LEN - CHUNK_OVERLAP

    sum_arr = numpy.zeros((Lbig, embed_dim), dtype=numpy.float32)
    cnt_arr = numpy.zeros((Lbig,), dtype=numpy.int32)
    starts = list(range(0, Lbig, stride))

    for start in starts:
        end = min(start + MAX_LEN, Lbig)
        window_seq = seq[start:end]

        labels, strs, toks = batch_converter([("win", window_seq)])
        toks = toks.to(device)
        with torch.no_grad():
            out = model(toks, repr_layers=[REPR_LAYER], return_contacts = False)
            rep = out["representations"][REPR_LAYER][0, 1:1+len(window_seq)].detach().to("cpu").numpy()

        sum_arr[start:end] += rep
        cnt_arr[start:end] += 1

    sum_arr /= cnt_arr[:,None]
    return sum_arr.mean(0)

pairs=[]
for rec in SeqIO.parse(FASTA, "fasta"):
    hdr = rec.description.strip()
    seq = str(rec.seq).replace("*","")
    pairs.append((hdr, seq))

all_ids, all_vecs = [], []
short_batch = []
for hdr, seq in pairs:
    if len(seq) <= MAX_LEN:
        short_batch.append((hdr, seq))
        if len(short_batch) == BATCH_SIZE:
            res = embed_batch_mean(short_batch)
            for (hdr2,_), v in zip(short_batch, res):
                all_ids.append(hdr2.split()[0])
                all_vecs.append(v)
            short_batch.clear()
    else:
        v = embed_long_with_overlap(seq)
        all_ids.append(hdr.split()[0])
        all_vecs.append(v)

if short_batch:
    res = embed_batch_mean(short_batch)
    for (hdr2,_), v in zip(short_batch, res):
        all_ids.append(hdr2.split()[0])
        all_vecs.append(v)

numpy.savez(f"{NAME}_embeddings.npz", ids = numpy.array(all_ids), embeddings = numpy.array(all_vecs))

with open(OUT_CSV, "w", newline = "") as f:
    w = csv.writer(f)
    w.writerow(["id"]+[f"emb_{k:03d}" for k in range(embed_dim)])
    for i,vec in zip(all_ids, all_vecs):
        w.writerow([i]+vec.tolist())