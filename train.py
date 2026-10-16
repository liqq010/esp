"""Epochal sawtooth experiment: a small byte-level GPT trained for many epochs on a TinyStories slice.

Records, for every step: the batch training loss (before the update), and for
every sample in the batch its per-sample loss and its gap Delta (steps since it
was last trained on). Also evaluates a fixed subset of training samples every
`eval_every` steps.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--mode", default="shuffle", choices=["shuffle", "noshuffle", "replace", "reverse"])
p.add_argument("--n", type=int, default=4096)       # number of sequences
p.add_argument("--seq", type=int, default=128)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--d", type=int, default=256)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--beta1", type=float, default=0.9)
p.add_argument("--beta2", type=float, default=0.999)
p.add_argument("--wd", type=float, default=0.0)
p.add_argument("--eval_every", type=int, default=8)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--data", default="data/TinyStoriesV2-GPT4-valid.txt")
p.add_argument("--out", required=True)
a = p.parse_args()

torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda"
raw = np.frombuffer(open(a.data, "rb").read(), dtype=np.uint8)
L = a.seq + 1
data = torch.tensor(raw[: a.n * L].reshape(a.n, L).astype(np.int64), device=dev)
K = a.n // a.bs


class GPT(nn.Module):
    def __init__(s, d, nl, T):
        super().__init__()
        s.tok = nn.Embedding(256, d); s.pos = nn.Embedding(T, d)
        layer = nn.TransformerEncoderLayer(d, d // 64, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        s.tf = nn.TransformerEncoder(layer, nl)
        s.ln = nn.LayerNorm(d); s.head = nn.Linear(d, 256, bias=False)
        s.register_buffer("mask", nn.Transformer.generate_square_subsequent_mask(T), persistent=False)

    def forward(s, x):
        T = x.shape[1]
        h = s.tok(x) + s.pos(torch.arange(T, device=x.device))
        h = s.tf(h, mask=s.mask[:T, :T], is_causal=True)
        return s.head(s.ln(h))


model = GPT(a.d, a.layers, a.seq).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(a.beta1, a.beta2), weight_decay=a.wd)


def per_sample_loss(idx):
    x, y = data[idx, :-1], data[idx, 1:]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x)
    l = F.cross_entropy(logits.float().reshape(-1, 256), y.reshape(-1), reduction="none")
    return l.view(len(idx), -1).mean(1)


g = torch.Generator().manual_seed(a.seed)
base = torch.randperm(a.n, generator=g)
eval_idx = torch.randperm(a.n, generator=g)[:512].to(dev)
last = torch.full((a.n,), -1, dtype=torch.long, device=dev)

steps = a.epochs * K
batch_loss = np.zeros(steps, np.float32)
rec_gap, rec_loss = [], []
evals = []
t0 = time.time()
for ep in range(a.epochs):
    if a.mode == "shuffle":
        order = torch.randperm(a.n, generator=g)
    elif a.mode == "noshuffle":
        order = base
    elif a.mode == "reverse":
        order = base if ep % 2 == 0 else base.flip(0)
    else:
        order = torch.randint(0, a.n, (a.n,), generator=g)
    order = order.to(dev)
    for q in range(K):
        t = ep * K + q
        if t % a.eval_every == 0:
            model.eval()
            with torch.no_grad():
                evals.append((t, per_sample_loss(eval_idx).mean().item()))
            model.train()
        idx = order[q * a.bs:(q + 1) * a.bs]
        ls = per_sample_loss(idx)
        loss = ls.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        batch_loss[t] = loss.item()
        gap = torch.where(last[idx] >= 0, t - last[idx], torch.full_like(idx, -1))
        rec_gap.append(gap.cpu()); rec_loss.append(ls.detach().float().cpu())
        last[idx] = t
    print(f"ep {ep} loss {batch_loss[t-K+1:t+1].mean():.4f} eval {evals[-1][1]:.4f} {time.time()-t0:.0f}s", flush=True)

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
np.savez(a.out, batch_loss=batch_loss, gap=torch.stack(rec_gap).numpy(), sample_loss=torch.stack(rec_loss).numpy(),
         evals=np.array(evals), K=K, args=json.dumps(vars(a)))
