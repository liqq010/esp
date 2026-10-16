"""Print the numbers quoted in the post."""
import argparse, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs")
args = ap.parse_args()
R = os.path.join(args.runs, "{}_d{}_s{}.npz")
E0, E1 = 25, 35  # epochs averaged over


def rise(x, K):  # last 10% of each epoch vs first 10%, in percent
    return (x[:, -K // 10:].mean() / x[:, :K // 10].mean() - 1) * 100


print(f"Loss change within an epoch (epochs {E0}-{E1 - 1}, last 10% vs first 10% of steps)")
for mode in ["shuffle", "noshuffle", "replace", "reverse"]:
    r = []
    for s in range(3):
        d = np.load(R.format(mode, 256, s)); K = int(d["K"])
        r.append(rise(d["batch_loss"].reshape(-1, K)[E0:E1], K))
    print(f"  {mode:10s} {np.mean(r):+6.1f}%   seeds: " + " ".join(f"{x:+.1f}%" for x in r))

print("\nShuffle, fixed set of 512 training samples")
r = []
for s in range(3):
    d = np.load(R.format("shuffle", 256, s)); K = int(d["K"]); t, l = d["evals"].T
    e, qf = t // K, (t % K) / K; m = (e >= E0) & (e < E1)
    r.append((l[m & (qf > .9)].mean() / l[m & (qf < .1)].mean() - 1) * 100)
print(f"  {np.mean(r):+6.1f}%   seeds: " + " ".join(f"{x:+.1f}%" for x in r))

print("\nShuffle, loss change within epoch 15, by width")
for w in [128, 256, 512]:
    r = []
    for s in range(3):
        d = np.load(R.format("shuffle", w, s)); K = int(d["K"])
        r.append(rise(d["batch_loss"].reshape(-1, K)[14:15], K))
    print(f"  width {w:3d} {np.mean(r):+6.1f}%")

print("\nShuffle, loss change within an epoch by batch size")
def bs_run(b, s):
    return np.load(R.format("shuffle", 256, s)) if b == 32 else np.load(os.path.join(args.runs, f"shuffle_bs{b}_s{s}.npz"))
for b in [16, 32, 64, 128]:
    r = []
    for s in range(3):
        d = bs_run(b, s); K = int(d["K"]); r.append(rise(d["batch_loss"].reshape(-1, K)[E0:E1], K))
    print(f"  batch {b:3d} (epochs {E0}-{E1 - 1}) {np.mean(r):+6.1f}%")
print("  at matched training loss (epochs whose mean loss is in the window, averaged over seeds):")
for lo, hi in [(0.19, 0.25), (0.35, 0.45)]:
    row = []
    for b in [16, 32, 64, 128]:
        r = []
        for s in range(3):
            d = bs_run(b, s); K = int(d["K"]); x = d["batch_loss"].reshape(-1, K); m = x.mean(1)
            r += [rise(x[e:e + 1], K) for e in range(len(x)) if lo <= m[e] <= hi]
        row.append(f"batch {b}: {np.mean(r):+.0f}%")
    print(f"    loss {lo}-{hi}: " + ", ".join(row))

print("\nShuffle, loss change within an epoch by beta2")
for b2 in ["0.999", "0.95", "0.9", "0.8", "0.7"]:
    r = []
    for s in range(3):
        f = R.format("shuffle", 256, s) if b2 == "0.999" else os.path.join(args.runs, f"shuffle_b2{b2}_s{s}.npz")
        d = np.load(f); K = int(d["K"]); r.append(rise(d["batch_loss"].reshape(-1, K)[E0:E1], K))
    print(f"  beta2 {b2:5s} {np.mean(r):+6.1f}%")
