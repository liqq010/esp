"""Make every figure of the blog post from the runs produced by run_all.sh."""
import argparse, os
import numpy as np, matplotlib, matplotlib.transforms
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs")
ap.add_argument("--out", default="figs")
args = ap.parse_args()
OUT = args.out
os.makedirs(OUT, exist_ok=True)
R = os.path.join(args.runs, "{}_d{}_s{}.npz")
C = {"shuffle": "#2a78d6", "noshuffle": "#eb6834", "reverse": "#1baf7a", "replace": "#6250d6",
     128: "#eb6834", 256: "#2a78d6", 512: "#1baf7a", "train": "#2a78d6", "fixed": "#eb6834"}
NAME = {"shuffle": "Shuffle each epoch", "noshuffle": "No shuffle", "replace": "With replacement",
        "reverse": "Reverse each epoch"}
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.titlesize": 12, "axes.titlecolor": INK,
})
SM = 4  # steps averaged per plotted point
E0, E1 = 30, 34  # zoom window (epochs)


def load(mode, d=256, s=0):
    return np.load(R.format(mode, d, s))


def smooth(bl, K):
    s = bl[: len(bl) // SM * SM].reshape(-1, SM).mean(1)
    return np.arange(len(s)) * SM / K, s


def save(fig, name):
    fig.savefig(f"{OUT}/{name}"); plt.close(fig)


# 1. the phenomenon
d = load("shuffle"); K = int(d["K"]); t, s = smooth(d["batch_loss"], K)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})
ax[0].plot(t, s, color=C["shuffle"], lw=1); ax[0].set_ylim(0, 2.0)
ax[0].axvspan(E0, E1, color=GRID, alpha=.6, lw=0)
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training loss"); ax[0].set_title("40 epochs")
m = (t >= E0) & (t <= E1)
ax[1].plot(t[m], s[m], color=C["shuffle"]); ax[1].set_xticks(range(E0, E1 + 1))
ax[1].set_xlabel("epoch"); ax[1].set_title(f"zoom: epochs {E0}–{E1}")
save(fig, "sawtooth.png")

# README teaser: full loss curve with an inset zooming into epochs Z0-Z1, plus the mean gap there
from matplotlib.patches import ConnectionPatch, Rectangle, FancyBboxPatch
Z0, Z1 = 20, 24  # zoom window (epochs)
d = load("shuffle"); K = int(d["K"]); t, s = smooth(d["batch_loss"], K)
_, gm = smooth(d["gap"].mean(1).astype(float), K)
m = (t >= Z0) & (t <= Z1)
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.plot(t, s, color=C["shuffle"], lw=0.9); ax.set_xlim(0, 40); ax.set_ylim(0, 2.0)
ax.set_xlabel("epoch", labelpad=1); ax.set_ylabel("training loss")
lo, hi = s[m].min() - 0.03, s[m].max() + 0.03
ax.add_patch(Rectangle((Z0, lo), Z1 - Z0, hi - lo, fill=False, ec=MUTED, lw=0.9))
# inset: zoomed loss (top) and mean gap (bottom), framed together
BOX = (0.45, 0.40, 0.54, 0.58)  # frame in axes coords: x, y, w, h
ax.add_patch(FancyBboxPatch(BOX[:2], BOX[2], BOX[3], boxstyle="round,pad=0,rounding_size=0.01",
                            transform=ax.transAxes, fc="white", ec=MUTED, lw=0.9, zorder=3))
a1 = ax.inset_axes([0.535, 0.705, 0.44, 0.21]); a2 = ax.inset_axes([0.535, 0.475, 0.44, 0.17], sharex=a1)
for a_ in (a1, a2):
    a_.set_zorder(4); a_.set_xlim(Z0, Z1); a_.margins(y=0.08); a_.tick_params(labelsize=8.5)
    a_.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
a1.plot(t[m], s[m], color=C["shuffle"], lw=1.2); a1.tick_params(labelbottom=False)
a1.set_ylabel("loss", fontsize=9)
a2.plot(t[m], gm[m], color=C["noshuffle"], lw=1.2)
qq = np.arange(Z0 * K, Z1 * K + 1)
a2.plot(qq / K, (K + 1) / 2 + qq % K, color=INK, lw=0.9, ls="--")
a2.set_xticks(range(Z0, Z1 + 1)); a2.set_ylabel("gap Δ", fontsize=9)
a1.set_title(f"Zoom: epochs {Z0}–{Z1}", fontsize=9, loc="left")
a2.set_title("orange: measured, dashed: K/2 + q", fontsize=7.5, loc="right", color=MUTED, pad=3)
for xd, xa in ((Z0, BOX[0]), (Z1, BOX[0] + BOX[2])):  # rectangle top corners -> frame bottom corners
    fig.add_artist(ConnectionPatch(xyA=(xd, hi), coordsA=ax.transData, xyB=(xa, BOX[1]), coordsB=ax.transAxes,
                                   color=MUTED, lw=0.8, ls=":"))
fig.savefig(f"{OUT}/teaser.png", pad_inches=0.02); plt.close(fig)

# 2. theory: gap vs step
K = 128; q = np.arange(3 * K); qe = q % K
fig, ax = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
band = {"shuffle": (qe + 1, qe + K, qe + (K + 1) / 2), "noshuffle": (np.full_like(q, K),) * 3,
        "reverse": (1 + 2 * qe,) * 3}
lo, hi = np.log(0.9) / np.log(1 - 1 / K), np.log(0.1) / np.log(1 - 1 / K)
band["replace"] = (np.full(len(q), lo), np.full(len(q), hi), np.full(len(q), K))
for i, mo in enumerate(NAME):
    a, b, mu = band[mo]
    ax[i].fill_between(q / K, a, b, color=C[mo], alpha=.18, lw=0, step="post")
    ax[i].plot(q / K, mu, color=C[mo], drawstyle="steps-post")
    ax[i].set_title(NAME[mo]); ax[i].set_xlabel("epoch"); ax[i].set_xticks([0, 1, 2, 3])
ax[0].set_ylabel(r"gap $\Delta$ (steps)")
ax[0].text(2.05, 2.15 * K, "range", color=MUTED, fontsize=9); ax[0].text(2.05, 1.12 * K, "mean", color=C["shuffle"], fontsize=9)
ax[list(NAME).index("replace")].text(0.05, 2.05 * K, "10–90% range", color=MUTED, fontsize=9)
save(fig, "gap_theory.png")

# 3. four sampling schemes
fig, ax = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
for i, mo in enumerate(NAME):
    d = load(mo); K = int(d["K"]); t, s = smooth(d["batch_loss"], K); m = (t >= E0) & (t <= E1)
    ax[i].plot(t[m], s[m], color=C[mo], lw=1.3); ax[i].set_title(NAME[mo]); ax[i].set_xlabel("epoch")
    ax[i].set_xticks(range(E0, E1 + 1))
ax[0].set_ylabel("training loss"); ax[0].set_ylim(0.12, 0.42)
save(fig, "ablation.png")


# 4. forgetting curve
def gap_curve(d, e0, e1, sel=None):
    K = int(d["K"]); g = d["gap"][e0 * K:e1 * K]; l = d["sample_loss"][e0 * K:e1 * K]
    if sel is not None:
        qq = np.arange(g.shape[0]) % K; keep = sel(qq / K); g, l = g[keep], l[keep]
    g, l = g.ravel(), l.ravel(); ok = g > 0; g, l = g[ok], l[ok]
    bins = np.unique(np.round(np.geomspace(1, 4 * K, 36)).astype(int))
    c = np.digitize(g, bins); x, y = [], []
    for k in range(1, len(bins)):
        if (c == k).sum() >= 50: x.append(g[c == k].mean()); y.append(l[c == k].mean())
    return np.array(x), np.array(y)


fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
d = load("shuffle"); K = int(d["K"])
for (lab, f), col in zip([("first third of epoch", lambda r: r < 1 / 3), ("middle third", lambda r: (r >= 1 / 3) & (r < 2 / 3)),
                          ("last third", lambda r: r >= 2 / 3)], ["#2a78d6", "#eb6834", "#1baf7a"]):
    x, y = gap_curve(d, 25, 35, f); ax[0].plot(x, y, "o-", color=col, ms=4, label=lab)
ax[0].set_xscale("log"); ax[0].set_xlabel(r"gap $\Delta$ (steps)"); ax[0].set_ylabel("per-sample loss")
ax[0].set_title("Shuffle: loss depends on Δ, not on position"); ax[0].legend(fontsize=9)
for mo in ["shuffle", "replace"]:
    x, y = gap_curve(load(mo), 25, 35); ax[1].plot(x, y, "o-", color=C[mo], ms=4, label=NAME[mo])
ax[1].axvline(K, color=MUTED, ls=":", lw=1); ax[1].text(K * 1.07, 0.37, "K", color=MUTED)
ax[1].set_xscale("log"); ax[1].set_xlabel(r"gap $\Delta$ (steps)"); ax[1].set_title("Forgetting curve ℓ(Δ), two sampling schemes")
ax[1].legend(fontsize=9, loc="upper left"); ax[1].set_ylim(0.05, 0.45)
save(fig, "forgetting.png")

# 5. training loss vs fixed subset
d = load("shuffle"); K = int(d["K"]); t, s = smooth(d["batch_loss"], K); ev = d["evals"]
m = (t >= E0) & (t <= E1); me = (ev[:, 0] / K >= E0) & (ev[:, 0] / K <= E1)
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(t[m], s[m], color=C["train"], lw=1.3, label="training loss (current batch)")
ax.plot(ev[me, 0] / K, ev[me, 1], color=C["fixed"], lw=1.8, label="loss on a fixed set of 512 training samples")
ax.set_xticks(range(E0, E1 + 1)); ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(fontsize=9, loc="upper left")
ax.set_ylim(0.1, 0.32)
save(fig, "fixed_subset.png")


# 6. capacity
def rise(bl, K):
    x = bl.reshape(-1, K); return (x[:, -K // 10:].mean(1) / x[:, :K // 10].mean(1) - 1) * 100


fig, ax = plt.subplots(figsize=(7, 3.4))
for w in [128, 256, 512]:
    r = np.stack([rise(load("shuffle", w, s)["batch_loss"], 128) for s in range(3)])
    e = np.arange(r.shape[1]) + 1
    ax.fill_between(e, r.min(0), r.max(0), color=C[w], alpha=.2, lw=0)
    ax.plot(e, r.mean(0), color=C[w], label=f"width {w}")
ax.axhline(0, color=MUTED, lw=1); ax.set_xlabel("epoch"); ax.set_ylabel("loss rise within epoch (%)")
ax.legend(fontsize=9, loc="upper left")
save(fig, "capacity.png")
print(f"figures written to {OUT}/")
