"""
Model/03_plot_attention.py
Plots the attention weights saved by 01_attention_fusion.py
(Model/attention_weights_test.npz) -> Model/attention_plot.png

Left : average attention per month, split by TRUE label (mining vs non-mining).
       Dashed line = 1/12, what "no preference" would look like.
Right: the single most confident CORRECT example of each class.
Run 01_attention_fusion.py first. Works from any folder.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "attention_weights_test.npz"))
w = d["attention_weights"]            # (n_test, T)
y = d["y_true"].astype(int)
p = d["proba"].ravel()
T = w.shape[1]
months = np.arange(1, T + 1)
MINE, OTHER = "#C0562B", "#2E8B57"

fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=150)

for lab, col, name in ((1, MINE, "Mining"), (0, OTHER, "Non-mining")):
    m = y == lab
    mean, sd = w[m].mean(0), w[m].std(0)
    ax[0].plot(months, mean, color=col, lw=2.2, marker="o", label=f"{name} (n={m.sum()})")
    ax[0].fill_between(months, mean - sd, mean + sd, color=col, alpha=0.15)
ax[0].axhline(1 / T, color="grey", ls="--", lw=1, label="Uniform (1/%d)" % T)
ax[0].set(title="Average attention per month", xlabel="Month in 12-month window",
          ylabel="Attention weight", xticks=months)
ax[0].legend(frameon=False)

for lab, col, name in ((1, MINE, "Mining"), (0, OTHER, "Non-mining")):
    ok = np.where((y == lab) & ((p >= 0.5) == bool(lab)))[0]
    if len(ok) == 0:
        continue
    conf = p[ok] if lab == 1 else 1 - p[ok]
    i = ok[np.argmax(conf)]
    ax[1].bar(months + (0.2 if lab else -0.2), w[i], width=0.4, color=col,
              label=f"{name} site, P(mining)={p[i]:.2f}")
ax[1].axhline(1 / T, color="grey", ls="--", lw=1)
ax[1].set(title="Most confident correct example per class", xlabel="Month in 12-month window",
          ylabel="Attention weight", xticks=months)
ax[1].legend(frameon=False)

for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = os.path.join(HERE, "attention_plot.png")
fig.savefig(out)
print("Saved", out)