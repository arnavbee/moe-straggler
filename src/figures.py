"""
Figures for the straggler experiment.  Matplotlib defaults, no seaborn.
Every caption states B, N, token count and dtype.
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "out")
FIG = os.path.join(HERE, "figures")

DOM_COLORS = {"code": "#1f77b4", "prose": "#2ca02c", "math": "#d62728", "chat": "#ff7f0e"}


def load():
    rows = []
    with open(os.path.join(OUT, "sweep.csv")) as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                if k != "domain":
                    r[k] = float(v)
            r["layer"] = int(r["layer"]); r["B"] = int(r["B"]); r["N"] = int(r["N"])
            rows.append(r)
    return rows


REAL = ["code", "prose", "math", "chat"]


def sel(rows, **kw):
    """domain="__within__" selects the four single-domain rows, never the pooled set.

    This matters: the pooled "all" trace concatenates the four domains, so its
    contiguous micro-batches are domain-homogeneous while its SHUFFLED ones are
    domain-mixed.  S_obs - S_shuf on the pooled set therefore measures domain
    composition, not token-to-token correlation.  Every correlation claim here
    uses within-domain rows.
    """
    out = rows
    for k, v in kw.items():
        if k == "domain" and v == "__within__":
            out = [r for r in out if r["domain"] in REAL]
            continue
        if isinstance(v, (list, tuple, range)):
            out = [r for r in out if r[k] in v]
        else:
            out = [r for r in out if r[k] == v]
    return out


def agg(rows, key):
    """Mean of `key` grouped by B, over whatever rows are passed."""
    bs = sorted({r["B"] for r in rows})
    return bs, [np.mean([r[key] for r in rows if r["B"] == b]) for b in bs]


# ---------------------------------------------------------------- Figure 1
def fig1(meta):
    E = meta["E"]
    marg = meta["marginals"]
    doms = [d for d in ["code", "prose", "math", "chat"] if d in marg]
    fig, axes = plt.subplots(1, len(doms), figsize=(4.0 * len(doms), 3.4), sharey=True)
    if len(doms) == 1:
        axes = [axes]
    for ax, d in zip(axes, doms):
        m = np.array(marg[d])                      # (L, E)
        srt = -np.sort(-m, axis=1)                 # descending within each layer
        for l in range(srt.shape[0]):
            ax.plot(np.arange(1, E + 1), srt[l], color="0.75", lw=0.7, zorder=1)
        ax.plot(np.arange(1, E + 1), np.median(srt, axis=0), color=DOM_COLORS[d],
                lw=2.2, zorder=3, label="median over 24 layers")
        ax.axhline(1.0 / E, ls="--", color="k", lw=1.0, zorder=2,
                   label="uniform 1/E = %.3f" % (1.0 / E))
        ax.set_yscale("log")
        ax.set_title(d)
        ax.set_xlabel("expert rank (sorted)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("share of top-k assignments")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Fig 1 - expert popularity is not uniform (context, not the result)\n"
                 "Granite-3.1-1B-A400M, E=32, k=8, 24 MoE layers, ~200k tokens/domain, "
                 "fp16 on MPS; faint lines are individual layers", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(os.path.join(FIG, "fig1_expert_marginals.png"), dpi=160)
    plt.close(fig)
    print("fig1 written")


# ---------------------------------------------------------------- Figure 2
def fig2(rows, N=8, domain="__within__"):
    groups = [("early  (layers 0-7)", range(0, 8)),
              ("mid    (layers 8-15)", range(8, 16)),
              ("late   (layers 16-23)", range(16, 24))]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.9), sharey=True)
    for ax, (title, lay) in zip(axes, groups):
        rs = sel(rows, domain=domain, N=N, layer=list(lay))
        bs, s_obs = agg(rs, "S_obs_mean")
        _, s_shuf = agg(rs, "S_shuf_mean")
        _, s_iid = agg(rs, "S_iid")
        _, p5 = agg(rs, "S_obs_p5")
        _, p95 = agg(rs, "S_obs_p95")
        ax.fill_between(bs, p5, p95, color="#1f77b4", alpha=0.16,
                        label="$S_{obs}$ 5-95th pct across micro-batches")
        ax.plot(bs, s_obs, "-o", color="#1f77b4", lw=2, ms=4, label="$S_{obs}$ observed")
        ax.plot(bs, s_shuf, "--s", color="#d62728", lw=1.8, ms=4,
                label="$S_{shuf}$ shuffled (marginals kept, correlation destroyed)")
        ax.plot(bs, s_iid, ":", color="0.35", lw=2.2,
                label="$S_{iid}$ closed form, measured marginal")
        _, s_asym = agg(rs, "S_iid_asym")
        ax.plot(bs, s_asym, "-.", color="#9467bd", lw=2.0,
                label="$S_{iid}$ assuming UNIFORM marginals\n(the \"batch harder\" prediction)")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("micro-batch size B (tokens)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        ax.axhline(1.0, color="k", lw=0.9)
        ax.set_ylim(0.98, 1.82)
        # annotate the gap at B=4096: observed vs the uniform-marginal prediction
        if 4096 in bs:
            i = bs.index(4096)
            ax.annotate("", xy=(4096, s_obs[i]), xytext=(4096, s_asym[i]),
                        arrowprops=dict(arrowstyle="<->", lw=1.3, color="#7f0000"))
            ax.text(4600, (s_obs[i] + s_asym[i]) / 2,
                    "B=4096\n$S_{obs}$=%.2f vs %.2f\n%.0fx the predicted\nexcess" %
                    (s_obs[i], s_asym[i], (s_obs[i] - 1) / (s_asym[i] - 1)),
                    fontsize=7.6, color="#7f0000", va="center")
    axes[0].set_ylabel("straggler factor $S = \\max_j L_j / (Bk/N)$")
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, fontsize=8.2, loc="lower center", ncol=5,
               frameon=False, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Fig 2 - does the all-to-all straggler batch away?  "
                 "N=8 devices (4 experts each, contiguous placement), k=8, fp16/MPS.\n"
                 "Mean of the four WITHIN-domain curves (~200k tokens each); the pooled "
                 "set is excluded because shuffling it also destroys domain composition.",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0.11, 1, 0.85))
    fig.savefig(os.path.join(FIG, "fig2_straggler_vs_batch.png"), dpi=160,
                bbox_inches="tight")
    plt.close(fig)
    print("fig2 written")


# ---------------------------------------------------------------- Figure 3
def fig3(rows, B=4096, N=8):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for d in ["code", "prose", "math", "chat"]:
        rs = sorted(sel(rows, domain=d, B=B, N=N), key=lambda r: r["layer"])
        if not rs:
            continue
        ax.plot([r["layer"] for r in rs], [r["S_obs_mean"] for r in rs],
                "-o", ms=3.5, color=DOM_COLORS[d], label="%s  $S_{obs}$" % d)
    lays = sorted({r["layer"] for r in rows})
    shuf = [np.mean([r["S_shuf_mean"] for r in sel(rows, domain="__within__", B=B, N=N, layer=l)])
            for l in lays]
    ax.plot(lays, shuf, "--", color="0.45", lw=1.8,
            label="within-domain $S_{shuf}$ (the marginal+placement floor)")
    ax.axhline(1.0298, ls=":", color="k", lw=1.6,
               label="naive uniform-marginal $S_{iid}$ = 1.030")
    ax.set_xlabel("MoE layer index")
    ax.set_ylabel("straggler factor $S$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Fig 3 - where in the stack the skew tax is paid\n"
                 "B=%d tokens, N=8 devices, k=8, contiguous placement, "
                 "~200k tokens/domain, fp16/MPS" % B, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_straggler_by_layer.png"), dpi=160)
    plt.close(fig)
    print("fig3 written")


# ---------------------------------------------------------------- Figure 4
def fig4(rows, B=4096, domain="__within__"):
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    ns = sorted({r["N"] for r in rows})
    obs = [np.mean([r["S_obs_mean"] for r in sel(rows, domain=domain, B=B, N=n)]) for n in ns]
    shf = [np.mean([r["S_shuf_mean"] for r in sel(rows, domain=domain, B=B, N=n)]) for n in ns]
    iid = [np.mean([r["S_iid"] for r in sel(rows, domain=domain, B=B, N=n)]) for n in ns]
    ax.plot(ns, obs, "-o", color="#1f77b4", lw=2, label="$S_{obs}$ device-level")
    ax.plot(ns, shf, "--s", color="#d62728", lw=1.8, label="$S_{shuf}$")
    ax.plot(ns, iid, ":", color="0.35", lw=2, label="$S_{iid}$")
    m = np.mean([r["M_obs_mean"] for r in sel(rows, domain=domain, B=B, N=8)])
    ax.axhline(m, color="0.2", ls="-.", lw=1.4,
               label="expert-level $M$ = %.3f (ceiling, = S at N=E=32)" % m)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("number of devices N (E/N experts pooled per device)")
    ax.set_ylabel("straggler factor $S$")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    ax.set_title("Fig 4 - how much imbalance survives pooling experts onto devices\n"
                 "B=%d tokens, k=8, E=32, contiguous placement, mean over 4 domains and\n"
                 "24 layers.  N=32 is one expert per device, where S = M." % B,
                 fontsize=9.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_pooling.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("fig4 written")


def fig5(rows, N=8):
    """The decomposition: how fast does each component of the excess decay?"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.3))
    bs = sorted({r["B"] for r in rows})
    for d in DOM_COLORS:
        y = [np.mean([r["S_obs_mean"] - r["S_shuf_mean"]
                      for r in sel(rows, domain=d, N=N, B=b)]) for b in bs]
        ax1.plot(bs, y, "-o", ms=4, color=DOM_COLORS[d], label="%s (within domain)" % d)
    y = [np.mean([r["S_obs_mean"] - r["S_shuf_mean"]
                  for r in sel(rows, domain="all", N=N, B=b)]) for b in bs]
    ax1.plot(bs, y, "-s", ms=4, color="k", lw=2,
             label="pooled = correlation + domain composition")
    ref = y[0] * (np.array(bs, float) / bs[0]) ** -0.5
    ax1.plot(bs, ref, ":", color="0.5", lw=1.6, label="$1/\\sqrt{B}$ reference")
    ax1.set_xscale("log", base=2); ax1.set_yscale("log")
    ax1.set_xlabel("micro-batch size B (tokens)")
    ax1.set_ylabel("$S_{obs} - S_{shuf}$   (the correlation term)")
    ax1.set_title("(a) the correlation term does batch away", fontsize=10)
    ax1.grid(alpha=0.25); ax1.legend(fontsize=7.5)

    for d in DOM_COLORS:
        y = [np.mean([r["S_shuf_mean"] - 1.0
                      for r in sel(rows, domain=d, N=N, B=b)]) for b in bs]
        ax2.plot(bs, y, "-o", ms=4, color=DOM_COLORS[d], label=d)
    ax2.plot(bs, [np.mean([r["S_iid"] - 1.0 for r in sel(rows, domain="__within__", N=N, B=b)])
                  for b in bs], ":", color="0.35", lw=2, label="$S_{iid} - 1$ closed form")
    ax2.plot(bs, [np.mean([r["S_iid_asym"] - 1.0 for r in sel(rows, domain="__within__", N=N, B=b)])
                  for b in bs], "-.", color="0.6", lw=1.6,
             label="naive uniform-marginal form")
    ax2.set_xscale("log", base=2); ax2.set_yscale("log")
    ax2.set_xlabel("micro-batch size B (tokens)")
    ax2.set_ylabel("$S_{shuf} - 1$   (the marginal+placement floor)")
    ax2.set_title("(b) the floor does not", fontsize=10)
    ax2.grid(alpha=0.25); ax2.legend(fontsize=7.5)
    fig.suptitle("Fig 5 - the decomposition. N=8, k=8, mean over 24 layers. "
                 "Left: correlation vanishes with batching. Right: expert popularity does not.",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(os.path.join(FIG, "fig5_decomposition.png"), dpi=160)
    plt.close(fig)
    print("fig5 written")


def fig6():
    """Placement, not batching, is where the straggler actually goes away."""
    import json as _json
    pl = _json.load(open(os.path.join(OUT, "placement.json")))
    names = list(pl.keys())
    short = ["mixed traffic\n(random half ->\nheld-out half)",
             "unseen domains\n(code+prose ->\nmath+chat)",
             "code -> math", "chat -> code",
             "same domain\n(math 1st half ->\n2nd half)"]
    x = np.arange(len(names)); w = 0.26
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ax.bar(x - w, [pl[n]["S_contig"] for n in names], w, label="contiguous placement",
           color="#d62728")
    ax.bar(x, [pl[n]["S_heldout"] for n in names], w,
           label="greedy placement fitted on held-out traffic", color="#1f77b4")
    ax.bar(x + w, [pl[n]["S_oracle"] for n in names], w,
           label="greedy placement fitted in-sample (oracle)", color="#7fbfff")
    for i, n in enumerate(names):
        ax.text(i, pl[n]["S_heldout"] + 0.02, "%.0f%%\nrecovered" % (100 * pl[n]["recovered"]),
                ha="center", fontsize=7.5)
    ax.axhline(1.0, color="k", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7.8)
    ax.set_ylabel("straggler factor $S$")
    ax.set_ylim(1.0, 1.75)
    ax.legend(fontsize=8.2); ax.grid(alpha=0.25, axis="y")
    ax.set_title("Fig 6 - the straggler is a placement problem, not a batching problem\n"
                 "N=8, B->inf (marginal loads), mean over 24 layers.  Greedy = "
                 "largest-expert-first into the least-loaded device, E/N per device.\n"
                 "Placement fitted on the right traffic mix removes most of the excess; "
                 "fitted on the wrong domain it does not.", fontsize=9.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_placement.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("fig6 written")


def main():
    os.makedirs(FIG, exist_ok=True)
    rows = load()
    meta = json.load(open(os.path.join(OUT, "marginals.json")))
    fig1(meta)
    fig2(rows)
    fig3(rows)
    fig4(rows)
    fig5(rows)
    fig6()


if __name__ == "__main__":
    main()
