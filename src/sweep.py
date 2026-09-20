"""
Run the full offline sweep over the collected traces and write a tidy CSV.

Axes: domain x layer x micro-batch size B x device count N.
Cost control: expert counts are computed once per (layer, B, ordering) and the
device pooling for every N is then just a reshape of the same count matrix.
"""
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as A

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")

BS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
NS = [2, 4, 8, 16, 32]
N_SHUFFLES = 20
DOMAINS = ["code", "prose", "math", "chat"]


def load_traces():
    tr = {}
    for d in DOMAINS:
        p = os.path.join(DATA, "trace_%s.npz" % d)
        if not os.path.exists(p):
            print("missing %s" % p, file=sys.stderr)
            continue
        z = np.load(p)
        tr[d] = z["topk"]                      # (T, L, K) int8
    if len(tr) == len(DOMAINS):
        tr["all"] = np.concatenate([tr[d] for d in DOMAINS], axis=0)
    return tr


def main():
    os.makedirs(OUT, exist_ok=True)
    tr = load_traces()
    if not tr:
        sys.exit("no traces found -- run collect.py first")
    any_d = next(iter(tr.values()))
    T0, L, K = any_d.shape
    E = int(max(a.max() for a in tr.values())) + 1
    print("L=%d K=%d E=%d  domains=%s" % (L, K, E, list(tr)))

    rows = []
    rng = np.random.default_rng(20260914)
    t0 = time.time()
    for dom, arr in tr.items():
        T = arr.shape[0]
        arr = arr.astype(np.int16)             # int8 would overflow the +offset trick
        # top-1 repetition rate, Mixtral's statistic, per layer
        rep = [A.adjacent_repeat_rate(arr[:, l, :]) for l in range(L)]
        perms = [rng.permutation(T) for _ in range(N_SHUFFLES)]
        for l in range(L):
            tl = arr[:, l, :]
            tot = np.bincount(tl.ravel().astype(np.int64), minlength=E)
            p_marg = tot / tot.sum()
            for B in BS:
                if T // B < 4:
                    continue
                c_obs = A.expert_counts(tl, B, None, E)
                c_shuf = [A.expert_counts(tl, B, pm, E) for pm in perms]
                for N in NS:
                    P = p_marg.reshape(N, E // N).sum(axis=1)
                    s_o = A.straggler(c_obs, N)
                    s_s = np.concatenate([A.straggler(c, N) for c in c_shuf])
                    rows.append({
                        "domain": dom, "layer": l, "B": B, "N": N,
                        "n_micro": int(c_obs.shape[0]),
                        "S_obs_mean": s_o.mean(), "S_obs_p5": np.percentile(s_o, 5),
                        "S_obs_p50": np.percentile(s_o, 50),
                        "S_obs_p95": np.percentile(s_o, 95), "S_obs_max": s_o.max(),
                        "S_shuf_mean": s_s.mean(), "S_shuf_p5": np.percentile(s_s, 5),
                        "S_shuf_p95": np.percentile(s_s, 95),
                        "S_iid": A.s_iid(B, K, N, P, E=E),
                        "S_iid_asym": A.s_iid_asymptotic(B, K, N),
                        "N_maxP": N * p_marg.reshape(N, E // N).sum(axis=1).max(),
                        "M_obs_mean": A.expert_max_ratio(c_obs).mean(),
                        "H_obs_mean": A.entropy_norm(c_obs).mean(),
                        "G_obs_mean": A.gini(c_obs).mean(),
                        "dead_obs_mean": A.dead_experts(c_obs).mean(),
                        "rep_top1": rep[l],
                    })
        print("%-6s done  %5.1f s" % (dom, time.time() - t0), flush=True)

    csv_path = os.path.join(OUT, "sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in r.items()})
    print("wrote %s (%d rows)" % (csv_path, len(rows)))

    # per-domain, per-layer expert marginals, for Figure 1
    marg = {}
    for dom, arr in tr.items():
        if dom == "all":
            continue
        a = arr.astype(np.int16)
        m = np.zeros((a.shape[1], E))
        for l in range(a.shape[1]):
            c = np.bincount(a[:, l, :].ravel().astype(np.int64), minlength=E)
            m[l] = c / c.sum()
        marg[dom] = m.tolist()
    json.dump({"E": E, "K": K, "L": L, "marginals": marg},
              open(os.path.join(OUT, "marginals.json"), "w"))
    print("wrote marginals.json")


if __name__ == "__main__":
    main()
