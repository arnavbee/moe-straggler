"""
Turn a measured straggler factor into bytes and milliseconds (scope note 5.5).

EVERYTHING HERE IS DERIVED, NOT MEASURED.  No cluster was involved.  The
bandwidth number is an assumption stated in the output.  This script prices what
the measured routing *implies* under a stated placement and bandwidth model.
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "out")

D_MODEL = 1024          # Granite-3.1-1B-A400M
K = 8
L_MOE = 24
BYTES = 2               # fp16
BW = 50e9               # ASSUMED per-device receive bandwidth, B/s (~400 Gb/s NIC)

# for the "why this is a cluster problem" contrast
DSV3 = dict(name="DeepSeek-V3-shaped", d=7168, k=8, layers=58)


def payload_table(B, N):
    per_tok_layer = 2 * K * D_MODEL * BYTES              # dispatch + combine
    per_tok_fwd = per_tok_layer * L_MOE
    step_total = per_tok_fwd * B
    recv_per_dev_layer = (B * K / N) * D_MODEL * BYTES   # one direction
    t_bal = recv_per_dev_layer / BW * L_MOE * 2
    return dict(per_tok_layer=per_tok_layer, per_tok_fwd=per_tok_fwd,
                step_total=step_total, recv_per_dev_layer=recv_per_dev_layer,
                t_balanced_s=t_bal)


def main():
    B, N = 4096, 8
    rows = []
    with open(os.path.join(OUT, "sweep.csv")) as f:
        rows = [r for r in csv.DictReader(f)
                if int(r["B"]) == B and int(r["N"]) == N
                and r["domain"] in ("code", "prose", "math", "chat")]
    S_obs = float(np.mean([float(r["S_obs_mean"]) for r in rows]))
    S_p95 = float(np.mean([float(r["S_obs_p95"]) for r in rows]))
    S_shuf = float(np.mean([float(r["S_shuf_mean"]) for r in rows]))
    S_iid = float(np.mean([float(r["S_iid"]) for r in rows]))

    t = payload_table(B, N)
    lines = []
    A = lines.append
    A("All-to-all payload and implied latency  (DERIVED, NOT MEASURED)")
    A("=" * 68)
    A("model        : Granite-3.1-1B-A400M, d=%d, k=%d, %d MoE layers, fp16"
      % (D_MODEL, K, L_MOE))
    A("micro-batch  : B = %d tokens,  N = %d devices, contiguous placement" % (B, N))
    A("ASSUMPTION   : per-device receive bandwidth %.0f GB/s (~400 Gb/s NIC)." % (BW / 1e9))
    A("               This is an assumption, not a measurement. No cluster was used.")
    A("")
    A("dispatch per token per layer : %8.0f B  (k*d*b)" % (K * D_MODEL * BYTES))
    A("combine  per token per layer : %8.0f B" % (K * D_MODEL * BYTES))
    A("per token per MoE layer      : %8.1f KiB" % (t["per_tok_layer"] / 1024))
    A("per token per forward pass   : %8.1f KiB  (x%d layers)"
      % (t["per_tok_fwd"] / 1024, L_MOE))
    A("per forward step, B=%-5d    : %8.2f GiB across %d collectives"
      % (B, t["step_total"] / 2**30, L_MOE * 2))
    A("inter-device share (1 - 1/N) : %8.1f %%" % (100 * (1 - 1.0 / N)))
    A("")
    A("per-device receive per layer : %8.2f MB" % (t["recv_per_dev_layer"] / 1e6))
    A("                             : %8.0f us at the assumed bandwidth"
      % (t["recv_per_dev_layer"] / BW * 1e6))
    A("balanced all-to-all per pass : %8.2f ms  (x%d layers x2)"
      % (t["t_balanced_s"] * 1e3, L_MOE))
    A("")
    A("MEASURED straggler factors at this B and N")
    A("(mean over 24 layers and the 4 WITHIN-domain traces, ~200k tokens each):")
    A("  S_obs  (what a server sees)      = %.4f  ->  %6.2f ms  (+%.2f ms skew tax)"
      % (S_obs, t["t_balanced_s"] * S_obs * 1e3, t["t_balanced_s"] * (S_obs - 1) * 1e3))
    A("  S_obs  95th pct micro-batch      = %.4f  ->  %6.2f ms  (+%.2f ms)"
      % (S_p95, t["t_balanced_s"] * S_p95 * 1e3, t["t_balanced_s"] * (S_p95 - 1) * 1e3))
    A("  S_shuf (correlation destroyed)   = %.4f  ->  %6.2f ms  (+%.2f ms)"
      % (S_shuf, t["t_balanced_s"] * S_shuf * 1e3, t["t_balanced_s"] * (S_shuf - 1) * 1e3))
    A("  S_iid  (closed form)             = %.4f  ->  %6.2f ms  (+%.2f ms)"
      % (S_iid, t["t_balanced_s"] * S_iid * 1e3, t["t_balanced_s"] * (S_iid - 1) * 1e3))
    A("")
    A("  attributable to expert popularity + placement : %.2f ms  (S_shuf - 1)"
      % (t["t_balanced_s"] * (S_shuf - 1) * 1e3))
    A("  attributable to token-to-token correlation    : %.2f ms  (S_obs - S_shuf)"
      % (t["t_balanced_s"] * (S_obs - S_shuf) * 1e3))
    A("")
    A("Scale contrast - why this is a cluster problem and not a laptop problem:")
    A("  %s: d=%d -> %.0f KiB per token per MoE layer, %.1fx this model's traffic"
      % (DSV3["name"], DSV3["d"], 2 * DSV3["k"] * DSV3["d"] * BYTES / 1024,
         DSV3["d"] / float(D_MODEL)))
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT, "latency_model.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
