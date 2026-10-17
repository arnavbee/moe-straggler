# Is the MoE all-to-all straggler sampling noise, or is it structural?

A measurement of how expert-routing load imbalance decays with micro-batch size
in a real Mixture-of-Experts model, with a shuffle control and a closed-form
independent-sampling null.

Everything here runs on one 8 GB M1 MacBook Air. No cluster, no GPU rental, no
API keys, no paid services. Total cost: a 2.67 GB model download.

---

## The question

Expert-parallel MoE inference pays an all-to-all every layer, and a collective
finishes when its slowest participant finishes. Define the **straggler factor**

```
S = max_j L_j / (B·k/N)
```

where `L_j` is the number of (token, expert) assignments landing on device `j`,
`B` is the micro-batch size in tokens, `k` the top-k, and `N` the number of
devices. `S` is the multiplier on all-to-all time *and* on the overloaded
device's expert GEMM time, and every other device blocks on it.

The folk answer to MoE imbalance is *"batch harder, it averages out."* Under
independent sampling that answer is correct and quantitatively strong: excess
imbalance decays as `1/√(B·k)`. But routing is not independent — Mixtral
measured 23.6–28.4% of adjacent tokens sharing a first-choice expert at layer 15
(arXiv:2401.04088, Table 5), against a 12.5% random baseline.

**So: does `S` actually fall at the `1/√B` rate independence predicts, or does
token-to-token correlation put a floor under it that you cannot batch away?**

## The control that makes this an experiment rather than a plot

Take the exact same multiset of routing assignments and **shuffle which token
lands in which micro-batch**. Shuffling destroys token-to-token correlation
while preserving every expert's marginal popularity *exactly*. Then:

| quantity | what it isolates |
|---|---|
| `S_shuf(B) − 1` | skew from **expert popularity + placement**. A real floor; converges to `N·max_j P_j`. |
| `S_obs(B) − S_shuf(B)` | skew from **token-to-token correlation**. Goes to 0 as B→∞ if correlation is short-range. |
| `S_shuf(B) − S_iid(B)` | sanity check on the closed form. Should be ≈0. |

## What is already published, and what is not

This measurement sits next to two 2026 papers and it is worth being precise
about the gap, because the gap is small.

- **arXiv:2604.23150** (*Scaling Multi-Node MoE Inference Using Expert
  Activation Patterns*) profiles Llama 4 Maverick, DeepSeek V3 and Qwen3 over
  100k+ activation traces and reports an Imbalance Factor per layer per dataset.
  **No micro-batch-size axis, no shuffle or i.i.d. control.**
- **arXiv:2601.17111** (*Least-Loaded Expert Parallelism*) shows real routing
  imbalance for gpt-oss-20b (Fig. 3, max load per expert and per GPU under 8-way
  EP) and sweeps batch size — but the batch sweep measures *LLEP's own speedup*
  under **synthetic** imbalance levels (30/50/80/95% of tokens concentrated),
  not measured imbalance versus B.

Both establish that expert load imbalance exists and hurts. Neither plots the
straggler factor against micro-batch size against an independence null. That
decomposition is what this repo measures. It is a small adjacent result, not a
new phenomenon, and the writeup says so.

Prior art for the premise itself (do not claim any of this as new): Switch
Transformer (2101.03961), GShard (2006.16668), ST-MoE (2202.08906), Mixtral
(2401.04088), OLMoE (2409.02060), DeepSeek-V3 / EPLB (2412.19437).

---

## Setup

| | |
|---|---|
| model | `ibm-granite/granite-3.1-1b-a400m-instruct` (Apache-2.0) |
| architecture | `GraniteMoeForCausalLM`, 24 MoE layers, E=32 experts, top-k=8, d=1024, no shared expert |
| runtime | HF `transformers` 4.57.6 on PyTorch 2.8.0, fp16 on Apple MPS |
| corpus | 4 domains × ~200k tokens: code (local Python source), prose (WikiText-2), math (GSM8K), chat (Dolly-15k) |
| chunking | fixed 1024-token sequences, prefill only |
| machine | MacBook Air M1, 8 GB unified memory |

**The one trick that makes this cheap: the model runs exactly once.** Per-token,
per-layer top-k expert indices are dumped to disk as `int8` (E=32 fits), and
every subsequent analysis — every B, every N, every shuffle, every re-plot — is
offline numpy over a ~154 MB array.

## Reproducing

```bash
python3 -m pip install --user "transformers==4.57.6" matplotlib pyarrow
python3 src/download.py       # 2.67 GB into the HF cache
python3 src/probe.py          # asserts the routing API actually works. Do this first.
python3 src/test_analysis.py  # validates the estimator on synthetic data. Also first.
python3 src/test_latency_model.py  # pins the derived payload arithmetic. No data needed.
python3 src/build_corpus.py   # -> data/tokens_<domain>.npy
python3 src/collect.py        # the one model run  -> data/trace_<domain>.npz
python3 src/sweep.py          # -> out/sweep.csv
python3 src/figures.py        # -> figures/*.png
python3 src/latency_model.py  # -> out/latency_model.txt
```

`collect.py` is resumable and checkpoints every 20 forwards.

### Files

| file | what it does |
|---|---|
| `src/probe.py` | Hour-1 gate. Asserts `output_router_logits=True` returns 24 tensors of `(n_tokens, 32)`, cross-checks against a `register_forward_hook` on the router module, and measures fp16/MPS vs fp32/CPU top-k agreement. |
| `src/build_corpus.py` | Four domains → fixed 1024-token sequences. All sources free and public. |
| `src/collect.py` | The single model run. Converts router logits to `int8` top-k indices and frees them **inside** the loop — forget that and an 8 GB machine swaps to death. |
| `src/analysis.py` | All metrics. `S`, `M`, entropy, Gini, dead experts, the shuffle control, the closed-form null. |
| `src/test_analysis.py` | Unit tests on synthetic data where the answer is known. **This caught a real bug in the closed form — see below.** |
| `src/sweep.py` | domain × layer × B × N → `out/sweep.csv`. |
| `src/figures.py` | Figures 1–4. |
| `src/latency_model.py` | Converts a measured `S` into implied bytes and milliseconds. Derived, not measured. |
| `src/test_latency_model.py` | Arithmetic tests for the derived model: dispatch+combine both counted, linear in `B`, `1/N` per device, and a missing sweep exits instead of printing `nan` milliseconds. Needs no traces. |

---

## A correction to the closed form, found by the unit test

The obvious closed form for the independent-sampling null is

```
S_iid ≈ 1 + √(2 ln N) · √((N−1)/(B·k))
```

taking `E[max_j L_j]` over `N` devices as `z_N = √(2 ln N)` standard deviations.
**This overestimates the true i.i.d. straggler by 1.42× to 1.68× on the excess
`S − 1`.** Tested against the shuffle control on synthetic i.i.d. data, it was
wrong by up to 0.14 in `S` — larger than the effect being measured.

Had it been used uncorrected, measured `S_obs` would have come in *below* the
"i.i.d. prediction" and read as a spurious *better than independent* result.

Three corrections, each verified against the shuffle control:

1. `√(2 ln N)` is the **asymptotic** expected max of N normals and is a loose
   upper bound at N ≤ 32. Use the exact `E[max of N standard normals]`
   (quadrature). This is the dominant error.
2. `× √(N/(N−1))`: the N device loads sum to exactly `B·k`, so they are
   negatively correlated, not independent.
3. `× √((E−k)/(E−1))`: each token picks k **distinct** experts, so sampling
   without replacement within a token shrinks the per-token variance below the
   binomial value.

With all three, the closed form matches the shuffle control to within **0.24%**
across N ∈ {4,8,16} and B ∈ {128…8192}. `src/test_analysis.py` prints both forms
side by side.

---

## Caveats — read these before believing any number

1. **No cluster. The straggler factor is derived, not observed.** This computes
   what the measured routing *implies* for an expert-parallel deployment under a
   stated placement and bandwidth model. **No latency was measured.** The
   50 GB/s per-device bandwidth in `latency_model.py` is an assumption.
2. **One model, and a small one.** 24 layers, E=32, k=8, d=1024. Frontier MoEs
   run E=256 with k=8 — a completely different E/k ratio, which changes the
   combinatorics of pooling. The mechanism may transfer; **the numbers will
   not.** Nothing here supports a sentence beginning "MoE models exhibit…".
3. **Placement is assumed, not learned.** Contiguous expert→device assignment.
   Real systems use EPLB and redundant experts. Contiguous is an honest
   baseline, not a strawman, but it is a baseline.
4. **Small token counts are noisy at large B.** ~200k tokens/domain means ~48
   micro-batches per domain at B=4096 and ~12 at B=16384. Percentile bands, not
   point estimates, above B=4096.
5. **Instruct checkpoint, not base.** Chosen because it matches a serving
   deployment. Routing may differ from the base model.
6. **fp16 on MPS changes some tie-breaking.** Measured against fp32 on CPU;
   the agreement number is in `out/probe.json` and reported in the results.
7. **Chunking at 1024 tokens destroys long-range context**, so any
   position-within-chunk cut measures distance from an artificial context reset,
   not true document position.
8. **Prefill only.** No decode traces.
9. **`router_aux_loss_coef = 0.0` in this checkpoint's config.** Released
   inference configs routinely zero out training-only terms. **Do not read this
   as "trained without a load-balancing loss"** — IBM's tech report was not
   checked, and nothing here depends on it.

## Licence

Code: do as you like. The model is Apache-2.0 from IBM. WikiText-2, GSM8K and
Dolly-15k carry their own licences.
