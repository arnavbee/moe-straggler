"""
Caveat 6: does fp16 on MPS change which experts get selected, versus fp32 on CPU?

The probe did this on 64 tokens.  This does it on ~1k real corpus tokens and
reports per-layer agreement, because a single number hides the worst layer.
Loads one model at a time -- 2.7 GB + 5.4 GB would swap an 8 GB machine.
"""
import gc
import json
import os

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

N_SEQ = 1          # 1024 tokens
DOMAIN = "prose"


def topk_for(dtype, device, ids, L, K):
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype)
    m.eval().to(device)
    with torch.no_grad():
        o = m(ids.to(device), output_router_logits=True)
    res = [o.router_logits[i].float().cpu().topk(K, -1).indices.sort(-1).values
           for i in range(L)]
    del o, m
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return res


def main():
    cfg = AutoConfig.from_pretrained(MODEL)
    L, K = cfg.num_hidden_layers, cfg.num_experts_per_tok
    toks = np.load(os.path.join(HERE, "data", "tokens_%s.npy" % DOMAIN))[:N_SEQ]
    ids = torch.from_numpy(toks.astype(np.int64))
    n = ids.numel()
    print("comparing %d tokens, %d layers" % (n, L))

    a = topk_for(torch.float16, "mps", ids, L, K)
    b = topk_for(torch.float32, "cpu", ids, L, K)

    per_layer = [float((a[i] == b[i]).all(-1).float().mean()) for i in range(L)]
    # also: fraction of individual (token, slot) assignments that match
    per_layer_slot = [float((a[i] == b[i]).float().mean()) for i in range(L)]
    print("\nlayer  identical top-8 SET   identical assignments")
    for i in range(L):
        print("  %2d        %.4f              %.4f" % (i, per_layer[i], per_layer_slot[i]))
    print("\nmean over layers : %.4f  (set)   %.4f  (assignment)"
          % (np.mean(per_layer), np.mean(per_layer_slot)))
    print("worst layer      : %.4f  (layer %d)"
          % (min(per_layer), int(np.argmin(per_layer))))

    json.dump({"n_tokens": n, "domain": DOMAIN,
               "per_layer_set": per_layer, "per_layer_assignment": per_layer_slot,
               "mean_set": float(np.mean(per_layer)),
               "mean_assignment": float(np.mean(per_layer_slot)),
               "worst_layer_set": float(min(per_layer)),
               "worst_layer_idx": int(np.argmin(per_layer))},
              open(os.path.join(HERE, "out", "precision_check.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
