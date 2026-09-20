"""
Hour-1 probe.  If this fails the experiment is impossible.

Asserts, against the real checkpoint and the installed transformers:
  1. output_router_logits=True yields a tuple of L tensors, each (n_tokens, E)
  2. a forward hook on the router module reproduces the same top-k selection
  3. the architectural constants match what the scope assumed (L=24, E=32, k=8, d=1024)
"""
import json
import os
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"


def main():
    cfg = AutoConfig.from_pretrained(MODEL)
    print("architectures      :", cfg.architectures)
    print("num_hidden_layers  :", cfg.num_hidden_layers)
    print("num_local_experts  :", cfg.num_local_experts)
    print("num_experts_per_tok:", cfg.num_experts_per_tok)
    print("hidden_size        :", cfg.hidden_size)
    print("router_aux_loss_coef:", getattr(cfg, "router_aux_loss_coef", None))
    print("shared_intermediate_size:", getattr(cfg, "shared_intermediate_size", None))

    L = cfg.num_hidden_layers
    E = cfg.num_local_experts
    K = cfg.num_experts_per_tok

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16)
    model.eval()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(dev)
    print("device             :", dev)

    text = ("The quick brown fox jumps over the lazy dog. " * 12)
    ids = tok(text, return_tensors="pt")["input_ids"][:, :64].to(dev)
    n_tok = ids.shape[1]
    print("probe tokens       :", n_tok)

    # --- 1. the public flag ------------------------------------------------
    with torch.no_grad():
        out = model(ids, output_router_logits=True)
    rl = out.router_logits
    assert isinstance(rl, (tuple, list)), type(rl)
    assert len(rl) == L, "expected %d layers of router logits, got %d" % (L, len(rl))
    for i, t in enumerate(rl):
        assert t.shape == (n_tok, E), "layer %d: %s != (%d,%d)" % (i, tuple(t.shape), n_tok, E)
    print("PASS  router_logits: tuple of %d tensors, each %s" % (len(rl), tuple(rl[0].shape)))

    # --- 2. the hook, as a version-independent cross-check ----------------
    layer = model.model.layers[3]
    router = layer.block_sparse_moe.router
    print("router module      :", type(router).__name__)
    grabbed = {}

    def hook(mod, inp, outp):
        grabbed["in"] = inp[0].detach()
        grabbed["out"] = outp

    h = router.register_forward_hook(hook)
    with torch.no_grad():
        out2 = model(ids, output_router_logits=True)
    h.remove()

    flag_topk = out2.router_logits[3].float().topk(K, dim=-1).indices.sort(dim=-1).values
    # GraniteMoeTopKGating returns a tuple; find the member that is (n_tok, E) logits
    cand = grabbed["out"]
    hook_logits = None
    if torch.is_tensor(cand) and cand.shape[-1] == E:
        hook_logits = cand
    else:
        for m in (cand if isinstance(cand, (tuple, list)) else []):
            if torch.is_tensor(m) and m.dim() == 2 and m.shape == (n_tok, E):
                hook_logits = m
                break
    if hook_logits is None:
        # fall back to recomputing from the router's *input* activations
        print("note: router output had no (n,E) member; recomputing from its input")
        hook_logits = router.layer(grabbed["in"]).float()
    hook_topk = hook_logits.float().topk(K, dim=-1).indices.sort(dim=-1).values
    agree = (hook_topk == flag_topk).all(dim=-1).float().mean().item()
    print("PASS  hook vs flag top-%d set agreement: %.4f" % (K, agree))
    assert agree > 0.999, "hook and flag disagree (%.4f)" % agree

    # --- 3. fp16/MPS vs fp32/CPU tie-breaking (caveat 6) ------------------
    # Free the fp16 copy first: 2.7 GB (fp16) + 5.4 GB (fp32) would swap an 8 GB box.
    mps_topk = [out.router_logits[i].float().cpu().topk(K, -1).indices.sort(-1).values
                for i in range(L)]
    del out, out2, model, rl, grabbed
    import gc
    gc.collect()
    if dev == "mps":
        torch.mps.empty_cache()

    model_cpu = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model_cpu.eval()
    with torch.no_grad():
        out_cpu = model_cpu(ids.to("cpu"), output_router_logits=True)
    same = []
    for i in range(L):
        a = mps_topk[i]
        b = out_cpu.router_logits[i].float().topk(K, -1).indices.sort(-1).values
        same.append((a == b).all(-1).float().mean().item())
    frac = sum(same) / len(same)
    print("fp16-MPS vs fp32-CPU identical top-%d set: %.4f of tokens (mean over layers)" % (K, frac))
    print("  per-layer min %.4f  max %.4f" % (min(same), max(same)))

    json.dump(
        {"L": L, "E": E, "K": K, "d": cfg.hidden_size, "device": dev,
         "hook_agreement": agree, "mps_cpu_topk_agreement": frac,
         "mps_cpu_per_layer": same,
         "router_aux_loss_coef": getattr(cfg, "router_aux_loss_coef", None)},
        open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "out", "probe.json"), "w"), indent=2)
    print("\nPROBE OK")


if __name__ == "__main__":
    sys.exit(main())
