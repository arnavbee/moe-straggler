"""
Collect the routing trace.  Runs the model ONCE over the whole corpus and dumps
per-token, per-layer top-k expert indices.  Every later analysis is offline numpy.

Output per domain:  data/trace_<domain>.npz
    topk   int8   (n_tokens, L, K)   the selected expert ids
    seq_id int32  (n_tokens,)        which 1024-token sequence each token came from
    pos    int16  (n_tokens,)        position within that sequence

Memory discipline (the thing that kills an 8 GB machine): router_logits for a
chunk is (tokens x L x E) float32 and transformers keeps all L layers live at
once.  We convert to int8 top-k indices and free the logits inside the loop.
"""
import gc
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")

SEQ_PER_FWD = int(os.environ.get("SEQ_PER_FWD", 2))   # 1024-token sequences per forward
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", 20))    # checkpoint every N forwards


def collect_domain(model, cfg, name, device):
    src = os.path.join(DATA, "tokens_%s.npy" % name)
    dst = os.path.join(DATA, "trace_%s.npz" % name)
    part = os.path.join(DATA, "trace_%s.partial.npy" % name)
    if os.path.exists(dst):
        print("%-6s trace exists, skipping" % name)
        return

    toks = np.load(src)                       # (n_seq, seq_len)
    n_seq, seq_len = toks.shape
    L, K = cfg.num_hidden_layers, cfg.num_experts_per_tok

    out = np.zeros((n_seq, seq_len, L, K), dtype=np.int8)
    start_seq = 0
    if os.path.exists(part):                  # resume
        prev = np.load(part)
        meta = json.load(open(part + ".json"))
        start_seq = meta["done_seq"]
        out[:start_seq] = prev[:start_seq]
        print("%-6s resuming at sequence %d/%d" % (name, start_seq, n_seq))

    t0 = time.time()
    for s in range(start_seq, n_seq, SEQ_PER_FWD):
        e = min(s + SEQ_PER_FWD, n_seq)
        batch = torch.from_numpy(toks[s:e].astype(np.int64)).to(device)
        with torch.no_grad():
            o = model(batch, output_router_logits=True)
            rl = o.router_logits
            # rl[i] is (bsz*seq_len, E) flattened in row-major (seq, pos) order
            stacked = torch.stack(
                [t.float().topk(K, dim=-1).indices for t in rl], dim=1
            )                                  # (bsz*seq_len, L, K)
            arr = stacked.to(torch.int8).cpu().numpy()
        del o, rl, stacked, batch
        out[s:e] = arr.reshape(e - s, seq_len, L, K)
        del arr
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

        if (e // SEQ_PER_FWD) % CKPT_EVERY == 0 or e == n_seq:
            np.save(part, out)
            json.dump({"done_seq": e}, open(part + ".json", "w"))
            el = time.time() - t0
            rate = (e - start_seq) * seq_len / max(el, 1e-9)
            eta = (n_seq - e) * seq_len / max(rate, 1e-9)
            print("%-6s %4d/%4d seq  %7.0f tok/s  eta %5.1f min"
                  % (name, e, n_seq, rate, eta / 60), flush=True)

    flat = out.reshape(n_seq * seq_len, L, K)
    seq_id = np.repeat(np.arange(n_seq, dtype=np.int32), seq_len)
    pos = np.tile(np.arange(seq_len, dtype=np.int16), n_seq)
    np.savez_compressed(dst, topk=flat, seq_id=seq_id, pos=pos)
    for f in (part, part + ".json"):
        if os.path.exists(f):
            os.remove(f)
    print("%-6s done  %s  -> %s (%.1f MB)"
          % (name, flat.shape, os.path.basename(dst), os.path.getsize(dst) / 1e6))


def main():
    domains = sys.argv[1:] or ["code", "prose", "math", "chat"]
    cfg = AutoConfig.from_pretrained(MODEL)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16)
    model.eval()
    model.to(device)
    print("loaded on %s | L=%d E=%d K=%d d=%d"
          % (device, cfg.num_hidden_layers, cfg.num_local_experts,
             cfg.num_experts_per_tok, cfg.hidden_size), flush=True)
    for d in domains:
        collect_domain(model, cfg, d, device)


if __name__ == "__main__":
    main()
