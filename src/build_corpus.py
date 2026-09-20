"""
Build four fixed-length token corpora (code / prose / math / chat) for the
routing-trace collection.

Every source is free and public.  Nothing here needs an account or a key.
Output: data/tokens_<domain>.npy, int32, shape (n_seq, SEQ_LEN).
"""
import io
import json
import os
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
RAW = os.path.join(DATA, "raw")
MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"

SEQ_LEN = 1024
TARGET_TOKENS = int(os.environ.get("TARGET_TOKENS", 200_000))
N_SEQ = TARGET_TOKENS // SEQ_LEN

WIKITEXT_PARQUET = (
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
    "wikitext-2-raw-v1/train-00000-of-00001.parquet"
)
GSM8K_JSONL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/train.jsonl"
)
DOLLY_JSONL = (
    "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/"
    "databricks-dolly-15k.jsonl"
)
CODE_ROOT = "/opt/homebrew/lib/python3.11"


def fetch(url, name):
    """Download once, cache on disk."""
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    req = urllib.request.Request(url, headers={"User-Agent": "moe-straggler/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r, open(path, "wb") as f:
        f.write(r.read())
    return path


def text_code():
    """Python source already on this machine. Zero download."""
    chunks, total = [], 0
    paths = []
    for root, _, files in os.walk(CODE_ROOT):
        for fn in sorted(files):
            if fn.endswith(".py"):
                paths.append(os.path.join(root, fn))
    paths.sort()
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                s = f.read()
        except OSError:
            continue
        if len(s) < 200:
            continue
        chunks.append(s)
        total += len(s)
        # ~4 chars/token, take a healthy margin over the target
        if total > TARGET_TOKENS * 6:
            break
    return "\n\n".join(chunks)


def text_prose():
    import pyarrow.parquet as pq

    path = fetch(WIKITEXT_PARQUET, "wikitext2_train.parquet")
    col = pq.read_table(path).column("text").to_pylist()
    out, total = [], 0
    for line in col:
        line = line.strip()
        if len(line) < 80:          # drop headings and blanks
            continue
        out.append(line)
        total += len(line)
        if total > TARGET_TOKENS * 6:
            break
    return "\n\n".join(out)


def text_math():
    path = fetch(GSM8K_JSONL, "gsm8k_train.jsonl")
    out, total = [], 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            s = d["question"].strip() + "\n" + d["answer"].strip()
            out.append(s)
            total += len(s)
            if total > TARGET_TOKENS * 6:
                break
    return "\n\n".join(out)


def text_chat():
    path = fetch(DOLLY_JSONL, "dolly15k.jsonl")
    out, total = [], 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            parts = [d.get("instruction", "").strip()]
            if d.get("context"):
                parts.append(d["context"].strip())
            parts.append(d.get("response", "").strip())
            s = "\n".join(p for p in parts if p)
            if len(s) < 40:
                continue
            out.append(s)
            total += len(s)
            if total > TARGET_TOKENS * 6:
                break
    return "\n\n".join(out)


DOMAINS = {
    "code": text_code,
    "prose": text_prose,
    "math": text_math,
    "chat": text_chat,
}


def main():
    from transformers import AutoTokenizer

    os.makedirs(DATA, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    manifest = {}
    for name, fn in DOMAINS.items():
        out_path = os.path.join(DATA, "tokens_%s.npy" % name)
        if os.path.exists(out_path):
            arr = np.load(out_path)
            print("%-6s cached  %s" % (name, arr.shape))
            manifest[name] = list(arr.shape)
            continue
        raw = fn()
        ids = tok(raw, add_special_tokens=False)["input_ids"]
        need = N_SEQ * SEQ_LEN
        if len(ids) < need:
            print("WARNING: %s produced only %d tokens, wanted %d"
                  % (name, len(ids), need), file=sys.stderr)
            n = len(ids) // SEQ_LEN
        else:
            n = N_SEQ
        arr = np.asarray(ids[: n * SEQ_LEN], dtype=np.int32).reshape(n, SEQ_LEN)
        np.save(out_path, arr)
        manifest[name] = list(arr.shape)
        print("%-6s built   %s  (%d chars of source)" % (name, arr.shape, len(raw)))
    with open(os.path.join(DATA, "corpus_manifest.json"), "w") as f:
        json.dump({"seq_len": SEQ_LEN, "model": MODEL, "shapes": manifest}, f, indent=2)


if __name__ == "__main__":
    main()
