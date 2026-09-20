"""Pre-fetch the Granite MoE checkpoint into the HF cache."""
from huggingface_hub import snapshot_download
MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"
p = snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
print("CACHED_AT", p)
