"""
Generate a real self-attention matrix for the Chapter 4 heatmap.

Loads DistilBERT (distilbert-base-uncased, ~260MB download on first run) via
HuggingFace transformers, runs one short legal-flavored sentence through it,
and dumps the attention weights from a single layer + head as JSON. That JSON
gets inlined into index.html so the page stays a single static file.

We pick a mid layer (layer index 3 of 0..5) and head 0 of 12 — a fixed choice,
documented alongside the visualization.

To run:
    /tmp/claude-502/embvenv/bin/python scripts/build_attention.py \
        > scripts/attention.json

Or from a fresh env:
    python -m venv .venv && .venv/bin/pip install transformers torch
    .venv/bin/python scripts/build_attention.py > scripts/attention.json
"""

import json
import sys

import torch
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "distilbert-base-uncased"
SENTENCE = "The judge ruled the contract was invalid."
LAYER = 3   # of 6 transformer blocks, 0-indexed
HEAD = 0    # of 12 attention heads, 0-indexed


def main():
    print(f"Loading {MODEL_NAME}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, output_attentions=True)
    model.eval()

    inputs = tokenizer(SENTENCE, return_tensors="pt")
    token_ids = inputs["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    print(f"Tokens: {tokens}", file=sys.stderr)

    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.attentions is a tuple of length num_layers. Each entry has
    # shape (batch, num_heads, seq_len, seq_len). Row i is token i's
    # attention distribution (sums to 1 over cols).
    attn = outputs.attentions[LAYER][0, HEAD].tolist()

    payload = {
        "model": MODEL_NAME,
        "layer": LAYER,
        "head": HEAD,
        "sentence": SENTENCE,
        "tokens": tokens,
        "attention": attn,
    }

    for i, row in enumerate(attn):
        s = sum(row)
        if abs(s - 1.0) > 1e-3:
            print(f"WARN: row {i} sums to {s}", file=sys.stderr)

    print(
        f"Done. {len(tokens)} tokens, layer {LAYER}, head {HEAD}.",
        file=sys.stderr,
    )

    json.dump(payload, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
