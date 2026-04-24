"""
Generate the attention + hidden-state data for the "charge" disambiguation
demo in Part 2, Chapter 2 (embeddings subsection).

Four sentences, each containing the word "charge" in a different sense:
    1. legal       — "The prosecutor dropped the charge in court."
    2. financial   — "I disputed the charge on my credit card statement."
    3. electrical  — "The charge built up on the wire and sparked."
    4. physical    — "The cavalry charge broke the enemy line."

For each sentence we extract:
    - tokens produced by distilbert-base-uncased's tokenizer
    - the attention from the `charge` token to every other token, at the final
      layer (layer 5 of 6), averaged across all 12 heads. One distribution
      per sentence (not the full (seq, seq) matrix).
    - the final-layer hidden state of the `charge` token.

We also grab the static input embedding of `charge` (before any layer
processes it). PCA is fit on a pool of ALL tokens' final hidden states
across all sentences, plus the static embedding; the 5 vectors we care
about are projected into that space. This gives a 2D plane whose axes
reflect how the model spreads out real token representations, not just
the 5 points we care about — so the "same word, different place" story
lands against a meaningful backdrop.

Output JSON is inlined into index.html as `const CHARGE_DATA = ...`.

Run:
    /tmp/claude-502/embvenv/bin/python scripts/build_charge_contexts.py \
        > scripts/charge_contexts.json
"""

import json
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "distilbert-base-uncased"
# Attention: layer 0 (first layer). Averaged across all 12 heads, this layer's
# attention from `charge` maps cleanly onto the sentence's disambiguating
# content words. Deeper layers in DistilBERT develop strong attention sinks
# (heavy weight on [SEP]/punctuation), which Chapter 4 demonstrates and
# explains — here we want the content-signal story instead.
# Hidden state: final layer (5), since the "where does charge land" panel
# is supposed to show the vector AFTER the transformer has done its work.
ATTN_LAYER = 0
HIDDEN_LAYER = 5
# Special / non-content tokens to exclude when computing the content-only
# normalized attention view.
SPECIAL_TOKENS = {"[CLS]", "[SEP]", ".", ",", ";", ":", "?", "!"}

SENTENCES = [
    {"label": "legal",      "text": "The prosecutor dropped the charge in court."},
    {"label": "financial",  "text": "I disputed the charge on my credit card statement."},
    {"label": "electrical", "text": "The charge built up on the wire and sparked."},
    {"label": "physical",   "text": "The cavalry charge broke the enemy line."},
]

CHARGE = "charge"


def find_charge_index(tokens):
    for i, t in enumerate(tokens):
        if t == CHARGE:
            return i
    raise RuntimeError(f"'{CHARGE}' not in tokens: {tokens}")


def main():
    print(f"Loading {MODEL_NAME}...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(
        MODEL_NAME, output_attentions=True, output_hidden_states=True
    )
    model.eval()

    # Static embedding of "charge" (word-embedding matrix row, pre-layers,
    # pre-positional).
    charge_id = tok.convert_tokens_to_ids(CHARGE)
    static_embedding = (
        model.embeddings.word_embeddings.weight[charge_id].detach().numpy()
    )

    contexts = []
    charge_hiddens = []      # final hidden state of `charge` per sentence
    pool_vectors = []        # all tokens' final hidden states across sentences

    for s in SENTENCES:
        inputs = tok(s["text"], return_tensors="pt")
        token_ids = inputs["input_ids"][0].tolist()
        tokens = tok.convert_ids_to_tokens(token_ids)
        idx = find_charge_index(tokens)

        with torch.no_grad():
            out = model(**inputs)

        # Attention row for `charge` at ATTN_LAYER, averaged across heads.
        # out.attentions[L]: (batch, heads, seq, seq)
        attn_layer = out.attentions[ATTN_LAYER][0]           # (heads, seq, seq)
        attn_row = attn_layer[:, idx, :].mean(dim=0).numpy()  # (seq,)

        # Hidden state at HIDDEN_LAYER. out.hidden_states has length
        # num_layers+1 (index 0 = post-embedding input to layer 0; index K+1
        # = output of layer K).
        final = out.hidden_states[HIDDEN_LAYER + 1][0].numpy()  # (seq, dim)
        charge_hiddens.append(final[idx])
        pool_vectors.append(final)

        # Also build a content-only, renormalized view of the attention row:
        # zero out special tokens + charge itself, normalize the remainder.
        # This is what the strip visualization uses; the raw row is kept in
        # the JSON for transparency.
        content_mask = [
            (t not in SPECIAL_TOKENS and i != idx) for i, t in enumerate(tokens)
        ]
        content_raw = [
            float(attn_row[i]) if content_mask[i] else 0.0 for i in range(len(tokens))
        ]
        total = sum(content_raw)
        content_norm = [v / total if total > 0 else 0.0 for v in content_raw]

        contexts.append({
            "label": s["label"],
            "text": s["text"],
            "tokens": tokens,
            "charge_index": idx,
            "attention_from_charge": [float(x) for x in attn_row],
            "attention_content_only": content_norm,
        })

    # Fit PCA on just the 5 points we plan to plot: the static embedding of
    # `charge` plus its four context-specific hidden states. With 5 points we
    # get a well-defined 2D projection that maximally spreads them — better
    # for a tiny 5-point scatter than fitting on a pool where the axes are
    # shaped by unrelated tokens.
    charge_points = np.vstack([static_embedding[None, :]] + charge_hiddens)
    mean_vec = charge_points.mean(axis=0)
    centered = charge_points - mean_vec
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pc2 = Vt[:2]  # top 2 principal components, shape (2, dim)

    def project(v):
        return ((v - mean_vec) @ pc2.T).tolist()

    static_xy = [float(x) for x in project(static_embedding)]
    for ctx, hs in zip(contexts, charge_hiddens):
        ctx["vector_xy"] = [float(x) for x in project(hs)]

    variance_2d = float((S[:2] ** 2).sum() / (S ** 2).sum())

    payload = {
        "model": MODEL_NAME,
        "attn_layer": ATTN_LAYER,
        "hidden_layer": HIDDEN_LAYER,
        "layers_total": 6,
        "heads": 12,
        "variance_2d": variance_2d,
        "static_embedding_xy": static_xy,
        "contexts": contexts,
    }

    print(
        f"Done. {len(contexts)} contexts. Attention layer {ATTN_LAYER}, "
        f"hidden-state layer {HIDDEN_LAYER}. "
        f"2D variance explained: {variance_2d:.1%}",
        file=sys.stderr,
    )

    json.dump(payload, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
