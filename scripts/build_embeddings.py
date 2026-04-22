"""
Generate real word embeddings for the Chapter 3 scatter plot.

Uses GloVe 50-dim (Wikipedia + Gigaword, 400k words, 66MB download on first run)
and projects to 2D via PCA. Output is a JSON blob that gets pasted into
index.html's inline JS so the page stays a single static file with no fetches.

To run:
    /tmp/claude-502/embvenv/bin/python scripts/build_embeddings.py > scripts/embeddings.json

Or from a fresh env:
    python -m venv .venv && .venv/bin/pip install gensim scikit-learn
    .venv/bin/python scripts/build_embeddings.py > scripts/embeddings.json
"""

import json
import sys

import gensim.downloader as api
from sklearn.decomposition import PCA


# Curated word list by category. All must be lowercase and in GloVe's vocab.
# Legal terms are the Clio nod; the others are classic "obvious cluster" picks
# that make the visualization land immediately.
WORDS = {
    "legal":     ["law", "court", "judge", "contract", "lawyer", "trial", "justice", "evidence"],
    "colors":    ["red", "blue", "green", "yellow", "black", "white"],
    "animals":   ["dog", "cat", "horse", "bird", "fish", "cow", "tiger"],
    "numbers":   ["one", "two", "three", "four", "ten"],
    "countries": ["france", "japan", "brazil", "germany", "china", "india"],
    "time":      ["day", "week", "month", "year", "hour"],
    "emotions":  ["happy", "sad", "angry", "love", "fear"],
}


def main():
    print("Loading GloVe 50-dim (first run downloads ~66MB)...", file=sys.stderr)
    model = api.load("glove-wiki-gigaword-50")

    rows = []
    missing = []
    for category, words in WORDS.items():
        for word in words:
            if word in model:
                rows.append({"word": word, "category": category, "vec": model[word].tolist()})
            else:
                missing.append(word)

    if missing:
        print(f"WARN: not in vocab: {missing}", file=sys.stderr)

    vectors = [r["vec"] for r in rows]
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(vectors)

    for r, (x, y) in zip(rows, coords_2d):
        r["x"] = float(x)
        r["y"] = float(y)

    explained = pca.explained_variance_ratio_
    print(
        f"Done. {len(rows)} words. "
        f"PCA explains {explained[0]*100:.1f}% + {explained[1]*100:.1f}% = {sum(explained)*100:.1f}% of variance.",
        file=sys.stderr,
    )

    json.dump(rows, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
