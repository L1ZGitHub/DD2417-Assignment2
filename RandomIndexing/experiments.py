"""
Runs experiments for Random Indexing hyperparameter exploration.
Tokenizes once, then tests multiple configurations.
"""
import os
import time
import numpy as np
import nltk
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors


TEST_WORDS = ["harry", "gryffindor", "chair", "wand", "good", "enter", "on", "school"]


def tokenize_all(data_dir):
    tokens = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            with open(os.path.join(root, file), encoding="utf-8", errors="ignore") as f:
                text = f.read()
            tokens.extend(nltk.word_tokenize(text))
    return [t.lower() for t in tokens]


def build_word2id(tokens, pad="<pad>"):
    word2id = defaultdict(int)
    id2word = []
    for t in tokens + [pad]:
        if t not in word2id:
            word2id[t] = len(id2word)
            id2word.append(t)
    return word2id, id2word


def get_context(tokens, i, lws, rws, pad):
    if i < lws:
        left = [pad] * (lws - i) + tokens[0:i]
    else:
        left = tokens[i - lws:i]
    if i + rws >= len(tokens):
        right = tokens[i + 1:] + [pad] * (rws - (len(tokens) - i - 1))
    else:
        right = tokens[i + 1:i + 1 + rws]
    return left + right


def build_datapoints(tokens, word2id, lws, rws, pad):
    datapoints = []
    for i, tok in enumerate(tokens):
        focus_id = word2id[tok]
        context = get_context(tokens, i, lws, rws, pad)
        context_ids = [word2id[w] for w in context]
        datapoints.append((focus_id, context_ids))
    return datapoints


def create_vectors(datapoints, vocab_size, dim, non_zero, non_zero_vals=(-1, 1), seed=42):
    rng = np.random.default_rng(seed)
    rv = np.zeros((vocab_size, dim))
    for i in range(vocab_size):
        positions = rng.choice(dim, non_zero, replace=False)
        values = rng.choice(non_zero_vals, non_zero)
        rv[i, positions] = values
    cv = np.zeros((vocab_size, dim))
    for focus_id, context in datapoints:
        for c in context:
            cv[focus_id] += rv[c]
    return cv


def normalize(cv):
    norms = np.linalg.norm(cv, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return cv / norms


def get_neighbors(cv, word2id, id2word, word, k=5, metric="cosine"):
    if word not in word2id:
        return []
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(cv)
    dists, idxs = nbrs.kneighbors([cv[word2id[word]]])
    return [(id2word[idxs[0][j]], dists[0][j]) for j in range(k)]


def format_neighbors(neighbors):
    return ", ".join([f"{w}({d:.3f})" for w, d in neighbors[1:]])


def run_config(datapoints, vocab_size, word2id, id2word,
               dim=2000, non_zero=100, lws=2, rws=2,
               metric="cosine", normalized=True, label=""):
    t0 = time.time()
    cv = create_vectors(datapoints, vocab_size, dim, non_zero)
    if normalized:
        cv = normalize(cv)
    elapsed = time.time() - t0
    print(f"\n=== {label} [dim={dim}, nz={non_zero}, lws={lws}, rws={rws}, metric={metric}, norm={normalized}] ({elapsed:.1f}s) ===")
    for w in TEST_WORDS:
        nbrs = get_neighbors(cv, word2id, id2word, w, k=5, metric=metric)
        print(f"  {w:12s} -> {format_neighbors(nbrs)}")


def main():
    print("Tokenizing...")
    data_dir = "data"
    tokens = tokenize_all(data_dir)
    print(f"Total tokens: {len(tokens)}")
    word2id, id2word = build_word2id(tokens)
    print(f"Vocab size: {len(id2word)}")

    # --- Default config (will reuse datapoints) ---
    pad = "<pad>"

    # Experiment 1: Metrics (fix default params)
    print("\n\n########## EXPERIMENT: METRICS ##########")
    datapoints = build_datapoints(tokens, word2id, 2, 2, pad)
    for metric in ["cosine", "euclidean", "manhattan"]:
        run_config(datapoints, len(id2word), word2id, id2word,
                   dim=2000, non_zero=100, lws=2, rws=2,
                   metric=metric, normalized=True, label=f"METRIC={metric}")

    # Experiment 2: Normalized vs not
    print("\n\n########## EXPERIMENT: NORMALIZATION ##########")
    for norm in [True, False]:
        run_config(datapoints, len(id2word), word2id, id2word,
                   dim=2000, non_zero=100, lws=2, rws=2,
                   metric="cosine", normalized=norm, label=f"NORM={norm}")

    # Experiment 3: Dimensionality x non-zero proportion
    print("\n\n########## EXPERIMENT: DIMENSION x NON-ZERO ##########")
    for dim in [10, 50, 100, 1000]:
        for prop in [0.05, 0.10, 0.20, 0.50]:
            nz = max(1, int(dim * prop))
            run_config(datapoints, len(id2word), word2id, id2word,
                       dim=dim, non_zero=nz, lws=2, rws=2,
                       metric="cosine", normalized=True,
                       label=f"dim={dim},prop={int(prop * 100)}%")

    # Experiment 4: Window sizes
    print("\n\n########## EXPERIMENT: WINDOW SIZES ##########")
    window_configs = [(0, 0), (3, 3), (10, 10), (0, 3), (3, 0), (0, 10), (10, 0)]
    for lws, rws in window_configs:
        dp = build_datapoints(tokens, word2id, lws, rws, pad)
        run_config(dp, len(id2word), word2id, id2word,
                   dim=2000, non_zero=100, lws=lws, rws=rws,
                   metric="cosine", normalized=True,
                   label=f"WINDOW L={lws}, R={rws}")


if __name__ == "__main__":
    main()
