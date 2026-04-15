"""
Find 3 English sentences and 3 Spanish sentences where the algorithm fails
to find the correct translation in the top-3 nearest neighbors.
"""
import pandas as pd
import numpy as np
import re
from gensim.models import KeyedVectors
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv("data/aligned_literature_en_es.csv")
en_model = KeyedVectors.load_word2vec_format("data/mini.en.vec")
es_model = KeyedVectors.load_word2vec_format("data/mini.es.vec")

# Use the best method: Mean-Centered + TF-IDF
tfidf_en = TfidfVectorizer().fit(df["english_text"])
tfidf_es = TfidfVectorizer().fit(df["spanish_text"])


def sentence_to_tfidf(sent, model, tfidf):
    words = re.findall(r'\w+', sent.lower())
    vocab = tfidf.vocabulary_
    idf = tfidf.idf_
    vecs, weights = [], []
    for w in words:
        if w in model and w in vocab:
            vecs.append(model[w])
            weights.append(idf[vocab[w]])
    if not vecs:
        return np.zeros(model.vector_size)
    return np.average(vecs, axis=0, weights=weights)


en_tfidf = np.array([sentence_to_tfidf(s, en_model, tfidf_en) for s in df["english_text"]])
es_tfidf = np.array([sentence_to_tfidf(s, es_model, tfidf_es) for s in df["spanish_text"]])

# Mean-centering
en_vecs = en_tfidf - en_tfidf.mean(axis=0)
es_vecs = es_tfidf - es_tfidf.mean(axis=0)


def find_failures(query_vecs, gallery_vecs, query_texts, gallery_texts, n=3):
    dists = cdist(query_vecs, gallery_vecs, metric="cosine")
    top_3 = np.argsort(dists, axis=1)[:, :3]
    failures = []
    for i in range(len(query_vecs)):
        if i not in top_3[i]:
            actual_rank = int(np.where(np.argsort(dists[i]) == i)[0][0])
            failures.append({
                "idx": i,
                "query": query_texts.iloc[i],
                "actual_rank": actual_rank,
                "top1_idx": int(top_3[i][0]),
                "top1_text": gallery_texts.iloc[int(top_3[i][0])],
                "top2_idx": int(top_3[i][1]),
                "top2_text": gallery_texts.iloc[int(top_3[i][1])],
                "top3_idx": int(top_3[i][2]),
                "top3_text": gallery_texts.iloc[int(top_3[i][2])],
                "correct_text": gallery_texts.iloc[i],
            })
    return failures


print("### EN -> ES failures (3 examples) ###\n")
en_failures = find_failures(en_vecs, es_vecs, df["english_text"], df["spanish_text"])
for f in en_failures[:3]:
    print(f"[idx {f['idx']}] Correct rank in results: #{f['actual_rank']}")
    print(f"  EN query:      {f['query']}")
    print(f"  Expected ES:   {f['correct_text']}")
    print(f"  Top-1 found:   [{f['top1_idx']}] {f['top1_text']}")
    print(f"  Top-2 found:   [{f['top2_idx']}] {f['top2_text']}")
    print(f"  Top-3 found:   [{f['top3_idx']}] {f['top3_text']}")
    print()

print(f"\nTotal EN->ES failures: {len(en_failures)} / {len(df)}\n")

print("\n### ES -> EN failures (3 examples) ###\n")
es_failures = find_failures(es_vecs, en_vecs, df["spanish_text"], df["english_text"])
for f in es_failures[:3]:
    print(f"[idx {f['idx']}] Correct rank in results: #{f['actual_rank']}")
    print(f"  ES query:      {f['query']}")
    print(f"  Expected EN:   {f['correct_text']}")
    print(f"  Top-1 found:   [{f['top1_idx']}] {f['top1_text']}")
    print(f"  Top-2 found:   [{f['top2_idx']}] {f['top2_text']}")
    print(f"  Top-3 found:   [{f['top3_idx']}] {f['top3_text']}")
    print()

print(f"\nTotal ES->EN failures: {len(es_failures)} / {len(df)}")
