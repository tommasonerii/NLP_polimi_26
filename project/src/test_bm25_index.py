import argparse
import joblib

from build_retrieval_index import tokenize, add_token_ngrams, STOPWORDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    obj = joblib.load(args.index)

    bm25 = obj["bm25"]
    docs = obj["docs"]

    remove_stopwords = obj.get("bm25_remove_stopwords", False)
    ngram_max = obj.get("bm25_ngram_max", 1)

    tokens = tokenize(args.query)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    tokens = add_token_ngrams(tokens, ngram_max)

    results, scores = bm25.retrieve([tokens], k=args.top_k)

    for rank, idx in enumerate(results[0], start=1):
        doc = docs[int(idx)]
        score = float(scores[0][rank - 1])

        print("=" * 100)
        print(f"Rank: {rank}")
        print(f"Score: {score:.4f}")
        print(f"ID: {doc.get('id')}")
        print(f"Title: {doc.get('title')}")
        print(f"Source: {doc.get('source')}")
        print()
        print(doc.get("text", "")[:1200])


if __name__ == "__main__":
    main()