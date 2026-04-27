"""Query a saved TF-IDF or BM25 retrieval index."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np

from build_retrieval_index import tokenize


def retrieve(index, query: str, top_k: int):
    kind = index["kind"]
    docs = index["docs"]
    if kind == "tfidf":
        query_vec = index["vectorizer"].transform([query])
        scores = (index["matrix"] @ query_vec.T).toarray().ravel()
    elif kind == "bm25":
        scores = np.asarray(index["bm25"].get_scores(tokenize(query)), dtype="float32")
    else:
        raise ValueError(f"Unsupported index kind: {kind}")

    if len(scores) == 0:
        return []
    top = np.argpartition(-scores, kth=min(top_k, len(scores) - 1))[:top_k]
    top = top[np.argsort(-scores[top])]
    return [(float(scores[i]), docs[int(i)]) for i in top if scores[i] > 0]


def query_from_log_row(row: dict[str, str]) -> str:
    question = row.get("question_text", "")
    options = []
    raw_options = row.get("options_json", "")
    if raw_options:
        try:
            for option in json.loads(raw_options):
                options.append(str(option.get("text", "")))
        except json.JSONDecodeError:
            pass
    return " ".join([question, *options]).strip()


def print_results(query: str, results) -> None:
    print(f"\nQUERY: {query}")
    for rank, (score, doc) in enumerate(results, start=1):
        title = f" | {doc['title']}" if doc.get("title") else ""
        text = doc["text"]
        if len(text) > 350:
            text = text[:347] + "..."
        print(f"{rank}. score={score:.4f} source={doc.get('source', '')}{title}")
        print(f"   {text}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("--query", default=None)
    parser.add_argument("--logs", type=Path, default=None, help="CSV log file to sample queries from")
    parser.add_argument("--limit", type=int, default=5, help="Number of log rows to query")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    index = joblib.load(args.index)
    if args.query:
        print_results(args.query, retrieve(index, args.query, args.top_k))
        return

    if args.logs:
        with args.logs.open("r", encoding="utf-8", newline="") as inp:
            for i, row in enumerate(csv.DictReader(inp)):
                if i >= args.limit:
                    break
                query = query_from_log_row(row)
                print_results(query, retrieve(index, query, args.top_k))
        return

    raise SystemExit("Pass --query or --logs")


if __name__ == "__main__":
    main()
