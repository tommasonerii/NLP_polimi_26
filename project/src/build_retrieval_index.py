"""Build TF-IDF or BM25 indexes from a chunked JSONL retrieval corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import joblib
import numpy as np


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+-]*", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text or "")]


def load_corpus(path: Path, limit: int | None = None) -> tuple[list[str], list[dict[str, Any]]]:
    texts: list[str] = []
    docs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as inp:
        for line in inp:
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text") or "")
            if not text:
                continue
            docs.append(
                {
                    "id": row.get("id"),
                    "doc_id": row.get("doc_id") or row.get("id"),
                    "chunk_id": row.get("chunk_id") if row.get("chunk_id") is not None else 0,
                    "title": row.get("title", ""),
                    "url": row.get("url", ""),
                    "source": row.get("source", ""),
                    "text": text,
                }
            )
            texts.append(text)
            if limit is not None and len(texts) >= limit:
                break
    return texts, docs


def build_tfidf(texts: list[str], max_features: int, min_df: int, ngram_max: int):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        strip_accents="unicode",
        max_features=max_features,
        min_df=min_df,
        ngram_range=(1, ngram_max),
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(texts)
    return {"kind": "tfidf", "vectorizer": vectorizer, "matrix": matrix}


def build_bm25(texts: list[str]):
    import bm25s

    print("Tokenizing corpus...")
    # Keep the same tokenizer logic
    tokenized = [tokenize(text) for text in texts]
    
    print("Building BM25 index...")
    bm25 = bm25s.BM25()
    bm25.index(tokenized)
    return {"kind": "bm25", "bm25": bm25}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path, help="Chunked JSONL corpus")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output .joblib index")
    parser.add_argument("--kind", choices=["tfidf", "bm25"], required=True)
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of chunks")
    parser.add_argument("--max-features", type=int, default=300_000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--compress", type=int, default=3)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    texts, docs = load_corpus(args.corpus, limit=args.limit)
    if not texts:
        raise SystemExit(f"No texts found in {args.corpus}")

    print(f"loaded {len(texts)} chunks")
    if args.kind == "tfidf":
        index = build_tfidf(texts, args.max_features, args.min_df, args.ngram_max)
    else:
        index = build_bm25(texts)

    index["docs"] = docs
    index["corpus_path"] = str(args.corpus)
    joblib.dump(index, args.output, compress=args.compress)
    print(f"done: wrote {args.kind} index to {args.output}")


if __name__ == "__main__":
    main()
