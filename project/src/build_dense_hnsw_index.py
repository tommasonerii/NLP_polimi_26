"""Build a dense HNSW index from a chunked JSONL retrieval corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import hnswlib
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"


def count_valid_rows(corpus_path: Path, limit: int | None = None) -> int:
    count = 0
    with corpus_path.open("r", encoding="utf-8") as inp:
        for line in inp:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("text") or "").strip():
                count += 1
                if limit is not None and count >= limit:
                    break
    return count


def doc_metadata(row: dict[str, Any], default_source: str) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "doc_id": row.get("doc_id") or row.get("id"),
        "chunk_id": row.get("chunk_id") if row.get("chunk_id") is not None else 0,
        "title": row.get("title", ""),
        "url": row.get("url", ""),
        "source": row.get("source", default_source),
        "text": str(row.get("text") or ""),
    }


def embedding_text(row: dict[str, Any], title_repeat: int = 1) -> str:
    text = str(row.get("text") or "").strip()
    title = str(row.get("title") or "").strip()
    if title and title_repeat > 0:
        return " ".join([title] * title_repeat + [text]).strip()
    return text


def iter_batches(
    corpus_path: Path,
    source: str,
    batch_size: int,
    limit: int | None = None,
    title_repeat: int = 1,
):
    texts: list[str] = []
    docs: list[dict[str, Any]] = []
    seen = 0
    with corpus_path.open("r", encoding="utf-8") as inp:
        for line in inp:
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text") or "").strip()
            if not text:
                continue

            texts.append(embedding_text(row, title_repeat=title_repeat))
            docs.append(doc_metadata(row, default_source=source))
            seen += 1

            if len(texts) >= batch_size:
                yield texts, docs
                texts, docs = [], []

            if limit is not None and seen >= limit:
                break

    if texts:
        yield texts, docs


def build_dense_hnsw(
    corpus_path: Path,
    out_index_path: Path,
    out_meta_path: Path,
    source: str,
    model_name: str = DEFAULT_MODEL,
    device: str | None = None,
    limit: int | None = None,
    add_batch_size: int = 2048,
    encode_batch_size: int = 64,
    title_repeat: int = 1,
    m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 128,
    compress: int = 3,
) -> tuple[Path, Path]:
    if not corpus_path.exists():
        raise FileNotFoundError(corpus_path)

    total = count_valid_rows(corpus_path, limit=limit)
    print(f"{source}: valid rows = {total}")
    if total == 0:
        raise ValueError(f"No valid rows in {corpus_path}")

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dim: {dim}")

    out_index_path.parent.mkdir(parents=True, exist_ok=True)
    out_meta_path.parent.mkdir(parents=True, exist_ok=True)

    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=total, ef_construction=ef_construction, M=m)

    all_docs: list[dict[str, Any]] = []
    offset = 0
    started = time.time()

    for batch_texts, batch_docs in iter_batches(
        corpus_path,
        source=source,
        batch_size=add_batch_size,
        limit=limit,
        title_repeat=title_repeat,
    ):
        embeddings = model.encode(
            batch_texts,
            batch_size=encode_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        ids = np.arange(offset, offset + len(batch_texts))
        index.add_items(embeddings, ids)
        all_docs.extend(batch_docs)
        offset += len(batch_texts)

        elapsed = time.time() - started
        rate = offset / elapsed if elapsed > 0 else 0.0
        print(f"{source}: {offset}/{total} docs, {rate:.1f} docs/s")

    index.set_ef(ef_search)
    index.save_index(str(out_index_path))

    joblib.dump(
        {
            "kind": "dense_hnsw",
            "model_name": model_name,
            "hnsw_path": str(out_index_path),
            "docs": all_docs,
            "dim": dim,
            "space": "cosine",
            "ef": ef_search,
            "corpus_path": str(corpus_path),
            "source": source,
            "title_repeat": title_repeat,
            "m": m,
            "ef_construction": ef_construction,
        },
        out_meta_path,
        compress=compress,
    )

    print(f"Saved index: {out_index_path}")
    print(f"Saved meta:  {out_meta_path}")
    return out_index_path, out_meta_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path, help="Chunked JSONL corpus")
    parser.add_argument("--index-output", type=Path, required=True, help="Output .index path")
    parser.add_argument("--meta-output", type=Path, required=True, help="Output .joblib metadata path")
    parser.add_argument("--source", required=True, help="Source id stored in metadata")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None, help="SentenceTransformer device, e.g. cpu or cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--add-batch-size", type=int, default=2048)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--title-repeat", type=int, default=1)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=128)
    parser.add_argument("--compress", type=int, default=3)
    args = parser.parse_args()

    build_dense_hnsw(
        corpus_path=args.corpus,
        out_index_path=args.index_output,
        out_meta_path=args.meta_output,
        source=args.source,
        model_name=args.model_name,
        device=args.device,
        limit=args.limit,
        add_batch_size=args.add_batch_size,
        encode_batch_size=args.encode_batch_size,
        title_repeat=args.title_repeat,
        m=args.m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
        compress=args.compress,
    )


if __name__ == "__main__":
    main()
