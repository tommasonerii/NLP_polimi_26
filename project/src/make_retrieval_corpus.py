"""Build chunked JSONL corpora for local retrieval.

Input rows must contain at least a text field. Output rows have stable fields:
id, text, title, url, source. Long documents are split into overlapping word
chunks so TF-IDF/BM25 retrieve focused snippets instead of entire articles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def words(text: str) -> list[str]:
    return compact_text(text).split()


def chunk_words(tokens: list[str], max_words: int, overlap_words: int):
    if len(tokens) <= max_words:
        yield 0, tokens
        return

    step = max(1, max_words - overlap_words)
    start = 0
    chunk_id = 0
    while start < len(tokens):
        end = min(start + max_words, len(tokens))
        yield chunk_id, tokens[start:end]
        if end == len(tokens):
            break
        start += step
        chunk_id += 1


def pick_text(row: dict[str, Any], field_names: list[str]) -> str:
    parts = []
    for name in field_names:
        value = row.get(name)
        if value:
            parts.append(str(value))
    return compact_text(" ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Input JSONL corpus")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output chunked JSONL corpus")
    parser.add_argument("--source", required=True, help="Corpus source label, e.g. simplewiki or kelm")
    parser.add_argument("--id-prefix", default=None, help="Optional output id prefix")
    parser.add_argument("--text-fields", nargs="+", default=["text"], help="Fields to concatenate as retrievable text")
    parser.add_argument("--title-field", default="title")
    parser.add_argument("--url-field", default="url")
    parser.add_argument("--max-words", type=int, default=160)
    parser.add_argument("--overlap-words", type=int, default=30)
    parser.add_argument("--min-words", type=int, default=20)
    parser.add_argument("--limit-docs", type=int, default=None)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    prefix = args.id_prefix or args.source

    read_docs = 0
    written = 0
    with args.input.open("r", encoding="utf-8") as inp, args.output.open("w", encoding="utf-8", newline="\n") as out:
        for line in inp:
            if not line.strip():
                continue
            row = json.loads(line)
            read_docs += 1

            text = pick_text(row, args.text_fields)
            token_list = words(text)
            if len(token_list) < args.min_words:
                continue

            base_id = str(row.get("id") or read_docs)
            title = str(row.get(args.title_field) or "")
            url = str(row.get(args.url_field) or "")

            for local_chunk_id, chunk in chunk_words(token_list, args.max_words, args.overlap_words):
                if len(chunk) < args.min_words:
                    continue
                out.write(
                    json.dumps(
                        {
                            "id": f"{prefix}_{base_id}_{local_chunk_id}",
                            "doc_id": base_id,
                            "chunk_id": local_chunk_id,
                            "title": title,
                            "url": url,
                            "source": args.source,
                            "text": " ".join(chunk),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
                if written % 50_000 == 0:
                    print(f"written {written} chunks")

            if args.limit_docs is not None and read_docs >= args.limit_docs:
                break

    print(f"done: read {read_docs} documents, wrote {written} chunks to {args.output}")


if __name__ == "__main__":
    main()
