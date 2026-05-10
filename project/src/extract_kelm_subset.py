"""Create a limited KELM JSONL subset for BM25/RAG experiments.

The full KELM dataset is larger than we need for quick PoliMillionaire tests.
This script streams rows from Hugging Face when possible and writes a compact
JSONL file with stable fields used by local retrieval code.
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
from pathlib import Path
import re
from typing import Any
from urllib.request import urlopen


DEFAULT_KELM_URL = "https://storage.googleapis.com/gresearch/kelm-corpus/updated-2021/kelm_generated_corpus.jsonl"


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def row_text(row: dict[str, Any]) -> tuple[str, str]:
    sentence = compact_text(str(row.get("sentence") or row.get("gen_sentence") or row.get("sentence2") or ""))
    triple_value = (
        row.get("triple")
        or row.get("serialized_triples")
        or row.get("serialized triples")
        or row.get("sentence1")
        or ""
    )
    if isinstance(triple_value, list):
        triple_value = " ".join(str(item) for item in triple_value)
    triple = compact_text(str(triple_value))
    return sentence, triple


def iter_jsonl_url(url: str):
    opener = gzip.open if url.endswith(".gz") else open
    if url.startswith(("http://", "https://")):
        with urlopen(url) as response:
            source = gzip.GzipFile(fileobj=response) if url.endswith(".gz") else response
            for raw_line in source:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if line:
                    yield json.loads(line)
    else:
        with opener(url, "rt", encoding="utf-8") as source:
            for line in source:
                line = line.strip()
                if line:
                    yield json.loads(line)


def write_subset(rows, output: Path, limit: int, min_chars: int, source_name: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    seen = 0
    with output.open("w", encoding="utf-8", newline="\n") as out:
        for row in rows:
            seen += 1
            sentence, triple = row_text(row)
            text = sentence or triple
            if len(text) < min_chars:
                continue

            out.write(
                json.dumps(
                    {
                        "id": f"kelm_{written}",
                        "text": html.unescape(text),
                        "sentence": html.unescape(sentence),
                        "triple": html.unescape(triple),
                        "source": source_name,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

            if written % 10_000 == 0:
                print(f"written {written} rows")
            if written >= limit:
                break

    print(f"done: saw {seen} rows, wrote {written} rows to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="google-research-datasets/kelm",
        help="Hugging Face dataset id. Use visoc/KELM for the lighter single-triple variant.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_KELM_URL,
        help="JSONL URL/path to stream. Use --url '' to read from Hugging Face instead.",
    )
    parser.add_argument("--split", default="train", help="Dataset split to read")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/kelm/kelm_subset_100k.jsonl"),
        help="Output JSONL file",
    )
    parser.add_argument("--limit", type=int, default=100_000, help="Maximum rows to write")
    parser.add_argument("--min-chars", type=int, default=30, help="Skip rows with shorter text")
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream the dataset instead of downloading/materializing the full split",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass trust_remote_code=True to older Hugging Face dataset loaders when needed",
    )
    args = parser.parse_args()

    if args.url:
        write_subset(
            iter_jsonl_url(args.url),
            output=args.output,
            limit=args.limit,
            min_chars=args.min_chars,
            source_name=args.url,
        )
        return

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency: install with `conda install -n polimillionaire -c conda-forge datasets`") from exc

    load_kwargs = {
        "path": args.dataset,
        "split": args.split,
        "streaming": args.streaming,
    }
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    ds = load_dataset(**load_kwargs)

    write_subset(
        ds,
        output=args.output,
        limit=args.limit,
        min_chars=args.min_chars,
        source_name=args.dataset,
    )


if __name__ == "__main__":
    main()
