from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from pypdf import PdfReader


BOILERPLATE_PATTERNS = [
    r"Access for free at openstax\.org",
    r"This OpenStax book is available for free.*",
    r"Download for free at.*",
    r"To learn more about OpenStax.*",
    r"Individual print copies and bulk orders.*",
    r"Attribution Non-Commercial ShareAlike.*",
    r"Creative Commons.*",
    r"Rice University",
    r"OpenStax",
    r"https://openstax\.org.*",
    r"support@openstax\.org",
    r"ISBN.*",
    r"Printed in.*",
    r"Revision.*",
]

PAGE_NUMBER_RE = re.compile(r"^\s*\d+\s*$")
MULTISPACE_RE = re.compile(r"[ \t]+")
WORD_RE = re.compile(r"\S+")

SECTION_RE = re.compile(
    r"""^\s*(
        Chapter\s+\d+.* |
        \d+(\.\d+)+\s+.{3,120} |
        [A-Z][A-Za-z0-9 ,:'’\-–—()]{6,120}
    )\s*$""",
    re.VERBOSE,
)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2010", "-").replace("\u2011", "-")
    text = text.replace("\u2012", "-").replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\xa0", " ")
    return text


def clean_line(line: str) -> str:
    line = normalize_text(line).strip()
    line = MULTISPACE_RE.sub(" ", line)

    if not line:
        return ""

    if PAGE_NUMBER_RE.match(line):
        return ""

    for pat in BOILERPLATE_PATTERNS:
        if re.search(pat, line, flags=re.IGNORECASE):
            return ""

    return line


def extract_pdf_lines(
    pdf_path: Path,
    start_page: int = 1,
    end_page: int | None = None,
) -> list[dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    rows: list[dict[str, Any]] = []

    for page_num, page in enumerate(reader.pages, start=1):
        if page_num < start_page:
            continue
        if end_page is not None and page_num > end_page:
            continue

        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""

        for line in raw.splitlines():
            line = clean_line(line)
            if line:
                rows.append({"page": page_num, "line": line})

    return rows


def is_probable_section_title(line: str) -> bool:
    if len(line) < 6 or len(line) > 140:
        return False

    if line.endswith("."):
        return False

    if len(line.split()) > 14:
        return False

    # Avoid obvious fragments from PDF extraction.
    lowered = line.lower()
    bad_fragments = [
        "html)",
        "then you must",
        "if you",
        "license",
        "bibliographic reference",
        "all rights reserved",
    ]
    if any(fragment in lowered for fragment in bad_fragments):
        return False

    return bool(SECTION_RE.match(line))


def split_into_chunks(
    rows: list[dict[str, Any]],
    book_id: str,
    book_title: str,
    source: str,
    chunk_words: int,
    overlap_words: int,
    min_chunk_words: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []

    section_title = book_title
    buffer_words: list[str] = []
    page_start: int | None = None
    page_end: int | None = None
    chunk_idx = 0

    def flush(force: bool = False) -> None:
        nonlocal buffer_words, page_start, page_end, chunk_idx

        if not buffer_words:
            return

        if not force and len(buffer_words) < chunk_words:
            return

        text_words = buffer_words[:chunk_words]

        if len(text_words) < min_chunk_words:
            buffer_words = []
            page_start = None
            page_end = None
            return

        chunk_id = f"{book_id}__chunk_{chunk_idx:05d}"

        chunks.append(
            {
                "id": chunk_id,
                "doc_id": book_id,
                "chunk_id": chunk_idx,
                "title": section_title,
                "source": source,
                "url": "",
                "page_start": page_start,
                "page_end": page_end,
                "text": " ".join(text_words),
            }
        )

        chunk_idx += 1

        if overlap_words > 0:
            buffer_words = buffer_words[max(0, chunk_words - overlap_words):]
        else:
            buffer_words = []

        page_start = page_end

    for row in rows:
        line = row["line"]
        page = row["page"]

        if is_probable_section_title(line):
            flush(force=True)
            section_title = line
            buffer_words = []
            page_start = page
            page_end = page
            continue

        words = WORD_RE.findall(line)
        if not words:
            continue

        if page_start is None:
            page_start = page

        page_end = page
        buffer_words.extend(words)

        while len(buffer_words) >= chunk_words:
            flush(force=False)

    flush(force=True)
    return chunks


def write_jsonl(chunks: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for row in chunks:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--book-id", required=True)
    parser.add_argument("--book-title", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument("--chunk-words", type=int, default=200)
    parser.add_argument("--overlap-words", type=int, default=50)
    parser.add_argument("--min-chunk-words", type=int, default=40)

    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=None)

    args = parser.parse_args()

    rows = extract_pdf_lines(
        pdf_path=args.pdf,
        start_page=args.start_page,
        end_page=args.end_page,
    )

    chunks = split_into_chunks(
        rows=rows,
        book_id=args.book_id,
        book_title=args.book_title,
        source=args.source,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
        min_chunk_words=args.min_chunk_words,
    )

    write_jsonl(chunks, args.output)

    print(f"PDF: {args.pdf}")
    print(f"Start page: {args.start_page}")
    print(f"End page: {args.end_page}")
    print(f"Lines extracted: {len(rows)}")
    print(f"Chunks written: {len(chunks)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()