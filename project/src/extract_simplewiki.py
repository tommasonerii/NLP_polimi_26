"""Extract Simple English Wikipedia dump into JSONL articles.

This is a lightweight fallback for Python versions where WikiExtractor fails.
It does not fully parse every MediaWiki template, but it produces clean enough
plain text for BM25/RAG experiments.
"""

from __future__ import annotations

import argparse
import bz2
import html
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def wiki_url(title: str) -> str:
    return "https://simple.wikipedia.org/wiki/" + title.replace(" ", "_")


def remove_balanced_templates(text: str) -> str:
    """Remove simple balanced {{...}} blocks with a bounded loop."""
    for _ in range(20):
        new_text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
        if new_text == text:
            return text
        text = new_text
    return text


def clean_wikitext(text: str) -> str:
    text = html.unescape(text or "")

    # Drop comments, refs, tables and common noisy blocks.
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>/]*/>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\{\|.*?\|\}", " ", text, flags=re.DOTALL)
    text = remove_balanced_templates(text)

    # Remove files/images/categories but keep ordinary wikilinks.
    text = re.sub(r"\[\[(?:File|Image|Category):[^\]]+\]\]", " ", text, flags=re.IGNORECASE)

    # Convert [[target|label]] -> label and [[target]] -> target.
    text = re.sub(r"\[\[[^|\]]+\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)

    # Convert external links [url label] -> label, [url] -> empty.
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://[^\]]+\]", " ", text)

    # Remove markup.
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"^=+\s*(.*?)\s*=+$", r"\1", text, flags=re.MULTILINE)
    text = re.sub(r"<[^>]+>", " ", text)

    # Keep text compact.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def iter_pages(input_path: Path):
    with bz2.open(input_path, "rb") as fh:
        context = ET.iterparse(fh, events=("end",))
        for _, elem in context:
            if strip_namespace(elem.tag) != "page":
                continue

            title = elem.findtext("./{*}title") or ""
            ns = elem.findtext("./{*}ns") or ""
            page_id = elem.findtext("./{*}id") or ""
            redirect = elem.find("./{*}redirect") is not None
            text = elem.findtext("./{*}revision/{*}text") or ""

            if ns == "0" and not redirect and title and text:
                yield {
                    "id": page_id,
                    "title": title,
                    "url": wiki_url(title),
                    "text": clean_wikitext(text),
                }

            elem.clear()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Path to simplewiki pages-articles xml.bz2 dump")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/wiki/simplewiki_articles.jsonl"),
        help="Output JSONL file",
    )
    parser.add_argument("--min-chars", type=int, default=300, help="Skip shorter articles")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of articles for testing")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as out:
        for page in iter_pages(args.input):
            if len(page["text"]) < args.min_chars:
                continue
            out.write(json.dumps(page, ensure_ascii=False) + "\n")
            written += 1
            if written % 10000 == 0:
                print(f"written {written} articles")
            if args.limit is not None and written >= args.limit:
                break

    print(f"done: wrote {written} articles to {args.output}")


if __name__ == "__main__":
    main()
