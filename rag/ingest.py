#!/usr/bin/env python3
"""Extract, chunk, and embed a PDF into a local Chroma collection."""

import argparse
import re
from pathlib import Path

import chromadb
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
STORE_DIR = Path(__file__).parent / "store"


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        pages.append((i, text.strip()))
    doc.close()
    return pages


def is_heading(line: str) -> bool:
    """A section/table heading: an all-caps line that isn't purely numbers/currency."""
    s = line.strip().strip("\xa0")
    if len(s) < 4 or s != s.upper():
        return False
    stripped = s.replace(",", "").replace(".", "").replace("$", "").replace("%", "").replace("(", "").replace(")", "").replace("\xa0", "").strip()
    return not stripped.isdigit() and any(c.isalpha() for c in s)


def split_into_sections(text: str) -> list[tuple[str | None, str]]:
    """Split page text into (heading, body) sections on detected heading lines.
    Text before the first heading has heading=None."""
    lines = text.split("\n")
    sections = []
    current_heading = None
    current_lines: list[str] = []
    for line in lines:
        if is_heading(line):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines)))
            current_heading = line.strip().strip("\xa0")
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_heading, "\n".join(current_lines)))
    return sections


def chunk_pages(pages: list[tuple[int, str]], chunk_size: int, overlap: int) -> list[dict]:
    """Chunk text, keeping each heading-anchored section (e.g. a financial table)
    together as its own chunk, labeled with its heading. Sections without a
    heading, or long narrative sections, fall back to a sliding word window."""
    chunks = []
    for page_num, text in pages:
        if not text:
            continue
        for heading, body in split_into_sections(text):
            body = body.strip()
            if not body:
                continue

            if heading is not None and len(body.split()) <= chunk_size * 2:
                # small heading-anchored section (typically a table): keep intact,
                # label it so the embedding captures what it's about.
                chunk_text = f"{heading}\n{body}"
                chunks.append({"text": chunk_text, "page": page_num})
                continue

            # narrative or oversized section: sliding word window, still labeled.
            prefix = f"{heading}\n" if heading else ""
            words = body.split(" ")
            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk_text = prefix + " ".join(words[start:end])
                if chunk_text.strip():
                    chunks.append({"text": chunk_text, "page": page_num})
                if end == len(words):
                    break
                start = end - overlap
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a PDF into a Chroma collection")
    parser.add_argument("pdf", help="path to the PDF file")
    parser.add_argument("--collection", required=True, help="chroma collection name")
    parser.add_argument("--chunk-size", type=int, default=220, help="words per chunk")
    parser.add_argument("--overlap", type=int, default=40, help="word overlap between chunks")
    parser.add_argument("--reset", action="store_true", help="delete the collection first if it exists")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    print(f"reading {pdf_path} ...")
    pages = extract_pages(pdf_path)
    print(f"  {len(pages)} pages extracted")

    chunks = chunk_pages(pages, args.chunk_size, args.overlap)
    print(f"  {len(chunks)} chunks (size={args.chunk_size}, overlap={args.overlap})")

    print(f"loading embedding model {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(path=str(STORE_DIR))
    if args.reset:
        try:
            client.delete_collection(args.collection)
        except Exception:
            pass
    collection = client.get_or_create_collection(args.collection)

    print("embedding and storing chunks ...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    ids = [f"{pdf_path.stem}-{i}" for i in range(len(chunks))]
    metadatas = [{"source": pdf_path.name, "page": c["page"]} for c in chunks]

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )

    print(f"done. collection '{args.collection}' now has {collection.count()} chunks.")


if __name__ == "__main__":
    main()
