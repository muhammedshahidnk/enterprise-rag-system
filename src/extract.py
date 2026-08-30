"""
STEP 1: Text Extraction
------------------------
Why per-page extraction matters: if you just concatenate the whole PDF into
one giant string, you lose the ability to answer "which page did this come
from" later (Experiment D). So we extract page-by-page and carry that
metadata all the way through the pipeline.
"""
from pypdf import PdfReader
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PageDoc:
    source: str      # filename
    page_num: int     # 1-indexed page number
    text: str


def extract_pdfs(folder: str) -> list[PageDoc]:
    pages = []
    for pdf_path in sorted(Path(folder).glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append(PageDoc(source=pdf_path.name, page_num=i + 1, text=text))
    return pages


if __name__ == "__main__":
    docs = extract_pdfs("documents")
    print(f"Extracted {len(docs)} pages from PDFs\n")
    for d in docs[:2]:
        print(f"--- {d.source} p.{d.page_num} ---")
        print(d.text[:200], "...\n")
