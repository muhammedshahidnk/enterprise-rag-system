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
        print(f"Extracting text from {pdf_path}...")

        try:
            with open(pdf_path, "rb") as f:

                header = f.read(4)

                if header != b"%PDF":
                    print(f"  Skipping: not a PDF (header={header!r})")
                    continue

                f.seek(0)

                try:
                    reader = PdfReader(f, strict=False)

                    for i, page in enumerate(reader.pages):
                        try:
                            text = page.extract_text() or ""

                        except Exception as e:
                            print(
                                f"  Warning extracting page "
                                f"{i + 1} of {pdf_path.name}: {e}"
                            )
                            text = ""

                        text = text.strip()

                        print(
                            f"  page {i + 1}: "
                            f"{len(text)} chars"
                        )

                        if text:
                            pages.append(
                                PageDoc(
                                    source=pdf_path.name,
                                    page_num=i + 1,
                                    text=text
                                )
                            )

                except Exception as e:
                    print(
                        f"  Error reading PDF "
                        f"{pdf_path.name}: {e}"
                    )
                    continue

        except Exception as e:
            print(
                f"  Could not open "
                f"{pdf_path}: {e}"
            )
            continue

        # print(pages)
        # break
    return pages


if __name__ == "__main__":
    docs = extract_pdfs("documents")
    print(f"Extracted {len(docs)} pages from PDFs\n")
    for d in docs:
        print(f"--- {d.source} p.{d.page_num} ---")
        print(d.text, "...\n")
