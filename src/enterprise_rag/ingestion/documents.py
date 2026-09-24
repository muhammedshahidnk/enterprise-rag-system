from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class PageDocument:
    source: str
    page_num: int
    text: str


def extract_pdfs(folder: Path) -> list[PageDocument]:
    pages: list[PageDocument] = []

    for pdf_path in sorted(folder.glob("*.pdf")):
        print(f"Extracting text from {pdf_path}...")
        with pdf_path.open("rb") as file:
            reader = PdfReader(file, strict=False)
            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                print(f"  page {page_number}: {len(text)} chars")
                if text:
                    pages.append(PageDocument(pdf_path.name, page_number, text))

    return pages