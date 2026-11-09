from pathlib import Path
from pypdf import PdfReader

total = 0

for file in Path("documents").glob("*.pdf"):
    pages = len(PdfReader(str(file)).pages)
    total += pages
    print(f"{file.name}: {pages} pages")

print(f"\nTOTAL PDF PAGES: {total}")
