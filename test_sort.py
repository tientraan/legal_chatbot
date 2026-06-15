import fitz
from pathlib import Path

pdf_path = Path("data/raw/luat_bhyt.pdf")
doc = fitz.open(str(pdf_path))
page = doc[1]  # Page 2 (0-indexed is 1)

print("--- WITHOUT SORT ---")
print(page.get_text()[:1000])

print("\n--- WITH SORT ---")
print(page.get_text(sort=True)[:1000])
