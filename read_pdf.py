import fitz
from pathlib import Path

pdf_path = Path("data/raw/luat_bhyt.pdf")
doc = fitz.open(str(pdf_path))
output_lines = []
output_lines.append(f"Total pages: {len(doc)}")
for i in range(min(5, len(doc))):
    output_lines.append(f"--- PAGE {i+1} ---")
    output_lines.append(doc[i].get_text())

with open("temp_pages.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print("Done writing to temp_pages.txt")
