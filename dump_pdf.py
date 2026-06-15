import fitz
from pathlib import Path

pdf_path = Path("data/raw/luat_bhyt.pdf")
doc = fitz.open(str(pdf_path))

output_lines = []
for idx, page in enumerate(doc):
    output_lines.append(f"\n==================== PAGE {idx+1} ====================\n")
    output_lines.append(page.get_text())

with open("full_text_bhyt.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print("Done writing to full_text_bhyt.txt")
