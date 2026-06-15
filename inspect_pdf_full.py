import fitz
from pathlib import Path

pdf_path = Path("data/raw/luat_bhyt.pdf")
doc = fitz.open(str(pdf_path))

output_lines = []
for idx, page in enumerate(doc):
    text = page.get_text()
    if "Điều 2." in text or "Giải thích từ ngữ" in text:
        output_lines.append(f"--- Page {idx+1} ---")
        output_lines.append(text)

with open("temp_inspect.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print("Done writing to temp_inspect.txt")
