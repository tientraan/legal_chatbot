import os
import sys
import logging
from pathlib import Path

# Thêm thư mục gốc vào path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from dotenv import load_dotenv
load_dotenv()

# Cấu hình encoding UTF-8 trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.WARNING)
from src.rag import get_vectordb

def main():
    print("=== KIỂM TRA CHUNKS TRONG VECTOR DATABASE ===")
    vectordb = get_vectordb()
    if vectordb is None:
        print("Lỗi: Không tìm thấy Vector Database. Vui lòng chạy nạp dữ liệu trước: python src/ingest.py")
        return

    try:
        count = vectordb._collection.count()
        print(f"Tổng số chunks hiện tại trong database: {count}")
        
        if count == 0:
            print("Cơ sở dữ liệu hiện đang trống.")
            return

        # Lấy tối đa 20 chunk đầu tiên
        results = vectordb._collection.get(limit=20, include=["documents", "metadatas"])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        print(f"\n--- HIỂN THỊ CHI TIẾT {len(documents)} CHUNKS ĐẦU TIÊN ---")
        for i, (doc_text, meta) in enumerate(zip(documents, metadatas), start=1):
            print(f"\n[Chunk {i}/{count}]")
            print(f"Metadata:")
            for k, v in meta.items():
                print(f"  - {k}: {v}")
            print("Nội dung (250 ký tự đầu):")
            preview = doc_text.replace('\n', ' ')[:250] + "..." if len(doc_text) > 250 else doc_text
            print(f"  {preview}")
            print("-" * 60)

    except Exception as e:
        print(f"Đã xảy ra lỗi khi truy xuất database: {str(e)}")

if __name__ == "__main__":
    main()
