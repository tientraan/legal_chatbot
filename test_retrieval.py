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
from src.rag import ask, expand_query

def main():
    print("=== KIỂM TRA TRUY XUẤT RAG PHÁP LUẬT Y TẾ ===")
    
    query = input("Nhập câu hỏi test: ").strip()
    if not query:
        print("Lỗi: Câu hỏi không được để trống.")
        return

    print(f"\n[1] Query gốc: '{query}'")
    
    # In danh sách các query mở rộng
    expanded = expand_query(query)
    print("[2] Danh sách query mở rộng:")
    for idx, eq in enumerate(expanded[1:], start=1):
        print(f"    {idx}. {eq}")
        
    print("\nĐang chạy hybrid search & reranking...")
    answer, docs = ask(query)
    
    print(f"\n[3] Kết quả truy xuất - Top {len(docs)} Documents sau Reranking:")
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "N/A")
        page = doc.metadata.get("page", "N/A")
        law_name = doc.metadata.get("law_name", "N/A")
        article = doc.metadata.get("article", "N/A")
        art_title = doc.metadata.get("article_title", "N/A")
        
        print(f"\n--- DOCUMENT {i} ---")
        print(f"Metadata:")
        for k, v in doc.metadata.items():
            print(f"  - {k}: {v}")
        print(f"Thông tin chi tiết:")
        print(f"  - Luật: {law_name}")
        print(f"  - Điều khoản: {article}")
        print(f"  - Tiêu đề Điều: {art_title}")
        print(f"  - File nguồn: {source}")
        print(f"  - Trang: {page}")
        
        # In 1000 đến 1500 ký tự đầu tiên
        content_preview = doc.page_content[:1200]
        suffix = "..." if len(doc.page_content) > 1200 else ""
        print(f"Nội dung (tối đa 1200 ký tự):")
        print(f"{content_preview}{suffix}")
        print("-" * 60)

    print("\n[4] Câu trả lời cuối cùng từ LLM:")
    print(answer)

if __name__ == "__main__":
    main()
