import os
import sys

# Khắc phục lỗi protobuf descriptor trên Cloud/Linux
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import re
import logging
from pathlib import Path

# Thêm thư mục gốc của dự án vào sys.path để python nhận diện gói 'src'
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from google import genai
from google.genai import types

# Cấu hình log
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

DB_DIR = Path("db")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PRIORITY_KEYWORDS = [
    "người bệnh",
    "quyền",
    "nghĩa vụ",
    "bảo hiểm y tế",
    "hồ sơ bệnh án",
    "giấy phép hành nghề",
    "thuốc",
    "dược"
]

# Đảm bảo in UTF-8 trên console Windows để tránh lỗi UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Khởi tạo embeddings và vector DB ở chế độ lười (lazy load)
_embeddings = None
_vectordb = None
_genai_client = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        logger.info(f"Đang khởi tạo mô hình nhúng: {EMBEDDING_MODEL}")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
    return _embeddings

def get_vectordb():
    global _vectordb
    if _vectordb is None:
        if not DB_DIR.exists():
            logger.warning(f"Thư mục database {DB_DIR} không tồn tại. Vui lòng chạy ingest dữ liệu trước.")
            return None
        logger.info(f"Đang tải ChromaDB từ {DB_DIR}")
        _vectordb = Chroma(
            persist_directory=str(DB_DIR),
            embedding_function=get_embeddings()
        )
    return _vectordb

def get_genai_client():
    global _genai_client
    if _genai_client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            # Thử tìm trong streamlit secrets
            try:
                import streamlit as st
                if "GOOGLE_API_KEY" in st.secrets:
                    api_key = st.secrets["GOOGLE_API_KEY"]
            except Exception:
                pass
        
        if not api_key:
            raise ValueError(
                "Không tìm thấy GOOGLE_API_KEY. "
                "Nếu chạy local, hãy cấu hình trong file .env. "
                "Nếu chạy trên Streamlit Cloud, hãy cấu hình trong phần App Settings -> Secrets bằng cú pháp:\n"
                "GOOGLE_API_KEY = \"your_api_key_here\""
            )
        _genai_client = genai.Client(api_key=api_key)
    return _genai_client

def tokenize_vietnamese(text: str) -> list[str]:
    """
    Tokenize đơn giản cho tiếng Việt bằng cách đưa về chữ thường,
    loại bỏ các ký tự đặc biệt và tách từ bằng khoảng trắng.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()

def perform_bm25_search(query: str, vectordb: Chroma, k: int = 10) -> list[Document]:
    """
    Thực hiện tìm kiếm từ khóa BM25 trên toàn bộ tài liệu trong ChromaDB.
    """
    try:
        # Lấy toàn bộ document trong collection
        all_docs_data = vectordb.get(include=["documents", "metadatas"])
        if not all_docs_data or not all_docs_data.get("documents"):
            return []
            
        all_docs = []
        for text, metadata in zip(all_docs_data["documents"], all_docs_data["metadatas"]):
            all_docs.append(Document(page_content=text, metadata=metadata))
            
        # Tokenize corpus và truy vấn
        corpus_tokens = [tokenize_vietnamese(doc.page_content) for doc in all_docs]
        bm25 = BM25Okapi(corpus_tokens)
        
        query_tokens = tokenize_vietnamese(query)
        scores = bm25.get_scores(query_tokens)
        
        # Lấy Top K kết quả có điểm số lớn hơn 0
        top_indices = sorted(
            [i for i, score in enumerate(scores) if score > 0],
            key=lambda idx: scores[idx],
            reverse=True
        )[:k]
        
        return [all_docs[idx] for idx in top_indices]
    except Exception as e:
        logger.error(f"Lỗi khi thực hiện BM25 Search: {str(e)}")
        return []

def rerank_documents(
    vector_docs: list[Document], 
    bm25_docs: list[Document], 
    priority_keywords: list[str], 
    k: int = 10
) -> list[Document]:
    """
    Kết hợp kết quả tìm kiếm Vector và BM25 (Hybrid Search),
    sau đó rerank và ưu tiên các đoạn văn bản có chứa từ khóa quan trọng.
    Sử dụng Reciprocal Rank Fusion (RRF) kết hợp với Keyword Boost để tính điểm cuối cùng.
    """
    # Lập bản đồ định danh duy nhất cho từng tài liệu tránh trùng lặp
    def get_doc_key(doc: Document) -> tuple[str, int, str]:
        return (doc.metadata.get("source", ""), doc.metadata.get("page", 0), doc.page_content)

    doc_map = {}
    vector_ranks = {}
    bm25_ranks = {}

    for rank, doc in enumerate(vector_docs, start=1):
        key = get_doc_key(doc)
        doc_map[key] = doc
        vector_ranks[key] = rank

    for rank, doc in enumerate(bm25_docs, start=1):
        key = get_doc_key(doc)
        doc_map[key] = doc
        bm25_ranks[key] = rank

    scored_candidates = []
    for key, doc in doc_map.items():
        # 1. Tính điểm RRF (Reciprocal Rank Fusion)
        rrf_score = 0.0
        if key in vector_ranks:
            rrf_score += 1.0 / (60 + vector_ranks[key])
        if key in bm25_ranks:
            rrf_score += 1.0 / (60 + bm25_ranks[key])

        # 2. Đếm số lượng từ khóa ưu tiên xuất hiện trong văn bản
        match_count = 0
        content_lower = doc.page_content.lower()
        for kw in priority_keywords:
            if kw in content_lower:
                match_count += 1

        # 3. Tính điểm Reranking cuối cùng: RRF làm gốc + cộng điểm thưởng đáng kể cho mỗi từ khóa khớp.
        # Điểm RRF tối đa là ~0.033. Cộng thêm 0.05 điểm cho mỗi từ khóa khớp giúp đẩy các đoạn có từ khóa
        # lên trên nhưng vẫn giữ thứ tự liên quan nội tại của chúng nhờ phần thập phân RRF.
        final_score = rrf_score + 0.05 * match_count
        
        scored_candidates.append((doc, final_score))

    # Sắp xếp theo điểm số giảm dần và chọn Top K
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in scored_candidates[:k]]

def ask(question: str) -> tuple[str, list[Document]]:
    """
    Pipeline xử lý câu hỏi:
    1. Tìm kiếm Vector (Top 10)
    2. Tìm kiếm BM25 (Top 10)
    3. Kết hợp Hybrid + Reranking (Ưu tiên từ khóa) -> Chọn ra Top 10 cuối cùng.
    4. Gửi ngữ cảnh vào Gemini 2.5 Flash để sinh câu trả lời tiếng Việt chính xác.
    """
    vectordb = get_vectordb()
    if vectordb is None:
        return "Hệ thống chưa được nạp cơ sở dữ liệu luật. Vui lòng bấm nút nạp dữ liệu ở sidebar hoặc chạy lệnh `python src/ingest.py` trước.", []

    # 1. Vector Search (Top 10)
    logger.info(f"Đang thực hiện Vector Search cho câu hỏi: '{question}'")
    try:
        vector_docs = vectordb.similarity_search(question, k=10)
    except Exception as e:
        logger.error(f"Lỗi Vector Search: {str(e)}")
        vector_docs = []

    # 2. BM25 Keyword Search (Top 10)
    logger.info("Đang thực hiện BM25 Search...")
    bm25_docs = perform_bm25_search(question, vectordb, k=10)

    # 3. Hybrid Search & Reranking
    logger.info("Đang thực hiện Reranking và chọn lọc Top 10...")
    retrieved_docs = rerank_documents(vector_docs, bm25_docs, PRIORITY_KEYWORDS, k=10)

    if not retrieved_docs:
        return "Tôi chưa tìm thấy căn cứ trong dữ liệu đã nạp.", []

    # 4. Tạo ngữ cảnh (Context) cho mô hình
    context_parts = []
    for i, doc in enumerate(retrieved_docs, start=1):
        source_name = doc.metadata.get("source", "Không rõ nguồn")
        page_num = doc.metadata.get("page", "Không rõ trang")
        part = (
            f"--- ĐOẠN VĂN BẢN THAM KHẢO {i} ---\n"
            f"Tài liệu nguồn: {source_name}\n"
            f"Số trang: {page_num}\n"
            f"Nội dung:\n{doc.page_content}"
        )
        context_parts.append(part)
    context_str = "\n\n".join(context_parts)

    # 5. Xây dựng Prompt chặt chẽ theo yêu cầu
    prompt = f"""
Bạn là chuyên gia pháp luật y tế Việt Nam.

Chỉ sử dụng thông tin trong phần "VĂN BẢN LUẬT CUNG CẤP".
Không dùng kiến thức bên ngoài.

Nếu câu hỏi hỏi về:
- đối tượng
- quyền
- nghĩa vụ
- điều kiện
- trường hợp
- hành vi bị cấm

thì phải liệt kê ĐẦY ĐỦ tất cả các ý tìm thấy trong văn bản.
Không được chỉ trả lời ý đầu tiên.
Không được tóm tắt quá ngắn.

Nếu không tìm thấy căn cứ phù hợp, trả lời:
"Tôi chưa tìm thấy căn cứ trong dữ liệu đã nạp."

VĂN BẢN LUẬT CUNG CẤP:
{context_str}

CÂU HỎI:
{question}

YÊU CẦU TRẢ LỜI:
- Trả lời bằng tiếng Việt.
- Liệt kê đầy đủ theo gạch đầu dòng hoặc đánh số.
- Nêu rõ điều, khoản nếu có.
- Nêu nguồn file PDF và số trang.
- Không bịa thông tin.

TRẢ LỜI:
"""

    # 6. Gọi LLM
    try:
        client = get_genai_client()
        logger.info("Đang gọi API Gemini 2.5 Flash...")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0
            )
        )
        answer = response.text
        return answer, retrieved_docs
    except Exception as e:
        logger.error(f"Lỗi khi gọi API Gemini: {str(e)}")
        return f"Có lỗi xảy ra khi gọi dịch vụ AI: {str(e)}", retrieved_docs

if __name__ == "__main__":
    print("=== CHATBOT PHÁP LUẬT KHÁM CHỮA BỆNH (DÒNG LỆNH) ===")
    print("Nhập 'q', 'exit' hoặc 'quit' để thoát.\n")
    
    # Kiểm tra xem db có tồn tại chưa
    if not DB_DIR.exists():
        print("Cảnh báo: Thư mục db/ chưa tồn tại. Hệ thống sẽ cố gắng tìm kiếm nhưng có thể thất bại.")
        print("Hãy chạy lệnh nạp dữ liệu trước nếu cần: python src/ingest.py\n")

    while True:
        try:
            question = input("Đặt câu hỏi: ").strip()
            if not question:
                continue
            if question.lower() in ["q", "exit", "quit"]:
                print("Tạm biệt!")
                break
                
            print("\nĐang xử lý câu hỏi, vui lòng đợi...")
            answer, docs = ask(question)
            
            print("\n=== TRẢ LỜI ===")
            print(answer)
            print("================\n")
            
            if docs:
                print("--- NGUỒN TÀI LIỆU TRÍCH DẪN (TOP 10 CHUNKS) ---")
                for i, doc in enumerate(docs, 1):
                    print(f"{i}. {doc.metadata.get('source')} - trang {doc.metadata.get('page')}")
                print("------------------------------------------------\n")
        except KeyboardInterrupt:
            print("\nThoát chương trình...")
            break
        except Exception as e:
            print(f"\nĐã xảy ra lỗi: {str(e)}\n")