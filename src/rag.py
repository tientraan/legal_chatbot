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

def expand_query(question: str) -> list[str]:
    """
    Mở rộng truy vấn dựa trên các mẫu câu hỏi pháp lý phổ biến về y tế
    để đảm bảo không bỏ sót context liên quan khi thực hiện similarity search.
    """
    queries = [question]
    q_lower = question.lower()
    
    # 1. Nhóm từ khóa Bảo hiểm y tế
    if "đối tượng" in q_lower and ("bảo hiểm y tế" in q_lower or "bhyt" in q_lower):
        queries.extend([
            "đối tượng tham gia bảo hiểm y tế",
            "nhóm đối tượng tham gia bảo hiểm y tế",
            "đối tượng được ngân sách nhà nước đóng bảo hiểm y tế",
            "đối tượng được ngân sách nhà nước hỗ trợ đóng bảo hiểm y tế"
        ])
    elif "mức hưởng" in q_lower and ("bảo hiểm y tế" in q_lower or "bhyt" in q_lower):
        queries.extend([
            "mức hưởng bảo hiểm y tế",
            "mức hưởng khám bệnh chữa bệnh bảo hiểm y tế",
            "phạm vi được hưởng bảo hiểm y tế",
            "thanh toán chi phí khám bệnh chữa bệnh bảo hiểm y tế"
        ])
    elif "bảo hiểm y tế" in q_lower or "bhyt" in q_lower:
        queries.extend([
            "đối tượng tham gia bảo hiểm y tế",
            "mức hưởng bảo hiểm y tế",
            "khám bệnh, chữa bệnh bảo hiểm y tế",
            "quỹ bảo hiểm y tế"
        ])
        
    # 2. Nhóm từ khóa Khám bệnh, chữa bệnh
    if "người bệnh" in q_lower:
        if "quyền" in q_lower:
            queries.extend([
                "quyền của người bệnh",
                "người bệnh có những quyền gì",
                "quyền được khám bệnh, chữa bệnh",
                "quyền được tôn trọng và đối xử bình đẳng"
            ])
        elif "nghĩa vụ" in q_lower or "trách nhiệm" in q_lower:
            queries.extend([
                "nghĩa vụ của người bệnh",
                "người bệnh có nghĩa vụ gì",
                "trách nhiệm của người bệnh",
                "nghĩa vụ tôn trọng người hành nghề"
            ])
        else:
            queries.extend([
                "quyền của người bệnh",
                "nghĩa vụ của người bệnh"
            ])
            
    if "người hành nghề" in q_lower:
        if "quyền" in q_lower:
            queries.extend([
                "quyền của người hành nghề",
                "quyền của người hành nghề khám bệnh, chữa bệnh",
                "quyền được hành nghề"
            ])
        elif "nghĩa vụ" in q_lower or "trách nhiệm" in q_lower:
            queries.extend([
                "nghĩa vụ của người hành nghề",
                "nghĩa vụ của người hành nghề khám bệnh, chữa bệnh",
                "nghĩa vụ đối với người bệnh"
            ])
        else:
            queries.extend([
                "quyền của người hành nghề",
                "nghĩa vụ của người hành nghề"
            ])
            
    if "giấy phép hành nghề" in q_lower or "chứng chỉ hành nghề" in q_lower or "cấp phép" in q_lower:
        queries.extend([
            "điều kiện cấp giấy phép hành nghề khám bệnh, chữa bệnh",
            "cấp giấy phép hành nghề khám bệnh, chữa bệnh",
            "yêu cầu cấp giấy phép hành nghề",
            "hồ sơ đề nghị cấp giấy phép hành nghề"
        ])
        
    if "cơ sở" in q_lower and ("khám" in q_lower or "chữa" in q_lower):
        queries.extend([
            "trách nhiệm của cơ sở khám bệnh, chữa bệnh",
            "điều kiện hoạt động của cơ sở khám bệnh, chữa bệnh",
            "hình thức tổ chức của cơ sở khám bệnh, chữa bệnh"
        ])
        
    if "cấm" in q_lower or "nghiêm cấm" in q_lower:
        queries.extend([
            "các hành vi bị nghiêm cấm trong khám bệnh, chữa bệnh",
            "hành vi bị nghiêm cấm",
            "các hành vi bị nghiêm cấm trong y tế"
        ])
        
    # 3. Nhóm từ khóa Dược
    if "thuốc" in q_lower or "dược" in q_lower:
        if "điều kiện" in q_lower or "kinh doanh" in q_lower:
            queries.extend([
                "điều kiện kinh doanh dược",
                "điều kiện cấp Giấy chứng nhận đủ điều kiện kinh doanh dược",
                "cơ sở kinh doanh dược"
            ])
        elif "đơn thuốc" in q_lower or "kê đơn" in q_lower:
            queries.extend([
                "kê đơn thuốc",
                "đơn thuốc",
                "quy định về kê đơn thuốc",
                "bán lẻ thuốc theo đơn"
            ])
        else:
            queries.extend([
                "kinh doanh dược",
                "thuốc cổ truyền",
                "cấp phát thuốc",
                "dược sĩ"
            ])

    # Loại bỏ truy vấn trùng lặp
    seen = set()
    unique_queries = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique_queries.append(q)
    return unique_queries

def perform_bm25_search(query: str, vectordb: Chroma, k: int = 15) -> list[Document]:
    """
    Thực hiện tìm kiếm từ khóa BM25 trên toàn bộ tài liệu trong ChromaDB.
    Nếu xảy ra lỗi khởi tạo (ví dụ: database rỗng), trả về danh sách rỗng thay vì làm crash app.
    """
    try:
        # Lấy toàn bộ document trong collection
        all_docs_data = vectordb.get(include=["documents", "metadatas"])
        if not all_docs_data or not all_docs_data.get("documents"):
            logger.warning("Không có dữ liệu trong ChromaDB để khởi tạo BM25.")
            return []
            
        all_docs = []
        for text, metadata in zip(all_docs_data["documents"], all_docs_data["metadatas"]):
            all_docs.append(Document(page_content=text, metadata=metadata))
            
        # Tokenize corpus và truy vấn
        corpus_tokens = [tokenize_vietnamese(doc.page_content) for doc in all_docs]
        if not corpus_tokens:
            return []
            
        bm25 = BM25Okapi(corpus_tokens)
        query_tokens = tokenize_vietnamese(query)
        scores = bm25.get_scores(query_tokens)
        
        # Lấy các index có điểm lớn hơn 0
        valid_indices = [i for i, score in enumerate(scores) if score > 0]
        if not valid_indices:
            return []
            
        # Sắp xếp và chọn Top K
        top_indices = sorted(
            valid_indices,
            key=lambda idx: scores[idx],
            reverse=True
        )[:k]
        
        return [all_docs[idx] for idx in top_indices]
    except Exception as e:
        logger.error(f"Lỗi khi thực hiện BM25 Search (Fallback về Vector Search): {str(e)}")
        return []

def get_doc_unique_key(doc: Document) -> str:
    """Tạo khóa duy nhất dựa trên metadata và nội dung để loại bỏ trùng lặp."""
    src = doc.metadata.get("source", "unknown")
    art = doc.metadata.get("article", "unknown")
    idx = doc.metadata.get("chunk_index", 0)
    # Rút gọn nội dung tránh key quá dài
    content_hash = doc.page_content[:150]
    return f"{src}_{art}_{idx}_{content_hash}"

def rerank_documents(
    candidates_with_rank: list[dict], 
    question: str, 
    k: int = 10
) -> list[Document]:
    """
    Reranking các tài liệu ứng viên dựa trên RRF (Reciprocal Rank Fusion) và
    châm điểm tương thích từ khóa (Keyword Boost), cấu trúc pháp luật cụ thể.
    """
    question_tokens = set(tokenize_vietnamese(question))
    
    scored_candidates = []
    for cand in candidates_with_rank:
        doc = cand["doc"]
        
        # 1. Tính điểm RRF cơ sở
        # Điểm rank gốc (nếu có)
        rrf_score = 0.0
        if cand["vector_rank"] is not None:
            rrf_score += 1.0 / (60 + cand["vector_rank"])
        if cand["bm25_rank"] is not None:
            rrf_score += 1.0 / (60 + cand["bm25_rank"])
            
        # Điểm rank mở rộng (nếu có)
        if cand["exp_vector_rank"] is not None:
            rrf_score += 0.5 / (60 + cand["exp_vector_rank"])
        if cand["exp_bm25_rank"] is not None:
            rrf_score += 0.5 / (60 + cand["exp_bm25_rank"])
            
        # 2. Keyword Match Boost
        # Đếm tỷ lệ các token của câu hỏi xuất hiện trong nội dung chunk
        content_tokens = tokenize_vietnamese(doc.page_content)
        match_count = sum(1 for tok in question_tokens if tok in content_tokens)
        keyword_boost = (match_count / len(question_tokens)) * 0.4 if question_tokens else 0.0
        
        # 3. Legal Structure Boost
        # Nếu câu hỏi có từ khóa pháp lý cốt lõi, và tiêu đề điều của chunk khớp trực tiếp
        meta_art_title = doc.metadata.get("article_title", "").lower()
        structure_boost = 0.0
        
        if "quyền" in question.lower() and "quyền" in meta_art_title:
            structure_boost += 0.3
        if "nghĩa vụ" in question.lower() and ("nghĩa vụ" in meta_art_title or "trách nhiệm" in meta_art_title):
            structure_boost += 0.3
        if "đối tượng" in question.lower() and "đối tượng" in meta_art_title:
            structure_boost += 0.3
        if "điều kiện" in question.lower() and "điều kiện" in meta_art_title:
            structure_boost += 0.3
        if "cấm" in question.lower() and ("cấm" in meta_art_title or "nghiêm cấm" in meta_art_title):
            structure_boost += 0.3
            
        # Tổng điểm rerank
        final_score = rrf_score + keyword_boost + structure_boost
        scored_candidates.append((doc, final_score))
        
    # Sắp xếp theo điểm giảm dần và lấy Top K
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in scored_candidates[:k]]

def ask(question: str) -> tuple[str, list[Document]]:
    """
    Pipeline xử lý câu hỏi tối ưu RAG pháp luật y tế:
    1. Tạo danh sách các câu hỏi mở rộng (Query Expansion).
    2. Truy xuất đa chiều Hybrid Search (Vector + BM25) cho cả query gốc và mở rộng.
    3. Hợp nhất, loại trùng và tính điểm xếp hạng lại (Reranking).
    4. Gửi ngữ cảnh chất lượng vào LLM sinh câu trả lời cấu trúc chặt chẽ.
    """
    vectordb = get_vectordb()
    if vectordb is None:
        return "Tôi chưa tìm thấy căn cứ trong dữ liệu đã nạp. (Cơ sở dữ liệu chưa được nạp hoặc rỗng. Hãy thực hiện nạp dữ liệu trước).", []

    # 1. Tạo các query mở rộng
    queries = expand_query(question)
    original_query = queries[0]
    expanded_queries = queries[1:]
    
    logger.info(f"Query gốc: '{original_query}'")
    if expanded_queries:
        logger.info(f"Các query mở rộng: {expanded_queries}")

    # Cấu trúc lưu trữ thông tin ứng viên
    # key: unique_key -> {doc: Document, vector_rank: int, bm25_rank: int, exp_vector_rank: int, exp_bm25_rank: int}
    candidate_map = {}

    def add_candidate(doc: Document, source_type: str, rank: int):
        key = get_doc_key_fn(doc)
        if key not in candidate_map:
            candidate_map[key] = {
                "doc": doc,
                "vector_rank": None,
                "bm25_rank": None,
                "exp_vector_rank": None,
                "exp_bm25_rank": None
            }
        
        # Ghi nhận thứ tự xếp hạng nhỏ nhất (tốt nhất)
        current_rank = candidate_map[key][source_type]
        if current_rank is None or rank < current_rank:
            candidate_map[key][source_type] = rank

    # Hàm tạo key cục bộ
    def get_doc_key_fn(doc: Document) -> str:
        return get_doc_unique_key(doc)

    # 2. Truy xuất cho Query Gốc
    try:
        orig_vector_docs = vectordb.similarity_search(original_query, k=15)
        for rank, doc in enumerate(orig_vector_docs, start=1):
            add_candidate(doc, "vector_rank", rank)
    except Exception as e:
        logger.error(f"Lỗi Vector Search trên query gốc: {str(e)}")

    orig_bm25_docs = perform_bm25_search(original_query, vectordb, k=15)
    for rank, doc in enumerate(orig_bm25_docs, start=1):
        add_candidate(doc, "bm25_rank", rank)

    # 3. Truy xuất cho các Query Mở rộng
    for exp_q in expanded_queries:
        try:
            exp_vector_docs = vectordb.similarity_search(exp_q, k=10)
            for rank, doc in enumerate(exp_vector_docs, start=1):
                add_candidate(doc, "exp_vector_rank", rank)
        except Exception as e:
            logger.error(f"Lỗi Vector Search trên query mở rộng '{exp_q}': {str(e)}")

        exp_bm25_docs = perform_bm25_search(exp_q, vectordb, k=10)
        for rank, doc in enumerate(exp_bm25_docs, start=1):
            add_candidate(doc, "exp_bm25_rank", rank)

    # Convert map to list
    candidates = list(candidate_map.values())
    
    if not candidates:
        return "Tôi chưa tìm thấy căn cứ trong dữ liệu đã nạp.", []

    # 4. Reranking và lấy Top 8-12 tài liệu tốt nhất (ở đây chọn k=10)
    retrieved_docs = rerank_documents(candidates, question, k=10)
    
    # 5. Tạo ngữ cảnh (Context) cho mô hình
    context_parts = []
    for i, doc in enumerate(retrieved_docs, start=1):
        source_name = doc.metadata.get("source", "Không rõ nguồn")
        page_num = doc.metadata.get("page", "Không rõ trang")
        law_name = doc.metadata.get("law_name", "Không rõ luật")
        article = doc.metadata.get("article", "")
        art_title = doc.metadata.get("article_title", "")
        
        header_text = f"Căn cứ: {law_name}"
        if article:
            header_text += f", {article}"
        if art_title:
            header_text += f" ({art_title})"
        header_text += f" [File: {source_name}, Trang: {page_num}]"

        part = (
            f"--- ĐOẠN LUẬT THAM KHẢO {i} ---\n"
            f"{header_text}\n"
            f"Nội dung:\n{doc.page_content}"
        )
        context_parts.append(part)
    context_str = "\n\n".join(context_parts)

    system_instruction = """Bạn là trợ lý pháp luật y tế Việt Nam. Nhiệm vụ của bạn là trả lời câu hỏi chỉ dựa trên phần VĂN BẢN PHÁP LUẬT CUNG CẤP.

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin trong văn bản được cung cấp. Tuyệt đối không tự suy diễn, không dùng kiến thức ngoài.
2. Nếu câu hỏi yêu cầu liệt kê/kể/nêu/trình bày/bao gồm những gì, phải liệt kê đầy đủ tất cả các ý xuất hiện trong văn bản. Không được rút gọn còn 1–2 ý nếu văn bản có nhiều ý.
3. Khi có căn cứ, phải nêu rõ tên văn bản, điều, khoản, điểm nếu metadata hoặc nội dung có.
4. Nếu không tìm thấy căn cứ hoặc context hoàn toàn không liên quan đến câu hỏi, bạn BẮT BUỘC chỉ trả lời đúng câu: "Tôi chưa tìm thấy căn cứ trong dữ liệu đã nạp." và không viết thêm bất kỳ điều gì khác.
5. Nếu context có thông tin liên quan nhưng chưa đầy đủ, hãy trả lời phần tìm thấy và nói rõ phần còn thiếu ở mục Ghi chú.
6. Trình bày rõ ràng bằng bullet hoặc đánh số.

Cấu trúc trả lời của bạn BẮT BUỘC phải tuân theo định dạng sau:
- **Trả lời trực tiếp**: [Tóm tắt câu trả lời ngắn gọn, trực diện]
- **Căn cứ pháp lý**: [Nêu rõ Luật, Điều, Khoản, Điểm trích dẫn, kèm tên file PDF và trang nếu có]
- **Nội dung chi tiết**: [Liệt kê đầy đủ và chi tiết các quy định dưới dạng danh sách gạch đầu dòng]
- **Ghi chú nếu dữ liệu chưa đủ**: [Ghi chú nếu thông tin trong context chưa đáp ứng trọn vẹn câu hỏi hoặc các lưu ý khác. Nếu đã đầy đủ thì ghi "Dữ liệu đã đầy đủ."]
"""

    prompt = f"""Dưới đây là các đoạn văn bản luật được hệ thống truy xuất liên quan đến câu hỏi.

VĂN BẢN PHÁP LUẬT CUNG CẤP:
{context_str}

CÂU HỎI CỦA NGƯỜI DÙNG:
{question}

YÊU CẦU TRẢ LỜI:
- Trả lời bằng tiếng Việt.
- Bắt buộc chia rõ 3 phần: "## Trả lời", "## Căn cứ từ văn bản luật", "## Giải thích ngắn gọn" như cấu trúc trong System Instruction.
- Gắn trích dẫn file và số trang ở cuối mỗi luận điểm.

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
                temperature=0.0,
                system_instruction=system_instruction
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