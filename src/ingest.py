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
import shutil
import logging
from pathlib import Path

# Thêm thư mục gốc của dự án vào sys.path để python nhận diện gói 'src'
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from dotenv import load_dotenv
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.utils import clean_text

# Cấu hình log
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

DATA_DIR = Path("data/raw")
DB_DIR = Path("db")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

LAW_MAP = {
    "luat_bhyt.pdf": "Luật Bảo hiểm y tế",
    "luat_kham_chua_benh.pdf": "Luật Khám bệnh, chữa bệnh",
    "luat_duoc.pdf": "Luật Dược"
}

def clean_line(line: str) -> str:
    """Làm sạch khoảng trắng thừa trong một dòng."""
    line = re.sub(r"\s+", " ", line)
    return line.strip()

def chunk_legal_pdf(pdf_path: Path, chunk_size: int = 1500, chunk_overlap: int = 300) -> list[Document]:
    """
    Phân tích file PDF theo cấu trúc Chương, Mục, Điều của văn bản luật Việt Nam.
    Tự động gắn tiêu đề ngữ cảnh vào đầu mỗi chunk con và lưu đầy đủ metadata.
    """
    import re
    logger.info(f"Bắt đầu phân tích cấu trúc luật file: {pdf_path.name}")
    doc = fitz.open(str(pdf_path))
    source = pdf_path.name
    law_name = LAW_MAP.get(source, source.replace(".pdf", "").replace("_", " ").title())
    
    current_chapter = ""
    current_section = ""
    articles = []
    
    # Regex nhận dạng cấu trúc pháp lý
    # Điều 1., Điều 12a. hoặc "Điều 12: ", "Điều 12 "
    article_pattern = re.compile(r"^(?:[\"“'«]?\s*)Điều\s+(\d+[a-z]?)\s*[\.\:-]?\s*(.*)$", re.IGNORECASE)
    chapter_pattern = re.compile(r"^\s*Chương\s+([I|V|X|L|C|D|M|\d]+)\s*(.*)$", re.IGNORECASE)
    section_pattern = re.compile(r"^\s*Mục\s+(\d+)\s*(.*)$", re.IGNORECASE)
    
    # 1. Đọc tất cả các dòng cùng với số trang của chúng
    lines_with_pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        for line in text.split("\n"):
            cleaned = clean_line(line)
            if cleaned:
                lines_with_pages.append((cleaned, page_num))
                
    # 2. Nhóm các dòng thành các Điều
    current_article = None
    
    for line, page_num in lines_with_pages:
        # Kiểm tra Chương
        chap_match = chapter_pattern.match(line)
        if chap_match:
            current_chapter = line
            continue
            
        # Kiểm tra Mục
        sec_match = section_pattern.match(line)
        if sec_match:
            current_section = line
            continue
            
        # Kiểm tra Điều
        art_match = article_pattern.match(line)
        if art_match:
            if current_article:
                articles.append(current_article)
            
            art_num = art_match.group(1)
            art_title = art_match.group(2).strip()
            
            current_article = {
                "article_num": art_num,
                "article_title": art_title,
                "chapter": current_chapter,
                "section": current_section,
                "page_start": page_num,
                "page_end": page_num,
                "lines": [line]
            }
            continue
            
        if current_article:
            current_article["lines"].append(line)
            current_article["page_end"] = page_num
            
            # Cập nhật tiêu đề điều nếu chưa có ở dòng đầu tiên
            if not current_article["article_title"] and len(current_article["lines"]) == 2:
                # Nếu dòng tiếp theo không bắt đầu bằng số (ví dụ: "1. ...") thì có thể là tiêu đề điều bị xuống dòng
                if not re.match(r"^\d+\.", line):
                    current_article["article_title"] = line
                    
    if current_article:
        articles.append(current_article)
        
    # 3. Phân mảnh (chunking) từng Điều và thêm ngữ cảnh
    final_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Nếu không tìm thấy Điều nào (trường hợp văn bản không có cấu trúc Điều), fallback về RecursiveCharacterTextSplitter
    if not articles:
        logger.warning(f"Không tìm thấy cấu trúc Điều nào trong {source}. Thực hiện fallback chia thô.")
        full_text_with_pages = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            full_text_with_pages.append(Document(
                page_content=text,
                metadata={"source": source, "page": page_num, "law_name": law_name}
            ))
        return splitter.split_documents(full_text_with_pages)
        
    for art in articles:
        art_num = art["article_num"]
        art_title = art["article_title"]
        chapter = art["chapter"]
        section = art["section"]
        page_start = art["page_start"]
        page_end = art["page_end"]
        
        # Tiêu đề chung cho mỗi chunk của Điều này
        header_lines = [f"Luật: {law_name}"]
        if chapter:
            header_lines.append(chapter)
        if section:
            header_lines.append(section)
        header_lines.append(f"Điều {art_num}: {art_title}")
        header_str = "\n".join(header_lines) + "\n\n"
        
        # Nội dung Điều thô
        article_body = "\n".join(art["lines"])
        
        # Metadata chung
        meta = {
            "source": source,
            "law_name": law_name,
            "chapter": chapter,
            "section": section,
            "article": f"Điều {art_num}",
            "article_title": art_title,
            "page": page_start if page_start == page_end else f"{page_start}-{page_end}"
        }
        
        # Nếu nội dung của Điều quá lớn, chia nhỏ
        if len(article_body) > chunk_size:
            sub_chunks = splitter.split_text(article_body)
            for idx, sub_txt in enumerate(sub_chunks):
                # Tiền tố header vào mỗi chunk con để giữ ngữ cảnh
                chunk_content = f"{header_str}[Tiếp theo]\n{sub_txt}" if idx > 0 else f"{header_str}{sub_txt}"
                
                # Copy metadata và thêm index
                sub_meta = meta.copy()
                sub_meta["chunk_index"] = idx
                
                final_chunks.append(Document(page_content=chunk_content, metadata=sub_meta))
        else:
            chunk_content = f"{header_str}{article_body}"
            meta["chunk_index"] = 0
            final_chunks.append(Document(page_content=chunk_content, metadata=meta))
            
    logger.info(f"Đã xử lý file {pdf_path.name} thành {len(final_chunks)} chunks.")
    return final_chunks

def run_ingestion(data_dir: Path = DATA_DIR, db_dir: Path = DB_DIR) -> tuple[int, int]:
    """
    Quy trình chính nạp dữ liệu:
    1. Xóa cơ sở dữ liệu cũ (nếu có) để chuẩn bị cho schema metadata mới.
    2. Đọc và phân tích cấu trúc toàn bộ file PDF trong thư mục dữ liệu thô.
    3. Nhúng vector (embeddings) và lưu vào ChromaDB.
    Trả về: (số lượng văn bản nguồn, số lượng chunks đã được lưu).
    """
    if not data_dir.exists():
        logger.warning(f"Thư mục dữ liệu nguồn {data_dir} không tồn tại. Đang tạo mới...")
        data_dir.mkdir(parents=True, exist_ok=True)
        return 0, 0

    # 1. Quét tìm tất cả các file PDF
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"Không tìm thấy file PDF nào trong {data_dir}")
        return 0, 0

    # 2. Xóa cơ sở dữ liệu cũ
    if db_dir.exists():
        logger.info(f"Phát hiện cơ sở dữ liệu cũ tại {db_dir}. Đang tiến hành xóa để nạp dữ liệu theo định dạng mới...")
        try:
            shutil.rmtree(db_dir, ignore_errors=True)
            # Chờ một chút để Windows giải phóng file handle
            import time
            time.sleep(1.5)
        except Exception as e:
            logger.error(f"Lỗi khi xóa thư mục database cũ: {str(e)}. Hãy chắc chắn dừng các tiến trình đang kết nối tới DB này trước.")

    # 3. Phân mảnh tài liệu theo cấu trúc pháp lý
    chunks = []
    for pdf_path in pdf_files:
        try:
            # Sử dụng cấu hình chunk_size=1500, chunk_overlap=300 theo đề xuất
            pdf_chunks = chunk_legal_pdf(pdf_path, chunk_size=1500, chunk_overlap=300)
            chunks.extend(pdf_chunks)
        except Exception as e:
            logger.error(f"Lỗi khi xử lý file {pdf_path.name}: {str(e)}")

    if not chunks:
        logger.warning("Không có chunks nào được trích xuất thành công.")
        return len(pdf_files), 0

    # 4. Khởi tạo Embeddings
    logger.info(f"Đang tải mô hình nhúng: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )

    # 5. Lưu vào ChromaDB
    logger.info(f"Đang lưu {len(chunks)} chunks vào ChromaDB tại {db_dir}...")
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(db_dir.resolve()))
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=client
        )
        logger.info("Hoàn thành nạp dữ liệu và lưu trữ vào database thành công.")
    except Exception as e:
        logger.error(f"Lỗi khi lưu trữ vào ChromaDB: {str(e)}")
        raise e

    return len(pdf_files), len(chunks)

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    logger.info("Bắt đầu chạy quy trình ingest dữ liệu...")
    num_docs, num_chunks = run_ingestion()
    print(f"\n--- KẾT QUẢ INGEST ---")
    print(f"Số lượng file PDF đã xử lý: {num_docs}")
    print(f"Tổng số chunks đã nạp: {num_chunks}")