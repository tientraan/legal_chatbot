import os
import sys
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

def process_pdf(pdf_path: Path) -> list[Document]:
    """
    Đọc một file PDF sử dụng PyMuPDF, làm sạch văn bản từng trang và tạo danh sách các Document thô.
    """
    raw_documents = []
    try:
        logger.info(f"Bắt đầu đọc: {pdf_path.name}")
        doc = fitz.open(str(pdf_path))
        
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if not text:
                continue
            
            # Làm sạch văn bản tiếng Việt
            cleaned = clean_text(text)
            if not cleaned:
                continue
                
            raw_doc = Document(
                page_content=cleaned,
                metadata={
                    "source": pdf_path.name,
                    "page": page_num
                }
            )
            raw_documents.append(raw_doc)
            
        logger.info(f"Đã trích xuất {len(raw_documents)} trang từ {pdf_path.name}")
    except Exception as e:
        logger.error(f"Lỗi khi xử lý file {pdf_path.name}: {str(e)}")
        
    return raw_documents

def run_ingestion(data_dir: Path = DATA_DIR, db_dir: Path = DB_DIR) -> tuple[int, int]:
    """
    Quy trình chính nạp dữ liệu:
    1. Xóa cơ sở dữ liệu cũ (nếu có).
    2. Đọc và trích xuất toàn bộ file PDF trong thư mục dữ liệu thô.
    3. Cắt văn bản thành các chunks có độ dài phù hợp.
    4. Nhúng vector (embeddings) và lưu vào ChromaDB.
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

    # 2. Xóa cơ sở dữ liệu cũ để tránh dữ liệu trùng lặp khi nạp lại
    if db_dir.exists():
        logger.info(f"Đang xóa cơ sở dữ liệu cũ tại {db_dir}...")
        try:
            shutil.rmtree(db_dir, ignore_errors=True)
            # Chờ một chút để Windows giải phóng file handle
            import time
            time.sleep(1)
        except Exception as e:
            logger.error(f"Không thể xóa thư mục database: {str(e)}")

    # 3. Trích xuất text từ các file PDF
    all_pages_docs = []
    for pdf_path in pdf_files:
        all_pages_docs.extend(process_pdf(pdf_path))

    if not all_pages_docs:
        logger.warning("Không có dữ liệu văn bản nào được trích xuất thành công.")
        return len(pdf_files), 0

    # 4. Chunking (Cắt nhỏ văn bản)
    # chunk_size = 1000, chunk_overlap = 200 theo yêu cầu đề bài
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_documents(all_pages_docs)
    logger.info(f"Đã chia {len(all_pages_docs)} trang tài liệu thành {len(chunks)} chunks.")

    # 5. Khởi tạo Embeddings
    logger.info(f"Đang tải mô hình nhúng: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )

    # 6. Lưu vào ChromaDB
    logger.info(f"Đang lưu {len(chunks)} chunks vào ChromaDB tại {db_dir}...")
    try:
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(db_dir)
        )
        logger.info("Hoàn thành lưu trữ vào database thành công.")
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