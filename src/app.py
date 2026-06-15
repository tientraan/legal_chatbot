import sys
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import glob
from pathlib import Path

# Thêm thư mục gốc của dự án vào sys.path để python nhận diện gói 'src'
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
from dotenv import load_dotenv

# Tải cấu hình
load_dotenv()

# Thiết lập tiêu đề và cấu hình trang trước khi import RAG
st.set_page_config(
    page_title="Hỏi Đáp Pháp Luật Khám Chữa Bệnh",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh giao diện cao cấp
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Cấu hình Font chữ chính */
    html, body, .stApp {
        font-family: 'Outfit', sans-serif !important;
        background-color: #0f172a !important;
    }
    
    /* Màu chữ chính mặc định là trắng */
    .stMarkdown, p, li, h1, h2, h3, h4, h5, h6, [data-testid="stHeader"], span {
        color: #ffffff !important;
    }
    
    /* Đối với ô nhập liệu có nền trắng thì chữ màu đen */
    div[data-baseweb="input"] input, 
    div[data-baseweb="textarea"] textarea,
    div[data-baseweb="select"] select,
    select,
    input {
        color: #0f172a !important; /* Chữ đen */
        background-color: #ffffff !important; /* Nền trắng */
    }
    [data-testid="stChatInput"] * {
    color: #ffffff !important;
    }
    
    /* Ô nhập chat input cuối trang (Nền tối, chữ trắng) */
    div[data-testid="stChatInput"] textarea {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #ffffff !important; /* Chữ trắng */
    }
    
    /* Custom Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155 !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown h1, 
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #38bdf8 !important;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    /* Thẻ thống kê (Metric cards) tự chế */
    .metric-container {
        display: flex;
        flex-direction: column;
        gap: 12px;
        margin-bottom: 24px;
        margin-top: 15px;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(8px);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
        box-shadow: 0 10px 15px -3px rgba(56, 189, 248, 0.15);
    }
    
    .metric-val {
        font-size: 28px;
        font-weight: 700;
        color: #38bdf8 !important;
        line-height: 1.2;
    }
    
    .metric-lbl {
        font-size: 12px;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
        font-weight: 500;
    }
    
    /* Thiết lập nút bấm Sidebar */
    .stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        box-shadow: 0 4px 6px -1px rgba(2, 132, 199, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
        margin-top: 10px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 15px -3px rgba(56, 189, 248, 0.4) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
    }
    
    /* Custom khung chat */
    div[data-testid="stChatMessage"] {
        background-color: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        padding: 15px !important;
        margin-bottom: 15px !important;
    }
    
    div[data-testid="stChatMessage"][data-classname="stChatMessage-user"] {
        background-color: rgba(2, 132, 199, 0.15) !important;
        border-color: rgba(56, 189, 248, 0.4) !important;
    }
    
    /* Vùng hiển thị nguồn */
    .sources-title {
        font-size: 14px;
        font-weight: 600;
        color: #38bdf8;
        margin-top: 12px;
        margin-bottom: 6px;
    }
    
    .source-item {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid #475569;
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 13px;
        color: #cbd5e1;
        margin-bottom: 5px;
        display: inline-block;
        margin-right: 8px;
    }
    /* Ô nhập câu hỏi */
[data-testid="stChatInputTextArea"] {
    color: #ffffff !important;
    background-color: #1e293b !important;
}

/* Placeholder */
[data-testid="stChatInputTextArea"]::placeholder {
    color: #ffffff !important;
    opacity: 1 !important;
}

/* Khi focus */
[data-testid="stChatInputTextArea"]:focus {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# Import các hàm nghiệp vụ
from src.rag import ask, get_vectordb
from src.ingest import run_ingestion

DATA_DIR = Path("data/raw")
DB_DIR = Path("db")

# Đếm số lượng văn bản nguồn
pdf_files = list(DATA_DIR.glob("*.pdf"))
num_docs = len(pdf_files)

# Đếm số lượng chunks trong ChromaDB
num_chunks = 0
vectordb = get_vectordb()
if vectordb is not None:
    try:
        num_chunks = vectordb._collection.count()
    except Exception:
        num_chunks = 0

# Tự động nạp dữ liệu nếu chưa có cơ sở dữ liệu hoặc trống (khi chạy lần đầu trên Cloud)
if num_chunks == 0 and num_docs > 0:
    st.info("Cơ sở dữ liệu trống hoặc chưa được nạp. Đang tự động nạp và phân mảnh dữ liệu (ingestion)...")
    try:
        processed_docs, processed_chunks = run_ingestion(DATA_DIR, DB_DIR)
        
        # Reset biến toàn cục để load lại database mới
        from src import rag
        rag._vectordb = None
        
        vectordb = get_vectordb()
        if vectordb is not None:
            num_chunks = vectordb._collection.count()
            
        st.success(f"Tự động nạp dữ liệu thành công! Đã xử lý {processed_docs} tài liệu thành {processed_chunks} chunks.")
    except Exception as e:
        st.error(f"Lỗi khi tự động nạp dữ liệu: {str(e)}")

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## ⚖️ Quản trị Cơ sở dữ liệu")
    st.markdown("Thống kê tài liệu pháp lý đã được nạp vào hệ thống RAG:")
    
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-card">
            <div class="metric-val">{num_docs}</div>
            <div class="metric-lbl">Văn bản luật nguồn (.pdf)</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">{num_chunks:,}</div>
            <div class="metric-lbl">Đoạn văn bản (Chunks)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Thao tác")
    if st.button("🔄 Làm mới Cơ sở Dữ liệu"):
        with st.spinner("Đang đọc lại tài liệu, cắt nhỏ và lưu vào cơ sở dữ liệu vector. Vui lòng đợi..."):
            try:
                # Chạy ingest lại
                processed_docs, processed_chunks = run_ingestion(DATA_DIR, DB_DIR)
                
                # Giải phóng biến vectordb toàn cục trong rag để load lại database mới
                from src import rag
                rag._vectordb = None
                
                st.success(f"Nạp dữ liệu thành công! Đã xử lý {processed_docs} văn bản thành {processed_chunks} chunks.")
                # Rerun để làm mới sidebar metrics và trạng thái
                st.rerun()
            except Exception as e:
                st.error(f"Đã xảy ra lỗi khi nạp dữ liệu: {str(e)}")
                
    st.markdown("---")
    st.markdown("<p style='font-size: 12px; color: #64748b; text-align: center;'>Hệ thống hỏi đáp Luật Khám chữa bệnh v1.0.0</p>", unsafe_allow_html=True)

# --- KHUNG CHAT CHÍNH ---
st.markdown("<h1 style='text-align: center; color: #38bdf8; font-weight: 700; margin-bottom: 30px;'>⚖️ Chatbot Hỏi Đáp Pháp Luật Khám Chữa Bệnh</h1>", unsafe_allow_html=True)

# Khởi tạo lịch sử chat trong session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử hội thoại
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Nếu có nguồn tham khảo đi kèm, hiển thị dạng expander đẹp mắt
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("📖 Xem căn cứ và nguồn luật trích dẫn"):
                st.markdown("<div class='sources-title'>Các đoạn luật liên quan được tìm thấy:</div>", unsafe_allow_html=True)
                for i, doc in enumerate(message["sources"], 1):
                    source_file = doc.metadata.get("source", "Không rõ nguồn")
                    page_num = doc.metadata.get("page", "Không rõ trang")
                    st.markdown(f"<span class='source-item'>📄 <b>{source_file}</b> - Trang {page_num}</span>", unsafe_allow_html=True)
                    st.markdown(f"```text\n{doc.page_content}\n```")

# Ô nhập câu hỏi từ người dùng
if question := st.chat_input("Nhập câu hỏi pháp lý của bạn ở đây..."):
    # Hiển thị câu hỏi của user
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
        
    # Xử lý và hiển thị câu trả lời từ chatbot
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        with st.spinner("Đang nghiên cứu các văn bản luật và soạn câu trả lời..."):
            answer, retrieved_docs = ask(question)
            
        # Hiển thị câu trả lời
        response_placeholder.write(answer)
        
        # Hiển thị nguồn trích dẫn
        if retrieved_docs:
            with st.expander("📖 Xem căn cứ và nguồn luật trích dẫn"):
                st.markdown("<div class='sources-title'>Các đoạn luật liên quan được tìm thấy:</div>", unsafe_allow_html=True)
                for i, doc in enumerate(retrieved_docs, start=1):
                    source_file = doc.metadata.get("source", "Không rõ nguồn")
                    page_num = doc.metadata.get("page", "Không rõ trang")
                    st.markdown(f"<span class='source-item'>📄 <b>{source_file}</b> - Trang {page_num}</span>", unsafe_allow_html=True)
                    st.markdown(f"```text\n{doc.page_content}\n```")
                    
        # Lưu vào lịch sử chat
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": retrieved_docs
        })
