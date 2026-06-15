# ⚖️ Chatbot Hỏi Đáp Pháp Luật Khám Chữa Bệnh tại Việt Nam (RAG)

Dự án này là hệ thống chatbot RAG (Retrieval-Augmented Generation) thông minh chuyên hỗ trợ giải đáp câu hỏi pháp lý dựa trên các văn bản luật: Luật Khám bệnh, chữa bệnh, Luật Bảo hiểm y tế và Luật Dược.

Hệ thống được phát triển bằng **Python 3.11**, sử dụng **ChromaDB** cho cơ sở dữ liệu Vector, mô hình nhúng cục bộ **sentence-transformers**, thuật toán tìm kiếm kết hợp **Hybrid Search (Vector + BM25)**, giải thuật **Reranking** ưu tiên từ khóa và mô hình ngôn ngữ lớn **Gemini 2.5 Flash** để tạo câu trả lời tự nhiên, chính xác và có nguồn trích dẫn cụ thể.

---

## 📂 Cấu trúc Thư mục

```text
kcb-rag-app/
├── data/
│   └── raw/              # Chứa các file PDF luật đầu vào (luat_kham_chua_benh.pdf, luat_bhyt.pdf, luat_duoc.pdf)
├── db/                   # Thư mục lưu trữ cơ sở dữ liệu vector ChromaDB sau khi ingest
├── src/
│   ├── utils.py          # Hàm tiền xử lý và làm sạch văn bản tiếng Việt trích xuất từ PDF
│   ├── ingest.py         # Script nạp dữ liệu: trích xuất, làm sạch, chunking, embed và lưu vào ChromaDB
│   ├── rag.py            # Lô-gíc truy vấn lai (Hybrid Search) + Reranking + Kết nối Gemini 2.5 Flash
│   └── app.py            # Giao diện người dùng Streamlit (Premium Custom Theme)
├── requirements.txt      # Danh sách thư viện phụ thuộc
├── .env                  # Cấu hình biến môi trường (API Key)
└── README.md             # Tài liệu hướng dẫn sử dụng (File này)
```

---

## 🛠️ Hướng dẫn Cài đặt & Cấu hình

### 1. Cài đặt Thư viện
Chạy lệnh sau để cài đặt đầy đủ các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

### 2. Cấu hình Khóa API Gemini
Tạo/Cập nhật file `.env` ở thư mục gốc của dự án với khóa API Google Gemini của bạn:
```env
GOOGLE_API_KEY=your_actual_gemini_api_key_here
```

---

## 🚀 Hướng dẫn Chạy ứng dụng

### Bước 1: Nạp và nhúng dữ liệu Luật vào Database
Trước khi sử dụng chatbot lần đầu, bạn cần chạy quy trình nạp dữ liệu để hệ thống quét các file PDF trong `data/raw/` và xây dựng cơ sở dữ liệu ChromaDB:

```bash
python -m src.ingest
```
*Lưu ý: Quá trình này sẽ tải mô hình nhúng `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` về máy và chạy nhúng cục bộ. Thời gian chạy khoảng từ 30 giây đến vài phút tùy theo CPU của bạn.*

### Bước 2: Khởi chạy Giao diện Chatbot Streamlit
Khởi động ứng dụng giao diện web Streamlit bằng lệnh:

```bash
streamlit run src/app.py
```
*Hoặc: `python -m streamlit run src/app.py`*

Giao diện sẽ tự động mở trong trình duyệt của bạn tại địa chỉ mặc định `http://localhost:8501`.

---

## 🧠 Kiến trúc Hệ thống & Điểm Nổi bật

### 1. Làm sạch dữ liệu nâng cao (`src/utils.py`)
Khi trích xuất văn bản từ PDF, dữ liệu thường bị ngắt quãng, chứa ký tự rác hoặc ngắt dòng vô nghĩa chia đôi câu làm giảm hiệu quả tìm kiếm. Hàm `clean_text` giải quyết việc này bằng cách:
*   Chuẩn hóa văn bản tiếng Việt sang dạng Unicode chuẩn (NFKC).
*   Khôi phục các từ bị ngắt dòng bởi dấu gạch nối (Ví dụ: `chứng-\nchỉ` -> `chứng chỉ`).
*   Hợp nhất các câu bị xuống dòng nửa chừng do định dạng cột trong PDF, đồng thời bảo toàn các ranh giới dòng của các đề mục cấu trúc luật như **"Điều..."**, **"Khoản..."**, hoặc các danh sách có dạng **"a)"**, **"b)"**, **"-"**.

### 2. Tìm kiếm Kết hợp - Hybrid Search (`src/rag.py`)
Chatbot không chỉ dựa vào Vector Search hay Keyword Search đơn lẻ, mà kết hợp cả hai để mang lại độ chính xác tối ưu:
*   **Vector Search (ChromaDB)**: Truy xuất 10 đoạn văn bản có sự tương đồng về ngữ nghĩa nhất với câu hỏi của người dùng (bất kể họ dùng từ đồng nghĩa hay cách diễn đạt khác).
*   **Keyword Search (BM25)**: Truy xuất 10 đoạn văn bản chứa chính xác các từ khóa chính xác từ câu hỏi. Cực kỳ hiệu quả khi tìm kiếm các thuật ngữ chuyên môn hoặc số hiệu điều luật cụ thể.

### 3. Reranking và Ưu tiên Từ khóa Luật định (`src/rag.py`)
Kết quả của Vector Search và BM25 Search được gộp lại, loại bỏ trùng lặp và chấm điểm lại:
*   Sử dụng giải thuật **Reciprocal Rank Fusion (RRF)** làm thước đo cơ sở để kết hợp thứ hạng của hai phương pháp.
*   **Keyword Boost**: Cộng thêm điểm thưởng rất lớn cho các đoạn văn bản có chứa các từ khóa trọng tâm: `người bệnh`, `quyền`, `nghĩa vụ`, `bảo hiểm y tế`, `hồ sơ bệnh án`, `giấy phép hành nghề`, `thuốc`, `dược`.
*   Chọn ra **Top 10** đoạn văn bản có điểm Rerank cao nhất làm ngữ cảnh đưa vào LLM.

### 4. Sinh Câu trả lời Nghiêm ngặt (`src/rag.py`)
Mô hình Gemini 2.5 Flash được cấu hình chặt chẽ để đóng vai trò là một Chuyên gia Pháp luật:
*   Chỉ trả lời dựa trên ngữ cảnh cung cấp.
*   Trích dẫn rõ ràng tên file luật nguồn và số trang cụ thể.
*   Nếu thông tin không nằm trong dữ liệu, hệ thống trả về chính xác câu: **"Tôi chưa tìm thấy căn cứ trong dữ liệu đã nạp."** để loại bỏ hoàn toàn hiện tượng ảo tưởng thông tin (hallucination).
