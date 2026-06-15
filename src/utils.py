import re
import unicodedata

def clean_text(text: str) -> str:
    """
    Làm sạch văn bản pháp luật trích xuất từ PDF:
    - Chuẩn hóa Unicode (NFKC).
    - Khôi phục các từ bị ngắt dòng bằng dấu gạch nối (ví dụ: 'chứng-\nchỉ' -> 'chứng chỉ').
    - Hợp nhất các dòng xuống dòng vô nghĩa thuộc cùng một câu/đoạn văn.
    - Giữ nguyên các ngắt dòng quan trọng định hình cấu trúc luật (Điều, Khoản, Điểm, bullet points).
    - Loại bỏ các khoảng trắng thừa.
    """
    if not text:
        return ""
    
    # 1. Chuẩn hóa Unicode (phục vụ tìm kiếm chính xác các dấu tiếng Việt)
    text = unicodedata.normalize("NFKC", text)
    
    # 2. Sửa các từ bị ngắt dòng bởi dấu gạch nối ở cuối dòng
    text = re.sub(r"(\w+)-\n\s*(\w+)", r"\1\2", text)
    
    # 3. Loại bỏ ngắt dòng vô nghĩa
    lines = text.split("\n")
    cleaned_lines = []
    
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        cleaned_lines.append(line_str)
        
    if not cleaned_lines:
        return ""
        
    merged_text = cleaned_lines[0]
    for i in range(1, len(cleaned_lines)):
        prev_line = cleaned_lines[i-1]
        curr_line = cleaned_lines[i]
        
        # Các dấu hiệu cho thấy dòng mới bắt đầu một mục cấu trúc pháp lý mới
        is_new_element = (
            re.match(r"^(điều|chương|mục|phần|nghị định|luật)\s+\d+", curr_line, re.IGNORECASE) or
            re.match(r"^\d+\.\s+", curr_line) or          # Ví dụ: "1. Người bệnh..."
            re.match(r"^[a-zđ]\)\s+", curr_line) or        # Ví dụ: "a) Có chứng chỉ...", "đ) ..."
            curr_line.startswith("-") or                   # Đầu dòng bằng gạch ngang
            curr_line.startswith("*")                      # Đầu dòng bằng dấu sao
        )
        
        # Dòng trước đó kết thúc bằng một dấu chấm câu ngắt đoạn rõ ràng
        ends_with_terminator = prev_line[-1] in [".", "?", "!", ":", ";"] if prev_line else False
        
        if is_new_element or ends_with_terminator:
            merged_text += "\n" + curr_line
        else:
            # Hợp nhất với một khoảng trắng
            merged_text += " " + curr_line
            
    # 4. Loại bỏ khoảng trắng thừa (giữa các từ và giữa các đoạn)
    merged_text = re.sub(r"[ \t]+", " ", merged_text)
    # Chuẩn hóa số dòng trống liên tiếp (tối đa 2 dòng trống liên tiếp)
    merged_text = re.sub(r"\n\s*\n", "\n\n", merged_text)
    
    return merged_text.strip()
