import os
import re
import json
from docx import Document

DOCX_FILE = "scripts/khong_img.docx"
OUTPUT_PARSED_QUESTIONS = "parsed_questions.json"

def parse_docx(file_path):
    print(f"🔄 Đang đọc file: {file_path}...")
    try:
        doc = Document(file_path)
    except Exception as e:
        print(f"❌ Lỗi khi mở file .docx: {e}")
        return None

    questions = []
    current_question = None
    
    # Regex để tìm câu hỏi (ví dụ: "Câu 209.")
    question_regex = re.compile(r"^(Câu \d+)\.(.+)", re.IGNORECASE)

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        q_match = question_regex.match(text)
        
        if q_match:
            # Nếu tìm thấy câu hỏi mới, lưu câu hỏi cũ (nếu có)
            if current_question:
                questions.append(current_question)
            
            # Bắt đầu câu hỏi mới
            current_question = {
                "id": q_match.group(1).strip(), # "Câu 209"
                "question": q_match.group(2).strip(), # "Khi khởi hành ô tô..."
                "answer": "" # Sẽ được điền ở dòng tiếp theo
            }
        
        # Giả định: Đáp án nằm ngay dòng bên dưới câu hỏi
        elif current_question and not current_question["answer"]:
             if text.lower().startswith("đáp án:"):
                 current_question["answer"] = text.split(":", 1)[1].strip()
             else:
                 # Xử lý trường hợp đáp án không có chữ "Đáp án:"
                 # (như ví dụ của bạn)
                 current_question["answer"] = text
                 
    # Lưu câu hỏi cuối cùng
    if current_question:
        questions.append(current_question)

    print(f"✅ Đã parse được {len(questions)} câu hỏi.")
    return questions

# --- Chạy script ---
parsed_data = parse_docx(DOCX_FILE)
if parsed_data:
    with open(OUTPUT_PARSED_QUESTIONS, 'w', encoding='utf-8') as f:
        json.dump(parsed_data, f, ensure_ascii=False, indent=4)
    print(f"✅ Đã lưu vào file: {OUTPUT_PARSED_QUESTIONS}")