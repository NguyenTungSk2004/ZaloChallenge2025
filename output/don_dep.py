import json
import re
import os

# Hàm này sẽ tìm và xóa khối chữ ký số
def clean_signature(text):
    if not isinstance(text, str):
        return text
    # Dùng regex để tìm và thay thế khối chữ ký
    pattern = r"Người ký: CỔNG THÔNG TIN[\s\S]*?\+07:00"
    return re.sub(pattern, "", text).strip()

# ----- SỬA TÊN FILE NẾU CẦN -----
input_file = 'E:/Zalo Challenge 2025/Build_RAG/output/luat_data_cleaned_2.json' # File gốc của bạn
output_file = 'E:/Zalo Challenge 2025/Build_RAG/output/luat_data_final_2.json' # File đã sạch

data = []
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

cleaned_data = []
for dieu in data:
    # Dọn dẹp ở cấp "khoan"
    for khoan in dieu.get("khoan", []):
        khoan["noi_dung"] = clean_signature(khoan.get("noi_dung"))
        
        # Dọn dẹp ở cấp "diem"
        for diem in khoan.get("diem", []):
            diem["noi_dung"] = clean_signature(diem.get("noi_dung"))
            
    cleaned_data.append(dieu)

# Lưu file JSON đã sạch hoàn toàn
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

print(f"Đã dọn dẹp và lưu vào file: {output_file}")

# (Bạn cũng nên làm tương tự cho file luat_data_cleaned_2.json)