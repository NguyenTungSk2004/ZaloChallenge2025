import json
import re

input_path = "E:/Zalo Challenge 2025/Build_RAG/output/luat_data_2.json"
output_path = "E:/Zalo Challenge 2025/Build_RAG/output/luat_data_cleaned_2.json"

def parse_dieu_content(text):
    """
    Chuẩn hóa nội dung Điều thành các khoản (1,2,3...) và điểm (a,b,c...)
    """
    # Bắt đầu dòng mới có số + dấu chấm (vd: "1. ", "2. ")
    khoan_pattern = r"^\s*(\d+)\.\s"
    khoan_splits = re.split(khoan_pattern, text, flags=re.MULTILINE)
    
    khoan_list = []
    if len(khoan_splits) <= 1:
        # Không có khoản, chỉ có toàn văn điều
        return [{"so": None, "noi_dung": text.strip(), "diem": []}]

    # Bỏ phần đầu trước khoản 1
    for i in range(1, len(khoan_splits), 2):
        so = khoan_splits[i]
        noi_dung = khoan_splits[i+1].strip()

        # Tách điểm a), b), c)
        diem_pattern = r"^\s*([a-z])\)\s"
        diem_splits = re.split(diem_pattern, noi_dung, flags=re.MULTILINE)
        diem_list = []
        if len(diem_splits) > 1:
            for j in range(1, len(diem_splits), 2):
                ky_tu = diem_splits[j]
                diem_text = diem_splits[j+1].strip()
                diem_list.append({"ky_tu": ky_tu, "noi_dung": diem_text})
        
        khoan_list.append({
            "so": so,
            "noi_dung": noi_dung if not diem_list else "",
            "diem": diem_list
        })
    return khoan_list

# ===== Chuẩn hóa toàn bộ file luật =====
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

cleaned = []
for dieu in data:
    dieu_clean = dieu.copy()
    dieu_clean["khoan"] = parse_dieu_content(dieu["noi_dung"])
    cleaned.append(dieu_clean)
    # Xóa phần "noi_dung" gốc nếu muốn
    del dieu_clean["noi_dung"]

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(cleaned, f, ensure_ascii=False, indent=2)

print(f"✅ Đã chuẩn hóa và lưu vào: {output_path}")
