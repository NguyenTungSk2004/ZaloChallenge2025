import pymupdf
import re
import json
import os

def clean_text(text):
    """
    Hàm dọn header, footer, số trang không cần thiết
    """
    print("🧹 Đang dọn dẹp văn bản...")
    text = re.sub(r"CÔNG BÁO/Số.*?\n", "", text)            # Xóa header
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)              # Xóa số trang
    text = re.sub(r"\(Xem tiếp Công báo.*?\)\n", "", text)   # Xóa footer
    text = re.sub(r"VĂN PHÒNG CHÍNH PHỦ XUẤT BẢN[\s\S]*", "", text)  # Xóa phần cuối
    text = re.sub(r"\n{3,}", "\n\n", text)                   # Giảm dòng trống
    return text.strip()

def parse_law_to_json(file_pdf_path, output_json_path):
    print(f"📂 Đang mở file: {file_pdf_path}")
    full_text = ""
    try:
        doc = pymupdf.open(file_pdf_path)
        for i, page in enumerate(doc):
            page_text = page.get_text("text")
            full_text += page_text + "\n"
            if i == 0:  # chỉ in thử trang đầu
                print("\n=== 🔍 50 dòng đầu của file sau khi đọc PDF ===")
                print("\n".join(page_text.splitlines()[:50]))
        doc.close()
    except Exception as e:
        print(f"❌ Lỗi khi đọc PDF: {e}")
        return

    # Dọn dẹp
    cleaned_text = clean_text(full_text)

    # DEBUG: xem có “Điều” xuất hiện không
    test_dieu = re.findall(r"Điều\s*\d+", cleaned_text)
    print(f"\n🔎 Phát hiện {len(test_dieu)} lần xuất hiện từ 'Điều' trong văn bản.")

    # Kiểm tra có “Chương” không
    test_chuong = re.findall(r"Chương\s+[IVXLCDM]+", cleaned_text)
    print(f"📘 Phát hiện {len(test_chuong)} lần xuất hiện từ 'Chương' trong văn bản.\n")

    # Nếu không thấy chương nào → in debug ra 200 ký tự đầu
    if not test_chuong:
        print("⚠️ Không tìm thấy từ 'Chương'! Có thể file PDF bị lỗi dòng hoặc không xuống dòng đúng cách.")
        print("Đoạn văn bản đầu tiên:")
        print(cleaned_text[:500])

    # Regex linh hoạt hơn
    pattern_chuong = re.compile(
    r"Chương\s*([IVXLCDM]+)\s*\n*(.*?)\n([\s\S]*?)(?=\nChương\s*[IVXLCDM]+|\Z)",
    re.IGNORECASE
    )   

    pattern_dieu = re.compile(
    r"Điều\s*(\d+)\.?\s*(.*?)\n([\s\S]*?)(?=\nĐiều\s*\d+\.|\nChương\s*[IVXLCDM]+|\Z)",
    re.IGNORECASE
    )

    # Cắt từ Chương I
    '''
    match_start = re.search(r"Chương I", cleaned_text, re.IGNORECASE)
    if not match_start:
        print("⚠️ Không tìm thấy 'Chương I' — có thể định dạng PDF khác nhau.")
        print("Thử lấy toàn bộ văn bản để tiếp tục phân tích.\n")
        meaningful_text = cleaned_text
    else:
        meaningful_text = cleaned_text[match_start.start():]
    '''
    meaningful_text = cleaned_text
    # ==== DEBUG CHƯƠNG II ====
    idx = meaningful_text.lower().find("chương ii")
    if idx != -1:
        print("\n===== 📜 ĐOẠN VĂN BẢN QUANH 'CHƯƠNG II' =====")
        print(meaningful_text[idx-100:idx+300])
    else:
        print("\n⚠️ Không tìm thấy 'CHƯƠNG II' trong text để debug.")
    
    idx2 = meaningful_text.lower().find("điều 1")
    if idx2 != -1:
        print("\n===== 📜 ĐOẠN VĂN BẢN QUANH 'ĐIỀU 1' =====")
        print(meaningful_text[idx2-50:idx2+300])
    else:
        print("\n⚠️ Không tìm thấy 'ĐIỀU 1' trong text để debug.")
    # Debug thêm
    chuong_test = re.findall(r"Chương\s*[IVXLCDM]+", meaningful_text)
    print(f"📘 Debug: Tìm thấy {len(chuong_test)} chương trong file")
    if len(chuong_test) > 0:
        print("   Ví dụ:", chuong_test[:5])

    dieu_test = re.findall(r"Điều\s*\n*\d+", meaningful_text)
    print(f"🔎 Debug: Tìm thấy {len(dieu_test)} điều (bằng regex linh hoạt)")
    if len(dieu_test) > 0:
        print("   Ví dụ:", dieu_test[:5])    

    # Tách các chương
    chuong_blocks = pattern_chuong.split(meaningful_text)
    print(f"🧱 Phát hiện {len(chuong_blocks)//3} chương trong văn bản.")

    results = []
    for chuong in pattern_chuong.finditer(meaningful_text):
        current_chuong_so = chuong.group(1).strip()
        current_chuong_ten = chuong.group(2).strip()
        chuong_content = chuong.group(3)

        print(f"➡️ Đang xử lý Chương {current_chuong_so}: {current_chuong_ten}")

        dieu_matches = pattern_dieu.finditer(chuong_content)
        count_in_chuong = 0
        for match in dieu_matches:
            dieu_so = match.group(1).strip()
            dieu_ten = match.group(2).strip()
            noi_dung = match.group(3).strip()
            count_in_chuong += 1

            if count_in_chuong <= 2:
                print(f"   🔹 Điều {dieu_so}: {dieu_ten[:50]}...")

            results.append({
                "nguon": "36/2024/QH15",
                "ten_luat": "LUẬT TRẬT TỰ, AN TOÀN GIAO THÔNG ĐƯỜNG BỘ",
                "chuong_so": current_chuong_so,
                "chuong_ten": current_chuong_ten,
                "dieu_so": dieu_so,
                "dieu_ten": dieu_ten,
                "noi_dung": noi_dung
            })
        print(f"   ↳ Tìm thấy {count_in_chuong} điều trong chương này.\n")

    # Lưu JSON
    print(f"✅ Tổng cộng {len(results)} điều được phân tích.")
    output_dir = os.path.dirname(output_json_path)
    if not os.path.exists(output_dir) and output_dir != "":
        os.makedirs(output_dir)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"📁 Đã lưu kết quả vào: {output_json_path}")

# === Chạy chính ===
file_pdf_path = "E:/Zalo Challenge 2025/36-2024-qh15_tiep.pdf"
output_json_path = "E:/Zalo Challenge 2025/Build_RAG/output/luat_data_2.json"

parse_law_to_json(file_pdf_path, output_json_path)
