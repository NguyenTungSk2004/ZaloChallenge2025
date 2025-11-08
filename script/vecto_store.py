from langchain_chroma import Chroma
from langchain_core.documents import Document
import json
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Chunk
json_file_path = [
    'E:/Zalo Challenge 2025/Build_RAG/output/luat_data_final.json',
    'E:/Zalo Challenge 2025/Build_RAG/output/luat_data_final_2.json'
]

data = []
for file_path in json_file_path:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = json.load(f)
            data.extend(file_content) # gộp list
    except Exception as e:
        print(f"Lỗi khi đọc file {file_path}: {e}")
print(f"--- Đã tải thành công {len(data)} 'Điều' {len(json_file_path)} file. ---")

all_documents = []
for dieu in data: 
    for khoan in dieu.get("khoan", []):
        
        diem_list = khoan.get("diem", [])
        
        if diem_list:
            # CHIẾN LƯỢC 1: Nếu Khoản có 'Điểm' (a, b, c...),
            # chúng ta sẽ chunk theo từng 'Điểm'.
            for diem in diem_list:
                # Tạo nội dung (context) đầy đủ
                content = f"Điều {dieu['dieu_so']}. {dieu['dieu_ten']}\n"
                content += f"Khoản {khoan['so']}.\n" 
                
                # Thêm nội dung của Khoản (nếu có) để làm rõ nghĩa
                if khoan.get("noi_dung"):
                    content += f"{khoan['noi_dung']}\n"
                    
                content += f"Điểm {diem['ky_tu']}) {diem['noi_dung']}"
                
                # Tạo Document
                doc = Document(
                    page_content=content.replace("\n", " ").strip(), # Xóa ngắt dòng
                    metadata={
                        "nguon": dieu["nguon"],
                        "chuong": dieu["chuong_ten"],
                        "dieu": f"Điều {dieu['dieu_so']}",
                        "khoan": f"Khoản {khoan['so']}",
                        "diem": f"Điểm {diem['ky_tu']}" # Metadata chi tiết
                    }
                )
                all_documents.append(doc)
        
        elif khoan.get("noi_dung"):
            # CHIẾN LƯỢC 2: Nếu Khoản KHÔNG có 'Điểm' (chỉ có nội dung chính),
            # chúng ta chunk theo 'Khoản' đó.
            content = f"Điều {dieu['dieu_so']}. {dieu['dieu_ten']}\n"
            
            # Xử lý trường hợp khoan['so'] là null (như ở Điều 1)
            khoan_so_text = f"Khoản {khoan['so']}." if khoan['so'] else ""
            content += f"{khoan_so_text} {khoan['noi_dung']}".strip()
            
            # Tạo Document
            doc = Document(
                page_content=content.replace("\n", " ").strip(), # Xóa ngắt dòng
                metadata={
                    "nguon": dieu["nguon"],
                    "chuong": dieu["chuong_ten"],
                    "dieu": f"Điều {dieu['dieu_so']}",
                    "khoan": f"Khoản {khoan['so']}" if khoan['so'] else "Nội dung chính"
                }
            )
            all_documents.append(doc)

print(f"--- Đã tạo được {len(all_documents)} chunks tài liệu. ---")

# Kiểm tra thử chunk đầu tiên
if all_documents:
    print("\n--- VÍ DỤ CHUNK ĐẦU TIÊN ---")
    print("NỘI DUNG:")
    print(all_documents[0].page_content)
    print("\nMETADATA:")
    print(all_documents[0].metadata)

# 2. Embedding
model_path = "E:/Zalo Challenge 2025/Build_RAG/model/vinai"

model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print("--- Đã tải mô hình embedding thành công. ---")

# Thử nghiệm
text = "Đây là câu thử nghiệm"
query_result = embeddings.embed_query(text)
print(f"Độ dài vector: {len(query_result)}")

# 3. Vector Store
persist_directory = "E:/Zalo Challenge 2025/Build_RAG/db_luat"

vectordb = Chroma.from_documents(
    documents=all_documents,
    embedding=embeddings,
    persist_directory=persist_directory
)

print("--- Đã tạo và lưu Vector Database thành công. ---")