from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Khởi tạo lại Embedding model
model_path = "E:/Zalo Challenge 2025/Build_RAG/model/vinai"
model_kwargs = {'device': 'cuda'}
enconde_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs=model_kwargs,
    encode_kwargs=enconde_kwargs
)

# Đường dẫn lưu vecto db
persist_directory = "E:/Zalo Challenge 2025/Build_RAG/db_luat"

# Tải vecto db
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

print("--- Đã tải Vector DB thành công. ---")

# Thử truy vấn
query = "Quy định về sử dụng đèn"
results = vectordb.similarity_search(query, k=3)

print(f"\n--- KẾT QUẢ TRUY VẤN---")
for i, doc in enumerate(results, 1):
    print(f"\n--- Kết quả {i} ---")
    print("NỘI DUNG:", doc.page_content)
    print("METADATA:", doc.metadata)
