import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import torch

# --- 1. TẢI CÁC THÀNH PHẦN RAG (TỪ qa.py) ---
print("🔄 Đang tải các model RAG...")
EMB_PATH = "models/bkai_vn_bi_encoder"
RERANK_PATH = "models/ViRanker"
KNOWLEDGE_BASE_PATH = "scripts/knowledge_base_final.json"
PARSED_QUESTIONS_PATH = "scripts/parsed_questions.json"
OUTPUT_GOLDEN_DATASET = "scripts/golden_dataset.json" # File tổng hợp cuối cùng

device = "cuda" if torch.cuda.is_available() else "cpu"

# Tải Embedder (BKAI)
embeddings = HuggingFaceEmbeddings(
    model_name=EMB_PATH,
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True} # Nên normalize cho BKAI
)

# Tải Reranker (ViRanker)
reranker = CrossEncoder(RERANK_PATH, device=device)

# --- 2. TẢI VÀ TẠO VECTOR DB TỪ JSON ---
print(f"🔄 Đang tải Knowledge Base từ {KNOWLEDGE_BASE_PATH}...")
try:
    with open(KNOWLEDGE_BASE_PATH, 'r', encoding='utf-8') as f:
        kb_data = json.load(f) #
except Exception as e:
    print(f"❌ Lỗi khi tải {KNOWLEDGE_BASE_PATH}: {e}")
    exit()

# Chuyển đổi dicts sang Document objects
all_kb_docs = [
    Document(
        page_content=item["page_content"],
        metadata=item["metadata"]
    ) for item in kb_data if "page_content" in item
]
print(f"✅ Đã tải {len(all_kb_docs)} chunks luật/biển báo.")

print("🔄 Đang tạo ChromaDB (in-memory)...")
# Tạo DB tạm thời trong RAM
vectordb = Chroma.from_documents(
    documents=all_kb_docs,
    embedding=embeddings
)
# Tạo retriever (bộ tìm kiếm thô)
retriever = vectordb.as_retriever(search_kwargs={"k": 10}) # Lấy top 10

# --- 3. TẢI CÁC CÂU HỎI ĐÃ PARSE ---
try:
    with open(PARSED_QUESTIONS_PATH, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    print(f"✅ Đã tải {len(questions_data)} câu hỏi từ {PARSED_QUESTIONS_PATH}.")
except Exception as e:
    print(f"❌ Lỗi khi tải {PARSED_QUESTIONS_PATH}: {e}")
    exit()
    
# --- 4. LIÊN KẾT CONTEXT VÀ LƯU GOLDEN DATASET ---
golden_dataset = []
print("\n🚀 Bắt đầu liên kết context cho từng câu hỏi...")

for i, item in enumerate(questions_data):
    query = item["question"]
    answer = item["answer"]
    
    # Bước 4.1: Retrieve (Tìm kiếm thô bằng BKAI)
    retrieved_docs = retriever.invoke(query)
    
    if not retrieved_docs:
        print(f"⚠️ Không tìm thấy context cho câu: {item['id']}")
        continue
        
    # Bước 4.2: Rerank (Tinh lọc bằng ViRanker)
    pairs = [(query, d.page_content) for d in retrieved_docs]
    scores = reranker.predict(pairs)
    
    # Sắp xếp và lấy context tốt nhất
    ranked = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
    best_doc = ranked[0][0] # Chỉ lấy 1 context tốt nhất
    
    # Tạo bản ghi mới
    golden_record = {
        "id": item["id"],
        "query": query,
        "answer": answer,
        "context": best_doc.page_content, # Đây là "đoạn luật"
        "context_metadata": best_doc.metadata
    }
    golden_dataset.append(golden_record)
    
    if (i+1) % 50 == 0:
        print(f"    ... Đã xử lý {i+1}/{len(questions_data)} câu hỏi ...")

print(f"✅ Đã liên kết context cho {len(golden_dataset)} câu hỏi.")

# Lưu file "vàng"
with open(OUTPUT_GOLDEN_DATASET, 'w', encoding='utf-8') as f:
    json.dump(golden_dataset, f, ensure_ascii=False, indent=4)
print(f"✅ Đã lưu bộ dữ liệu Vàng (Golden Dataset) vào: {OUTPUT_GOLDEN_DATASET}")