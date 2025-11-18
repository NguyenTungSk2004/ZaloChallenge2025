import json

GOLDEN_DATASET_PATH = "scripts/golden_dataset.json"
OUTPUT_BKAI_TRAIN = "scripts/train_bkai.jsonl"
OUTPUT_GEMMA_TRAIN = "scripts/train_gemma.jsonl" # Đổi tên cho Gemma

try:
    with open(GOLDEN_DATASET_PATH, 'r', encoding='utf-8') as f:
        golden_data = json.load(f)
    print(f"✅ Đã tải {len(golden_data)} mẫu từ Golden Dataset.")
except Exception as e:
    print(f"❌ Lỗi khi tải {GOLDEN_DATASET_PATH}: {e}")
    exit()

# Mở 2 file để ghi
with open(OUTPUT_BKAI_TRAIN, 'w', encoding='utf-8') as f_bkai, \
     open(OUTPUT_GEMMA_TRAIN, 'w', encoding='utf-8') as f_gemma:
    
    for item in golden_data:
        query = item["query"]
        context = item["context"]
        answer = item["answer"]
        
        # --- Format cho BKAI (Embedder) ---
        # (câu_hỏi, đoạn_luật_liên_quan)
        bkai_data = {
            "query": query,
            "positive": context 
        }
        f_bkai.write(json.dumps(bkai_data, ensure_ascii=False) + "\n")
        
        # --- Format cho Gemma (LLM) ---
        # (câu_hỏi, đoạn_luật, câu_trả_lời)
        # Chúng ta dùng định dạng chat mà Gemma (hoặc Phi-3) hiểu
        gemma_text = f"<|system|>\nBạn là trợ lý AI chuyên về luật giao thông Việt Nam.\n<|end|>\n"
        gemma_text += f"<|user|>\nBỐI CẢNH LUẬT:\n{context}\n\nCÂU HỎI:\n{query}\n<|end|>\n"
        gemma_text += f"<|assistant|>\n{answer}\n<|end|>"
        
        gemma_data = {"text": gemma_text}
        f_gemma.write(json.dumps(gemma_data, ensure_ascii=False) + "\n")

print(f"\n--- HOÀN TẤT ---")
print(f"✅ Đã lưu file training cho BKAI tại: {OUTPUT_BKAI_TRAIN}")
print(f"✅ Đã lưu file training cho Gemma tại: {OUTPUT_GEMMA_TRAIN}")