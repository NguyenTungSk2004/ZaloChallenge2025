import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from transformers import GenerationConfig

# --- 1. ĐỊNH NGHĨA ĐƯỜNG DẪN ---
LLM_PATH = "models/google/gemma-3-4b-it" 
KNOWLEDGE_BASE_PATH = "scripts/knowledge_base_final.json" 
OUTPUT_BKAI_TRAIN = "json_file/train_bkai.jsonl"
OUTPUT_PHI3_TRAIN = "json_file/train_phi3.jsonl"

# --- 2. LOAD GEMMA 3 4B ---
print(f"🔄 Đang tải Model từ: {LLM_PATH}...")
try:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        LLM_PATH,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
    )

    # --- 1️⃣ Tạo GenerationConfig chuẩn cho Gemma 3 4B ---
    model.generation_config = GenerationConfig(
        max_new_tokens=384,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    print("✅ Đã tải thành công Model Gemma 3 4B.")

except Exception as e:
    print(f"❌ LỖI: Không thể tải model tại: {LLM_PATH}")
    print(f"Chi tiết lỗi: {e}")
    exit()

# --- 3. HÀM TIỆN ÍCH ---
def load_knowledge_base(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Đã tải {len(data)} chunks từ {file_path}")
        return data
    except Exception as e:
        print(f"❌ Lỗi khi tải {file_path}: {e}")
        return None

def generate_qa_for_chunk(chunk_content):
    """
    Version ổn định cho Gemma 3 4B:
    - Model sinh Q&A dạng plain text
    - Hậu kỳ parse JSON
    """
    messages = [
        {"role": "system",
         "content": (
            "Bạn là chuyên gia tạo câu hỏi – câu trả lời từ văn bản luật.\n"
            "Nhiệm vụ duy nhất: sinh ra đúng 3 cặp Q&A.\n"
            "Mỗi Q&A phải liên quan 100% đến context.\n"
            "Không bịa thông tin ngoài context.\n"
            "Không giải thích, không tóm tắt.\n"
            "Trả về đúng format:\n"
            "Q1: <câu hỏi 1>\nA1: <câu trả lời 1>\n"
            "Q2: <câu hỏi 2>\nA2: <câu trả lời 2>\n"
            "Q3: <câu hỏi 3>\nA3: <câu trả lời 3>"
        )},
        {"role": "user",
         "content": f"Context:\n{chunk_content}\n\nHãy tạo đúng 3 Q&A theo định dạng."}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(**inputs)  # ⚡ Chỉ cần inputs, config đã gán

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # --- Hậu kỳ: parse sang JSON ---
    qa_pairs = []
    lines = generated.splitlines()
    pair = {}
    for line in lines:
        line = line.strip()
        if line.startswith("Q"):
            if pair:
                qa_pairs.append(pair)
            pair = {"question": line.split(":", 1)[1].strip()}
        elif line.startswith("A") and pair:
            pair["answer"] = line.split(":", 1)[1].strip()
    if pair:
        qa_pairs.append(pair)

    qa_pairs = qa_pairs[:3]
    if len(qa_pairs) != 3:
        print("⚠️ Không đọc được 3 Q&A:", generated[:200])
        return []

    return qa_pairs

# --- 4. HÀM CHÍNH ---
def create_and_save_training_files(knowledge_chunks):
    with open(OUTPUT_BKAI_TRAIN, 'w', encoding='utf-8') as f_bkai, \
         open(OUTPUT_PHI3_TRAIN, 'w', encoding='utf-8') as f_phi3:

        print("\n🚀 Bắt đầu tạo dữ liệu huấn luyện...")
        total_qa_pairs = 0

        for i, chunk in enumerate(knowledge_chunks):
            chunk_content = chunk.get("page_content")
            if not chunk_content:
                continue

            print(f"    Đang xử lý chunk {i+1}/{len(knowledge_chunks)} (ID: {chunk['metadata'].get('doc_id')})...")

            qa_pairs = generate_qa_for_chunk(chunk_content)
            if not qa_pairs:
                print(f"    Không tạo được Q&A cho chunk {i+1}.")
                continue

            for pair in qa_pairs:
                instruction = pair["question"]
                response = pair["answer"]

                # --- BKAI ---
                bkai_data = {"query": instruction, "positive": chunk_content}
                f_bkai.write(json.dumps(bkai_data, ensure_ascii=False) + "\n")

                # --- Phi-3 format ---
                phi3_text = f"<|system|>\nBạn là trợ lý AI chuyên về luật giao thông Việt Nam.\n<|end|>\n"
                phi3_text += f"<|user|>\nBỐI CẢNH LUẬT:\n{chunk_content}\n\nCÂU HỎI:\n{instruction}\n<|end|>\n"
                phi3_text += f"<|assistant|>\n{response}\n<|end|>"

                phi3_data = {"text": phi3_text}
                f_phi3.write(json.dumps(phi3_data, ensure_ascii=False) + "\n")

                total_qa_pairs += 1

            print(f"    Đã tạo {len(qa_pairs)} cặp Q&A. Tổng cộng: {total_qa_pairs}")

    print("\n--- HOÀN TẤT ---")
    print(f"✅ Đã lưu file training cho BKAI tại: {OUTPUT_BKAI_TRAIN}")
    print(f"✅ Đã lưu file training cho Phi-3 tại: {OUTPUT_PHI3_TRAIN}")
    print(f"Tổng cộng {total_qa_pairs} mẫu huấn luyện đã được tạo.")

# --- 5. CHẠY SCRIPT ---
if __name__ == "__main__":
    kb_chunks = load_knowledge_base(KNOWLEDGE_BASE_PATH)
    if kb_chunks:
        create_and_save_training_files(kb_chunks)
