from utils.cached_helper import save_json

def rerank(reranker, query, docs, k=3):
    if not docs:
        return []
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:k]]

# 5. PROMPT
def format_docs(docs):
    out = ""
    for d in docs:
        bien = d.metadata.get("bien_so", "")
        out += f"[Biển số: {bien}]\n{d.page_content.strip()}\n\n"
    return out.strip()

def create_messages(context, video_description, question, choices):
    """
    Tạo messages cho Qwen format chuẩn ChatML, sử dụng tiếng Anh để tối ưu hóa khả năng reasoning.
    """
    
    # 1. SYSTEM PROMPT (ENGLISH - Đã thêm Quy tắc ưu tiên Context)
    system_prompt = """You are a professional traffic assistant. Your task is to answer traffic law questions.

STRICT RULES:
1. Exclusive: Only select the correct answer (A, B, C, or D).
2. No Explanation: Absolutely do not explain or add any text besides the answer letter.
3. Single Source: Must answer based on the CONTEXT and VIDEO DESCRIPTION. **IMPORTANT: If specific information (like street names or sign details) in the VIDEO DESCRIPTION contradicts or is not supported by the CONTEXT, you must prioritize the accurate information from the CONTEXT.**
4. Format: Return only 1 capitalized character. Example: "A"."""

    # 2. FEW-SHOT EXAMPLES (ENGLISH - Giả lập lỗi VLM)
    # --- Example 1 (Simulating VLM Hallucination) ---
    # Giả định VLM đọc sai 'GƯƠNG ĐỖ XUÂN KẾP', nhưng Context chứa 'Đường Đỗ Xuân Hợp'
    ex1_content = f"""video_description: Night view. On the overhead gantry, there are 02 blue signs: The left sign reads DAU GIAY LONG THANH (straight), the right sign reads DUONG **GUONG DO XUAN KEP** (turn right). The road has yellow speed reduction strips and dashed lane dividers. A 16-seater vehicle is overtaking on the right.
question: Theo trong video, nếu ô tô đi hướng chếch sang phải là hướng vào đường nào?
choices:
"A. Không có thông tin",
"B. Dầu Giây Long Thành",
"C. Đường Đỗ Xuân Hợp",
"D. Xa Lộ Hà Nội"

<context>FAKE_CONTEXT</context>

Please select the correct answer."""
    
    # --- Example 2 ---
    ex2_content = """video_description: The red traffic light ahead is lit.
question: Theo trong video, nếu ô tô đi hướng chếch sang phải là hướng vào đường Xa Lộ Hà Nội. Đúng hay sai??
choices:
"A. Đúng",
"B. Sai"

<context>FAKE_CONTEXT</context>

Please select the correct answer."""

    # 3. REAL USER QUESTION (ENGLISH format, retaining Vietnamese content)
    real_content = f"""video_description: {video_description}
question: {question}
choices: {choices}

<context>
{context}
</context>

Please select the correct answer."""

    # 4. TỔNG HỢP MESSAGES
    messages = [
        {"role": "system", "content": system_prompt},
        
        # Lượt hội thoại mẫu 1
        {"role": "user", "content": ex1_content},
        {"role": "assistant", "content": "C"},
        
        # Lượt hội thoại mẫu 2
        {"role": "user", "content": ex2_content},
        {"role": "assistant", "content": "A"},
        
        # Câu hỏi thực tế cần trả lời
        {"role": "user", "content": real_content}
    ]
    
    return messages

# Hàm llm_choise_answer (Đã sửa prompt reranker sang tiếng Anh)
def llm_choise_answer(models, vlm_description: str, question_data, box_info: str = "") -> str:
    llm = models['llm']
    tokenizer = models['llm_tokenizer']
    retriever = models['retriever']
    reranker = models['reranker']

    question = question_data["question"]
    choices = question_data["choices"]

    # Chuyển đổi choices từ list thành string
    if isinstance(choices, list):
        choices = "\n".join(choices)

    docs = retriever.invoke(vlm_description)

    # Prompt Reranker (Đã chuyển sang tiếng Anh)
    rank_prompt = f"""
        Based on the video description: "{vlm_description}", find the relevant information in the following documents to answer the question: "{question}" 
        Information from YOLO sensor: [{box_info}]
    """
    top_docs = rerank(reranker, rank_prompt, docs, k=3)
    context = format_docs(top_docs)
    
    # Tạo messages format
    messages = create_messages(context, vlm_description, question, choices)
    
    # Apply chat template với enable_thinking=False để tắt thinking mode
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # Tắt thinking mode
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(llm.device)
    
    generated_ids = llm.generate(
        **model_inputs,
        max_new_tokens=10,  # Chỉ cần 1-2 ký tự cho đáp án
        do_sample=False,
        temperature=0.0
    )

    # Decode output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    answer = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    
    data_cache = {
        "vlm_description": vlm_description,
        "context": context,
        "question": question,
        "choices": choices,
        "output": answer
    }
    save_json(data_cache, f"{question_data['id']}.json")
    print("LLM Output:", repr(answer))
    return answer