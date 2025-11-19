from langchain_core.prompts import PromptTemplate

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
    Tạo messages cho Qwen format chuẩn ChatML.
    Tách Few-shot examples thành các lượt User/Assistant riêng biệt để model học pattern tốt hơn.
    """
    
    # 1. SYSTEM PROMPT (Luật nghiêm ngặt)
    system_prompt = """Bạn là trợ lý giao thông chuyên nghiệp. Nhiệm vụ của bạn là trả lời các câu hỏi về luật giao thông.

QUY TẮC BẮT BUỘC (STRICT RULES):
1. Độc quyền: Chỉ chọn đáp án đúng (A, B, C, hoặc D).
2. Không giải thích: Tuyệt đối không giải thích, không thêm bất kỳ văn bản nào ngoài chữ cái đáp án.
3. Nguồn duy nhất: Bắt buộc trả lời dựa trên CONTEXT và VIDEO DESCRIPTION.
4. Định dạng: Chỉ trả về duy nhất 1 ký tự viết hoa. Ví dụ: "A"."""

    # 2. FEW-SHOT EXAMPLES (Định nghĩa mẫu hội thoại)
    # Lưu ý: Format của User content trong ví dụ PHẢI GIỐNG HỆT format của câu hỏi thật.
    
    # --- Example 1 ---
    ex1_content = """video_description: Góc nhìn đêm. Trên giá long môn có 02 biển xanh: Biển trái ghi ĐẦU GIÂY LONG THÀNH (đi thẳng), biển phải ghi ĐƯỜNG ĐỖ XUÂN HỢP (rẽ phải). Mặt đường có vạch giảm tốc màu vàng và vạch phân làn nét đứt. Bên phải có xe 16 chỗ đang vượt.
question: Theo trong video, nếu ô tô đi hướng chếch sang phải là hướng vào đường nào?
choices:
"A. Không có thông tin",
"B. Dầu Giây Long Thành",
"C. Đường Đỗ Xuân Hợp",
"D. Xa Lộ Hà Nội"

<context>FAKE_CONTEXT</context>

Hãy chọn đáp án đúng."""
    
    # --- Example 2 ---
    ex2_content = """video_description: Đèn đỏ phía trước đang sáng.
question: Phía trước có đèn giao thông không?
choices:
"A. Có",
"B. Không"

<context>FAKE_CONTEXT</context>

Hãy chọn đáp án đúng."""

    # 3. REAL USER QUESTION (Câu hỏi thực tế)
    real_content = f"""video_description: {video_description}
question: {question}
choices: {choices}

<context>
{context}
</context>

Hãy chọn đáp án đúng."""

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

# Sửa đổi hàm lm_generate để tối ưu hóa cho Qwen và trích xuất đáp án
def lm_generate(*, models, vlm_description: str, question: str, choices: str = "") -> str:
    """Hàm public để team gọi từ pipeline chính, sử dụng messages format với enable_thinking=False."""
    llm = models['llm']
    tokenizer = models['llm_tokenizer']
    retriever = models['retriever']
    reranker = models['reranker']

    docs = retriever.invoke(vlm_description)
    top_docs = rerank(reranker, vlm_description, docs, k=3)
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
    full_output = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    
    print("LLM Output:", repr(full_output))
    
    # Tìm đáp án A, B, C hoặc D trong output
    import re
    matches = re.findall(r'\b([A-D])\b', full_output)
    if matches:
        final_answer = matches[0]  # Lấy đáp án đầu tiên
        print("Final Answer:", final_answer)
        return final_answer
    
    # Fallback: lấy ký tự đầu tiên nếu là A-D
    first_char = full_output.strip()[0] if full_output.strip() else ""
    if first_char in "ABCD":
        print("Fallback Answer:", first_char)
        return first_char
    
    # Default fallback
    print("Default Answer: A")
    return "A"
