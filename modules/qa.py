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

TEMPLATE = """
<|im_start|>system
Bạn là trợ lý giao thông chuyên nghiệp. Nhiệm vụ của bạn là trả lời các câu hỏi về luật giao thông dựa trên các dữ liệu đầu vào.

QUY TẮC BẮT BUỘC (STRICT RULES):
1. Độc quyền: **Chỉ chọn đáp án đúng (A, B, C, hoặc D).**
2. Không giải thích: **Tuyệt đối không giải thích, không thêm bất kỳ văn bản nào ngoài chữ cái đáp án.**
3. Nguồn duy nhất & Bắt buộc trả lời: **Bắt buộc phải chọn một đáp án (A, B, C, hoặc D)** dựa trên CONTEXT và VIDEO DESCRIPTION. Không suy diễn hoặc thêm kiến thức bên ngoài. **Tuyệt đối không sử dụng câu trả lời mặc định/lỗi.**
<|im_end|>

<|im_start|>user
### FEW-SHOT EXAMPLES ###

video_description: Frame 1: Xe đang chạy trên đoạn đường có ba làn. Làn ngoài cùng bên phải là làn hỗn hợp cho cả xe máy và ô tô. Phía trước có biển báo cho phép xe đi thẳng hoặc rẽ phải. [Loại đối tượng là R.412, độ tin cậy: 0.985.]
question: Nếu xe ô tô đang chạy ở làn ngoài cùng bên phải trong video này thì xe đó chỉ được phép rẽ phải?
choices:
A. Đúng
B. Sai

<context>FAKE_CONTEXT</context>

Hãy chọn đáp án đúng.
<|im_end|>
<|im_start|>assistant
B<|im_end|>

<|im_start|>user
video_description: Frame 2: Đoạn đường chứa ba làn xe. Phía trước có biển báo R.411 chỉ dẫn các hướng được phép: đi thẳng, rẽ trái và rẽ phải. Mặt đường thông thoáng và có nhiều xe máy gần đó. [Loại đối tượng là R.411, độ tin cậy: 0.992.]
question: Phần đường trong video cho phép các phương tiện đi theo hướng nào khi đến nút giao?
choices:
A. Đi thẳng
B. Đi thẳng và rẽ phải
C. Đi thẳng, rẽ trái và rẽ phải
D. Rẽ trái và rẽ phải

<context>FAKE_CONTEXT</context>

Hãy chọn đáp án đúng.
<|im_end|>
<|im_start|>assistant
C<|im_end|>

### END FEW-SHOT EXAMPLES ###

### REAL USER QUESTION ###

video_description: {video_description}
question: {question}
choices: {choices}

<context>
{context}
</context>

Hãy chọn đáp án đúng.
<|im_end|>
<|im_start|>assistant
"""

prompt = PromptTemplate.from_template(TEMPLATE)

# Sửa đổi hàm lm_generate để tối ưu hóa cho Qwen và trích xuất đáp án
def lm_generate(*, models, vlm_description: str, question: str, choices: str = "") -> str:
    """Hàm public để team gọi từ pipeline chính, đã tối ưu cho Qwen Parsing."""
    llm = models['llm']
    tokenizer = models['llm_tokenizer']
    retriever = models['retriever']
    reranker = models['reranker']

    docs = retriever.invoke(vlm_description)
    top_docs = rerank(reranker, vlm_description, docs, k=3)
    context = format_docs(top_docs)
    
    final_prompt = prompt.format(
        context=context, 
        video_description=vlm_description, 
        question=question, 
        choices=choices
    )
    model_inputs = tokenizer([final_prompt], return_tensors="pt", return_attention_mask=False).to(llm.device)
    
    generated_ids = llm.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=False,
        temperature=0.0
    )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    answer = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("thinking content:", thinking_content)
    print("Answer:", answer)
    
    # QUY TẮC BẮT BUỘC: Đảm bảo chỉ trả về chữ cái đáp án đầu tiên
    return answer.split()[0] if answer else "A"