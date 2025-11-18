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
<|system|>
Bạn là trợ lý giao thông. Chỉ trả lời dựa trên CONTEXT và VIDEO DESCRIPTION.
STRICT RULE :
1. Chỉ chọn đáp án đúng.
2. Không giải thích gì thêm.
3. Không suy diễn.
4.Nếu không có thông tin → trả lời đúng câu:
"Tôi không tìm thấy biển báo phù hợp trong cơ sở dữ liệu."
Không suy diễn. Không thêm kiến thức ngoài context.
<|end|>

### FEW-SHOT EXAMPLES ###

<|user|>
video_description: The road section has three lanes. The outermost right lane is a mixed lane for both motorbikes and cars. Ahead, there is a sign allowing vehicles to go straight or turn right.
question: Nếu xe ô tô đang chạy ở làn ngoài cùng bên phải trong video này thì xe đó chỉ được phép rẽ phải?
choices:
- A. Đúng
- B. Sai

<context>FAKE_CONTEXT</context>

Hãy chọn đáp án đúng.
<|assistant|>
B

<|user|>
video_description: The road section contains three lanes. A traffic sign R.411 ahead shows arrows indicating allowed directions: straight, left turn, and right turn.
question: Phần đường trong video cho phép các phương tiện đi theo hướng nào khi đến nút giao?
choices:
- A. Đi thẳng
- B. Đi thẳng và rẽ phải
- C. Đi thẳng, rẽ trái và rẽ phải
- D. Rẽ trái và rẽ phải

<context>FAKE_CONTEXT</context>

Hãy chọn đáp án đúng.
<|assistant|>
C

### END FEW-SHOT EXAMPLES ###


### REAL USER QUESTION ###

<|user|>
video_description: {video_description}
{question}

<context>
{context}
</context>

Hãy chọn đáp án đúng.
<|assistant|>
"""

prompt = PromptTemplate.from_template(TEMPLATE)

# 6. PUBLIC FUNCTION: lm_generate()
def lm_generate(*,llm, tokenizer,retriever, reranker, vlm_description: str, question: str) -> str:
    """Hàm public để team gọi từ pipeline chính"""

    # 1. Retrieve
    docs = retriever.invoke(vlm_description)

    # 2. Rerank
    top_docs = rerank(reranker, vlm_description, docs, k=3)

    # 3. Kiểm tra context rỗng
    if len(top_docs) == 0:
        return "Tôi không tìm thấy biển báo phù hợp trong cơ sở dữ liệu."

    # 4. Format context
    context = format_docs(top_docs)

    # 5. Run LLM
    final_prompt = prompt.format(context=context, video_description=vlm_description, question=question)
    # 5.1. Tokenize prompt
    inputs = tokenizer(final_prompt, return_tensors="pt", return_attention_mask=False).to(llm.device)
    
    # 5.2. Generate answer
    # Lưu ý: Cần thêm max_new_tokens để giới hạn độ dài câu trả lời của LLM
    outputs = llm.generate(
        **inputs,
        max_new_tokens=256, # Giới hạn 256 token mới cho câu trả lời
        pad_token_id=tokenizer.pad_token_id, # Cần thiết cho mô hình Phi-3
        eos_token_id=tokenizer.eos_token_id
    )

    # 5.3. Decode (Chỉ giải mã phần câu trả lời)
    # Do mô hình sinh ra cả prompt, chúng ta cần loại bỏ độ dài của prompt gốc.
    prompt_length = inputs['input_ids'].shape[1]
    
    # Debug: in ra thông tin generation
    print(f"🔍 Prompt length: {prompt_length}")
    print(f"🔍 Output length: {outputs[0].shape}")
    
    # Giải mã phần token được sinh ra (sau prompt)
    if outputs[0].shape[0] > prompt_length:
        generated_tokens = outputs[0][prompt_length:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"🔍 Generated tokens: {generated_tokens[:10]}")  # In 10 token đầu
        print(f"🔍 Raw answer: '{answer}'")
    else:
        # Fallback: decode toàn bộ rồi loại bỏ prompt
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompt_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        
        print(f"🔍 Full output: {full_output[:300]}...")
        print(f"🔍 Prompt text: {prompt_text[-200:]}")  # In 200 ký tự cuối của prompt
        
        if prompt_text in full_output:
            answer = full_output.replace(prompt_text, "", 1).strip()
        else:
            answer = full_output.strip()
        
        print(f"🔍 Final answer after cleanup: '{answer}'")

    # Làm sạch answer - chỉ lấy ký tự đầu tiên nếu đó là A, B, C, D
    answer_clean = answer.strip()
    if answer_clean and answer_clean[0] in ['A', 'B', 'C', 'D']:
        answer_clean = answer_clean[0]
    elif 'A' in answer_clean:
        answer_clean = 'A'
    elif 'B' in answer_clean:
        answer_clean = 'B' 
    elif 'C' in answer_clean:
        answer_clean = 'C'
    elif 'D' in answer_clean:
        answer_clean = 'D'
    else:
        print(f"⚠️ No valid answer found, defaulting to A. Raw: '{answer_clean}'")
        answer_clean = 'A'
    
    print(f"🔍 Final cleaned answer: '{answer_clean}'")
    return answer_clean
