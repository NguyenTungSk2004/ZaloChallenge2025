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
Nếu không có thông tin → trả lời đúng câu:
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
B. Sai

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
C. Đi thẳng, rẽ trái và rẽ phải

### END FEW-SHOT EXAMPLES ###


### REAL USER QUESTION ###

<|user|>
video_description: {video_description}
{query}

<context>
{context}
</context>

Hãy chọn đáp án đúng.
<|assistant|>
"""

prompt = PromptTemplate.from_template(TEMPLATE)

# 6. PUBLIC FUNCTION: lm_generate()
def lm_generate(*,llm, retriever, reranker, vlm_description: str, question: str) -> str:
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
    answer = llm.invoke(final_prompt)

    return answer.strip()