# qa.py (Đã sửa)

import torch
import time
from langchain_community.llms import LlamaCpp
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder

# --- 1-3. LOAD MODELS (Giữ nguyên) ---
# (Tải Embeddings, Chroma, Reranker)
print("🚀 Đang tải Embedder, ChromaDB, Reranker...")
EMB_PATH = "E:/Zalo Challenge 2025/Build_RAG/model/bkai_vn_bi_encoder"
embeddings = HuggingFaceEmbeddings(
    model_name=EMB_PATH,
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': False}
)
DB_PATH = "E:/Zalo Challenge 2025/module_rag/Vecto_Database/db_bienbao_2"
vectordb = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)
retriever = vectordb.as_retriever(search_kwargs={"k": 8})
RERANK_PATH = "E:/Zalo Challenge 2025/Build_RAG/model/ViRanker"
device = "cuda" if torch.cuda.is_available() else "cpu"
reranker = CrossEncoder(RERANK_PATH, device=device)
print("✅ Tải RAG (retriever, reranker) thành công.")

def rerank(query, docs, k=3):
    if not docs: return []
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:k]]

# --- 4. LOAD PHI-3 (Giữ nguyên) ---
print("🚀 Đang tải LLM (Phi-3)...")
LLM_PATH = "E:/Zalo Challenge 2025/Build_RAG/model/Phi-3-gguf/Phi-3-mini-4k-instruct-q4.gguf"
llm = LlamaCpp(
    model_path=LLM_PATH,
    n_gpu_layers=-1, n_batch=512, n_ctx=4096,
    temperature=0.1, top_p=0.9, max_tokens=256,
    verbose=False
)
print("✅ Tải LLM (Phi-3) thành công.")

# --- 5. PROMPT (Đã sửa) ---
def format_docs(docs):
    out = ""
    for d in docs:
        bien = d.metadata.get("bien_so", "")
        out += f"[Biển số: {bien}]\n{d.page_content.strip()}\n\n"
    return out.strip()

# Các ví dụ few-shot (Giữ nguyên)
question_1 = "Trong video, biển báo nào xuất hiện đầu tiên?"
choices_1 = """A. Biển giữ khoảng cách an toàn
B. Biển giới hạn tốc độ tối đa
C. Biển cấm dừng đỗ
D. Biển cấm xe tải"""
answer_1 = "A. Biển giữ khoảng cách an toàn"
question_2 = "Biển báo 'Stop' yêu cầu người lái xe làm gì?"
choices_2 = """A. Giảm tốc độ và quan sát
B. Dừng lại hoàn toàn, nhường đường
C. Chỉ dừng lại khi có xe khác
D. Bấm còi cảnh báo"""
answer_2 = "B. Dừng lại hoàn toàn, nhường đường"


# SỬA LỖI QUAN TRỌNG: Thêm placeholders {vlm_context}, {rag_context}, {query}, {choices}
# (Lưu ý: dùng {{}} để giữ chỗ cho f-string, vì PromptTemplate sẽ dùng .format())
TEMPLATE = f"""<|system|>
Bạn là một trợ lý AI chuyên về luật giao thông và phân tích video.
Nhiệm vụ của bạn là tổng hợp BẰNG CHỨNG TỪ VIDEO (VLM) và KIẾN THỨC VỀ LUẬT (RAG) để trả lời câu hỏi trắc nghiệm.
Chỉ trả lời bằng lựa chọn đúng (ví dụ: 'A. [Nội dung]').
<|end|>

<|user|>
Question: "{question_1}"
Choices:
{choices_1}
<|end|>
<|assistant|>
{answer_1}
<|end|>

<|user|>
Question: "{question_2}"
Choices:
{choices_2}
<|end|>
<|assistant|>
{answer_2}
<|end|>

<|user|>
BẰNG CHỨNG TỪ VIDEO (VLM):
{{vlm_context}}

KIẾN THỨC VỀ LUẬT (RAG):
{{rag_context}}

Question: "{{query}}"
Choices:
{{choices}}
<|end|>
<|assistant|>
"""

prompt = PromptTemplate.from_template(TEMPLATE)

# --- 6. PUBLIC FUNCTION (Đã sửa) ---
def lm_generate(vlm_context: str, query: str, choices: list[str]) -> str:
    """
    Hàm public để team gọi từ pipeline chính.
    Nhận context VLM, câu hỏi, và các lựa chọn.
    """

    # 1. Retrieve (Dựa trên câu hỏi)
    docs = retriever.invoke(query)

    # 2. Rerank
    top_docs = rerank(query, docs, k=3)

    # 3. Format RAG context (từ vectordb)
    rag_context = format_docs(top_docs)
    if len(top_docs) == 0:
        rag_context = "Không tìm thấy luật liên quan."

    # 4. Format choices (từ input)
    choices_str = "\n".join(choices)

    # 5. Run LLM
    final_prompt_str = prompt.format(
        vlm_context=vlm_context,
        rag_context=rag_context,
        query=query,
        choices=choices_str
    )
    
    print("--- DEBUG: PROMPT CUỐI CÙNG GỬI TỚI PHI-3 ---")
    print(final_prompt_str)
    print("------------------------------------------")

    answer = llm.invoke(final_prompt_str)
    return answer.strip()