import torch
import time
from langchain_community.llms import LlamaCpp
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder

# =========================================================
# 1. LOAD EMBEDDING ONCE
# =========================================================
EMB_PATH = "E:/Zalo Challenge 2025/Build_RAG/model/bkai_vn_bi_encoder"
embeddings = HuggingFaceEmbeddings(
    model_name=EMB_PATH,
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': False}
)

# =========================================================
# 2. LOAD CHROMA VECTOR DB (BIỂN BÁO)
# =========================================================
DB_PATH = "E:/Zalo Challenge 2025/module_rag/Vecto_Database/db_bienbao_2"
vectordb = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)
retriever = vectordb.as_retriever(search_kwargs={"k": 8})


# =========================================================
# 3. RERANKER
# =========================================================
RERANK_PATH = "E:/Zalo Challenge 2025/Build_RAG/model/ViRanker"
device = "cuda" if torch.cuda.is_available() else "cpu"
reranker = CrossEncoder(RERANK_PATH, device=device)

def rerank(query, docs, k=3):
    if not docs:
        return []
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:k]]


# =========================================================
# 4. LOAD PHI-3 MINI GGUF – SAFE LOADING
# =========================================================
LLM_PATH = "E:/Zalo Challenge 2025/Build_RAG/model/Phi-3-gguf/Phi-3-mini-4k-instruct-q4.gguf"

llm = LlamaCpp(
    model_path=LLM_PATH,
    n_gpu_layers=-1,
    n_batch=512,
    n_ctx=4096,
    temperature=0.1,
    top_p=0.9,
    max_tokens=256,
    verbose=False
)

# =========================================================
# 5. PROMPT
# =========================================================
def format_docs(docs):
    out = ""
    for d in docs:
        bien = d.metadata.get("bien_so", "")
        out += f"[Biển số: {bien}]\n{d.page_content.strip()}\n\n"
    return out.strip()

TEMPLATE = """
<|system|>
Bạn là trợ lý giao thông. Chỉ trả lời dựa trên CONTEXT.
Nếu không có thông tin → trả lời đúng câu:
"Tôi không tìm thấy biển báo phù hợp trong cơ sở dữ liệu."
Không suy diễn. Không thêm kiến thức ngoài context.
<|end|>

<context>
{context}
</context>

<|user|>
{query}
<|assistant|>
"""

prompt = PromptTemplate.from_template(TEMPLATE)

# =========================================================
# 6. PUBLIC FUNCTION: lm_generate()
# =========================================================
def lm_generate(question: str) -> str:
    """Hàm public để team gọi từ pipeline chính"""

    # 1. Retrieve
    docs = retriever.invoke(question)

    # 2. Rerank
    top_docs = rerank(question, docs, k=3)

    # 3. Kiểm tra context rỗng
    if len(top_docs) == 0:
        return "Tôi không tìm thấy biển báo phù hợp trong cơ sở dữ liệu."

    # 4. Format context
    context = format_docs(top_docs)

    # 5. Run LLM
    final_prompt = prompt.format(context=context, query=question)
    answer = llm.invoke(final_prompt)

    return answer.strip()
