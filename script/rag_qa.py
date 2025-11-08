from langchain_community.llms import LlamaCpp
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import time

# === 1. Embedding: PHOBERT LOCAL + GPU ===
embedding_model_path = "E:/Zalo Challenge 2025/Build_RAG/model/vinai"  # Folder chứa pytorch_model.bin, config.json
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_path,
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

# === 2. Vector DB ===
persist_directory = "E:/Zalo Challenge 2025/Build_RAG/db_luat"
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# === 3. Load Phi-3 GGUF (TỐI ƯU TỐC ĐỘ) ===
gguf_path = "E:/Zalo Challenge 2025/Build_RAG/model/Phi-3-gguf/Phi-3-mini-4k-instruct-q4.gguf"

start = time.time()
llm = LlamaCpp(
    model_path=gguf_path,
    n_gpu_layers=999,
    n_batch=1024,
    n_ctx=4096,
    temperature=0.2,
    max_tokens=512,
    f16_kv=True,
    use_mlock=True,
    use_mmap=False,
    flash_attn=True,
    verbose=False
)
print(f"--- Load model: {time.time() - start:.2f}s ---")

# === 4. Prompt SIÊU MẠNH (ÉP DỰA VÀO CONTEXT) ===
def format_docs(docs):
    return "\n\n".join([
        f"[Nguồn: {doc.metadata.get('nguon', '')} - {doc.metadata.get('dieu', '')} {doc.metadata.get('khoan', '')} {doc.metadata.get('diem', '')}]\n{doc.page_content.strip()}"
        for doc in docs
    ])

system_prompt = (
    "Bạn là trợ lý pháp lý chuyên nghiệp. "
    "Chỉ trả lời dựa đúng vào nội dung sau đây: {context}. "
    "Trích dẫn rõ: Điều X, Khoản Y, Điểm Z. "
    "Không suy luận, không thêm thắt, không dùng kiến thức bên ngoài. "
    "Trả lời ngắn gọn, tối đa 2 câu. "
    "Nếu không có trong context, nói: 'Tôi không tìm thấy quy định liên quan trong luật hiện hành'."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# === 5. RAG Chain ===
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# === 6. TEST RETRIEVAL ===
'''
test_query = "người điều khiển giao thông đường bộ"
docs = vectordb.similarity_search(test_query, k=3)
print("\n=== TOP 3 KẾT QUẢ TỪ VECTOR DB ===")
for i, doc in enumerate(docs):
    print(f"\n--- Doc {i+1} ---")
    print(doc.page_content[:500])
    print("Metadata:", doc.metadata)
'''
# === 7. TEST RAG ===
query = "Người tham gia giao thông không được vượt xe trong những trường hợp nào"
start = time.time()
answer = rag_chain.invoke(query)
infer_time = time.time() - start

print(f"\nThời gian suy luận: {infer_time:.2f}s")
print("Câu hỏi:", query)
print("Trả lời:", answer)