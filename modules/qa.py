from langchain_core.prompts import PromptTemplate
import torch

def rerank(reranker, query, docs, k=3):
    if not docs:
        return []
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:k]]

def format_docs(docs):
    out = ""
    for d in docs:
        bien = d.metadata.get("bien_so", "")
        out += f"[Biển số: {bien}]\n{d.page_content.strip()}\n\n"
    return out.strip()

# PROMPT TỐI ƯU HƠN - NGẮN GỌN HƠN ĐỂ TIẾT KIỆM TOKENS
TEMPLATE = """<|system|>
Bạn là trợ lý giao thông. Trả lời dựa trên CONTEXT và VIDEO.
RULES:
1. Chỉ trả lời: A hoặc B hoặc C hoặc D
2. Không giải thích, không thêm text
<|end|>

<|user|>
VIDEO: {video_description}

CONTEXT: {context}

{question}

Trả lời chỉ 1 chữ cái:
<|assistant|>
"""

prompt = PromptTemplate.from_template(TEMPLATE)


def lm_generate(*, llm, tokenizer, retriever, retriever_luat, reranker, vlm_description: str, question: str) -> str:
    """
    Generate answer - TỐI ƯU CHO 6GB VRAM
    """
    try:
        # 1. Retrieve documents
        docs = retriever.invoke(vlm_description) # Bien bao
        docs_luat = retriever_luat.invoke(vlm_description) # Luat
        all_docs = docs+docs_luat
        # 2. Rerank
        top_docs = rerank(reranker, vlm_description, all_docs, k=2)  # Giảm từ 3->2 để tiết kiệm
        
        # 3. Format context
        context = format_docs(top_docs) if top_docs else "Không có thông tin."
        
        # Truncate context nếu quá dài
        if len(context) > 800:
            context = context[:800] + "..."
        
        # 4. Build prompt
        final_prompt = prompt.format(
            context=context, 
            video_description=vlm_description[:1500],  # Giới hạn VLM description
            question=question
        )
        
        # 5. Tokenize với truncation
        inputs = tokenizer(
            final_prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=3000,  # Giảm xuống 3000 tokens thay vì 3500
            return_attention_mask=True,
            padding=False  # Không padding để tiết kiệm
        )
        
        # Move to device
        inputs = {k: v.to(llm.device) for k, v in inputs.items()}
        
        # 6. Generate với các tham số tối ưu cho memory
        with torch.no_grad():
            # Set model to eval mode
            llm.eval()
            
            outputs = llm.generate(
                **inputs,
                max_new_tokens=5,  # Chỉ cần 1-2 tokens cho A/B/C/D
                do_sample=False,  # Greedy decoding
                num_beams=1,  # Không dùng beam search để tiết kiệm memory
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                temperature=1.0,
                top_p=1.0,
                repetition_penalty=1.0
            )
        
        # 7. Decode
        prompt_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][prompt_length:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # 8. Extract answer
        answer = answer.strip()
        
        # Tìm chữ cái A/B/C/D
        for char in answer.upper():
            if char in ['A', 'B', 'C', 'D']:
                return char
        
        # Fallback
        if len(answer) > 0:
            first_char = answer[0].upper()
            if first_char in ['A', 'B', 'C', 'D']:
                return first_char
        
        return answer if answer else "UNK"
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"  [OOM ERROR] Out of memory: {e}")
        torch.cuda.empty_cache()
        return "OOM"
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  [OOM ERROR] Runtime OOM: {e}")
            torch.cuda.empty_cache()
            return "OOM"
        else:
            print(f"  [RUNTIME ERROR] {e}")
            raise
            
    except Exception as e:
        print(f"  [ERROR in lm_generate] {type(e).__name__}: {str(e)}")
        raise

