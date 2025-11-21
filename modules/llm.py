from utils.cached_helper import save_json
import re

def rerank(reranker, query, docs, k=5):
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

def create_messages(context, video_description, question, choices):
    """
    FIXED: Trust YOLO detections for existence questions
    """
    
    system_prompt = """You are a Vietnamese traffic law expert.

TASK: Answer multiple choice questions about traffic laws.

RULES:
1. Output: ONLY one letter (A, B, C, or D)
2. NO explanation, NO extra text
3. **For "existence" questions (Có... không?): Trust YOLO detections FIRST**
4. For "law/regulation" questions: Trust CONTEXT first
5. Priority depends on question type:
   - "Có [object] không?" → YOLO > VIDEO > CONTEXT
   - "Được phép... không?" → CONTEXT > YOLO > VIDEO
   - "Phải làm gì?" → CONTEXT > YOLO > VIDEO"""

    # ✅ NEW EXAMPLE: YOLO detection for existence question
    ex_yolo_detection = """video_description: YOLO: - TRAFFIC_LIGHT_GREEN
Scene: City street at night.

question: Phía trước có đèn giao thông không?
choices:
A. Có
B. Không

<context>
[Biển số: Đèn vàng]
Tín hiệu đèn màu vàng phải dừng lại...
</context>

Answer:"""

    # Example 2: YOLO + Law question
    ex_law_with_yolo = """video_description: YOLO: - TRAFFIC_LIGHT_RED
Scene: Intersection.

question: Khi đèn đỏ, xe có được đi tiếp không?
choices:
A. Được
B. Không được

<context>
[Biển số: Đèn đỏ]
Khi đèn đỏ bật, phải dừng lại trước vạch dừng.
</context>

Answer:"""

    # Example 3: No YOLO detection
    ex_no_detection = """video_description: YOLO: None
Scene: Highway, no signs visible.

question: Có biển cấm rẽ trái không?
choices:
A. Có
B. Không

<context>
[No relevant information]
</context>

Answer:"""

    # Example 4: YOLO detects sign
    ex_sign_detection = """video_description: YOLO: - P.102: Cấm đi ngược chiều
Scene: City street.

question: Có biển cấm không?
choices:
A. Có
B. Không

<context>
[Biển số: P.102]
Biển P.102 là biển cấm đi ngược chiều.
</context>

Answer:"""

    # Real question
    real_content = f"""video_description: {video_description}

question: {question}
choices:
{choices}

<context>
{context}
</context>

Answer:"""

    messages = [
        {"role": "system", "content": system_prompt},
        
        # YOLO detection examples
        {"role": "user", "content": ex_yolo_detection},
        {"role": "assistant", "content": "A"},  # ← Trust YOLO!
        
        {"role": "user", "content": ex_law_with_yolo},
        {"role": "assistant", "content": "B"},  # ← Trust CONTEXT for law
        
        {"role": "user", "content": ex_no_detection},
        {"role": "assistant", "content": "B"},  # ← No detection = No
        
        {"role": "user", "content": ex_sign_detection},
        {"role": "assistant", "content": "A"},  # ← Trust YOLO
        
        {"role": "user", "content": real_content}
    ]
    
    return messages

def llm_choise_answer(models, vlm_description: str, question_data, box_info: str = "") -> str:
    llm = models['llm']
    tokenizer = models['llm_tokenizer']
    retriever = models['retriever']
    reranker = models['reranker']

    question = question_data["question"]
    choices = question_data["choices"]

    if isinstance(choices, list):
        choices = "\n".join(choices)

    # ✅ FIX: RETRIEVE BY QUESTION, not VLM description
    docs = retriever.invoke(question)
    
    # ✅ Extract YOLO keywords from vlm_description
    yolo_keywords = []
    if "YOLO:" in vlm_description:
        yolo_part = vlm_description.split("YOLO:")[1].split("Scene:")[0]
        # Parse YOLO objects
        for line in yolo_part.split("\n"):
            if line.strip().startswith("-"):
                obj = line.strip()[1:].strip()
                if obj and obj != "None":
                    yolo_keywords.append(obj)
    
    # ✅ RERANK: Question + YOLO keywords
    yolo_str = ", ".join(yolo_keywords) if yolo_keywords else "None"
    rank_prompt = f"Question: {question}\nYOLO detected: {yolo_str}"
    
    top_docs = rerank(reranker, rank_prompt, docs, k=5)
    context = format_docs(top_docs)
    
    # Create messages
    messages = create_messages(context, vlm_description, question, choices)
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(llm.device)
    
    # ✅ Generation
    generated_ids = llm.generate(
        **model_inputs,
        max_new_tokens=3,        # Giảm từ 10 xuống 3
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    raw_answer = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
    
    # ✅ EXTRACT A/B/C/D
    match = re.search(r'\b[ABCD]\b', raw_answer.upper())
    if match:
        answer = match.group(0)
    else:
        answer = raw_answer[0].upper() if raw_answer and raw_answer[0].upper() in "ABCD" else "A"
    
    # Save cache
    data_cache = {
        "vlm_description": vlm_description,
        "yolo_keywords": yolo_keywords,
        "context": context,
        "question": question,
        "choices": choices,
        "raw_output": raw_answer,
        "final_answer": answer
    }
    save_json(data_cache, f"{question_data['id']}.json")
    
    print(f"LLM: {repr(raw_answer)} → {answer}")
    
    return answer