import json
import csv
import time
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from ultralytics import YOLO
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig
)
from sentence_transformers import CrossEncoder

from modules.tracker import BestFrameTracker
from modules.extract_frames import extract_frames_to_queue
from modules.vlm import describe_frame_with_prompt
from modules.qa import lm_generate

def load_models():
    """Load tất cả các models cần thiết"""
    print("Đang load models...")
    
    # 1. NF4 QUANTIZATION CONFIG
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    # 2. LOAD BLIP2 VLM (4-BIT NF4)
    model_path = "models/blip2-opt-2.7b"
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    # 3. YOLO
    model_path_yolo = "models/yolo/best.pt"
    yolo_detector = YOLO(model_path_yolo)

    # 4. EMBEDDING + CHROMA
    EMB_PATH = "models/bkai-foundation-models/vietnamese-bi-encoder"
    embeddings = HuggingFaceEmbeddings(
        model_name=EMB_PATH,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': False}
    )

    DB_PATH = "Vecto_Database/db_bienbao_2"
    vectordb = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # 5. RERANKER
    RERANK_PATH = "models/namdp-ptit/ViRanker"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = CrossEncoder(RERANK_PATH, device=device)

    # 6. LOAD PHI-3 MINI (TRANSFORMERS, 4BIT NF4)
    LLM_PATH = "models/microsoft/Phi-3-mini-4k-instruct"

    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        LLM_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    print("Tất cả models đã load xong!")
    
    return {
        'processor': processor,
        'model': model,
        'yolo_detector': yolo_detector,
        'retriever': retriever,
        'reranker': reranker,
        'llm': llm,
        'tokenizer': tokenizer
    }

def process_single_question(question_data, models):
    """Xử lý một câu hỏi và trả về kết quả"""
    
    try:
        # Khởi tạo tracker cho video này
        tracker = BestFrameTracker()
        
        # Extract frames
        frames_queue = extract_frames_to_queue(question_data['video_path'])
        frame_count = 0

        # Xử lý từng frame
        while True:
            frame = frames_queue.get()
            if frame is None:
                break

            frame_count += 1

            # YOLO detection
            results = models['yolo_detector'].track(frame, tracker="bytetrack.yaml", verbose=False)
            if not results or len(results) == 0:
                continue

            # Cập nhật tracker
            for box in results[0].boxes:
                if box.id is None:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                track_id = int(box.id)
                conf = float(box.conf)
                cls_id = int(box.cls)
                cls_name = results[0].names[cls_id]

                tracker.update_track(frame, track_id, (x1, y1, x2, y2), conf, cls_name)

        # VLM Captioning
        all_caption = ""
        for track_id, frameData in tracker.best_frames.items():
            box = frameData.box_info

            prompt = (
                f"Question: Describe the environment and context of the car. "
                f"Also, what is the **exact text or symbol** visible on the traffic sign associated with the bounding box {box.bbox}? Answer:"
            )

            caption = describe_frame_with_prompt(frameData.frame, prompt, models['processor'], models['model'])

            all_caption += (
                f"\n Caption Frame {track_id}: {caption} "
                f"Information the traffic sign:[label: '{box.class_name}', score: '{frameData.score}']"
            )

        # Chuẩn bị input cho LLM
        vlm_description = all_caption
        question = question_data["question"]
        choices = question_data["choices"]

        # Gọi LLM để trả lời
        final_answer = lm_generate(
            llm=models['llm'],
            tokenizer=models['tokenizer'],
            retriever=models['retriever'],
            reranker=models['reranker'],
            vlm_description=vlm_description,
            question=question + "\n" + "\n".join(choices),
        )

        return final_answer
        
    except Exception as e:
        print(f"Lỗi khi xử lý câu hỏi {question_data['id']}: {str(e)}")
        return "A"  # Trả về A mặc định nếu có lỗi

def main():
    """Hàm chính để xử lý tất cả câu hỏi"""
    
    # Load input data
    print("Đang đọc dữ liệu từ public_test.json...")
    with open('public_test/public_test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    questions = test_data['data']
    print(f"Tìm thấy {len(questions)} câu hỏi cần xử lý")
    
    # Load models một lần
    models = load_models()
    
    # Xử lý từng câu hỏi
    results = []
    total_time = 0
    
    for i, question in enumerate(questions, 1):
        print(f"\n--- Xử lý câu hỏi {i}/{len(questions)} ---")
        print(f"ID: {question['id']}")
        
        # Đo thời gian xử lý câu hỏi
        start_time = time.time()
        answer = process_single_question(question, models)
        end_time = time.time()
        processing_time = end_time - start_time
        total_time += processing_time
        
        # Chỉ lấy ký tự đầu tiên từ answer (A, B, C, D)
        clean_answer = answer.strip()[0] if answer.strip() else "A"
        
        results.append({
            'id': question['id'],
            'answer': clean_answer
        })
        
        print(f"✅ Đã xử lý xong câu hỏi {i}/{len(questions)}: {clean_answer}")
        print(f"⏱️  Thời gian xử lý: {processing_time:.2f} giây ({processing_time/60:.2f} phút)")
        
        # Tính thời gian trung bình và ước tính thời gian còn lại
        avg_time = total_time / i
        remaining_questions = len(questions) - i
        estimated_remaining_time = avg_time * remaining_questions
        print(f"📊 Thời gian TB: {avg_time:.2f}s | Còn lại ước tính: {estimated_remaining_time/60:.1f} phút")
        
        # Lưu tạm sau mỗi 10 câu hỏi
        if i % 10 == 0:
            temp_output_path = f'public_test/temp_submission_{i}.csv'
            with open(temp_output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['id', 'answer'])
                writer.writeheader()
                writer.writerows(results)
            print(f"💾 Đã lưu tạm kết quả tại: {temp_output_path}")
    
    # Lưu kết quả ra file CSV
    output_path = 'public_test/submission.csv'
    print(f"\nĐang lưu kết quả ra file: {output_path}")
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✅ Đã hoàn thành! Kết quả được lưu tại: {output_path}")
    print(f"📊 Tổng số câu hỏi đã xử lý: {len(results)}")
    print(f"⏱️  Tổng thời gian xử lý: {total_time:.2f} giây ({total_time/60:.2f} phút)")
    print(f"📈 Thời gian trung bình mỗi câu hỏi: {total_time/len(results):.2f} giây")

if __name__ == "__main__":
    main()
