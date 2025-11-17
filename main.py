import json
import csv
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import pandas as pd
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

# Thread-safe lock
file_lock = threading.Lock()
print_lock = threading.Lock()

def thread_safe_print(*args, **kwargs):
    """In an toàn với đa luồng"""
    with print_lock:
        print(*args, **kwargs)

def load_models_fast():
    """Load models với tối ưu hóa tốc độ"""
    thread_safe_print("🚀 Đang load models (tối ưu tốc độ)...")
    
    # Clear GPU cache trước khi load
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 1. NF4 QUANTIZATION CONFIG
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,  # Tắt để tăng tốc
        bnb_4bit_compute_dtype=torch.float16
    )

    # 2. LOAD BLIP2 VLM
    model_path = "models/blip2-opt-2.7b"
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    model.eval()  # Chỉ eval, không .half() cho quantized model

    # 3. YOLO
    model_path_yolo = "models/yolo/best.pt"
    yolo_detector = YOLO(model_path_yolo)
    yolo_detector.model.eval()

    # 4. EMBEDDING + CHROMA
    EMB_PATH = "models/bkai-foundation-models/vietnamese-bi-encoder"
    embeddings = HuggingFaceEmbeddings(
        model_name=EMB_PATH,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
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

    # 6. LOAD PHI-3 MINI
    LLM_PATH = "models/microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        LLM_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    llm.eval()  # Chỉ eval, không .half() cho quantized model

    thread_safe_print("✅ Tất cả models đã load xong!")
    
    return {
        'processor': processor,
        'model': model,
        'yolo_detector': yolo_detector,
        'retriever': retriever,
        'reranker': reranker,
        'llm': llm,
        'tokenizer': tokenizer
    }

def process_single_question_fast(args):
    """Xử lý một câu hỏi với tối ưu hóa tốc độ"""
    question_data, models, question_index, total_questions = args
    
    start_time = time.time()
    
    try:
        # Khởi tạo tracker
        tracker = BestFrameTracker()
        frames_queue = extract_frames_to_queue(question_data['video_path'])
        
        frame_count = 0
        skip_factor = 2  # Skip mỗi frame thứ 2 để tăng tốc
        
        while True:
            frame = frames_queue.get()
            if frame is None:
                break
                
            frame_count += 1
            
            # Skip frames để tăng tốc
            if frame_count % skip_factor != 0:
                continue

            # YOLO detection với tối ưu
            with torch.no_grad():
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

        # VLM Captioning nhanh
        all_caption = ""
        for track_id, frameData in tracker.best_frames.items():
            box = frameData.box_info

            # Prompt ngắn gọn
            prompt = f"Traffic sign text/symbol in {box.bbox}:"

            with torch.no_grad():
                caption = describe_frame_with_prompt(
                    frameData.frame, 
                    prompt, 
                    models['processor'], 
                    models['model']
                )

            all_caption += f" {caption} [{box.class_name}]"

        # LLM processing
        vlm_description = all_caption
        question = question_data["question"]
        choices = question_data["choices"]

        with torch.no_grad():
            final_answer = lm_generate(
                llm=models['llm'],
                tokenizer=models['tokenizer'],
                retriever=models['retriever'],
                reranker=models['reranker'],
                vlm_description=vlm_description,
                question=question + "\n" + "\n".join(choices),
            )

        end_time = time.time()
        processing_time = end_time - start_time
        
        clean_answer = final_answer.strip()[0] if final_answer.strip() else "A"
        
        thread_safe_print(f"✅ [{question_index:3d}/{total_questions}] {question_data['id']}: {clean_answer} ({processing_time:.1f}s)")
        
        return {
            'id': question_data['id'],
            'answer': clean_answer,
            'processing_time': processing_time,
            'index': question_index
        }
        
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        
        thread_safe_print(f"❌ [{question_index:3d}/{total_questions}] {question_data['id']}: {str(e)[:50]}... ({processing_time:.1f}s)")
        
        return {
            'id': question_data['id'],
            'answer': "A",
            'processing_time': processing_time,
            'index': question_index
        }

def save_temp_results(results, temp_file_path):
    """Lưu kết quả tạm thời"""
    with file_lock:
        sorted_results = sorted(results, key=lambda x: x['index'])
        csv_data = [{'id': r['id'], 'answer': r['answer']} for r in sorted_results]
        
        with open(temp_file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'answer'])
            writer.writeheader()
            writer.writerows(csv_data)
        
        thread_safe_print(f"💾 Backup: {len(results)} kết quả -> {temp_file_path}")

def main():
    """Hàm chính tối ưu tốc độ cho 405 câu hỏi"""
    # Load data
    with open('public_test/public_test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    questions = test_data['data']
    
    # Load models
    models = load_models_fast()
    
    # Cấu hình tối ưu cho 405 câu hỏi
    max_workers = 3  # Số luồng song song
    
    # Đọc file backup mới nhất
    backup_file = 'public_test/backup_140.csv'
    backup_df = pd.read_csv(backup_file)

    # Giả sử backup_df đọc từ backup CSV
    results = backup_df.to_dict(orient='records')

    # Lọc câu chưa xử lý
    answered_ids = set(r['id'] for r in results)
    args_list = []
    for i, question in enumerate(questions, 1):
        if question['id'] in answered_ids:
            continue
        args_list.append((question, models, i, len(questions)))

    total_time = 0
    start_total = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_args = {executor.submit(process_single_question_fast, args): args for args in args_list}
        
        completed_count = 0
        
        for future in as_completed(future_to_args):
            try:
                result = future.result()
                results.append(result)
                completed_count += 1
                total_time += result['processing_time']
                
                # Progress update mỗi 5 câu hỏi
                if completed_count % 5 == 0:
                    avg_time = total_time / completed_count
                    remaining = len(questions) - completed_count
                    estimated_remaining = (remaining * avg_time) / max_workers
                    throughput = completed_count / ((time.time() - start_total) / 60)  # câu/phút
                    
                    thread_safe_print(f"📈 {completed_count:3d}/{len(questions)} | "
                                    f"TB: {avg_time:4.1f}s | "
                                    f"Tốc độ: {throughput:4.1f} câu/phút | "
                                    f"Còn lại: {estimated_remaining/60:4.1f} phút")
                
                # Backup mỗi 20 câu hỏi
                if completed_count % 20 == 0:
                    temp_file = f'public_test/backup_{completed_count}.csv'
                    save_temp_results(results, temp_file)
                    
            except Exception as exc:
                thread_safe_print(f'❌ Lỗi: {exc}')
    
    end_total = time.time()
    total_wall_time = end_total - start_total
    
    # Lưu kết quả cuối cùng
    output_path = 'public_test/submission.csv'
    thread_safe_print(f"\n💾 Lưu kết quả cuối cùng: {output_path}")
    
    sorted_results = sorted(results, key=lambda x: x['index'])
    csv_data = [{'id': r['id'], 'answer': r['answer']} for r in sorted_results]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer'])
        writer.writeheader()
        writer.writerows(csv_data)
    
    # Ước tính cho các dataset khác
    if len(results) < 405:
        estimated_405 = (405 * total_wall_time/len(results)) / 60
        thread_safe_print(f"💡 Ước tính cho 405 câu hỏi: {estimated_405:.1f} phút")
    
    print("="*60)

if __name__ == "__main__":
    main()
