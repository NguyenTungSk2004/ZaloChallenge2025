import json
import csv
import threading
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from load_models import load_models
from modules.tracker import BestFrameTracker
from modules.extract_frames import extract_frames_to_queue
from modules.vlm import describe_frame_with_prompt
from modules.qa import lm_generate

# Thread-safe lock
file_lock = threading.Lock()
print_lock = threading.Lock()

# Cache VLM vào disk
def get_vlm_cache(video_path):
    """Load VLM description từ cache"""
    cache_dir = "cached_vlm"
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = hashlib.md5(video_path.encode()).hexdigest()[:8]
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)['vlm_description']
    return None

def save_vlm_cache(video_path, vlm_description):
    """Lưu VLM description vào cache"""
    cache_dir = "cached_vlm"
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = hashlib.md5(video_path.encode()).hexdigest()[:8]
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump({'vlm_description': vlm_description}, f, ensure_ascii=False)

def thread_safe_print(*args, **kwargs):
    """In an toàn với đa luồng"""
    with print_lock:
        print(*args, **kwargs)

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

def process_single_question_fast(args):
    """Xử lý một câu hỏi với cache VLM"""
    question_data, models, question_index, total_questions = args
    video_path = question_data['video_path']
    
    try:
        # Kiểm tra cache VLM trước
        vlm_description = get_vlm_cache(video_path)
        
        if not vlm_description:
            # Khởi tạo tracker
            tracker = BestFrameTracker()
            frames_queue = extract_frames_to_queue(video_path)
            
            frame_count = 0
            
            while True:
                frame = frames_queue.get()
                if frame is None:
                    break
                    
                frame_count += 1

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

            # VLM processing nhanh
            all_caption = ""
            for track_id, frameData in tracker.best_frames.items():
                box = frameData.box_info

                prompt = (
                    f"Question: Describe the environment and context of the car. "
                    f"Also, what is the **exact text or symbol** visible on the traffic sign associated with the bounding box {box.bbox}? Answer:"
                )

                with torch.no_grad():
                    caption = describe_frame_with_prompt(
                        frameData.frame, 
                        prompt, 
                        models['processor'], 
                        models['model']
                    )

                all_caption += f" {caption} [{box.class_name}]"

            vlm_description = all_caption
            # Lưu cache
            save_vlm_cache(video_path, vlm_description)

        # LLM processing
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
        
        clean_answer = final_answer.strip()[0] if final_answer.strip() else "A"
        
        thread_safe_print(f"✅ [{question_index:3d}/{total_questions}] {question_data['id']}: {clean_answer}")
        
        return {
            'id': question_data['id'],
            'answer': clean_answer,
            'index': question_index
        }
        
    except Exception as e:
        thread_safe_print(f"❌ [{question_index:3d}/{total_questions}] {question_data['id']}: {str(e)[:50]}...")
        
        return {
            'id': question_data['id'],
            'answer': "A",
            'index': question_index
        }

def main():
    """Hàm chính tối ưu tốc độ cho 405 câu hỏi"""
    # Load data
    with open('public_test/public_test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    questions = test_data['data']
    
    # Load models
    models = load_models()
    
    # Cấu hình tối ưu cho 405 câu hỏi
    max_workers = 5  # Số luồng song song
    results = []
    args_list = []
    for i, question in enumerate(questions, 1):
        args_list.append((question, models, i, len(questions)))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_args = {executor.submit(process_single_question_fast, args): args for args in args_list}
        
        completed_count = 0
        
        for future in as_completed(future_to_args):
            try:
                result = future.result()
                results.append(result)
                completed_count += 1
                
                # Backup mỗi 20 câu hỏi
                if completed_count % 20 == 0:
                    temp_file = f'public_test/backup_{completed_count}.csv'
                    save_temp_results(results, temp_file)
                    
            except Exception as exc:
                thread_safe_print(f'❌ Lỗi: {exc}')
    
    # Lưu kết quả cuối cùng
    output_path = 'public_test/submission.csv'
    thread_safe_print(f"\n💾 Lưu kết quả cuối cùng: {output_path}")
    
    sorted_results = sorted(results, key=lambda x: x['index'])
    csv_data = [{'id': r['id'], 'answer': r['answer']} for r in sorted_results]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer'])
        writer.writeheader()
        writer.writerows(csv_data)

if __name__ == "__main__":
    main()
