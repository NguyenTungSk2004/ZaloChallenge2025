import time
import json
import csv
import os
import hashlib
import torch
from load_models import load_models
from modules.ocr_helper import TrafficSignOCR
from modules.tracker import BestFrameTracker
from modules.extract_frames import extract_frames_to_queue
from modules.vlm import describe_frame_with_prompt
from modules.qa import lm_generate

ocr_reader = None
try:
    ocr_reader = TrafficSignOCR(backend='easy', use_gpu=True)
except Exception as e:
    print(f"  Warning: OCR failed({e})")

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

def save_temp_results(results, temp_file_path):
    """Lưu kết quả tạm thời"""
    sorted_results = sorted(results, key=lambda x: x['index'])
    csv_data = [{'id': r['id'], 'answer': r['answer']} for r in sorted_results]
    
    with open(temp_file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer'])
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"💾 Backup: {len(results)} kết quả -> {temp_file_path}")

def process_single_question_fast(question_data, models, question_index, total_questions):
    """Xử lý một câu hỏi với cache VLM"""
    video_path = question_data['video_path']
    
    try:
        # Kiểm tra cache VLM trước
        vlm_description = get_vlm_cache(video_path)
        
        if not vlm_description:
            # Kiểm tra video có mở được không
            try:
                frames_queue = extract_frames_to_queue(video_path)
            except Exception as video_error:
                # Trả về default answer nếu video lỗi
                return {
                    'id': question_data['id'],
                    'answer': "A",
                    'index': question_index
                }
            
            # Khởi tạo tracker
            tracker = BestFrameTracker()
            frame_count = 0
            
            while True:
                frame = frames_queue.get()
                if frame is None:
                    break
                    
                frame_count += 1

                # YOLO detection với error handling
                try:
                    with torch.no_grad():
                        results = models['yolo_detector'].track(frame, tracker="bytetrack.yaml", verbose=False)
                        
                    if not results or len(results) == 0:
                        continue
                except Exception as yolo_error:
                    # Skip frame nếu YOLO lỗi
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
                    
                # OCR Text
                ocr_text = ""
                if ocr_reader is not None:
                    try:
                        ocr_text = ocr_reader.read_text_from_roi(
                            frame=frameData.frame,
                            bbox=box.bbox,
                            conf_threshold=0.5,  # Confidence threshold
                            enhance=True,        # Tăng cường ảnh
                            padding=5            # Padding xung quanh bbox
                        )
                        if ocr_text:
                            ocr_text = ocr_reader.clean_traffic_text(ocr_text)
                    except Exception as e:
                        print(f"    [OCR Warning] Track {track_id}: {e}")
                        ocr_text = ""

                all_caption += f" {caption} [The traffic sign class is {box.class_name}"
                if ocr_text != "":
                    all_caption += f" The sign contains the text '{ocr_text}'."

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
        
        print(f"✅ [{question_index:3d}/{total_questions}] {question_data['id']}: {clean_answer}")
        
        return {
            'id': question_data['id'],
            'answer': clean_answer,
            'index': question_index
        }
        
    except Exception as e:
        print(f"❌ [{question_index:3d}/{total_questions}] {question_data['id']}: {str(e)[:50]}...")
        
        return {
            'id': question_data['id'],
            'answer': "A",
            'index': question_index
        }

def main():
    """Hàm chính xử lý tuần tự từng câu hỏi"""
    # Load data
    with open('public_test/public_test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    questions = test_data['data']
    
    # Load models
    models = load_models()
    
    # Xử lý tuần tự từng câu hỏi
    results = []
    
    for i, question in enumerate(questions, 1):
        result = process_single_question_fast(question, models, i, len(questions))
        results.append(result)
        
        # Backup mỗi 20 câu hỏi
        if i % 20 == 0:
            temp_file = f'public_test/backup_{i}.csv'
            save_temp_results(results, temp_file)
    
    # Lưu kết quả cuối cùng
    output_path = 'public_test/submission.csv'
    print(f"\n💾 Lưu kết quả cuối cùng: {output_path}")
    
    sorted_results = sorted(results, key=lambda x: x['index'])
    csv_data = [{'id': r['id'], 'answer': r['answer']} for r in sorted_results]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer'])
        writer.writeheader()
        writer.writerows(csv_data)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time() - start_time
    hours = int(end_time // 3600)
    minutes = int((end_time % 3600) // 60)
    seconds = int(end_time % 60)
    print(f"Execution time: {hours:02d}:{minutes:02d}:{seconds:02d}")
