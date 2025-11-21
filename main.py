import time
import json 
import csv
import torch
from load_models import load_models
from modules.extract_frames import extract_frames_to_queue
from modules.vlm import generate_video_description
from modules.llm import llm_choise_answer
from utils.cached_helper import *
from ultralytics import YOLO
import os
from modules.tracker import BestFrameTracker 

# --- HẰNG SỐ ĐƯỜNG DẪN MỚI ---
YOLO_JSON_DIR = 'yolo_json'

# --- HÀM LƯU JSON MỚI ---
def save_yolo_json(question_id, yolo_data_json):
    """Lưu chuỗi JSON YOLO vào thư mục yolo_json theo ID câu hỏi."""
    os.makedirs(YOLO_JSON_DIR, exist_ok=True)
    file_path = os.path.join(YOLO_JSON_DIR, f"{question_id}.json")
    
    # Ghi chuỗi JSON đã được format đẹp vào file
    with open(file_path, 'w', encoding='utf-8') as f:
        # Chúng ta dùng json.loads vì yolo_data_json đã là một chuỗi JSON
        data = json.loads(yolo_data_json)
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Đã lưu YOLO JSON: {file_path}")
# -----------------------------

def process_yolo_tracker(frames_queue, model: YOLO, tracker: BestFrameTracker):
    # Khởi tạo lại tracker cho mỗi video
    tracker.__init__()
    
    while True:
        frame = frames_queue.get()
        if frame is None:
            break
            
        try:
            with torch.no_grad():
                results = model.track(frame, tracker="bytetrack.yaml", verbose=False)
                
            if not results or len(results) == 0:
                continue
        except Exception:
            continue

        # Cập nhật tracker với các đối tượng được phát hiện
        for box in results[0].boxes:
            if box.id is None: continue
    
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            track_id = int(box.id)
            conf = float(box.conf)
            cls_id = int(box.cls)
            # LẤY TÊN LỚP GỐC BẰNG TIẾNG VIỆT
            cls_name = results[0].names[cls_id] 

            # Gọi BestFrameTracker để tìm frame tốt nhất cho mỗi đối tượng
            # LƯU TÊN LỚP TIẾNG VIỆT vào tracker
            tracker.update_track(frame, track_id, bbox, conf, cls_name) 
            
    # Lấy ra các frames tốt nhất từ tracker
    frames = [data.frame for data in tracker.best_frames.values()]
    
    # TẠO MẢNG JSON DỮ LIỆU YOLO (KEYS TIẾNG ANH, VALUES TIẾNG VIỆT)
    yolo_data_list = []
    for track_id, frame_data in tracker.best_frames.items():
        yolo_data_list.append({
            "track_id": track_id,
            "object_type": frame_data.box_info.class_name, # <-- Tên lớp Tiếng Việt
            "bbox": frame_data.box_info.bbox.tolist(), # Chuyển numpy array sang list
            "confidence": round(frame_data.box_info.confidence, 3),
            "sharpness": round(frame_data.box_info.sharpness, 2)
        })

    if yolo_data_list:
        # Sử dụng json.dumps để chuyển mảng Python sang chuỗi JSON
        video_info = json.dumps(yolo_data_list, ensure_ascii=False, indent=2)
    else:
        # Trả về mảng JSON rỗng
        video_info = "[]" 
        
    return frames, video_info

# Sửa đổi process_single_question để lưu file JSON
def process_single_question(question_data, models, tracker: BestFrameTracker, question_index, total_questions):
    video_path = question_data['video_path']
    question_id = question_data['id']
    
    try:
        vlm_description, video_info = get_vlm_cache(video_path)
        
        if vlm_description is None:  # Chỉ xử lý khi không có cache
            frames_queue = extract_frames_to_queue(video_path)
            # TRUYỀN TRACKER VÀO HÀM
            frames, video_info = process_yolo_tracker(frames_queue, models['yolo'], tracker)
            
            # --- LƯU FILE JSON YOLO VÀO THƯ MỤC MỚI ---
            save_yolo_json(question_id, video_info)

            # 4. Gọi VLM
            vlm_description = generate_video_description(frames, models, video_info, question_data['question'] + "\n".join(question_data['choices']))
            save_vlm_cache(video_path, vlm_description, video_info)
        
        return {
            'id': question_id,
            'answer': llm_choise_answer(models, vlm_description, question_data, video_info),
            'index': question_index
        }
        
    except Exception as e:
        print(f"[{question_index:3d}/{total_questions}] {question_data['id']}: {str(e)[:50]}...")
        return {
            'id': question_data['id'],
            'answer': "A",
            'index': question_index
        }

def main():
    """Hàm chính xử lý tuần tự từng câu hỏi"""
    os.makedirs('Results', exist_ok=True)
        
    with open('public_test/public_test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    questions = test_data['data']
    
    # KHỞI TẠO TRACKER 
    tracker = BestFrameTracker() 
    models = load_models()
    
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\nĐang xử lý câu hỏi {i}/{len(questions)}: {question['id']}")
        start_time = time.time()
        # TRUYỀN TRACKER VÀO process_single_question
        result = process_single_question(question, models, tracker, i, len(questions)) 
        results.append(result)
        end_time = time.time() - start_time
        print(f"Thời gian xử lý: {end_time:.2f} giây")
        print(f"[{i:3d}/{len(questions)}] {result['id']}: {result['answer']}")
        if i % 20 == 0:
            temp_file = f'Results/submission_{i}.csv'
            save_temp_results(results, temp_file)
    
    # Lưu kết quả cuối cùng
    output_path = 'Results/submission.csv'
    print(f"\nLưu kết quả cuối cùng: {output_path}")
    
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