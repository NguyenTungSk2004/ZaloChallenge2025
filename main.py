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

def process_yolo_tracker(frames_queue, model: YOLO):
    frames = []
    video_info_list = []

    while True:
        frame = frames_queue.get()
        box_info_list = []
        if frame is None:
            break
        try:
            with torch.no_grad():
                results = model.track(frame, tracker="bytetrack.yaml", verbose=False)
                
            if not results or len(results) == 0:
                continue
        except Exception:
            continue

        # Cập nhật tracker
        for box in results[0].boxes:
            if box.id is None: continue
    
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            track_id = int(box.id)
            conf = float(box.conf)
            cls_id = int(box.cls)
            cls_name = results[0].names[cls_id]

            info_str = f"Đối tượng {track_id} [Loại đối tượng là {cls_name}, vị trí: {bbox}, độ tin cậy: {conf:.3f}.]"
            box_info_list.append(info_str)

        if box_info_list:
            box_info_str = " ".join(box_info_list)
            frames.append(frame)
            video_info_list.append(box_info_str)
        else:
            box_info_str = "Không phát hiện đối tượng nào quan trọng trong frame."

    if video_info_list:
        video_info = " ".join(video_info_list)
    else:
        video_info = "Không phát hiện đối tượng nào quan trọng trong video."
    return frames, video_info

def process_single_question(question_data, models, question_index, total_questions):
    video_path = question_data['video_path']
    
    try:
        vlm_description, video_info = get_vlm_cache(video_path)
        
        if vlm_description is None:  # Chỉ xử lý khi không có cache
            frames_queue = extract_frames_to_queue(video_path)
            frames, video_info = process_yolo_tracker(frames_queue, models['yolo'])

            # 4. Gọi VLM
            vlm_description = generate_video_description(frames, models, video_info, question_data['question'] + "\n".join(question_data['choices']))
            save_vlm_cache(video_path, vlm_description, video_info)
        
        return {
            'id': question_data['id'],
            'answer': llm_choise_answer(models, vlm_description, question_data, video_info),
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
    os.makedirs('Results', exist_ok=True)
        
    with open('public_test/public_test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    questions = test_data['data']
    
    models = load_models()
    
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n🔍 Đang xử lý câu hỏi {i}/{len(questions)}: {question['id']}")
        start_time = time.time()
        result = process_single_question(question, models, i, len(questions))
        results.append(result)
        end_time = time.time() - start_time
        print(f"⏱️ Thời gian xử lý: {end_time:.2f} giây")
        print(f"✅ [{i:3d}/{len(questions)}] {result['id']}: {result['answer']}")
        if i % 20 == 0:
            temp_file = f'Results/submission_{i}.csv'
            save_temp_results(results, temp_file)
    
    # Lưu kết quả cuối cùng
    output_path = 'Results/submission.csv'
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
