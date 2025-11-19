import shutil
import time
import json
import csv
import torch
from load_models import load_models
from modules.tracker import BestFrameTracker
from modules.extract_frames import extract_frames_to_queue
from modules.vlm import generate_video_description
from modules.qa import lm_generate
from utils.cached_helper import *
from ultralytics import YOLO
import os

def process_yolo_tracker(frames_queue, model: YOLO) -> BestFrameTracker:
    tracker = BestFrameTracker()
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

        # Cập nhật tracker
        for box in results[0].boxes:
            if box.id is None: continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            track_id = int(box.id)
            conf = float(box.conf)
            cls_id = int(box.cls)
            cls_name = results[0].names[cls_id]

            tracker.update_track(frame, track_id, (x1, y1, x2, y2), conf, cls_name)
    return tracker

def choise_answer(models, vlm_description, question_data):
    question = question_data["question"]
    choices = question_data["choices"]

    with torch.no_grad():
        final_answer = lm_generate(
            llm=models['llm'],
            tokenizer=models['tokenizer'],
            retriever=models['retriever'],
            reranker=models['reranker'],
            vlm_description=vlm_description,
            question=question,
            choices=choices
        )
    
    return final_answer


def process_single_question(question_data, models, question_index, total_questions):
    video_path = question_data['video_path']
    
    try:
        vlm_description = get_vlm_cache(video_path)
        if vlm_description:
            return {
                'id': question_data['id'],
                'answer': choise_answer(models, vlm_description, question_data),
                'index': question_index
            }
        
        frames_queue = extract_frames_to_queue(video_path)
        tracker = process_yolo_tracker(frames_queue, models['yolo'])

        frames = []
        box_info_list = [] # Dùng list để chứa các đoạn text, nhanh hơn cộng chuỗi

        for track_id, frameData in tracker.best_frames.items():
            box = frameData.box_info
            frames.append(frameData.frame)
            info_str = f"Frame {track_id} [Loại đối tượng là {box.class_name}, vị trí: {box.bbox}, điểm đánh giá với 40% độ tin cậy và 60% độ nét: {frameData.score:.3f}.]"
            box_info_list.append(info_str)

        if box_info_list:
            box_info = "".join(box_info_list) # Python tối ưu hóa memory cực tốt cho hàm join
        else:
            box_info = "Không phát hiện đối tượng nào quan trọng trong video."

        # 4. Gọi VLM
        vlm_description = generate_video_description(frames, models, box_info)
        save_vlm_cache(video_path, vlm_description)

        return {
            'id': question_data['id'],
            'answer': choise_answer(models, vlm_description, question_data),
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
