import shutil
import time
import json
import csv
import torch
from load_models import load_models
from modules.ocr_helper import TrafficSignOCR
from modules.tracker import BestFrameTracker
from modules.extract_frames import extract_frames_to_queue
from modules.vlm import describe_frame_with_prompt
from modules.qa import lm_generate
from utils.SaveFrame import save_frame
from utils.cached_helper import *
from ultralytics import YOLO
import os

ocr_reader = None
try:
    ocr_reader = TrafficSignOCR(backend='easy', use_gpu=True)
except Exception as e:
    print(f"  Warning: OCR failed({e})")

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

def process_single_question(question_data, models, question_index, total_questions):
    """Xử lý một câu hỏi với cache VLM"""
    video_path = question_data['video_path']
    frames_queue = extract_frames_to_queue(video_path)
    tracker = process_yolo_tracker(frames_queue, models['yolo_detector'])

    for track_id, frameData in tracker.best_frames.items():
        box = frameData.box_info

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
                print(f"[OCR Warning] Track {track_id}: {e}")
                ocr_text = ""

        save_frame(frameData=frameData, track_id=track_id, output_path=f"frames_track/{track_id}_{frameData.id}.jpg")
        print(f" The sign contains the text '{ocr_text}'.")

def main():
    """Hàm chính xử lý tuần tự từng câu hỏi"""
    os.makedirs('Results', exist_ok=True)
        
    # Load data
    with open('public_test/public_test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    questions = test_data['data']
    question = questions[0]

    # Load models
    models = load_models(['yolo'])
    process_single_question(question, models, 1, len(questions))


def cleanup_folders():
    """Xóa toàn bộ thư mục frames_extract và frames_track"""
    folders_to_clean = ["frames_extract", "frames_track"]
    
    for folder in folders_to_clean:
        if os.path.exists(folder):
            print(f"🗑️ Đang xóa thư mục: {folder}")
            shutil.rmtree(folder)
            print(f"✅ Đã xóa thành công: {folder}")
        else:
            print(f"ℹ️ Thư mục không tồn tại: {folder}")

if __name__ == "__main__":
    cleanup_folders()
    start_time = time.time()
    main()
    end_time = time.time() - start_time
    hours = int(end_time // 3600)
    minutes = int((end_time % 3600) // 60)
    seconds = int(end_time % 60)
    print(f"Execution time: {hours:02d}:{minutes:02d}:{seconds:02d}")

