import time
import json
import torch
from load_models import load_models
from modules.ocr_helper import TrafficSignOCR
from modules.tracker import BestFrameTracker
from modules.extract_frames import extract_frames_to_queue
from utils.SaveFrame import save_frame


ocr_reader = None
try:
    ocr_reader = TrafficSignOCR(backend='easy', use_gpu=True)
except Exception as e:
    print(f"  Warning: OCR failed({e})")

def main():
    """Hàm chính xử lý tuần tự từng câu hỏi"""
    # Load data
    with open('public_test/public_test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    questions = test_data['data'][0]
    video_path = questions['video_path']

    frames_queue = extract_frames_to_queue(video_path)
    # Khởi tạo tracker
    tracker = BestFrameTracker()
    frame_count = 0

    models = load_models(['yolo'])
    
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
        save_frame(frameData=frameData, track_id=track_id, output_path=f"track_frames/{track_id}_{frameData.id}.jpg")
        print(f"OCR Text for track_frames/{track_id}_{frameData.id}.jpg:", ocr_text)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time() - start_time
    hours = int(end_time // 3600)
    minutes = int((end_time % 3600) // 60)
    seconds = int(end_time % 60)
    print(f"Execution time: {hours:02d}:{minutes:02d}:{seconds:02d}")
