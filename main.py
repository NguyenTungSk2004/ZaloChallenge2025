import os
from ultralytics import YOLO
from modules.extract_frames import extract_frames_to_queue
from modules.tracker import BestFrameTracker
from utils.SaveFrame import save_track_frame

def main():
    model_path = "models/best.pt"
    video_path = r"train/videos/00b9d4a3_129_clip_002_0009_0015_N.mp4"
    output_dir = r"check_frame/00b9d4a3_129_clip_002_0009_0015_N_best"
    os.makedirs(output_dir, exist_ok=True)
    
    model = YOLO(model_path)
    frames = extract_frames_to_queue(video_path)
    tracker = BestFrameTracker()
    
    frame_count = 0
    while True:
        frame = frames.get()
        if frame is None:
            break
        
        frame_count += 1
        
        # Object detection và tracking
        results = model.track(frame, tracker="bytetrack.yaml", verbose=False)
        
        if not results or len(results) == 0:
            continue

        # Xử lý từng object được detect
        for box in results[0].boxes:
            if box.id is None:  # Bỏ qua nếu không có track ID
                continue
            
            # Lấy thông tin box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            track_id = int(box.id)
            confidence = float(box.conf)
            cls_id = int(box.cls)
            cls_name = results[0].names[cls_id] if hasattr(results[0], "names") else str(cls_id)

            # Cập nhật tracker
            bbox = (x1, y1, x2, y2)
            print(f"Processing Frame {frame_count} Track ID {track_id}, BBox: {bbox}, Conf: {confidence:.3f}, Class: {cls_name}")
            tracker.update_track(frame, track_id, bbox, confidence, cls_name)

    for track_id, frameData in tracker.best_frames.items():
        save_track_frame(frameData.frame, frameData=frameData, track_id=track_id, output_path=f"{output_dir}/best_track_{track_id}.jpg")

    # Hiển thị kết quả phân tích
    print(f"📊 Đã phân tích {frame_count} frame, tìm thấy {len(tracker.best_frames)} đối tượng unique")


if __name__ == "__main__":
    main()
