import cv2
import os
import numpy as np
from queue import Queue
import hashlib

def extract_frames_to_queue(
    video_path: str,
    base_fps: float = 2,
    high_fps: float = 10,
    motion_threshold: float = 30,
    motion_persist_time: float = 1.5,
    frame_diff_interval: int = 1,
    max_queue_size: int = 100
) -> Queue:

    q = Queue(maxsize=max_queue_size)

    # Cache directory
    cache_dir = "cached_frames"
    os.makedirs(cache_dir, exist_ok=True)

    video_hash = hashlib.md5(video_path.encode()).hexdigest()[:12]
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cached_video_path = os.path.join(cache_dir, f"{video_name}_{video_hash}.mp4")

    # Nếu đã có cache, đọc lại
    if os.path.exists(cached_video_path):
        print(f"📥 Đang đọc cache từ: {cached_video_path}")
        
        cap = cv2.VideoCapture(cached_video_path)
        if not cap.isOpened():
            os.remove(cached_video_path)
            print(f"❌ File cache bị lỗi, đã xóa.")
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                q.put(frame)
            cap.release()
            q.put(None)
            return q

    # Xử lý video và tạo cache mới
    if not os.path.exists(video_path):
        print(f"❌ Video không tồn tại: {video_path}")
        q.put(None)
        return q

    print(f"🎥 Đang xử lý: {os.path.basename(video_path)}")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened() or fps == 0:
        print(f"❌ Không thể mở video")
        q.put(None)
        return q

    fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
    out = None

    frame_interval = int(max(1, fps / base_fps))
    high_interval = int(max(1, fps / high_fps))

    prev_gray = None
    frame_id = 0
    motion_mode = False
    motion_countdown = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Phát hiện motion
        if prev_gray is not None and frame_id % frame_diff_interval == 0:
            diff = cv2.absdiff(gray, prev_gray)
            motion_score = np.mean(diff)

            if motion_score > motion_threshold:
                motion_mode = True
                motion_countdown = int(fps * motion_persist_time)
            else:
                motion_countdown -= 1
                if motion_countdown <= 0:
                    motion_mode = False

        # Chọn interval
        interval = high_interval if motion_mode else frame_interval

        if frame_id % interval == 0:
            # Khởi tạo writer
            if out is None:
                h, w = frame.shape[:2]
                out = cv2.VideoWriter(cached_video_path, fourcc, high_fps, (w, h))
                if not out.isOpened():
                     print(f"❌ Không thể khởi tạo VideoWriter")
                     cap.release()
                     q.put(None)
                     return q

            out.write(frame)
            q.put(frame)

        prev_gray = gray

    cap.release()
    
    if out is not None:
        out.release()
        print(f"💾 Đã tạo cache: {cached_video_path}")
    
    q.put(None)
    return q
