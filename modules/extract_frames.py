import cv2
import os
import numpy as np
from queue import Queue
import threading

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

    # cache directory
    cache_dir = "cached_frames"
    os.makedirs(cache_dir, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cached_video_path = os.path.join(cache_dir, f"{video_name}.mp4")

    # ====================================================
    # 1) Nếu đã có file cache rồi → chỉ đọc lại là xong
    # ====================================================
    if os.path.exists(cached_video_path):
        def load_cache_worker():
            cap = cv2.VideoCapture(cached_video_path)
            if not cap.isOpened():
                print(f"❌ Không mở được cached video: {cached_video_path}!")
                q.put(None)
                return

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                q.put(frame)

            cap.release()
            q.put(None)

        threading.Thread(target=load_cache_worker, daemon=True).start()
        return q

    # ====================================================
    # 2) Chưa có cache → xử lý video + ghi ra file MP4
    # ====================================================
    def worker():
        if not os.path.exists(video_path):
            print(f"❌ Video không tồn tại: {video_path}")
            q.put(None)
            return

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        if not cap.isOpened() or fps == 0:
            print(f"❌ Không thể mở video hoặc lấy FPS")
            q.put(None)
            return

        # Chuẩn bị writer để cache
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = None

        # Tính khoảng frame
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
                # Khởi tạo writer lần đầu tiên
                if out is None:
                    h, w = frame.shape[:2]
                    out = cv2.VideoWriter(cached_video_path, fourcc, high_fps, (w, h))

                out.write(frame)     # Ghi vào file cache
                q.put(frame)         # Gửi vào queue

            prev_gray = gray

        cap.release()
        if out is not None:
            out.release()

        q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    return q
