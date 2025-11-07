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
    """
    Trích xuất frame từ video, tăng tốc khi có chuyển động, và đưa vào hàng đợi (queue).

    Args:
        video_path (str): Đường dẫn video.
        base_fps (float): FPS khi không có chuyển động.
        high_fps (float): FPS khi có chuyển động.
        motion_threshold (float): Ngưỡng phát hiện chuyển động (độ chênh lệch pixel trung bình).
        motion_persist_time (float): Thời gian duy trì high_fps sau khi hết chuyển động.
        frame_diff_interval (int): Khoảng so sánh frame để tính biến thiên.
        max_queue_size (int): Kích thước tối đa của queue.

    Returns:
        queue.Queue: Hàng đợi chứa các frame (numpy.ndarray).
    """

    q = Queue(maxsize=max_queue_size)

    def worker():
        if not os.path.exists(video_path):
            print(f"❌ Video không tồn tại: {video_path}")
            q.put(None)
            return

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or fps == 0:
            print(f"❌ Không thể mở video hoặc lấy FPS: {video_path}")
            q.put(None)
            return

        frame_interval = int(max(1, fps / base_fps))
        high_frame_interval = int(max(1, fps / high_fps))

        prev_gray = None
        frame_id = 0
        motion_mode = False
        motion_countdown = 0

        print(f"🎞️ Xử lý video {os.path.basename(video_path)} | FPS gốc: {fps:.1f}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

                interval = high_frame_interval if motion_mode else frame_interval

                if frame_id % interval == 0:
                    if not q.full():
                        print("✅ Đưa vào queue:", frame_id)
                        q.put(frame.copy())  # copy để tránh lỗi bộ nhớ
                    else:
                        print("⚠️ Queue đầy, bỏ qua frame")

            prev_gray = gray

        cap.release()
        q.put(None)  # báo hiệu kết thúc video
        print(f"✅ Hoàn tất: {os.path.basename(video_path)}")

    # chạy trong thread riêng để queue nhận frame bất đồng bộ
    threading.Thread(target=worker, daemon=True).start()

    return q
