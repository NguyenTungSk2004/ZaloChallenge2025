import os
import cv2
from modules.extract_frames import extract_frames_to_queue

if __name__ == "__main__":
    video_path = r"train/videos/faf14ff0_094_clip_010_0066_0075_Y.mp4"
    output_dir = r"check_frames/faf14ff0_094_clip_010_0066_0075_Y"  # thư mục lưu frame
    os.makedirs(output_dir, exist_ok=True)

    # Lấy frame theo FPS mong muốn
    frames = extract_frames_to_queue(video_path)

    frame_id = 0
    while True:
        frame = frames.get()
        if frame is None:
            break

        # Tên file ví dụ: frame_000001.jpg
        filename = os.path.join(output_dir, f"frame_{frame_id:06d}.jpg")
        cv2.imwrite(filename, frame)
        frame_id += 1

    print(f"✅ Hoàn tất, đã lưu {frame_id} frame vào '{output_dir}'")
