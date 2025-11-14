import os
from ultralytics import YOLO
from modules.extract_frames import extract_frames_to_queue
from modules.tracker import BestFrameTracker
from utils.SaveFrame import save_track_frame
from modules.vlm import describe_frame_with_prompt
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

# Đường dẫn đến thư mục chứa mô hình của bạn
model_path = './models/blip2-opt-2.7b'

# Tải bộ xử lý và mô hình
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model_path_yolo = "models/best.pt" # Đường dẫn cho mô hình YOLO
video_path = r"train/videos/00b9d4a3_129_clip_002_0009_0015_N.mp4"
output_dir = r"check_frame/00b9d4a3_129_clip_002_0009_0015_N_best"
os.makedirs(output_dir, exist_ok=True)

yolo_detector = YOLO(model_path_yolo) # Đổi tên biến mô hình YOLO
frames_queue = extract_frames_to_queue(video_path)
tracker = BestFrameTracker()

frame_count = 0
while True:
    frame = frames_queue.get()
    if frame is None:
        break

    frame_count += 1

    # Object detection và tracking
    results = yolo_detector.track(frame, tracker="bytetrack.yaml", verbose=False)

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
        tracker.update_track(frame, track_id, bbox, confidence, cls_name)

# Sau khi tracking, chúng ta sẽ điền vào list frames toàn cục
prompt_for_logging = "Hãy mô tả bối cảnh của các frame sau:"
for track_id, frameData in tracker.best_frames.items():
    box = frameData.box_info
    
    # Thay đổi prompt thành chuỗi rỗng để khuyến khích mô hình tự do mô tả hình ảnh
    vlm_instruction_prompt = f"Question: Describe the surrounding environment and context of the car. Furthermore, what is the location of the traffic sign associated with the bounding box {box.bbox}? Answer:"
    caption_from_vlm = describe_frame_with_prompt(frameData.frame, vlm_instruction_prompt, processor, model) # Sử dụng VLM model và processor toàn cục
    
    print(f"Caption {track_id}: {caption_from_vlm}")