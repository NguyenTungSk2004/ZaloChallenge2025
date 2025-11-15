import torch
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForImageTextToText

from modules.tracker import BestFrameTracker
from modules.extract_frames import extract_frames_to_queue
from modules.vlm import describe_frame_with_prompt

# Đường dẫn đến thư mục chứa mô hình của bạn
model_path = './models/blip2-opt-2.7b'

# Tải bộ xử lý và mô hình
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model_path_yolo = "models/best.pt" # Đường dẫn cho mô hình YOLO
video_path = r"train/videos/03cde2e3_322_clip_017_0123_0129_N.mp4"

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


all_caption = ""
# Sau khi tracking, chúng ta sẽ điền vào list frames toàn cục
for track_id, frameData in tracker.best_frames.items():
    box = frameData.box_info
    vlm_instruction_prompt = f"Question: Describe the surrounding environment and context of the car. Furthermore, what is the location of the traffic sign associated with the bounding box {box.bbox}? Answer:"
    caption_from_vlm = describe_frame_with_prompt(frameData.frame, vlm_instruction_prompt, processor, model) # Sử dụng VLM model và processor toàn cục
    all_caption += f"\n Caption Frame {track_id}: {caption_from_vlm} Information the traffic sign:[label: '{box.class_name}', score: '{frameData.score}']"
    
from modules.qa import lm_generate

lm_generate(all_caption)