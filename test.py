import torch
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForImageTextToText

from modules.tracker import BestFrameTracker
from modules.extract_frames import extract_frames_to_queue
from modules.vlm_test import describe_frame_with_prompt
from config import USE_GPU, USE_BLIP2_GPU

test_case = {
    "id": "train_0211",
    "question": "Trong video, biển báo nào xuất hiện đầu tiên?",
    "choices": [
        "A. Biển giữ khoảng cách an toàn",
        "B. Biển giới hạn tốc độ tối đa",
        "C. Biển cấm dừng đỗ",
        "D. Biển cấm xe tải"
    ],
    "video_path": "E:/Zalo Challenge 2025/traffic_buddy_train+public_test/train/videos/03cde2e3_322_clip_017_0123_0129_N.mp4"
}

# ========================================
# 1. Load BLIP2 (ép về CPU khi test)
# ========================================
model_path = "./models/blip2-opt-2.7b"
print("🔄 Loading BLIP2...")

device_blip = "cuda" if (torch.cuda.is_available() and USE_BLIP2_GPU) else "cpu"

processor = AutoProcessor.from_pretrained(model_path)

model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if device_blip == "cuda" else torch.float32,
    low_cpu_mem_usage=True
).to(device_blip)

print(f"✅ BLIP2 loaded on: {device_blip}")


# ========================================
# 2. Load YOLO detector (GPU nếu có)
# ========================================
model_path_yolo = "models/yolo/best.pt"
yolo_detector = YOLO(model_path_yolo)
print("✅ YOLO loaded")


# ========================================
# 3. Load video frames
# ========================================
video_path = r"E:/Zalo Challenge 2025/traffic_buddy_train+public_test/train/videos/03cde2e3_322_clip_017_0123_0129_N.mp4"
frames_queue = extract_frames_to_queue(video_path)
tracker = BestFrameTracker()


# ========================================
# 4. Tracking
# ========================================
frame_count = 0

while True:
    frame = frames_queue.get()
    if frame is None:
        break

    frame_count += 1

    results = yolo_detector.track(frame, tracker="bytetrack.yaml", verbose=False)
    if not results or len(results) == 0:
        continue

    for box in results[0].boxes:
        if box.id is None:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        track_id = int(box.id)
        conf = float(box.conf)
        cls_id = int(box.cls)
        cls_name = results[0].names[cls_id]

        tracker.update_track(frame, track_id, (x1, y1, x2, y2), conf, cls_name)


# ========================================
# 5. Gọi VLM để mô tả khung hình tốt nhất
# ========================================
all_caption = ""

for track_id, frameData in tracker.best_frames.items():
    box = frameData.box_info

    prompt = (
        f"Question: Describe the surrounding environment and context of the car. "
        f"Furthermore, what is the location of the traffic sign associated with the bounding box {box.bbox}? Answer:"
    )

    caption = describe_frame_with_prompt(frameData.frame, prompt, processor, model)

    all_caption += (
        f"\n Caption Frame {track_id}: {caption} "
        f"Information the traffic sign:[label: '{box.class_name}', score: '{frameData.score}']"
    )


# ========================================
# 6. Gọi RAG/Phi-3
# ========================================

# Import hàm 'lm_generate' đã sửa
from modules.qa import lm_generate

print("\n🔮 Running Orchestrator (RAG + VLM)...")

# Lấy thông tin từ test_case (đã định nghĩa ở đầu file)
vlm_context = all_caption  # BẰNG CHỨNG VLM (từ Bước 5)
query = test_case["question"]
choices = test_case["choices"]

# Gọi hàm tổng hợp
final_answer = lm_generate(
    vlm_context=vlm_context,
    query=query,
    choices=choices
)

print("\n🧠 FINAL ANSWER:\n", final_answer)
