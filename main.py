from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from ultralytics import YOLO
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModelForImageTextToText
from sentence_transformers import CrossEncoder
from langchain_community.llms import LlamaCpp

from modules.tracker import BestFrameTracker
from modules.extract_frames import extract_frames_to_queue
from modules.vlm import describe_frame_with_prompt
from config import USE_GPU, USE_BLIP2_GPU
from modules.qa import lm_generate

model_path = "models/blip2-opt-2.7b"
device_blip = "cuda" if (torch.cuda.is_available() and USE_BLIP2_GPU) else "cpu"
processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype=torch.float16 if device_blip == "cuda" else torch.float32,
    low_cpu_mem_usage=True
).to(device_blip)

model_path_yolo = "models/yolo/best.pt"
yolo_detector = YOLO(model_path_yolo)

tracker = BestFrameTracker()

# LOAD EMBEDDING
EMB_PATH = "models/bkai_vn_bi_encoder"
embeddings = HuggingFaceEmbeddings(
    model_name=EMB_PATH,
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': False}
)

# LOAD CHROMA VECTOR DB (BIỂN BÁO)
DB_PATH = "Vecto_Database/db_bienbao_2"
vectordb = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# RERANKER
RERANK_PATH = "models/ViRanker"
device = "cuda" if (torch.cuda.is_available() and USE_BLIP2_GPU) else "cpu"
reranker = CrossEncoder(RERANK_PATH, device=device)

# 4. LOAD PHI-3 MINI GGUF – SAFE LOADING
LLM_PATH = "models/Phi-3-gguf/Phi-3-mini-4k-instruct-q4.gguf"

# 1. Load model
print("Đang load model...")
llm = AutoModelForCausalLM.from_pretrained(
    LLM_PATH,
    model_type="phi3",       # Chỉ định model type là 'phi3'
    gpu_layers=50,           # Số layer offload lên GPU (thử nghiệm số này, 
    context_length=4096      # Tương tự n_ctx
)
print("Model đã được load xong!")

test_case =         {
            "id": "testa_0001",
            "question": "Theo trong video, nếu ô tô đi hướng chếch sang phải là hướng vào đường nào?",
            "choices": [
                "A. Không có thông tin",
                "B. Dầu Giây Long Thành",
                "C. Đường Đỗ Xuân Hợp",
                "D. Xa Lộ Hà Nội"
            ],
            "video_path": "public_test/videos/efc9909e_908_clip_001_0000_0009_Y.mp4"
        },

frames_queue = extract_frames_to_queue(test_case['video_path'])
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
# Lấy thông tin từ test_case (đã định nghĩa ở đầu file)
vlm_description = all_caption  # BẰNG CHỨNG VLM (từ Bước 5)
question = test_case["question"]
choices = test_case["choices"]

# Gọi hàm tổng hợp
final_answer = lm_generate(
    llm=llm,
    retriever=retriever,
    reranker=reranker,
    vlm_description=vlm_description,
    question=question + choices,
)