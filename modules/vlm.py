from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

def describe_frame_with_prompt(frame_np, prompt, vlm_processor, vlm_model):
    raw_image = Image.fromarray(frame_np.astype("uint8"))

    inputs = vlm_processor(
        images=raw_image,
        text=prompt,
        return_tensors="pt"
    )

    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        # Thêm max_new_tokens để đảm bảo câu trả lời không bị cắt ngắn
        out = vlm_model.generate(**inputs, max_new_tokens=100) # Có thể điều chỉnh giá trị này

    caption = vlm_processor.decode(out[0], skip_special_tokens=True)
    
    # Loại bỏ tiền tố 'Question: ... Answer:' nếu có
    if "Answer: " in caption:
        caption = caption.split("Answer: ", 1)[1].strip()
    elif "question:" in caption.lower() and "answer:" in caption.lower():
        # Xử lý trường hợp chữ hoa/thường khác nhau
        answer_start_index = caption.lower().find("answer:")
        if answer_start_index != -1:
            caption = caption[answer_start_index + len("answer:"):].strip()

    return caption