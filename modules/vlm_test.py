import torch

def describe_frame_with_prompt(frame, prompt, processor, vlm_model):
    # xác định thiết bị của model
    device = next(vlm_model.parameters()).device

    # encode text + image
    inputs = processor(images=frame, text=prompt, return_tensors="pt")

    # Nếu model chạy trên CUDA → ép dtype float16
    # Nếu model chạy CPU → giữ float32 để tránh lỗi dtype mismatch
    if device.type == "cuda":
        inputs = {k: v.to(device, dtype=torch.float16) for k, v in inputs.items()}
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate caption
    with torch.no_grad():
        out = vlm_model.generate(
            **inputs,
            max_new_tokens=100
        )

    return processor.decode(out[0], skip_special_tokens=True)
