from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch

def describe_frame_with_prompt(frame_np, prompt, vlm_processor, vlm_model):
    raw_image = Image.fromarray(frame_np.astype("uint8")).convert("RGB")

    # Qwen2-VL format theo official documentation
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": raw_image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    try:
        # Preparation for inference theo official way
        text = vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = vlm_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        with torch.no_grad():
            # Generation với parameters đơn giản hơn
            generated_ids = vlm_model.generate(
                **inputs, 
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                eos_token_id=vlm_processor.tokenizer.eos_token_id,
                pad_token_id=vlm_processor.tokenizer.pad_token_id or vlm_processor.tokenizer.eos_token_id
            )
            
            # Decode toàn bộ output rồi loại bỏ input
            full_output = vlm_processor.decode(generated_ids[0], skip_special_tokens=True)
            input_text = vlm_processor.decode(inputs["input_ids"][0], skip_special_tokens=True)
            
            # Loại bỏ phần input để lấy response
            if input_text in full_output:
                output_text = full_output.replace(input_text, "", 1).strip()
            else:
                # Fallback: decode chỉ phần response tokens
                input_token_len = inputs["input_ids"].shape[1]
                response_tokens = generated_ids[0][input_token_len:]
                output_text = vlm_processor.decode(response_tokens, skip_special_tokens=True)
            
            # Kiểm tra output có hợp lệ không
            if len(output_text.strip()) < 5 or output_text.strip() in ['<', '>', '<|', '|>', '<|im_start|>', '<|im_end|>']:
                return "Không thể tạo mô tả cho hình ảnh này."
                
        return output_text.strip()
        
    except Exception as e:
        print(f"❌ Error in describe_frame_with_prompt: {str(e)}")
        import traceback
        traceback.print_exc()
        return "Lỗi khi xử lý hình ảnh."
