from transformers import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch

def describe_frame_with_prompt(frame_np, prompt, vlm_processor: Qwen2VLProcessor, vlm_model):
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


def generate_video_description(frames, models, box_info):
    """
    Predict answer using Qwen3-VL-2B model - Updated với official API
    """
    try:
        model = models['vlm']
        processor = models['vlm_processor']

        # Chuyển đổi frames nếu cần thiết
        if frames and hasattr(frames[0], 'shape'):  # numpy array
            from PIL import Image
            frames = [Image.fromarray(frame.astype("uint8")).convert("RGB") for frame in frames]

        prompt_text = (
            f"Bạn là một chuyên gia phân tích giao thông phục vụ cho việc thi bằng lái xe.\n"
            f"Hệ thống nhận diện (YOLO) đã phát hiện các đối tượng sau trong video: [{box_info}].\n"
            f"Hãy quan sát video và mô tả thật chi tiết để trả lời câu hỏi trắc nghiệm:\n"
            f"- Mô tả chính xác các biển báo giao thông xuất hiện (hình dáng, màu sắc, nội dung chữ/số trên biển).\n"
            f"- Xác định số làn đường, loại vạch kẻ đường (nét đứt, nét liền) và vị trí xe đang đi.\n"
            f"- Quan sát các phương tiện xung quanh và tình huống giao thông.\n"
            f"Lưu ý: Kết hợp thông tin YOLO gợi ý với những gì bạn nhìn thấy để đảm bảo độ chính xác cao nhất."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames, 
                    },
                    {
                        "type": "text", 
                        "text": prompt_text
                    },
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        print("vlm description: ",output_text)

        return output_text[0].strip()
    except Exception as e:
        print(f"❌ Error in generate_video_description: {str(e)}")
        return f"Lỗi khi xử lý video: {str(e)}"
