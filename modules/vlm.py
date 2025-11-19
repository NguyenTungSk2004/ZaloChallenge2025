from qwen_vl_utils import process_vision_info

def generate_video_description(frames, models, box_info, question):
    """
    Predict answer using Qwen3-VL-2B model - Updated với official API
    """
    try:
        model = models['vlm']
        processor = models['vlm_processor']

        # Chuyển đổi frames nếu cần thiết và validate
        if not frames or len(frames) == 0:
            return "Không có frames để xử lý trong video."
            
        if hasattr(frames[0], 'shape'):  # numpy array
            from PIL import Image
            frames = [Image.fromarray(frame.astype("uint8")).convert("RGB") for frame in frames]
        
        # Đảm bảo có ít nhất 1 frame hợp lệ
        if len(frames) == 0:
            return "Không có frames hợp lệ để xử lý trong video."

        prompt_text = (
            f"Tôi cung cấp cho bạn một video giao thông và thông tin từ cảm biến YOLO: [{box_info}].\n"
            f"Nhiệm vụ: Hãy đóng vai trò là camera thông minh, quan sát và mô tả chi tiết khung cảnh trong video để làm bằng chứng trả lời cho câu hỏi: \"{question}\".\n\n"
            f"Hãy tập trung mô tả thật kỹ 3 yếu tố sau:\n"
            f"- Nội dung chữ và số trên các biển báo giao thông (đặc biệt là biển treo trên cao hoặc bên phải).\n"
            f"- Số lượng làn đường và loại vạch kẻ đường (nét đứt hay liền, vạch vàng hay trắng).\n"
            f"- Hành vi của các phương tiện xung quanh.\n\n"
            f"Lưu ý quan trọng: Chỉ mô tả khách quan những gì nhìn thấy. Tuyệt đối KHÔNG trả lời câu hỏi, KHÔNG chọn đáp án A/B/C/D."
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
        generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        print("vlm description: ",output_text)

        # Đảm bảo trả về string, không phải list
        if isinstance(output_text, list):
            return output_text[0].strip() if output_text else "Không có mô tả."
        else:
            return str(output_text).strip()
    except Exception as e:
        print(f"❌ Error in generate_video_description: {str(e)}")
        return f"Lỗi khi xử lý video: {str(e)}"
