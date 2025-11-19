from qwen_vl_utils import process_vision_info

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
        generated_ids = model.generate(**inputs)
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
