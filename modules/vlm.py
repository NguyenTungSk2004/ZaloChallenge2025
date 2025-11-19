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
            f"Bạn là hệ thống thị giác máy tính (Computer Vision). Nhiệm vụ của bạn là trích xuất thông tin thị giác thô (Raw Visual Data).\n\n"
            f"CONTEXT:\n"
            f"- Người dùng sắp hỏi câu hỏi: \"{question}\"\n"
            f"- Gợi ý từ YOLO: [{box_info}]\n\n"
            f"NHIỆM VỤ BẮT BUỘC:\n"
            f"1. QUAN SÁT video thật kỹ.\n"
            f"2. LIỆT KÊ các bằng chứng thị giác liên quan đến câu hỏi trên.\n"
            f"3. MÔ TẢ CHI TIẾT các biển báo (nội dung chữ, số, hình vẽ), vạch kẻ đường và tình huống xe.\n\n"
            f"QUY TẮC CẤM (NEGATIVE CONSTRAINTS):\n"
            f"- TUYỆT ĐỐI KHÔNG trả lời câu hỏi (Không chọn A, B, C, D).\n"
            f"- TUYỆT ĐỐI KHÔNG trả lời Có/Không.\n"
            f"- Chỉ đưa ra mô tả khách quan về hình ảnh.\n\n"
            f"Ví dụ output đúng: 'Trong video có biển báo tròn viền đỏ, nền trắng, con số 60 màu đen. Xe đang đi làn giữa, vạch kẻ đứt.'\n"
            f"Bắt đầu mô tả:"
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
