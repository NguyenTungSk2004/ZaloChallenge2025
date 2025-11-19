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
            f"Đóng vai trò là 'Mắt thần' hỗ trợ lái xe an toàn. Nhiệm vụ của bạn là trích xuất thông tin thị giác từ video để trả lời câu hỏi: \"{question}\"\n\n"
            f"THÔNG TIN THAM KHẢO TỪ YOLO:\n"
            f"- Các vật thể đã phát hiện: {box_info if box_info else 'Không có đối tượng đặc biệt'}\n\n"
            f"YÊU CẦU MÔ TẢ CHI TIẾT (Ưu tiên độ chính xác thực tế):\n"
            f"1. BIỂN BÁO GIAO THÔNG: Tìm và đọc CHÍNH XÁC nội dung chữ/số trên biển báo liên quan đến câu hỏi. Mô tả hình dáng (tròn/vuông/tam giác) và màu sắc (xanh/đỏ/vàng).\n"
            f"2. LÀN ĐƯỜNG & VẠCH KẺ: Xác định số lượng làn, loại vạch (nét đứt/liền/vạch vàng/vạch trắng) và hướng mũi tên trên đường (nếu có).\n"
            f"3. NGỮ CẢNH: Vị trí xe đang đi (làn trái/phải/giữa) và hành vi các xe xung quanh.\n\n"
            f"LƯU Ý QUAN TRỌNG: Nếu thông tin YOLO mâu thuẫn với những gì bạn nhìn thấy rõ trong video, hãy tin vào video."
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
