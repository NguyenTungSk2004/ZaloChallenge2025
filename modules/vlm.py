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
            
        # Lọc bỏ frames None và validate
        valid_frames = [frame for frame in frames if frame is not None]
        if not valid_frames:
            return "Không có frames hợp lệ để xử lý trong video."
            
        if hasattr(valid_frames[0], 'shape'):  # numpy array
            from PIL import Image
            frames = [Image.fromarray(frame.astype("uint8")).convert("RGB") for frame in valid_frames]
        else:
            frames = valid_frames
        
        # Đảm bảo có ít nhất 1 frame hợp lệ sau conversion
        if len(frames) == 0:
            return "Không có frames hợp lệ để xử lý trong video."
            
        messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là hệ thống phân tích video camera hành trình. "
                    "Nhiệm vụ: Mô tả chính xác và chi tiết khung cảnh giao thông để cung cấp context cho hệ thống AI phân tích.\n\n"
                    
                    "PHƯƠNG PHÁP MÔ TẢ:\n"
                    "1. VỊ TRÍ & LÀN ĐƯỜNG:\n"
                    "   - Xe đang ở làn nào (trái/giữa/phải)?\n"
                    "   - Loại vạch kẻ: liền (không được chuyển làn) hay đứt (được phép chuyển làn)?\n"
                    "   - Mũi tên chỉ hướng trên mặt đường (nếu có).\n\n"
                    
                    "2. BIỂN BÁO & CHỈ DẪN:\n"
                    "   - Tất cả biển báo nhìn thấy (màu sắc, hình dạng, vị trí).\n"
                    "   - Nội dung văn bản CHÍNH XÁC trên biển (tên đường, số km, giới hạn tốc độ).\n"
                    "   - Biển trên giá long môn, cột đèn tín hiệu, biển bên đường.\n\n"
                    
                    "3. PHƯƠNG TIỆN XUNG QUANH:\n"
                    "   - Vị trí tương đối: phía trước/sau/bên trái/bên phải.\n"
                    "   - Loại xe: ô tô con, xe tải, xe máy, xe buýt.\n"
                    "   - Hành vi: đi thẳng, chuyển làn, rẽ, dừng, tăng/giảm tốc.\n\n"
                    
                    "4. MÔI TRƯỜNG:\n"
                    "   - Thời tiết: nắng/mưa/sương mù.\n"
                    "   - Ánh sáng: ban ngày/hoàng hôn/ban đêm.\n"
                    "   - Loại đường: cao tốc/đường phố/ngã tư/vòng xuyến.\n\n"
                    
                    "NGUYÊN TẮC:\n"
                    "✓ Mô tả khách quan những gì nhìn thấy\n"
                    "✓ Ưu tiên thông tin quan trọng (biển báo, xe cản trở, tín hiệu đèn)\n"
                    "✓ Sử dụng ngôn ngữ rõ ràng, cụ thể\n"
                    "✗ KHÔNG suy luận hành động tiếp theo\n"
                    "✗ KHÔNG đưa ra đánh giá đúng/sai\n"
                    "✗ KHÔNG trả lời câu hỏi trực tiếp\n\n"
                    
                    "Ví dụ:\n"
                    "\"Xe đang ở làn giữa của đường 3 làn. Vạch kẻ bên trái là nét đứt màu trắng. "
                    "Phía trước 20m có biển báo giới hạn tốc độ 60km/h. Giá long môn phía trước có "
                    "2 biển xanh: trái ghi 'QL1A HÀ NỘI', phải ghi 'VÀNH ĐAI 3'. Bên phải có xe tải "
                    "đang đi song song. Trời nắng, tầm nhìn tốt.\""
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames
                    },
                    {
                        "type": "text",
                        "text": (
                            f"THÔNG TIN THAM KHẢO:\n"
                            f"- Câu hỏi cần phân tích: \"{question}\"\n"
                            f"- Đối tượng phát hiện bởi YOLO: {box_info}\n\n"
                            
                            "Hãy mô tả chi tiết khung cảnh giao thông trong video:"
                        )
                    }
                ]
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
        generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False, temperature=0.7)
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
