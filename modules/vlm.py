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

        messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là tài xế đang trực tiếp điều khiển chiếc xe này từ góc nhìn camera hành trình. "
                    "Nhiệm vụ duy nhất của bạn: mô tả CHÍNH XÁC những gì đang xuất hiện trong video, "
                    "không suy luận, không dự đoán, không trả lời thay cho người dùng."
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
                            "BỐI CẢNH (CONTEXT):\n"
                            f"- Tình huống tôi đang cần quan sát: \"{question}\".\n"
                            f"- Hệ thống YOLO phát hiện các đối tượng: [{box_info}].\n\n"

                            "NHIỆM VỤ MÔ TẢ (TASK):\n"
                            "Hãy mô tả lại KHUNG CẢNH GIAO THÔNG y như một biên bản hiện trường, theo 3 mục sau:\n"
                            "1) TRƯỚC MŨI XE & HƯỚNG DI CHUYỂN:\n"
                            "- Xe tôi đang ở làn nào?\n"
                            "- Vạch kẻ đường dưới bánh xe là nét liền hay nét đứt?\n"
                            "- Có mũi tên chỉ hướng nào trên mặt đường (nếu thấy)?\n\n"

                            "2) BIỂN BÁO, KÝ HIỆU & CHỮ/SỐ:\n"
                            "- Liệt kê TẤT CẢ biển báo nhìn thấy.\n"
                            "- Đọc to và chính xác chữ/số trên biển báo.\n"
                            "- Ưu tiên biển trên giá long môn, bên phải đường, hoặc biển giới hạn tốc độ.\n\n"

                            "3) CÁC PHƯƠNG TIỆN KHÁC:\n"
                            "- Xe phía trước/2 bên đang đi thế nào?\n"
                            "- Có xe tạt đầu, xi nhan, sang làn, phanh gấp, hoặc cản trở không?\n\n"

                            "QUY TẮC BẮT BUỘC (NEGATIVE CONSTRAINTS):\n"
                            "- Chỉ mô tả những gì nhìn thấy. KHÔNG được tự suy luận.\n"
                            "- KHÔNG được trả lời câu hỏi trắc nghiệm.\n"
                            "- KHÔNG đưa ra nhận xét đúng/sai, nên/không nên, dự đoán tương lai.\n"
                            "- KHÔNG mô tả các hình phản chiếu trên kính lái.\n"
                            "- KHÔNG thêm thông tin không nhìn thấy rõ.\n\n"

                            "MẪU MÔ TẢ CHUẨN (EXAMPLE):\n"
                            "“Trời tối. Trên giá long môn có 2 biển xanh: biển trái ghi ĐẦU GIÂY LONG THÀNH (đi thẳng), "
                            "biển phải ghi ĐƯỜNG ĐỖ XUÂN HỢP (rẽ phải). Mặt đường có vạch giảm tốc màu vàng và "
                            "vạch phân làn nét đứt. Bên phải có xe 16 chỗ đang vượt.”\n\n"

                            "BÁO CÁO QUAN SÁT THỰC TẾ CỦA BẠN:"
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
        generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.7)
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
