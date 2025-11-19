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
                    "Bạn là AI thị giác máy tính (Vision Encoder) chuyên trách của hệ thống xe tự lái. "
                    "Nhiệm vụ của bạn là chuyển đổi dữ liệu hình ảnh từ camera hành trình thành văn bản mô tả khách quan (Scene Captioning).\n\n"
                    "QUY TẮC AN TOÀN (SAFETY PROTOCOLS):\n"
                    "1. TRUNG THỰC: Chỉ mô tả những gì nhìn thấy rõ ràng. Nếu không rõ, hãy báo 'Không rõ'.\n"
                    "2. KHÁCH QUAN: Không đưa ra ý kiến cá nhân, không dự đoán, không trả lời câu hỏi trắc nghiệm (A,B,C,D).\n"
                    "3. CHÍNH XÁC: Đọc chính xác từng ký tự chữ/số trên biển báo."
                    "4. CHI TIẾT: Cung cấp mô tả chi tiết về các yếu tố quan trọng liên quan đến giao thông.\n"
                    "5. TẬP TRUNG: Ưu tiên mô tả các biển báo giao thông và tín hiệu đèn đường quan trọng ảnh hưởng đến việc lái xe an toàn.\n"
                    "6. NGẮN GỌN: Mô tả ngắn gọn, súc tích trong giới hạn 500 từ.\n"
                    "7. KHÔNG LẶP LẠI PROMPT: Không bao giờ lặp lại hoặc nhắc lại nội dung của prompt trong mô tả.\n"
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
                            f"PHÂN TÍCH DỮ LIỆU THỊ GIÁC:\n"
                            f"- Mục tiêu tìm kiếm: Các manh mối liên quan đến câu hỏi \"{question}\".\n"
                            f"- Dữ liệu cảm biến (YOLO) gợi ý: [{box_info}].\n\n"

                            "YÊU CẦU: Hãy quét video và điền thông tin chi tiết vào BÁO CÁO HIỆN TRƯỜNG sau:\n\n"
                            
                            "1. [CẤU TRÚC ĐƯỜNG]:\n"
                            "- Số lượng làn xe? Xe chủ đang đi làn nào?\n"
                            "- Loại vạch kẻ đường (nét liền/đứt, màu sắc)?\n"
                            "- Mũi tên chỉ hướng trên mặt đường (nếu có)?\n\n"

                            "2. [HỆ THỐNG BIỂN BÁO] (Quan trọng nhất):\n"
                            "- Quét kỹ biển báo trên giá long môn (trên cao) và lề đường bên phải.\n"
                            "- TRÍCH XUẤT NGUYÊN VĂN chữ và số trên biển báo (VD: 'Cấm đi ngược chiều', '60', 'Đồng Nai').\n\n"

                            "3. [TÌNH HUỐNG GIAO THÔNG]:\n"
                            "- Thời điểm (Ngày/Đêm)?\n"
                            "- Hành vi của các phương tiện xung quanh (nếu ảnh hưởng đến xe chủ).\n\n"

                            "Bắt đầu báo cáo:"
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
