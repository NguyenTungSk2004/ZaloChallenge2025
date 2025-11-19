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
            f"Bạn là tài xế đang trực tiếp điều khiển chiếc xe này (góc nhìn thứ nhất từ camera hành trình).\n\n"
            
            f"BỐI CẢNH (CONTEXT):\n"
            f"- Tôi đang cần bạn quan sát để giải quyết tình huống: \"{question}\"\n"
            f"- Hệ thống cảnh báo (YOLO) đang báo hiệu có: [{box_info}]\n\n"
            
            f"NHIỆM VỤ CỦA BẠN:\n"
            f"Hãy nhìn qua kính lái và mô tả lại thật chi tiết khung cảnh giao thông hiện tại như một biên bản hiện trường:\n"
            f"1. TRƯỚC MŨI XE & HƯỚNG DI CHUYỂN: Bạn đang đi ở làn nào? Vạch kẻ đường dưới bánh xe là nét liền hay đứt? Hướng mũi tên trên mặt đường chỉ đi đâu?\n"
            f"2. QUAN SÁT BIỂN BÁO: Đọc to và chính xác nội dung chữ/số trên các biển báo xuất hiện (đặc biệt là bên phải đường hoặc giá long môn phía trên).\n"
            f"3. CÁC PHƯƠNG TIỆN KHÁC: Có xe nào đang tạt đầu, xi nhan hay cản trở không?\n\n"
            
            f"⛔ QUY TẮC TUYỆT ĐỐI (NEGATIVE CONSTRAINTS):\n"
            f"- Chỉ mô tả những gì mắt thấy. KHÔNG được tự ý trả lời câu hỏi trắc nghiệm.\n"
            f"- KHÔNG đưa ra kết luận Đúng/Sai hay Có/Không.\n"
            f"- Bỏ qua các hình ảnh phản chiếu trên kính lái (nếu có).\n\n"
            
            f"Ví dụ mô tả chuẩn (Mẫu): 'Góc nhìn đêm. Trên giá long môn có 02 biển xanh: Biển trái ghi ĐẦU GIÂY LONG THÀNH (đi thẳng), biển phải ghi ĐƯỜNG ĐỖ XUÂN HỢP (rẽ phải). Mặt đường có vạch giảm tốc màu vàng và vạch phân làn nét đứt. Bên phải có xe 16 chỗ đang vượt.'\n"
            f"Báo cáo quan sát thực tế của bạn:"
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
