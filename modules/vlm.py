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
            f"Bạn là 'Mắt Thần' của xe tự lái. Nhiệm vụ là quan sát VIDEO THỰC TẾ ngay trước mắt để trích xuất dữ liệu.\n\n"
            
            f"BỐI CẢNH:\n"
            f"- Cần tìm thông tin cho câu hỏi: \"{question}\"\n"
            f"- Gợi ý từ cảm biến (YOLO): [{box_info}]\n\n"
            
            f"YÊU CẦU MÔ TẢ (Hãy điền thông tin video hiện tại vào các mục sau):\n"
            f"1. [BIỂN BÁO]: Quét toàn bộ khung hình. Đọc to nội dung chữ/số trên biển báo (nếu có). Chú ý biển trên giá long môn và bên phải đường.\n"
            f"2. [MẶT ĐƯỜNG]: Xe đang đi làn nào? Vạch kẻ là nét liền hay đứt? Có mũi tên chỉ hướng không?\n"
            f"3. [TÌNH HUỐNG]: Thời gian (Ngày/Đêm)? Có xe nào đang cản trở không?\n\n"
            
            f"LUẬT CẤM:\n"
            f"- KHÔNG ĐƯỢC TRẢ LỜI CÂU HỎI (Không chọn A,B,C,D).\n"
            f"- KHÔNG mô tả lại các ví dụ cũ. Hãy nhìn vào video hiện tại.\n"
            f"- Nếu không nhìn rõ biển báo, hãy nói 'Biển báo bị mờ'.\n\n"
            
            f"KẾT QUẢ QUAN SÁT THỰC TẾ:"
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
