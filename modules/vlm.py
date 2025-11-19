from qwen_vl_utils import process_vision_info

def generate_video_description(frames, models, box_info, question):
    """
    Mô tả chi tiết từng yếu tố giao thông theo frame cụ thể
    """
    try:
        model = models['vlm']
        processor = models['vlm_processor']
        
        # Validate frames
        if not frames or len(frames) == 0:
            return "Không có frames để xử lý trong video."
            
        valid_frames = [frame for frame in frames if frame is not None]
        if not valid_frames:
            return "Không có frames hợp lệ để xử lý trong video."
            
        # Convert to PIL Images
        if hasattr(valid_frames[0], 'shape'):
            from PIL import Image
            frames = [Image.fromarray(frame.astype("uint8")).convert("RGB") 
                     for frame in valid_frames]
        else:
            frames = valid_frames
        
        if len(frames) == 0:
            return "Không có frames hợp lệ để xử lý trong video."
        
        # ✅ PROMPT CHI TIẾT - Yêu cầu mô tả cụ thể từng yếu tố
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {
                        "type": "text",
                        "text": (
                            "Bạn là tài xế đang lái xe ở việt nam, hãy mô tả video theo hướng dẫn: "
                            f"Phát hiện từ YOLO: {box_info}\n\n"
                            "Hãy mô tả chi tiết theo danh sách:\n"
                            "1. Đèn giao thông: Màu gì (đỏ/vàng/xanh), vị trí, thời điểm xuất hiện\n"
                            "2. Biển báo: Loại biển, nội dung (số km/h, hướng đi), frame nào thấy rõ\n"
                            "3. Vạch kẻ đường: Vạch liền/đứt, số làn, màu sắc\n"
                            "4. Biển trên giá long môn: Nội dung, hướng chỉ dẫn\n"
                            "5. Biển bên lề đường phải: Tên đường, địa danh, cảnh báo\n"
                            "6. Phương tiện: Loại xe, vị trí (trước/sau/bên), khoảng cách ước tính\n"
                            "7. Tầm nhìn: Thông thoáng/bị che, điều kiện thời tiết\n"
                            "8. Chướng ngại vật: Công trình, người đi bộ, vật cản\n\n"
                            "Chỉ nói những gì thấy rõ trong video, sử dụng tiếng việt, không suy diễn thêm."
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
        
        # ✅ Inference với parameters chống lặp + chi tiết hơn
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=384,      # Tăng lên để đủ chi tiết
            do_sample=True,
            temperature=0.4,         # Cân bằng giữa chi tiết và ổn định
            top_p=0.85,
            repetition_penalty=1.4,  # Chống lặp mạnh
            no_repeat_ngram_size=5,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id
        )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        print("VLM description:", output_text)
        
        result = output_text[0].strip() if isinstance(output_text, list) else str(output_text).strip()
        
        # ✅ Post-processing: Loại bỏ văn vẻ không cần thiết
        result = result.replace("Ví dụ về", "").replace("Để đảm bảo an toàn", "")
        result = result.replace("hãy nhớ rằng", "").replace("Ngoài ra", "")
        
        # Loại bỏ câu lặp
        sentences = [s.strip() for s in result.split('.') if s.strip()]
        unique_sentences = []
        seen = set()
        
        for sent in sentences:
            normalized = sent.lower().strip()
            # Chỉ giữ câu > 15 ký tự và chưa thấy
            if len(normalized) > 15 and normalized not in seen:
                unique_sentences.append(sent)
                seen.add(normalized)
        
        final_result = '. '.join(unique_sentences)
        if final_result and not final_result.endswith('.'):
            final_result += '.'
            
        return final_result
        
    except Exception as e:
        print(f"❌ Error in generate_video_description: {str(e)}")
        return f"Lỗi khi xử lý video: {str(e)}"