from qwen_vl_utils import process_vision_info

def generate_video_description(frames, models, box_info, question):
    """
    Predict answer using Qwen3-VL-2B model - Fixed repetition issue
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
            
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {
                        "type": "text",
                        "text": (
                            f"YOLO phát hiện: {box_info}\n"
                            "Tập trung: biển báo (đọc chữ/số), vạch đường, làn xe, đèn giao thông, "
                            "xe máy, ô tô, xe tải, xe buýt, người đi bộ, cảnh sát giao thông, "
                            "vật cản, công trình, tình trạng thời tiết và tầm nhìn.\n"
                            "Chỉ mô tả thấy gì trong video, không suy đoán thêm.\n"
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
        
        # ✅ FIXED: Inference với anti-repetition parameters
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=256,  # Giảm xuống để tránh lặp dài
            do_sample=True,      # Bật sampling
            temperature=0.3,     # Giảm temperature cho stable hơn
            top_p=0.9,
            repetition_penalty=1.3,  # Tăng cao để tránh lặp
            no_repeat_ngram_size=4,  # Không lặp cụm 4 từ
            eos_token_id=processor.tokenizer.eos_token_id,  # Dừng sớm nếu có EOS
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
        
        # ✅ Post-processing: Loại bỏ đoạn lặp nếu còn sót
        sentences = result.split('. ')
        unique_sentences = []
        seen = set()
        for sent in sentences:
            if sent.lower() not in seen and len(sent) > 10:
                unique_sentences.append(sent)
                seen.add(sent.lower())
        
        return '. '.join(unique_sentences) if unique_sentences else result
        
    except Exception as e:
        print(f"❌ Error in generate_video_description: {str(e)}")
        return f"Lỗi khi xử lý video: {str(e)}"
