from qwen_vl_utils import process_vision_info
from PIL import Image 

def generate_video_description(frames, models, box_info, question):
    """
    Predict answer using Qwen3-VL-2B model - English response, Vietnamese sign text.
    """
    try:
        model = models['vlm']
        processor = models['vlm_processor']
        
        # ... (Frame validation and conversion code remains the same) ...
        
        # Validate frames
        if not frames or len(frames) == 0:
            return "No frames available for processing in the video."
            
        valid_frames = [frame for frame in frames if frame is not None]
        if not valid_frames:
            return "No valid frames available for processing in the video."
            
        # Convert to PIL Images
        if hasattr(valid_frames[0], 'shape'):
            frames = [Image.fromarray(frame.astype("uint8")).convert("RGB") 
                     for frame in valid_frames]
        else:
            frames = valid_frames
            
        if len(frames) == 0:
            return "No valid frames available for processing in the video."
            
        # SỬA ĐỔI: Prompt hoàn toàn bằng Tiếng Anh, yêu cầu nội dung biển báo là Tiếng Việt.
        instruction_text_en = (
            "You are a smart driver's assistant specializing in describing traffic situations in Vietnam based on the video provided. "
            f"**YOLO detection data**: {box_info}\n"
            "Focus on checking the existence and status of the following objects: **traffic signs** (read the text/numbers on the signs and **MUST REPORT THEM IN VIETNAMESE**), **road markings**, **lanes**, **traffic lights**, motorcycles, cars, trucks, buses, pedestrians, traffic police, obstacles, construction, weather, and visibility.\n"
            "**Strictly describe only what is seen in the video, do not speculate or add external information.**\n"
            "Analyze the scene and provide the final description **ENTIRELY IN ENGLISH**, but ensure the sign text you read is in **VIETNAMESE**." # Yêu cầu trả lời bằng tiếng Anh, nhưng nội dung biển báo là tiếng Việt
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {
                        "type": "text",
                        # Chèn placeholder <video> vào đầu hướng dẫn text
                        "text": f"<video>{instruction_text_en}" 
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
        
        # ✅ Inference with anti-repetition parameters (Unchanged)
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=256, 
            do_sample=True, 
            temperature=0.3, 
            top_p=0.9,
            repetition_penalty=1.3, 
            no_repeat_ngram_size=4, 
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
        
        # ✅ Post-processing: Loại bỏ đoạn lặp nếu còn sót (Unchanged)
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
        return f"Error processing video: {str(e)}"