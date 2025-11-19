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
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames
                    },
                    {
                        "type": "text",
                        "text": (
                            "Bạn là tài xế lái xe.\n"
                            f"Dữ liệu từ cảm biến YOLO cho biết:\n{box_info}\n\n"
                            f"Nhiệm vụ: Mô tả bối cảnh xe, tình huống hiện tại của xe, các biển báo, vạch kẻ đường và phương tiện xung quanh để cho luật sư trả lời câu hỏi: {question}\n"
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
