import cv2
from data_types import FrameData
from data_types import BoxInfo
from utils.SaveFrame import save_frame
import random

class BestFrameTracker:
    def __init__(self):
        self.best_frames: dict[int, FrameData] = {}  # track_id: FrameData
        
    def update_track(self, frame, track_id, bbox, confidence, cls_name):
        """Cập nhật track với frame mới nếu chất lượng tốt hơn"""
            # Trích xuất vùng đối tượng
        x1, y1, x2, y2 = bbox
        object_region = frame[y1:y2, x1:x2]
        if object_region.size == 0:
            return None
        
        # Tính toán các metrics
        sharpness = self.calculate_sharpness(object_region)
        quality_score = self.calculate_quality_score(confidence, sharpness)
        
        frame = FrameData(
            id=random.randint(100000, 999999),
            frame=frame.copy(),
            score=quality_score,
            box_info=BoxInfo(
                bbox=bbox,
                confidence=confidence,
                class_name=cls_name,
                sharpness=sharpness
            )
        )
        save_frame(frameData=frame, track_id=track_id, output_path=f"extract_frames/{track_id}_{frame.id}.jpg")
        # Cập nhật frame tốt nhất cho track này
        if (track_id not in self.best_frames or quality_score > self.best_frames[track_id].score):
            self.best_frames[track_id] = frame
            return True
        
        return False
    
    def calculate_sharpness(self, image_region):
        """
        Tính độ rõ nét (sharpness) của vùng ảnh bằng Laplacian variance.

        Args:
            image_region: numpy array (BGR) - vùng bounding box của đối tượng.

        Returns:
            sharpness: float, giá trị variance của Laplacian. Giá trị càng cao -> vùng ảnh càng nét.
        """
        
        gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        return sharpness

    def calculate_quality_score(self, confidence, sharpness, max_sharpness=2000):
        """
        Tính điểm chất lượng tổng hợp cho một object/frame.
        Phổ dụng, dễ hiểu và áp dụng trong YOLO + tracking pipelines.
        """
        
        # Chuẩn hóa các yếu tố về 0-1
        sharp_norm = min(sharpness / max_sharpness, 1.0)   # 0-1
        conf_norm = min(max(confidence, 0.0), 1.0)         # 0-1

        # Weighted sum: confidence 60%, sharpness 40%
        quality_score = 0.6*conf_norm + 0.4*sharp_norm
        
        return quality_score