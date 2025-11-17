import cv2
from data_types import FrameData
from data_types import BoxInfo

class BestFrameTracker:
    def __init__(self):
        self.best_frames: dict[int, FrameData] = {}  # track_id: FrameData
    

    def clear_frames(self):
        self.best_frames = {}
        
    def update_track(self, frame, track_id, bbox, confidence, cls_name):
        """Cập nhật track với frame mới nếu chất lượng tốt hơn"""
            # Trích xuất vùng đối tượng
        x1, y1, x2, y2 = bbox
        object_region = frame[y1:y2, x1:x2]
        if object_region.size == 0:
            return None
        
        # Tính toán các metrics
        bbox_area = (x2 - x1) * (y2 - y1)
        sharpness = self.calculate_sharpness(object_region)
        quality_score = self.calculate_quality_score(bbox_area, confidence, sharpness)
        
        # Cập nhật frame tốt nhất cho track này
        if (track_id not in self.best_frames or quality_score > self.best_frames[track_id].score) and quality_score >= 0.4:
            self.best_frames[track_id] = FrameData(
                frame=frame.copy(),
                score=quality_score,
                box_info=BoxInfo(
                    bbox=bbox,
                    confidence=confidence,
                    class_name=cls_name,
                    sharpness=sharpness,
                    area=bbox_area
                )
            )
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

    def calculate_quality_score(self, bbox_area, confidence, sharpness,
                                max_area=250000, max_sharpness=2000):
        """
        Tính điểm chất lượng tổng hợp cho một object/frame.
        Phổ dụng, dễ hiểu và áp dụng trong YOLO + tracking pipelines.
        """
        
        # Chuẩn hóa các yếu tố về 0-1
        area_norm = min(bbox_area / max_area, 1.0)         # 0-1
        sharp_norm = min(sharpness / max_sharpness, 1.0)   # 0-1
        conf_norm = min(max(confidence, 0.0), 1.0)         # 0-1

        # Weighted sum: confidence 40%, sharpness 40%, area 20%
        quality_score = 0.4*conf_norm + 0.4*sharp_norm + 0.2*area_norm
        
        return quality_score