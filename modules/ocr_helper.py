"""
OCR Module cho biển báo giao thông Việt Nam
Hỗ trợ: PaddleOCR (recommend) và EasyOCR
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

class TrafficSignOCR:
    """
    OCR Reader cho biển báo giao thông
    Tối ưu cho text tiếng Việt + số
    """
    
    def __init__(self, backend='easy', use_gpu=True):
        """
        Args:
            backend: 'paddle' (recommended) hoặc 'easy'
            use_gpu: Sử dụng GPU hay không
        """
        self.backend = backend
        self.use_gpu = use_gpu
        
        if backend == 'paddle':
            self._init_paddle()
        elif backend == 'easy':
            self._init_easy()
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _init_paddle(self):
        """
        Khởi tạo PaddleOCR
        pip install paddlepaddle-gpu paddleocr
        hoặc: pip install paddlepaddle paddleocr (CPU only)
        """
        try:
            from paddleocr import PaddleOCR
            
            print("Đang load PaddleOCR (Vietnamese)...")
            self.reader = PaddleOCR(
                use_angle_cls=True,  # Xoay text nếu cần
                lang='vi',           # Tiếng Việt
                use_gpu=self.use_gpu,
                show_log=False,
                det_db_thresh=0.3,   # Threshold detection thấp hơn cho biển nhỏ
                det_db_box_thresh=0.5,
                rec_batch_num=6      # Batch size cho recognition
            )
            print("✓ PaddleOCR loaded!")
            
        except ImportError:
            raise ImportError(
                "PaddleOCR not installed. Run:\n"
                "pip install paddlepaddle-gpu paddleocr  # GPU\n"
                "pip install paddlepaddle paddleocr      # CPU"
            )
    
    def _init_easy(self):
        """
        Khởi tạo EasyOCR (fallback)
        pip install easyocr
        """
        try:
            import easyocr
            
            print("Đang load EasyOCR (Vietnamese + English)...")
            self.reader = easyocr.Reader(
                ['vi', 'en'], 
                gpu=self.use_gpu,
                verbose=False
            )
            print("✓ EasyOCR loaded!")
            
        except ImportError:
            raise ImportError("EasyOCR not installed. Run: pip install easyocr")
    
    def preprocess_roi(self, roi: np.ndarray, enhance: bool = True) -> np.ndarray:
        """
        Tiền xử lý ROI để OCR tốt hơn
        
        Args:
            roi: Region of interest (BGR format)
            enhance: Có tăng cường chất lượng không
        
        Returns:
            Ảnh đã xử lý
        """
        if roi.size == 0:
            return roi
        
        # 1. Resize nếu quá nhỏ (OCR cần ít nhất 32px)
        h, w = roi.shape[:2]
        if h < 32 or w < 32:
            scale = max(32/h, 32/w, 2.0)
            new_w = int(w * scale)
            new_h = int(h * scale)
            roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        if not enhance:
            return roi
        
        # 2. Chuyển sang grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        # 3. Tăng độ tương phản (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 4. Adaptive thresholding (tốt cho text với nền không đều)
        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # 5. Denoise
        denoised = cv2.fastNlMeansDenoising(binary, h=10)
        
        # Convert back to BGR for PaddleOCR
        if len(roi.shape) == 3:
            return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        
        return denoised
    
    def read_text_from_roi(
        self, 
        frame: np.ndarray, 
        bbox: Tuple[int, int, int, int],
        conf_threshold: float = 0.5,
        enhance: bool = True,
        padding: int = 5
    ) -> str:
        """
        Đọc text từ bounding box
        
        Args:
            frame: Frame gốc (BGR)
            bbox: (x1, y1, x2, y2)
            conf_threshold: Ngưỡng confidence
            enhance: Có tăng cường ảnh không
            padding: Padding thêm xung quanh bbox
        
        Returns:
            String các text được phát hiện, cách nhau bởi dấu cách
        """
        x1, y1, x2, y2 = bbox
        
        # Expand bbox với padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        
        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return ""
        
        # Preprocess
        if enhance:
            roi_processed = self.preprocess_roi(roi, enhance=True)
        else:
            roi_processed = roi
        
        # OCR
        texts = self._ocr_inference(roi_processed, conf_threshold)
        
        return " ".join(texts)
    
    def _ocr_inference(self, image: np.ndarray, conf_threshold: float) -> List[str]:
        """
        Thực hiện OCR inference
        
        Returns:
            List các text được phát hiện
        """
        texts = []
        
        try:
            if self.backend == 'paddle':
                results = self.reader.ocr(image, cls=True)
                
                # PaddleOCR format: [[[bbox], (text, conf)], ...]
                if results and results[0]:
                    for line in results[0]:
                        if line and len(line) >= 2:
                            text, conf = line[1]
                            if conf >= conf_threshold:
                                text = text.strip()
                                if text:
                                    texts.append(text)
            
            elif self.backend == 'easy':
                results = self.reader.readtext(image)
                
                # EasyOCR format: [(bbox, text, conf), ...]
                for (bbox, text, conf) in results:
                    if conf >= conf_threshold:
                        text = text.strip()
                        if text:
                            texts.append(text)
        
        except Exception as e:
            print(f"    [OCR Error] {e}")
        
        return texts
    
    def clean_traffic_text(self, text: str) -> str:
        """
        Làm sạch text từ biển báo giao thông
        
        - Loại bỏ ký tự lạ
        - Chuẩn hóa số
        - Giữ lại chữ Việt + số + ký tự đặc biệt phổ biến
        """
        # Common patterns trong biển báo Việt Nam
        # Ví dụ: "50", "P.102", "R.301", "W.201", "KM/H"
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # TODO: Thêm logic clean nếu cần
        # Ví dụ: chuyển "5O" -> "50" (O thành 0)
        
        return text


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================
def extract_text_with_ocr(
    ocr_reader: TrafficSignOCR,
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    conf_threshold: float = 0.5
) -> str:
    """
    Wrapper function để dễ sử dụng
    
    Usage:
        ocr_reader = TrafficSignOCR(backend='paddle')
        text = extract_text_with_ocr(ocr_reader, frame, (x1,y1,x2,y2))
    """
    return ocr_reader.read_text_from_roi(
        frame, 
        bbox, 
        conf_threshold=conf_threshold,
        enhance=True,
        padding=5
    )