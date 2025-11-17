"""
OCR Module cho biển báo giao thông Việt Nam
Tối ưu theo CONFIG OCR tốt nhất (vi+en, min_size=10, text_threshold=0.3)
"""

import cv2
import numpy as np
import re
from typing import List, Tuple, Optional

class TrafficSignOCR:
    """
    OCR Reader cho biển báo giao thông
    Tối ưu cho text tiếng Việt + số
    """
    
    def __init__(self, backend='easy', use_cuda=True):
        self.backend = backend
        self.use_cuda = use_cuda
        
        if backend == 'easy':
            self._init_easy()
    
    def _init_easy(self):
        """
        Khởi tạo EasyOCR với CONFIG tối ưu hiện tại
        """
        try:
            import easyocr
            
            print("Đang load EasyOCR (Vietnamese + English)...")
            self.reader = easyocr.Reader(
                ['vi', 'en'],
                gpu=self.use_cuda,
                verbose=False,
                download_enabled=True
            )
            print("✓ EasyOCR loaded (vi+en)!")
        
        except ImportError:
            raise ImportError("EasyOCR not installed. Run: pip install easyocr")
    
    def preprocess_roi(self, roi: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý ROI theo hướng đơn giản (tránh gây nhiễu)
        """
        if roi.size == 0:
            return roi
        
        h, w = roi.shape[:2]
        if h < 32 or w < 32:
            scale = max(32/h, 32/w, 1.5)
            roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        return roi
    
    def extract_traffic_numbers(self, text: str) -> List[str]:
        numbers = []
        single_numbers = re.findall(r'\b(\d{2,3})\b', text)
        numbers.extend(single_numbers)

        sign_codes = re.findall(r'[PRWI]\.?\d{3}', text, re.IGNORECASE)
        numbers.extend(sign_codes)

        distances = re.findall(r'\d+\s?(?:m|km)', text, re.IGNORECASE)
        numbers.extend(distances)

        return numbers
    
    def parse_ocr_results(self, raw_texts: List[str]) -> str:
        if not raw_texts:
            return ""
        
        filtered = [t for t in raw_texts if len(t) >= 1]
        if not filtered:
            return ""
        
        all_numbers = []
        for text in filtered:
            nums = self.extract_traffic_numbers(text)
            all_numbers.extend(nums)

        if all_numbers:
            unique_numbers = list(dict.fromkeys(all_numbers))
            return " ".join(unique_numbers)

        stopwords = ['the', 'a', 'an', 'and', 'or', 'but']
        cleaned = []
        for text in filtered:
            words = text.lower().split()
            words = [w for w in words if w not in stopwords]
            if words:
                cleaned.append(" ".join(words))
        
        result = " ".join(cleaned)
        return result[:50]
    
    def read_text_from_roi(
        self, 
        frame: np.ndarray, 
        bbox: Tuple[int, int, int, int],
        conf_threshold: float = 0.3,   # CHỈNH THEO CONFIG TỐT NHẤT
        padding: int = 3
    ) -> str:
        
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return ""
        
        roi_processed = self.preprocess_roi(roi)
        raw_texts = self._ocr_inference(roi_processed, conf_threshold)
        parsed_text = self.parse_ocr_results(raw_texts)
        
        return parsed_text
    
    def _ocr_inference(self, image: np.ndarray, conf_threshold: float) -> List[str]:
        texts = []
        
        try:
            if self.backend == 'easy':
                # CONFIG TỐT NHẤT THEO KẾT QUẢ TEST
                results = self.reader.readtext(
                    image,
                    paragraph=False,
                    min_size=10,
                    text_threshold=0.3,
                    low_text=0.3
                )
                
                for (bbox_coords, text, conf) in results:
                    if conf >= conf_threshold:
                        text = text.strip()
                        if text:
                            texts.append(text)
        
        except Exception as e:
            print(f"[OCR Error] {e}")
        
        return texts
    
    def clean_traffic_text(self, text: str) -> str:
        text = " ".join(text.split())
        text = text.replace('O', '0').replace('o', '0')
        text = text.replace('I', '1').replace('l', '1')
        text = text.replace('S', '5')
        return text


# ============================================================
# SIMPLIFIED FUNCTION
# ============================================================
def extract_text_simple(
    ocr_reader: TrafficSignOCR,
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int]
) -> str:
    return ocr_reader.read_text_from_roi(frame, bbox, conf_threshold=0.3)
