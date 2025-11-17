"""
OCR Module cho biển báo giao thông Việt Nam
Optimized cho RTX 3060 (12GB VRAM)
"""

import cv2
import numpy as np
import re
from typing import List, Tuple, Optional

class TrafficSignOCR:
    """
    OCR Reader cho biển báo giao thông
    Chỉ đọc text TRONG BIỂN BÁO (không đọc text nền)
    """
    
    def __init__(self, backend='easy', use_cuda=True):
        """
        Args:
            backend: 'easy' (recommended) hoặc 'paddle'
            use_cuda: Sử dụng GPU
        """
        self.backend = backend
        self.use_cuda = use_cuda
        
        if backend == 'easy':
            self._init_easy()
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _init_easy(self):
        """Khởi tạo EasyOCR"""
        try:
            import easyocr
            
            print("Đang load EasyOCR (English for numbers)...")
            # Chỉ dùng English để tập trung vào SỐ
            self.reader = easyocr.Reader(
                ['en'],  # Chỉ English (đủ cho số và mã biển)
                gpu=self.use_cuda,
                verbose=False,
                quantize=False  # RTX 3060 đủ VRAM, không cần quantize
            )
            print("✓ EasyOCR loaded!")
            
        except ImportError:
            raise ImportError("EasyOCR not installed. Run: pip install easyocr")
    
    def preprocess_sign_roi(self, roi: np.ndarray) -> List[np.ndarray]:
        """
        Tiền xử lý ROI biển báo - TẠO NHIỀU VARIANTS
        Returns: List các ảnh đã xử lý
        """
        variants = []
        
        if roi.size == 0:
            return variants
        
        # 1. Resize nếu QUÁ NHỎ (biển báo thường nhỏ trong frame)
        h, w = roi.shape[:2]
        if h < 64 or w < 64:
            scale = max(64/h, 64/w, 2.0)
            new_w = int(w * scale)
            new_h = int(h * scale)
            roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        # 2. Original resized gray
        variants.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        
        # 3. CLAHE (tăng contrast - tốt cho text mờ)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        variants.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))
        
        # 4. Binary threshold (tốt cho text đen trên nền trắng)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
        
        # 5. Inverted binary (tốt cho text trắng trên nền đen)
        inverted = cv2.bitwise_not(binary)
        variants.append(cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR))
        
        # 6. Resize 2x (cho text rất nhỏ)
        if h < 100 or w < 100:
            large = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            variants.append(cv2.cvtColor(large, cv2.COLOR_GRAY2BGR))
        
        return variants
    
    def extract_traffic_info(self, text: str) -> List[str]:
        """
        Trích xuất thông tin QUAN TRỌNG từ text
        Ưu tiên: SỐ > MÃ BIỂN > TEXT
        """
        results = []
        
        # 1. SỐ TỐC ĐỘ (30, 40, 50, 60, 70, 80, 100, 120)
        speed_numbers = re.findall(r'\b([3-9]0|100|120)\b', text)
        results.extend(speed_numbers)
        
        # 2. MÃ BIỂN (P.102, R.301, W.201, I.407, SA1, ...)
        sign_codes = re.findall(r'[PRWISA]\.?\d{1,3}', text, re.IGNORECASE)
        results.extend(sign_codes)
        
        # 3. KHOẢNG CÁCH (50m, 100m, 1km, 2km)
        distances = re.findall(r'\d+\s?(?:m|km)', text, re.IGNORECASE)
        results.extend(distances)
        
        # 4. CÁC SỐ KHÁC (nếu không có gì ở trên)
        if not results:
            other_numbers = re.findall(r'\b\d{1,3}\b', text)
            results.extend(other_numbers)
        
        return results
    
    def is_background_text(self, text: str) -> bool:
        """
        Kiểm tra xem có phải text NỀN (địa danh, ...) không
        """
        # Loại bỏ text dài (địa danh thường dài)
        if len(text) > 15:
            return True
        
        # Từ khóa địa danh phổ biến
        background_keywords = [
            'xa', 'huyen', 'tinh', 'thanh', 'pho',
            'duong', 'phuong', 'quan', 'long', 'tan',
            'vinh', 'hoa', 'tay', 'nam', 'bac', 'dong',
            'khu', 'vuc', 'city', 'district'
        ]
        
        text_lower = text.lower().replace(' ', '')
        for keyword in background_keywords:
            if keyword in text_lower:
                return True
        
        return False
    
    def read_text_from_roi(
        self, 
        frame: np.ndarray, 
        bbox: Tuple[int, int, int, int],
        conf_threshold: float = 0.3,  # Medium threshold cho RTX 3060
        padding: int = 5,
        debug: bool = False
    ) -> str:
        """
        Đọc text từ ROI biển báo - CHỈ CROP BBOX, KHÔNG LẤY TOÀN FRAME
        
        Args:
            frame: Frame gốc (BGR)
            bbox: (x1, y1, x2, y2) - BBOX CỦA BIỂN BÁO
            conf_threshold: Ngưỡng confidence
            padding: Padding xung quanh bbox
            debug: In debug info
        
        Returns:
            String thông tin quan trọng đã parse
        """
        x1, y1, x2, y2 = bbox
        
        # Expand bbox với padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        
        # QUAN TRỌNG: CHỈ CROP ROI BIỂN BÁO
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return ""
        
        if debug:
            print(f"    ROI shape: {roi.shape}")
            cv2.imwrite("debug_roi_input.jpg", roi)
        
        # Tạo các variants
        variants = self.preprocess_sign_roi(roi)
        
        if debug:
            for i, var in enumerate(variants):
                cv2.imwrite(f"debug_variant_{i}.jpg", var)
        
        # OCR từng variant và gom kết quả
        all_texts = {}  # text -> max_confidence
        
        for variant in variants:
            try:
                results = self._ocr_inference(variant, conf_threshold)
                
                for text, conf in results:
                    # Filter background text
                    if self.is_background_text(text):
                        if debug:
                            print(f"    Filtered background: '{text}'")
                        continue
                    
                    # Lưu confidence cao nhất
                    if text not in all_texts or conf > all_texts[text]:
                        all_texts[text] = conf
            
            except Exception as e:
                if debug:
                    print(f"    OCR error: {e}")
                continue
        
        if not all_texts:
            return ""
        
        # Parse thông tin quan trọng
        important_info = []
        for text in all_texts.keys():
            info = self.extract_traffic_info(text)
            important_info.extend(info)
        
        # Remove duplicates, keep order
        unique_info = list(dict.fromkeys(important_info))
        
        result = " ".join(unique_info) if unique_info else ""
        
        if debug:
            print(f"    All texts: {list(all_texts.keys())}")
            print(f"    Extracted info: {unique_info}")
            print(f"    Final: '{result}'")
        
        return result
    
    def _ocr_inference(self, image: np.ndarray, conf_threshold: float) -> List[Tuple[str, float]]:
        """
        Thực hiện OCR inference
        Returns: List of (text, confidence)
        """
        results = []
        
        try:
            if self.backend == 'easy':
                raw_results = self.reader.readtext(
                    image,
                    detail=1,
                    paragraph=False,
                    min_size=10,              # Medium size cho RTX 3060
                    text_threshold=0.3,       # Medium threshold
                    low_text=0.3,
                    link_threshold=0.3,
                    width_ths=0.5,
                    height_ths=0.5,
                    # Cho phép số + chữ cái
                    allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.-/: ',
                )
                
                for (bbox_coords, text, conf) in raw_results:
                    if conf >= conf_threshold:
                        text = text.strip()
                        if text:
                            results.append((text, conf))
            
            elif self.backend == 'paddle':
                raw_results = self.reader.ocr(image, cls=True)
                
                if raw_results and raw_results[0]:
                    for line in raw_results[0]:
                        if line and len(line) >= 2:
                            text, conf = line[1]
                            if conf >= conf_threshold:
                                text = text.strip()
                                if text:
                                    results.append((text, conf))
        
        except Exception as e:
            pass
        
        return results
    
    def clean_traffic_text(self, text: str) -> str:
        """Làm sạch text"""
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Fix common OCR errors
        replacements = {
            'O': '0', 'o': '0',  # O -> 0
            'I': '1', 'l': '1',  # I,l -> 1
            'S': '5', 's': '5',  # S -> 5 (đôi khi)
        }
        
        # Chỉ replace trong context số
        result = []
        for char in text:
            if char.isdigit() or char in replacements:
                result.append(replacements.get(char, char))
            else:
                result.append(char)
        
        return ''.join(result)


# ============================================================
# SIMPLE FUNCTION
# ============================================================
def extract_sign_text(
    ocr_reader: TrafficSignOCR,
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    debug: bool = False
) -> str:
    """
    Wrapper đơn giản
    
    Usage:
        ocr = TrafficSignOCR(backend='easy')
        text = extract_sign_text(ocr, frame, (x1, y1, x2, y2))
    """
    return ocr_reader.read_text_from_roi(
        frame, 
        bbox, 
        conf_threshold=0.3,
        padding=5,
        debug=debug
    )