from modules.ocr_helper import TrafficSignOCR
import cv2

ocr = TrafficSignOCR(backend='easy', use_cuda=True)
frame = cv2.imread(r'E:\Project\images.png')

text = ocr.reader.readtext(frame, detail=0)   # Bỏ qua ROI để test cơ bản
print(text)
