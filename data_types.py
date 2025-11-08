from dataclasses import dataclass

@dataclass
class BoxInfo:
    bbox: tuple
    confidence: float
    class_name: str
    sharpness: float
    area: float

@dataclass
class FrameData:
    frame: any
    score: float
    box_info: BoxInfo