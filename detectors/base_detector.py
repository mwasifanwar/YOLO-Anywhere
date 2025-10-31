import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

class BaseDetector(ABC):
    def __init__(self, conf_threshold: float = 0.25, iou_threshold: float = 0.7, device: str = 'auto'):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = self._setup_device(device)
        self.model = None
        self.class_names = []
        
    def _setup_device(self, device):
        if device == 'auto':
            return 'cuda:0' if torch.cuda.is_available() else 'cpu'
        return device
    
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        pass
    
    @abstractmethod
    def postprocess(self, predictions: Any, original_shape: tuple) -> List[Dict]:
        pass
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        if self.model is None:
            self.load_model()
        
        input_tensor = self.preprocess(image)
        
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        detections = self.postprocess(predictions, image.shape[:2])
        return detections
    
    def update_classes(self, new_classes: List[str]):
        self.class_names = new_classes
        if hasattr(self, 'set_classes'):
            self.set_classes(new_classes)

class Detection:
    def __init__(self, bbox: List[float], confidence: float, class_id: int, class_name: str):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        
    def __repr__(self):
        return f"Detection({self.class_name}, conf={self.confidence:.2f}, bbox={self.bbox})"