import torch
import numpy as np
from ultralytics import YOLOWorld
from .base_detector import BaseDetector, Detection
from typing import List, Dict
import cv2

class YOLOWorldDetector(BaseDetector):
    def __init__(self, model_size: str = 'yolo_world_l', classes: List[str] = None, 
                 conf_threshold: float = 0.25, iou_threshold: float = 0.7, device: str = 'auto'):
        super().__init__(conf_threshold, iou_threshold, device)
        self.model_size = model_size
        self.class_names = classes or ['object']
        self.model = None
        self.load_model()
    
    def load_model(self):
        print(f"Loading YOLO-World {self.model_size} model...")
        
        model_map = {
            'yolo_world_s': 'yolov8s-world.pt',
            'yolo_world_m': 'yolov8m-world.pt', 
            'yolo_world_l': 'yolov8l-world.pt',
            'yolo_world_x': 'yolov8x-world.pt'
        }
        
        model_name = model_map.get(self.model_size, 'yolov8l-world.pt')
        
        try:
            self.model = YOLOWorld(model_name)
            self.model.set_classes(self.class_names)
            self.model.to(self.device)
            self.model.conf = self.conf_threshold
            self.model.iou = self.iou_threshold
            print(f"Model loaded successfully with {len(self.class_names)} classes")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def set_classes(self, classes: List[str]):
        self.class_names = classes
        if self.model is not None:
            self.model.set_classes(classes)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return image
    
    def postprocess(self, predictions, original_shape: tuple) -> List[Detection]:
        detections = []
        
        if hasattr(predictions, 'boxes') and predictions.boxes is not None:
            boxes = predictions.boxes.xyxy.cpu().numpy()
            confidences = predictions.boxes.conf.cpu().numpy()
            class_ids = predictions.boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes)):
                bbox = boxes[i].tolist()
                confidence = float(confidences[i])
                class_id = int(class_ids[i])
                
                if class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                    detection = Detection(bbox, confidence, class_id, class_name)
                    detections.append(detection)
        
        return detections
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        if self.model is None:
            self.load_model()
        
        results = self.model(image)
        detections = self.postprocess(results[0], image.shape[:2])
        return detections
    
    def get_available_models(self) -> List[str]:
        return ['yolo_world_s', 'yolo_world_m', 'yolo_world_l', 'yolo_world_x']

class AdvancedYOLOWorldDetector(YOLOWorldDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.track_history = {}
    
    def detect_with_tracking(self, image: np.ndarray, track: bool = True) -> List[Detection]:
        if self.model is None:
            self.load_model()
        
        if track:
            results = self.model.track(image, persist=True)
        else:
            results = self.model(image)
        
        detections = self.postprocess(results[0], image.shape[:2])
        
        if track and hasattr(results[0], 'boxes') and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            for i, track_id in enumerate(track_ids):
                if i < len(detections):
                    detections[i].track_id = track_id
        
        return detections