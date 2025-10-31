import cv2
import numpy as np
from typing import List, Dict
import random

class DetectionVisualizer:
    def __init__(self):
        self.colors = self._generate_colors(100)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
        
    def _generate_colors(self, n: int) -> List[tuple]:
        random.seed(42)
        colors = []
        for i in range(n):
            colors.append((
                random.randint(0, 255),
                random.randint(0, 255), 
                random.randint(0, 255)
            ))
        return colors
    
    def draw_detections(self, image: np.ndarray, detections: List, class_names: List[str] = None) -> np.ndarray:
        result_image = image.copy()
        
        for detection in detections:
            bbox = detection.bbox
            confidence = detection.confidence
            class_id = detection.class_id
            class_name = getattr(detection, 'class_name', f'class_{class_id}')
            
            color = self.colors[class_id % len(self.colors)]
            
            x1, y1, x2, y2 = map(int, bbox)
            
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name}: {confidence:.2f}"
            if hasattr(detection, 'track_id'):
                label += f" ID: {detection.track_id}"
            
            label_size = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)[0]
            
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       self.font, self.font_scale, (255, 255, 255), self.thickness)
        
        return result_image
    
    def draw_detection_info(self, image: np.ndarray, detections: List, fps: float = None) -> np.ndarray:
        result_image = self.draw_detections(image, detections)
        
        info_text = f"Detections: {len(detections)}"
        if fps:
            info_text += f" | FPS: {fps:.1f}"
        
        cv2.putText(result_image, info_text, (10, result_image.shape[0] - 10),
                   self.font, 0.7, (0, 255, 0), 2)
        
        return result_image

class AdvancedVisualizer(DetectionVisualizer):
    def __init__(self):
        super().__init__()
        self.track_history = {}
    
    def draw_tracking_lines(self, image: np.ndarray, detections: List) -> np.ndarray:
        result_image = image.copy()
        
        for detection in detections:
            if hasattr(detection, 'track_id'):
                track_id = detection.track_id
                center_x = int((detection.bbox[0] + detection.bbox[2]) / 2)
                center_y = int((detection.bbox[1] + detection.bbox[3]) / 2)
                
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                
                self.track_history[track_id].append((center_x, center_y))
                
                if len(self.track_history[track_id]) > 30:
                    self.track_history[track_id].pop(0)
                
                color = self.colors[track_id % len(self.colors)]
                
                for i in range(1, len(self.track_history[track_id])):
                    cv2.line(result_image, 
                            self.track_history[track_id][i-1],
                            self.track_history[track_id][i],
                            color, 2)
        
        return result_image

class HeatmapVisualizer:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.heatmap = None
    
    def update_heatmap(self, detections: List, image_shape: tuple):
        if self.heatmap is None:
            self.heatmap = np.zeros(image_shape[:2], dtype=np.float32)
        
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.bbox)
            self.heatmap[y1:y2, x1:x2] += detection.confidence
    
    def draw_heatmap(self, image: np.ndarray) -> np.ndarray:
        if self.heatmap is None:
            return image
        
        heatmap_normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        
        result = cv2.addWeighted(image, 1 - self.alpha, heatmap_colored, self.alpha, 0)
        return result