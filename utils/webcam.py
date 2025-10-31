import cv2
import time
from pathlib import Path
from typing import Union, Iterator
import numpy as np

class WebcamStream:
    def __init__(self, source: Union[str, int] = 0, output_path: str = None, 
                 frame_width: int = 1280, frame_height: int = 720, fps: int = 30):
        self.source = source
        self.output_path = output_path
        self.cap = None
        self.output_writer = None
        self.is_file = False
        self.stopped = False
        self.frame_count = 0
        
        self._initialize_stream()
        
        if output_path:
            self._initialize_output_writer(frame_width, frame_height, fps)
    
    def _initialize_stream(self):
        if isinstance(self.source, int) or self.source.isdigit():
            self.source = int(self.source)
            self.cap = cv2.VideoCapture(self.source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            source_path = Path(self.source)
            if source_path.exists():
                self.cap = cv2.VideoCapture(str(source_path))
                self.is_file = True
            else:
                raise FileNotFoundError(f"Source file {self.source} not found")
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source {self.source}")
    
    def _initialize_output_writer(self, width: int, height: int, fps: int):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_writer = cv2.VideoWriter(
            self.output_path, fourcc, fps, (width, height)
        )
    
    def __iter__(self) -> Iterator[np.ndarray]:
        return self
    
    def __next__(self) -> np.ndarray:
        if self.stopped:
            raise StopIteration
        
        ret, frame = self.cap.read()
        
        if not ret:
            if self.is_file:
                print("End of video file reached")
            else:
                print("Failed to capture frame from camera")
            self.stopped = True
            raise StopIteration
        
        self.frame_count += 1
        return frame
    
    def get_frame_size(self) -> tuple:
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)
    
    def get_fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS)
    
    def get_frame_count(self) -> int:
        if self.is_file:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return self.frame_count
    
    def release(self):
        if self.cap:
            self.cap.release()
        if self.output_writer:
            self.output_writer.release()
        cv2.destroyAllWindows()

class ImageProcessor:
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image from {image_path}")
        return image
    
    @staticmethod
    def save_image(image: np.ndarray, output_path: str):
        cv2.imwrite(output_path, image)
        print(f"Image saved to {output_path}")
    
    @staticmethod
    def resize_image(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        return image

class VideoProcessor:
    @staticmethod
    def get_video_info(video_path: str) -> dict:
        cap = cv2.VideoCapture(video_path)
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        }
        cap.release()
        return info