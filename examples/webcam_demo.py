import cv2
import time
from detectors.yolo_world import YOLOWorldDetector
from utils.visualization import AdvancedVisualizer
from utils.webcam import WebcamStream

def webcam_demo():
    print("YOLO-Anywhere Webcam Demo")
    print("Available models: yolo_world_s, yolo_world_m, yolo_world_l, yolo_world_x")
    
    model_size = input("Enter model size (default: yolo_world_l): ") or "yolo_world_l"
    custom_classes = input("Enter custom classes (comma-separated, default: person,car,dog,cat): ")
    
    if custom_classes:
        classes = [cls.strip() for cls in custom_classes.split(',')]
    else:
        classes = ['person', 'car', 'dog', 'cat']
    
    detector = YOLOWorldDetector(model_size=model_size, classes=classes)
    visualizer = AdvancedVisualizer()
    stream = WebcamStream(source=0)
    
    print("Starting webcam detection...")
    print("Press 'q' to quit, 'c' to change classes")
    
    try:
        while True:
            start_time = time.time()
            
            frame = next(stream)
            detections = detector.detect(frame)
            
            fps = 1.0 / (time.time() - start_time)
            
            frame_with_detections = visualizer.draw_detection_info(frame, detections, fps)
            frame_with_tracking = visualizer.draw_tracking_lines(frame_with_detections, detections)
            
            cv2.imshow('YOLO-Anywhere Webcam Demo', frame_with_tracking)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                new_classes = input("Enter new classes (comma-separated): ").split(',')
                new_classes = [cls.strip() for cls in new_classes if cls.strip()]
                if new_classes:
                    detector.update_classes(new_classes)
                    print(f"Updated classes: {new_classes}")
    
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    finally:
        stream.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    webcam_demo()