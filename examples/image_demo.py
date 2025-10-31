import cv2
from pathlib import Path
from detectors.yolo_world import YOLOWorldDetector
from utils.visualization import DetectionVisualizer
from utils.webcam import ImageProcessor

def image_demo():
    print("YOLO-Anywhere Image Demo")
    
    image_path = input("Enter image path: ").strip()
    custom_classes = input("Enter custom classes (comma-separated): ")
    
    classes = [cls.strip() for cls in custom_classes.split(',')] if custom_classes else ['object']
    
    detector = YOLOWorldDetector(classes=classes)
    visualizer = DetectionVisualizer()
    
    try:
        image = ImageProcessor.load_image(image_path)
        image = ImageProcessor.resize_image(image, 1024)
        
        print("Detecting objects...")
        detections = detector.detect(image)
        
        result_image = visualizer.draw_detections(image, detections)
        
        print(f"Found {len(detections)} objects:")
        for det in detections:
            print(f"  - {det.class_name}: {det.confidence:.2f}")
        
        output_path = f"detected_{Path(image_path).name}"
        ImageProcessor.save_image(result_image, output_path)
        
        cv2.imshow('YOLO-Anywhere Image Detection', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    image_demo()