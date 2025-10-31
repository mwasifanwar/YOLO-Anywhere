import argparse
import cv2
import torch
import time
from pathlib import Path
from detectors.yolo_world import YOLOWorldDetector
from utils.webcam import WebcamStream
from utils.visualization import DetectionVisualizer
from utils.config import load_config, save_config

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-Anywhere: Zero-Shot Object Detection')
    parser.add_argument('--source', type=str, default='0', help='Webcam index, video file, or image file')
    parser.add_argument('--classes', nargs='+', default=['person', 'car', 'dog', 'cat'], help='Custom classes to detect')
    parser.add_argument('--model', type=str, default='yolo_world_l', help='Model size: yolo_world_s, yolo_world_m, yolo_world_l')
    parser.add_argument('--conf-threshold', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.7, help='IOU threshold for NMS')
    parser.add_argument('--device', type=str, default=None, help='Device: cuda, cpu, or auto-detect')
    parser.add_argument('--output', type=str, default=None, help='Output file path for saving results')
    parser.add_argument('--no-display', action='store_true', help='Run without display')
    parser.add_argument('--save-config', type=str, default=None, help='Save configuration to file')
    parser.add_argument('--load-config', type=str, default=None, help='Load configuration from file')
    return parser.parse_args()

def setup_device(device_arg):
    if device_arg:
        return device_arg
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'

def main():
    args = parse_args()
    
    if args.load_config:
        config = load_config(args.load_config)
        source = config.get('source', '0')
        classes = config.get('classes', ['person', 'car', 'dog', 'cat'])
        model_size = config.get('model', 'yolo_world_l')
        conf_threshold = config.get('conf_threshold', 0.25)
        iou_threshold = config.get('iou_threshold', 0.7)
    else:
        source = args.source
        classes = args.classes
        model_size = args.model
        conf_threshold = args.conf_threshold
        iou_threshold = args.iou_threshold
    
    if args.save_config:
        config = {
            'source': source,
            'classes': classes,
            'model': model_size,
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold
        }
        save_config(config, args.save_config)
        print(f"Configuration saved to {args.save_config}")
    
    device = setup_device(args.device)
    print(f"Using device: {device}")
    print(f"Detection classes: {classes}")
    
    detector = YOLOWorldDetector(
        model_size=model_size,
        classes=classes,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        device=device
    )
    
    visualizer = DetectionVisualizer()
    stream = WebcamStream(source=source, output_path=args.output)
    
    print("Starting YOLO-Anywhere detection...")
    print("Press 'q' to quit, 'c' to change classes, 's' to save frame")
    
    try:
        for frame in stream:
            start_time = time.time()
            
            detections = detector.detect(frame)
            processed_frame = visualizer.draw_detections(frame, detections, detector.class_names)
            
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(processed_frame, f'FPS: {fps:.1f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if not args.no_display:
                cv2.imshow('YOLO-Anywhere: Zero-Shot Object Detection', processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    new_classes = input("Enter new classes (comma-separated): ").split(',')
                    new_classes = [cls.strip() for cls in new_classes if cls.strip()]
                    if new_classes:
                        detector.update_classes(new_classes)
                        print(f"Updated classes: {new_classes}")
                elif key == ord('s'):
                    timestamp = int(time.time())
                    filename = f"capture_{timestamp}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"Frame saved as {filename}")
            
            if stream.output_writer:
                stream.output_writer.write(processed_frame)
                
    except KeyboardInterrupt:
        print("\nDetection stopped by user")
    except Exception as e:
        print(f"Error during detection: {str(e)}")
    finally:
        stream.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()