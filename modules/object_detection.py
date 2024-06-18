'''
'''

import numpy as np
from time import time
import torch
import cv2
from ultralytics import YOLO


class ObjectDetection:

    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Current device:', self.device)
        self.model = self.load_model()
        self.classes = self.model.model.names
    

    def load_model(self):
        model = YOLO('modules/training/models/yolov8m.pt')
        model.fuse()
        return model
    

    def predict(self, frame):
        results = self.model.predict(frame)
        return results
    

    def get_results(self, results):
        detections = []

        for result in results[0]:
            bboxes = result.boxes

            x1, y1, x2, y2 = [int(b) for b in bboxes.xyxy[0].cpu().tolist()]
            cls = self.classes[int(result.boxes.cls[0].cpu().numpy())]
            conf = np.round(float(result.boxes.conf[0].cpu().numpy()), 2)

            detections.append([x1, y1, x2, y2, cls, conf])
        
        return detections
    
    def draw_bounding_boxes(self, img, detections):
        for detection in detections:
            x1, y1, x2, y2, cls, conf = detection

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f'ID: {cls} {conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return img
    

    def run(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:            
            time_start = time()
            
            ret, frame = cap.read()
            assert ret

            results = self.predict(frame)
            detections = self.get_results(results)

            frame = self.draw_bounding_boxes(frame, detections)

            time_end = time()
            fps = 1 / np.round(time_end - time_start, 2)

            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imshow('YOLOv8 Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()