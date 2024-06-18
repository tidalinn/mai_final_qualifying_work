'''
'''

import numpy as np
from time import time
import torch
import cv2
from ultralytics import YOLO
from enum import Enum

from modules.user import User
from modules.deepsort_tracker import Tracker


class ModelType(Enum):
    detection = 0
    tracking = 1
    keypoints = 2


class UserTrackingDeepsort:

    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Current device:', self.device)
        # detection
        self.model_detection = self.load_model(ModelType.detection)
        self.classes_detection = self.model_detection.model.names
        self.detection_threshold = 0.3
        # tracking
        self.model_tracking = self.load_model(ModelType.tracking)
        # keypoints detector
        #self.model_keypoints = self.load_model(ModelType.keypoints)
        self.user = User()
    

    def load_model(self, model_type):
        match model_type:
            case ModelType.detection:
                model = YOLO('modules/training/runs/detect/train/weights/best.pt')
                model.fuse()
            case ModelType.tracking:
                model = Tracker()

        return model
    

    def predict(self, frame, model_type):
        match model_type:
            case ModelType.detection:
                results = self.model_detection(frame, verbose=False)[0]

        return results
    

    def get_bboxes(self, results):
        detections = []
        
        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls_id = box
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)
            cls_id = int(cls_id)
            
            if conf > self.detection_threshold:
                detections.append([x1, y1, x2, y2, conf, cls_id])
        
        return detections
    

    def draw_bboxes(self, frame):
        for track in self.model_tracking.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)
            track_id = track.track_id

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    

    def get_keypoints(self):
        pass
    

    def track(self, frame):
        results = self.predict(frame, ModelType.detection)
        detections = self.get_bboxes(results)
        self.model_tracking.update(frame, detections)
        self.draw_bboxes(frame)
    

    def save_data(self):
        pass
    

    def run(self):
        cap = cv2.VideoCapture(self.capture_index)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while cap.isOpened():
            success, frame = cap.read()

            if success:
                time_start = time()

                self.track(frame)

                time_end = time()
                fps = 1 / np.round(time_end - time_start, 2)

                cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                cv2.imshow('YOLOv8 + DEEPSort Tracking', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        
        cap.release()
        cv2.destroyAllWindows()