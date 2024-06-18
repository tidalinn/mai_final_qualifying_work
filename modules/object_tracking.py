'''
'''

import numpy as np
from time import time
from collections import defaultdict
import torch
import cv2
from ultralytics import YOLO


class ObjectTracking:

    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Current device:', self.device)
        self.model = self.load_model()
        self.classes = self.model.model.names
        self.track_history = defaultdict(lambda: [])
    

    def load_model(self):
        model = YOLO('modules/training/models/yolov8m.pt')
        model.fuse()
        return model
    

    def predict(self, frame):
        results = self.model.track(frame, conf=0.3, iou=0.5, tracker='modules/training/models/bytetrack.yaml', persist=True)
        return results[0]
    

    def get_results(self, results):
        boxes = results.boxes
        detections = []

        if boxes.id != None:
            self.watch_object(boxes)

            for box in boxes:
                x1, y1, x2, y2 = [round(b) for b in box.xyxy[0].cpu().tolist()]
                track_id = int(box.id[0].int().cpu().numpy())
                cls_id = int(box.cls.cpu().numpy())
                conf = round(float(box.conf.cpu().numpy()), 2)

                detections.append([x1, y1, x2, y2, track_id, cls_id, conf])
        
        return detections
    

    def watch_object(self, boxes):
        bboxs = boxes.xywh.cpu()
        track_ids = boxes.id.int().cpu().tolist()

        for box, track_id in zip(bboxs, track_ids):
            x, y, w, h = box
            track = self.track_history[track_id]
            track.append((float(x), float(y))) # center point

            if len(track) > 30: # retrain 90 tracks for 90 frames
                track.pop(0)
    

    def draw_bounding_boxes(self, img, detections):
        for detection in detections:
            x1, y1, x2, y2, track_id, cls_id, conf = detection

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f'ID: {track_id} {self.classes[cls_id]} ({cls_id}) {conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return img
    

    def run(self):
        cap = cv2.VideoCapture(self.capture_index)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while cap.isOpened():
            success, frame = cap.read()

            if success:
                time_start = time()

                results = self.predict(frame)
                detections = self.get_results(results)

                frame = self.draw_bounding_boxes(frame, detections)

                time_end = time()
                
                fps = 1 / np.round(time_end - time_start, 2)

                cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                cv2.imshow('YOLOv8 Tracking', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        
        cap.release()
        cv2.destroyAllWindows()