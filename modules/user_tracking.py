'''
'''

import numpy as np
from time import time
import torch
import cv2
from collections import defaultdict
from ultralytics import YOLO
from enum import Enum
from modules.user import User


class ModelType(Enum):
    tracking = 0
    keypoints = 1


class UserTracking:

    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Current device:', self.device)
        # tracker
        self.model_tracking = self.load_model(ModelType.tracking)
        self.classes_tracking = self.model_tracking.model.names
        print(self.classes_tracking)
        self.track_history = defaultdict(lambda: [])
        # keypoints detector
        #self.model_keypoints = self.load_model(ModelType.keypoints)
        self.user = User()
    

    def load_model(self, model_type):
        match model_type:
            case ModelType.tracking:
                model = YOLO('modules/training/runs/detect/train222/weights/best.pt')
                model.fuse()
            #case ModelType.keypoints:
            #    pass

        return model
    

    def predict(self, frame, model_type):
        match model_type:
            case ModelType.tracking:
                results = self.model_tracking.track(
                    frame, conf=0.3, iou=0.8, persist=True #, tracker='modules/training/models/bytetrack.yaml'
                )[0]
            case ModelType.keypoints:
                #if self.user.hand_L.is_captured and \
                #   self.user.hand_R.is_captured or \
                #   self.user.hand_L.is_captured or self.user.hand_R.is_captured:
                pass

        return results
    

    def get_bboxes(self, results):
        boxes = results.boxes
        detections = []

        if boxes.id != None:        
            for box in boxes:
                cls_id = int(box.cls.cpu().numpy())
                x1, y1, x2, y2 = [round(b) for b in box.xyxy[0].cpu().tolist()]
                track_id = int(box.id[0].int().cpu().numpy())
                conf = round(float(box.conf.cpu().numpy()), 2)

                detection = [x1, y1, x2, y2, track_id, cls_id, conf]
                detections.append(detection)
            
            '''
            match cls_id:
                case 0:
                    self.user.foot_L.detect(detection)
                case 1:
                    self.user.foot_R.detect(detection)
                case 2:
                    self.user.hand_L.detect(detection)
                case 3:
                    self.user.hand_R.detect(detection)
        
        detections = [
            self.user.foot_L.get_detection(), 
            self.user.foot_R.get_detection(),
            self.user.hand_L.get_detection(),
            self.user.hand_R.get_detection()
        ]
        '''
        
        return detections
    

    def get_keypoints(self):
        pass
    

    def watch_limb(self, results):
        boxes = results.boxes
        print(boxes)

        if boxes.id != None:
            bboxs = boxes.xywh.cpu()
            track_ids = boxes.id.int().cpu().tolist()
            cls_ids = boxes.cls.int().cpu().tolist()

            for box, track_id, cls in zip(bboxs, track_ids, cls_ids):
                x, y, w, h = box

                track = self.track_history[track_id]
                track.append((float(x), float(y)))

                if len(track) > 30:
                    track.pop(0)

                '''
                match cls:
                    case 0:
                        self.user.foot_L.track(x, y, track_id)
                    case 1:
                        self.user.foot_R.track(x, y, track_id)
                    case 2:
                        self.user.hand_L.track(x, y, track_id)
                    case 3:
                        self.user.hand_R.track(x, y, track_id)
                '''
    

    def save_data(self):
        pass


    def draw_bboxes(self, img, detections):
        for detection in detections:
            if len(detection) > 0:
                x1, y1, x2, y2, track_id, cls_id, conf = detection

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f'ID: {track_id} {self.classes_tracking[cls_id]} ({cls_id}) {conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return img
    

    def run(self):
        cap = cv2.VideoCapture(self.capture_index)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while cap.isOpened():
            success, frame = cap.read()

            if success:
                time_start = time()

                results_tracking = self.predict(frame, ModelType.tracking)
                #results_keypoints = self.predict(frame, ModelType.keypoints)
                #print(results_keypoints)
                # if results_keypoints is not None
                detections = self.get_bboxes(results_tracking)
                self.watch_limb(results_tracking)

                frame = self.draw_bboxes(frame, detections)

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