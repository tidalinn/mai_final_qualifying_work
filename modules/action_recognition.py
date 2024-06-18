'''
'''

import numpy as np
from time import time
import random
from collections import defaultdict
import torch
import cv2
from ultralytics import YOLO
from pytorchvideo.models.hub import slowfast_r50_detection
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from enum import Enum


class ModelType(Enum):
    tracking = 0
    action = 1


class ActionRecognition:

    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Current device:', self.device)
        self.model_tracking = self.load_model(ModelType.detection)
        self.classes = self.model.model.names
        print(self.classes)
        self.model_action = self.load_model(ModelType.action)
        self.track_history = defaultdict(lambda: [])
        self.ava_labelnames,_ = AvaLabeledVideoFramePaths.read_label_map('modules/training/models/temp.pbtxt')
        print(self.ava_labelnames)
        self.coco_color_map = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]
    

    def load_model(self, model_type):
        match model_type:
            case ModelType.tracking:
                model = YOLO('modules/training/models/yolov8m.pt')
            case ModelType.action:
                model = slowfast_r50_detection(True).eval().to(self.device)
        model.fuse()
        return model
    

    def predict(self, frame):
        results = self.model.track(frame, conf=0.3, iou=0.5, tracker='modules/training/models/bytetrack.yaml', persist=True)
        return results[0]
    

    def get_bboxes(self, results):
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
            track.append((float(x), float(y)))

            if len(track) > 30:
                track.pop(0)
    

    def get_actions(self, cap, results):
        if len(cap.stack) == 25:
            print(f"processing {cap.idx // 25}th second clips")
            clip = cap.get_video_clip()

            '''
            if results.pred[0].shape[0]:
                inputs, inp_boxes, _=ava_inference_transform(clip, yolo_preds.pred[0][:,0:4], crop_size=imsize)
                inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)
                if isinstance(inputs, list):
                    inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
                else:
                    inputs = inputs.unsqueeze(0).to(device)
                with torch.no_grad():
                    slowfaster_preds = video_model(inputs, inp_boxes.to(device))
                    slowfaster_preds = slowfaster_preds.cpu()
                for tid,avalabel in zip(yolo_preds.pred[0][:,5].tolist(), np.argmax(slowfaster_preds, axis=1).tolist()):
                    id_to_ava_labels[tid] = ava_labelnames[avalabel+1]
            '''
    

    def draw_bboxes(self, img, detections):
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
                detections = self.get_bboxes(results)
                actions = self.get_actions(cap, results)

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