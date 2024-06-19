'''
'''

import numpy as np
from time import time
from enum import Enum
import random
import socket
import cv2
import torch
from ultralytics import YOLO
import mediapipe as mp
import pytorchvideo
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample, short_side_scale_with_boxes, clip_boxes_to_image
)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection


from modules.user import User


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # udp -> _DGRAM | dcp -> _STREAM
server_address_port = ('127.0.0.1', 5052)

class ModelType(Enum):
    tracking = 0
    keypoints = 1
    actions = 2


class UserTracking:

    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Current device:', self.device)

        # tracker
        self.model_tracking = self.load_model(ModelType.tracking)
        self.classes_tracking = self.model_tracking.model.names
        print(self.classes_tracking)

        self.mp_hands_L = mp.solutions.hands
        self.mp_drawing_L = mp.solutions.drawing_utils
        self.mp_hands_R = mp.solutions.hands
        self.mp_drawing_R = mp.solutions.drawing_utils
        
        # keypoints detector
        self.model_keypoints_L = self.load_model(ModelType.keypoints, self.mp_hands_L)
        self.model_keypoints_R = self.load_model(ModelType.keypoints, self.mp_hands_R)

        # action recognition
        self.model_action = self.load_model(ModelType.actions)
        self.ava_labelnames,_ = AvaLabeledVideoFramePaths.read_label_map('modules/training/models/temp.pbtxt')
        self.coco_color_map = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]
        self.id_to_ava_labels = {}

        # user
        self.user = User()
    

    def load_model(self, model_type, mp_hands = None):
        match model_type:
            case ModelType.tracking:
                model = YOLO('modules/training/runs/detect/train222/weights/best.pt')
                model.fuse()
            case ModelType.keypoints:
                model = mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3
                )
            case ModelType.actions:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model = slowfast_r50_detection(True).eval().to(device)

        return model
    

    def predict(self, frame, model_type, model_keypoints = None):
        results = []

        match model_type:
            case ModelType.tracking:
                results = self.model_tracking.track(
                    frame, conf=0.3, iou=0.8, persist=True, tracker='modules/training/models/bytetrack.yaml'
                )[0]
            case ModelType.keypoints:
                landmarks = model_keypoints.process(frame)

                if landmarks.multi_hand_landmarks:
                    results = landmarks.multi_hand_landmarks

        return results
    

    def get_bboxes(self, results):
        boxes = results.boxes
        detections = []
        track_ids = []

        if boxes.id != None:        
            for box in boxes:
                cls_id = int(box.cls.cpu().numpy())
                x1, y1, x2, y2 = [round(b) for b in box.xyxy[0].cpu().tolist()]
                track_id = int(box.id[0].int().cpu().numpy())
                conf = round(float(box.conf.cpu().numpy()), 2)

                if len(track_ids) == 0 or track_id not in track_ids:
                    track_ids.append(track_id)
                else:
                    break

                detection = [x1, y1, x2, y2, track_id, cls_id, conf]
            
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
                *self.user.foot_L.detections, 
                *self.user.foot_R.detections,
                *self.user.hand_L.detections,
                *self.user.hand_R.detections
            ]
            print()
        
        return detections


    def draw_bboxes(self, frame, detections):
        for detection in detections:
            if len(detection) > 0:
                x1, y1, x2, y2, track_id, cls_id, conf = detection

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f'ID: {track_id} {self.classes_tracking[cls_id]} ({cls_id}) {conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    

    def watch_limb(self, results):
        boxes = results.boxes

        if boxes.id != None:
            bboxs = boxes.xywh.cpu()
            track_ids = boxes.id.int().cpu().tolist()
            cls_ids = boxes.cls.int().cpu().tolist()

            for box, track_id, cls in zip(bboxs, track_ids, cls_ids):
                x, y, w, h = box

                match cls:
                    case 0:
                        self.user.foot_L.track(x, y, track_id)
                    case 1:
                        self.user.foot_R.track(x, y, track_id)
                    case 2:
                        self.user.hand_L.track(x, y, track_id)
                    case 3:
                        self.user.hand_R.track(x, y, track_id)
    

    def get_keypoints(self, frame, detections):
        for detection in detections:
            x1, y1, x2, y2, track_id, cls_id, conf = detection

            frame_cropped = frame #[y1:y2, x1:x2]
            frame_cropped_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)

            point_bottom = (int(x1 + (x2 - x1) / 2), int(y1))
            point_top = (int(x1 + (x2 - x1) / 2), int(y2))

            match cls_id:
                case 0:
                    self.user.foot_L.foot_keypoints(point_bottom, point_top)
                case 1:
                    self.user.foot_R.foot_keypoints(point_bottom, point_top)
                case 2:
                    self.user.hand_L.hand_keypoints(frame_cropped_rgb, self.model_keypoints_L, self.predict, ModelType.keypoints)
                case 3:
                    self.user.hand_R.hand_keypoints(frame_cropped_rgb, self.model_keypoints_R, self.predict, ModelType.keypoints)
    

    def draw_keypoints(self, frame, detections):
        for detection in detections:
            x1, y1, x2, y2, track_id, cls_id, conf = detection

            match cls_id:
                case 0:
                    for kps in self.user.foot_L.keypoints:
                        cv2.circle(frame, kps, 5, (0, 0, 255), -1)
                    cv2.line(frame, self.user.foot_L.keypoints[0], self.user.foot_L.keypoints[1], (255, 255, 255), 1)
                case 1:
                    for kps in self.user.foot_R.keypoints:
                        cv2.circle(frame, kps, 5, (0, 0, 255), -1)
                    cv2.line(frame, self.user.foot_R.keypoints[0], self.user.foot_R.keypoints[1], (255, 255, 255), 1)
                case 2:
                    for kps in self.user.hand_L.keypoints:
                        self.mp_drawing_L.draw_landmarks(frame, kps, self.mp_hands_L.HAND_CONNECTIONS)
                case 3:
                    for kps in self.user.hand_R.keypoints:
                        self.mp_drawing_R.draw_landmarks(frame, kps, self.mp_hands_R.HAND_CONNECTIONS)
        
        return frame
    

    def get_actions(self, frame, ):
        pass
    '''
        if yolo_preds.pred[0].shape[0]:
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

    def save_data(self, frame):
        data = [[], [], [], []]

        #   MediaPipe 0 -> ---- <- Unity max
        #                  |  |
        # MediaPipe max -> ---- <- Unity 0

        h, w, _ = frame.shape
        coords_foot = lambda x, y, z: [round(x, 3), round(y, 3), round(z, 3)]
        coords_hand = lambda x, y, z: [round(x, 3) * w, h - round(y, 3) * h, round(z, 3) * w]

        if len(self.user.foot_L.keypoints) > 0:
            for kp in self.user.foot_L.keypoints:
                data[0].extend(coords_foot(kp[0], 0, kp[1]))
        
        if len(self.user.foot_R.keypoints) > 0:
            for kp in self.user.foot_R.keypoints:
                data[1].extend(coords_foot(kp[0], 0, kp[1]))
        
        if len(self.user.hand_L.keypoints) > 0:
            for kps in self.user.hand_L.keypoints:
                for kp in kps.landmark:
                    data[2].extend(coords_hand(kp.x, kp.y, kp.z))
        
        if len(self.user.hand_R.keypoints) > 0:
            for kps in self.user.hand_R.keypoints:
                for kp in kps.landmark:
                    data[3].extend(coords_hand(kp.x, kp.y, kp.z))
        
        # foot_L, foot_R, hand_L, hand_R
        sock.sendto(str.encode(str(data)), server_address_port)
    

    def run(self):
        cap = cv2.VideoCapture(self.capture_index)
        count = 0

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 780)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 220)

        while cap.isOpened():
            success, frame = cap.read()

            if success:
                time_start = time()

                # tracking
                results_tracking = self.predict(frame, ModelType.tracking)

                if results_tracking is not None:
                    detections = self.get_bboxes(results_tracking)
                    self.watch_limb(results_tracking)

                    # keypoints
                    self.get_keypoints(frame, detections)

                    # showing
                    frame = self.draw_bboxes(frame, detections)
                    frame = self.draw_keypoints(frame, detections)
                
                if count < 30:
                    count += 1
                else:
                    self.get_actions(frame)
                    count = 0

                # sending to unity
                self.save_data(frame)

                # fps
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