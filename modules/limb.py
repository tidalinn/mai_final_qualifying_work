'''
'''

from collections import defaultdict
from enum import Enum


class BodyPart(Enum):
    HAND = 0
    FOOT = 1


class Limb:
    
    def __init__(self, 
        body_part: BodyPart,
        is_right: bool = False
    ):
        self.type = body_part
        self.is_right = is_right
        self.is_captured = False
        self.detections = []
        self.keypoints = []
        self.track_history = defaultdict(lambda: [])
    

    def detect(self, detection):
        if len(self.detections) == 0:
            self.detections.append(detection)
    

    def track(self, x, y, track_id):
        track = self.track_history[track_id]
        track.append((float(x), float(y)))
        self.is_captured = True

        if len(track) > 10:
            track.pop(0)
            self.is_captured = False
            self.detections = []
    
    
    def hand_keypoints(self, frame, model_keypoints, predict_method, model_type):
        self.keypoints = predict_method(frame, model_type, model_keypoints)


    def foot_keypoints(self, point_bottom, point_top):
        self.keypoints = [point_bottom, point_top]