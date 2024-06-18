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
        self.track_history = defaultdict(lambda: [])
    

    def detect(self, detection):
        self.detections.append(detection)
    

    def get_detection(self):
        if len(self.detections) > 0:
            self.detections = [sorted(self.detections, key=lambda x: x[5], reverse=True)[0]]
            return self.detections[0]
        else:
            return self.detections

    
    def track(self, x, y, track_id):
        track = self.track_history[track_id]
        track.append((float(x), float(y)))
        self.is_captured = True

        if len(track) > 30:
            track.pop(0)
            self.is_captured = False
            self.detections = []
    
    
    def estimate_keypoints(self):
        pass