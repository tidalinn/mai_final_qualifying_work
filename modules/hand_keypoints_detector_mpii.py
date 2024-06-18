'''
'''

import numpy as np
import os
import cv2
from time import time
import torch


class HandKeypointsDetector:

    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Current device:', self.device)
        self.nPoints = 22
        self.pose_pairs = [
            [0,1], [1,2], [2,3], [3,4],
            [0,5], [5,6], [6,7], [7,8],
            [0,9], [9,10], [10,11], [11,12],
            [0,13], [13,14], [14,15], [15,16],
            [0,17], [17,18], [18,19], [19,20]
        ]
        self.inWidth = 368
        self.inHeight = 368
        self.threshold = 0.1
        self.model = self.load_model()
    

    def load_model(self):
        protoFile = 'modules/training/models/pose_deploy.prototxt'
        weightsFile = 'modules/training/models/pose_iter_102000.caffemodel'
        model = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
        return model
    

    def predict(self, frame):
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (frame.shape[1], frame.shape[0]), (0, 0, 0), swapRB=False, crop=False)
        self.model.setInput(inpBlob)
        results = self.model.forward()
        self.position_width = results.shape[3]
        self.position_height = results.shape[2]
        return results
    

    def get_points(self, frame, results):
        points = []

        for i in range(self.nPoints):
            conf_map = results[0, i, :, :]
            #conf_map = cv2.resize(conf_map, (self.frameWidth, self.frameHeight))

            _, conf, _, point = cv2.minMaxLoc(conf_map)

            x = int((frame.shape[1] * point[0]) / self.position_width)
            y = int((frame.shape[0] * point[1]) / self.position_height)

            if conf > self.threshold :
                points.append((x, y))
            else :
                points.append(None)
        
        return points
    

    def draw_points(self, frame, points):
        for i in range(len(points)):
            if points[i] is not None:
                cv2.circle(frame, points[i], 8, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frame, '{}'.format(i), points[i], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    

    def draw_skeleton(self, frame, points):
        for pair in self.pose_pairs:
            partA, partB = pair[0], pair[1]

            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
                

    def run(self):
        cap = cv2.VideoCapture(self.capture_index)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while cap.isOpened():
            success, frame = cap.read()

            if success:
                time_start = time()

                results = self.predict(frame)
                points = self.get_points(frame, results)
                self.draw_points(frame, points)
                self.draw_skeleton(frame, points)

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