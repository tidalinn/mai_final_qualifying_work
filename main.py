'''
'''

from modules.user_tracking import UserTracking
from modules.user_tracking_deepsort import UserTrackingDeepsort
from modules.human_pose_detection import HumanPoseDetection
from modules.action_recognition import ActionRecognition
from modules.hand_keypoints_detector_mpii import HandKeypointsDetector


u = UserTracking(0)
u.run()

#u = UserTrackingDeepsort(0)
#u.run()

# 0 - person, 39 bottle, 67 cell phone, 77 teddy bear

#h = HumanPoseDetection(0)
#h.run()

#a = ActionRecognition()
#a.run()

#h = HandKeypointsDetector(0)
#h.run() 