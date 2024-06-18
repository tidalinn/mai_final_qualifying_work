'''
'''

from modules.limb import Limb, BodyPart

class User:

    def __init__(self):
        self.hand_L = Limb(
            body_part = BodyPart.HAND,
            is_right = False
        )
        self.hand_R = Limb(
            body_part = BodyPart.HAND,
            is_right = True
        )
        self.foot_L = Limb(
            body_part = BodyPart.FOOT,
            is_right = False
        )
        self.foot_R = Limb(
            body_part = BodyPart.FOOT,
            is_right = True
        )