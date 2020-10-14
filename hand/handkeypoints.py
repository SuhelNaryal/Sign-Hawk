from common.keypoint import KeyPoint
from hand.finger import Finger


class HandKeyPoints:

    def __init__(self, wrist=None, index=None, middle=None, ring=None, little=None, thumb=None):
        if wrist is None:
            wrist = KeyPoint()
        else:
            wrist = KeyPoint(wrist)

        if index is None:
            index = Finger()
        else:
            index = Finger(index)

        if middle is None:
            middle = Finger()
        else:
            middle = Finger(middle)

        if ring is None:
            ring = Finger()
        else:
            ring = Finger(ring)

        if little is None:
            little = Finger()
        else:
            little = Finger(little)

        if thumb is None:
            thumb = Finger()
        else:
            thumb = Finger(thumb)

        self.wrist = wrist
        self.index = index
        self.middle = middle,
        self.ring = ring
        self.little = little
        self.thumb = thumb

    def __init__(self, handkp):
        self.__init__(handkp.wrist, handkp.index, handkp.middle, handkp.ring, handkp.little, handkp.thumb)

    def set_wrist(self, x, y, z):
        self.wrist.set_xyz(x, y, z)

    def set_wrist(self, kp):
        self.set_wrist(kp.x, kp.y, kp.z)

    def set_index(self, index):
        self.index.set_all_joints(index)

    def set_index(self, tip, ipupper, iplower, mcp):
        self.index.set_all_joints(tip, ipupper, iplower, mcp)

    def set_middle(self, middle):
        self.middle.set_all_joints(middle)

    def set_middle(self, tip, ipupper, iplower, mcp):
        self.middle.set_all_joints(tip, ipupper, iplower, mcp)

    def set_ring(self, ring):
        self.ring.set_all_joints(ring)

    def set_ring(self, tip, ipupper, iplower, mcp):
        self.ring.set_all_joints(tip, ipupper, iplower, mcp)

    def set_little(self, little):
        self.little.set_all_joints(little)

    def set_little(self, tip, ipupper, iplower, mcp):
        self.little.set_all_joints(tip, ipupper, iplower, mcp)

    def set_thumb(self, thumb):
        self.thumb.set_all_joints(thumb)

    def set_thumb(self, tip, ipupper, iplower, mcp):
        self.thumb.set_all_joints(tip, ipupper, iplower, mcp)

    def get_wrist(self):
        return KeyPoint(self.wrist)

    def get_index(self):
        return Finger(self.index)

    def get_middle(self):
        return Finger(self.middle)

    def get_ring(self):
        return Finger(self.ring)

    def get_little(self):
        return Finger(self.little)

    def get_thumb(self):
        return Finger(self.thumb)

    def get_hand_keypoints(self):
        return {
            'wrist': self.wrist.get_keypoints(),
            'index': self.index.get_finger_keypoints(),
            'middle': self.middle.get_finger_keypoints(),
            'ring': self.ring.get_finger_keypoints(),
            'little': self.little.get_finger_keypoints(),
            'thumb': self.thumb.get_finger_keypoints()
        }