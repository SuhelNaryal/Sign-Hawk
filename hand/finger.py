from common.keypoint import KeyPoint

class Finger:
    def __init__(self, tip=None, ipupper=None, iplower=None, mcp=None):
        if tip is None:
            tip = KeyPoint()
        else:
            tip = KeyPoint(tip)

        if ipupper is None:
            ipupper = KeyPoint()
        else:
            ipupper = KeyPoint(ipupper)

        if iplower is None:
            iplower = KeyPoint()
        else:
            iplower = KeyPoint(iplower)

        if mcp is None:
            mcp = KeyPoint()
        else:
            mcp = KeyPoint(mcp)

        self.tip = tip
        self.IPupper = ipupper #interphalangeal joint upper
        self.IPlower = iplower #interphalangeal joint lower
        self.MCP = mcp #metacarpophalangeal joint

    def __init__(self, finger):
        self.__init__(finger.tip, finger.IPupper, finger.IPlower, self.MCP)

    def set_all_joints(self, finger):
        self.set_tip(finger.tip)
        self.set_IPupper(finger.IPupper)
        self.set_IPlower(finger.IPlower)
        self.set_MCP(finger.MCP)

    def set_all_joints(self, tip, ipupper, iplower, mcp):
        self.set_tip(tip)
        self.set_IPupper(ipupper)
        self.set_IPlower(iplower)
        self.set_MCP(mcp)

    def set_tip(self, kp):
        self.tip.set_xyz(kp.x, kp.y, kp.z)

    def set_tip(self, x, y, z):
        self.tip.set_xyz(x, y, z)

    def set_IPupper(self, kp):
        self.IPupper.set_xyz(kp.x, kp.y, kp.z)

    def set_IPupper(self, x, y, z):
        self.IPupper.set_xyz(x, y, x)

    def set_IPlower(self, kp):
        self.IPlower.set_xyz(kp.x, kp.y, kp.z)

    def set_IPlower(self, x, y, z):
        self.IPlower.set_xyz(x, y, z)

    def set_MCP(self, kp):
        self.MCP.set_xyz(kp.x, kp.y, kp.z)

    def set_MCP(self, x, y, z):
        self.MCP.set_xyz(x, y, z)

    def get_tip(self):
        return KeyPoint(self.tip)

    def get_IPupper(self):
        return KeyPoint(self.IPupper)

    def get_IPlower(self):
        return KeyPoint(self.IPlower)

    def get_MCP(self):
        return KeyPoint(self.MCP)

    def get_finger_keypoints(self):
        return {
            'tip': self.tip.get_keypoints(),
            'IPupper': self.IPupper.get_keypoints(),
            'IPlower': self.IPlower.get_keypoints(),
            'MCP': self.MCP.get_keypoints()
        }
