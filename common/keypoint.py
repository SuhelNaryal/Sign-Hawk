
class KeyPoint:

    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __init__(self, kp):
        self.x = kp.x
        self.y = kp.y
        self.z = kp.z

    def set_xyz(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def set_x(self, x):
        self.x = x

    def set_y(self, y):
        self.y = y

    def set_z(self, z):
        self.z = z

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_z(self):
        return self.z

    def get_keypoints(self):
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z
        }