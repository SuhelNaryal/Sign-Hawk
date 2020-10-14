import cv2
import tensorflow as tf
import numpy as np


class SingleHandWrapper:

    def __init__(self, path_to_model):
        self.model = tf.lite.Interpreter(path_to_model)
        self.model.allocate_tensors()

    def load_img(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if not isinstance(img, np.ndarray):
            raise IOError("Fail to read %s" % path)

        return self.process_img(img)

    def preprocess_img(self, img):
        # fit the image into a 256x256 square
        shape = np.r_[img.shape]
        pad = (shape.max() - shape[:2]).astype('uint32') // 2
        img_pad = np.pad(
            img,
            ((pad[0], pad[0]), (pad[1], pad[1]), (0, 0)),
            mode='constant')
        img_small = cv2.resize(img_pad, (256, 256))
        img_small = np.ascontiguousarray(img_small)

        img_norm = np.ascontiguousarray(2 * ((img_small / 255) - 0.5).astype('float32'))
        return img_pad, img_norm, pad

    def get_predictions(self, img):
        pass