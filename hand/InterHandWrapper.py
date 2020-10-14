from hand.InterHand.main.config import cfg
from torch.nn.parallel.data_parallel import DataParallel
from hand.InterHand.main import model
import cv2
import numpy as np
import torch
from common.keypoint import KeyPoint
from hand.handkeypoints import HandKeyPoints

def load_img(path, order='RGB'):
    # load
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order == 'RGB':
        img = img[:, :, ::-1].copy()

    img = img.astype(np.float32)
    return img


img = load_img('test.jpg')
img = np.transpose(img, (2, 0, 1))
inputs = {'img': torch.from_numpy(np.array([img]))}

handModel = model.get_model(mode='results', joint_num=21)
handModel = DataParallel(handModel).cuda()
print('load model start')
ckpt = torch.load('InterHand2.6M_all/snapshot_20.pth.tar')
handModel.load_state_dict(ckpt['network'], strict=False)
print('load model finish')
with torch.no_grad():
    out = handModel(inputs=inputs, mode='results')
print(out)

class InterHandWrapper:
    def __init__(self):
        handModel = model.get_model(mode='results', joint_num=21)
        handModel = DataParallel(handModel).cuda()
        ckpt = torch.load('hand/models/interhand/interhand.pth.tar')
        handModel.load_state_dict(ckpt['network'], strict=False)

    def process_img(self, img, order='RGB'):
        if order == 'RGB':
            img = img[:, :, ::-1].copy()

        img = img.astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
        return img

    def load_img(self, path, order='RGB'):
        # load
        img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if not isinstance(img, np.ndarray):
            raise IOError("Fail to read %s" % path)

        return self.process_img(img)

    def get_predictions(self, imgs):
        processed_imgs = []
        for img in imgs:
            processed_imgs.append(self.process_img(img))
        inputs = {'img': torch.from_numpy(np.array(processed_imgs))}

        with torch.no_grad():
            out = handModel(inputs=inputs, mode='results')

        preds = {
                'joint_coord': out['joint_coord'].cpu().numpy(),
                'rel_root_depth': out['rel_root_depth'].cpu().numpy(),
                'hand_type': out['hand_type'].cpu().numpy()
            }

        return preds

    def process_predictions(self, preds):

        joint_coord = preds['joint_coord'].copy()
        rel_root_depth = preds['rel_root_depth'].copy()[0]
        left_hand_score, right_hand_score = preds['hand_type'][0], preds['hand_type'][1]

        joint_coord[:, 0] = joint_coord[:, 0] / 64 * 256
        joint_coord[:, 1] = joint_coord[:, 1] / 64 * 256
        joint_coord[:, 2] = (joint_coord[:, 2] / 128 - 1)
        rel_root_depth = (rel_root_depth / 128 - 1)

        joint_coord[21:, 2] += rel_root_depth

        return {
                'right_joint_coord': joint_coord[:21],
                'left_joint_coord': joint_coord[21:],
                'rel_root_depth': rel_root_depth,
                'left_hand_score': left_hand_score,
                'right_hand_score': right_hand_score
            }

    def generate_handkeypoints(self, joint_coords):
        handkeypoints = HandKeyPoints()
        
        handkeypoints.wrist.set_xyz(joint_coords[20][0], joint_coords[20][1], joint_coords[20][2])
        
        handkeypoints.thumb.set_tip(joint_coords[0][0], joint_coords[0][1], joint_coords[0][2])
        handkeypoints.thumb.set_IPupper(joint_coords[1][0], joint_coords[1][1], joint_coords[1][2])
        handkeypoints.thumb.set_IPlower(joint_coords[2][0], joint_coords[2][1], joint_coords[2][2])
        handkeypoints.thumb.set_MCP(joint_coords[3][0], joint_coords[3][1], joint_coords[3][2])

        handkeypoints.index.set_tip(joint_coords[4][0], joint_coords[4][1], joint_coords[4][2])
        handkeypoints.index.set_IPupper(joint_coords[5][0], joint_coords[5][1], joint_coords[5][2])
        handkeypoints.index.set_IPlower(joint_coords[6][0], joint_coords[6][1], joint_coords[6][2])
        handkeypoints.index.set_MCP(joint_coords[7][0], joint_coords[7][1], joint_coords[7][2])

        handkeypoints.middle.set_tip(joint_coords[8][0], joint_coords[8][1], joint_coords[8][2])
        handkeypoints.middle.set_IPupper(joint_coords[9][0], joint_coords[9][1], joint_coords[9][2])
        handkeypoints.middle.set_IPlower(joint_coords[10][0], joint_coords[10][1], joint_coords[10][2])
        handkeypoints.middle.set_MCP(joint_coords[11][0], joint_coords[11][1], joint_coords[11][2])

        handkeypoints.ring.set_tip(joint_coords[12][0], joint_coords[12][1], joint_coords[12][2])
        handkeypoints.ring.set_IPupper(joint_coords[13][0], joint_coords[13][1], joint_coords[13][2])
        handkeypoints.ring.set_IPlower(joint_coords[14][0], joint_coords[14][1], joint_coords[14][2])
        handkeypoints.ring.set_MCP(joint_coords[15][0], joint_coords[15][1], joint_coords[15][2])

        handkeypoints.little.set_tip(joint_coords[16][0], joint_coords[16][1], joint_coords[16][2])
        handkeypoints.little.set_IPupper(joint_coords[17][0], joint_coords[17][1], joint_coords[17][2])
        handkeypoints.little.set_IPlower(joint_coords[18][0], joint_coords[18][1], joint_coords[18][2])
        handkeypoints.little.set_MCP(joint_coords[19][0], joint_coords[19][1], joint_coords[19][2])
        
        return handkeypoints
