import numpy as np
from mtcnn import MTCNN
from PIL import Image
from kalmanFilter import kalmanFilter

from mtcnn_pytorch.src.align_trans import warp_and_crop_face


class MTCNNTracker(MTCNN):

    def __init__(self, num_frames):
        super(MTCNNTracker, self).__init__()

        self.num_frames = num_frames
        # state transition matrix for warp params
        A = np.eye(12)
        A[0:6, 6:] = np.eye(6)

        # covariances of driving and observation noise
        sigma_u = np.sqrt(0.0001)
        Q = np.zeros((12, 12))
        Q[6:, 6:] = np.eye(6) * sigma_u**2

        sigma_warp = np.sqrt(0.1)
        sigma_warps = sigma_warp**2 * np.ones(4)
        sigma_translation = np.sqrt(1)
        sigma_translations = sigma_translation**2 * np.ones(2)
        sigmas_observations = np.concatenate((sigma_warps, sigma_translations))

        C = sigmas_observations @ np.eye(6)

        # setup initial values for params and their derivatives
        initial_signal_estimate = np.zeros(12)
        initial_MSE_estimate = 100 * np.eye(12)

        H = np.zeros((6, 12))
        H[:6, :6] = np.eye(6)

        self.kalman = kalmanFilter(A, Q, H, C, self.num_frames,
                                   initial_signal_estimate,
                                   initial_MSE_estimate)


    def align_multi(self, img, limit=None):
        boxes, landmarks = self.detect_faces(
            img, self.min_face_size, self.thresholds, self.nms_thresholds, self.factor)
        if limit:
            boxes = boxes[:limit]
            landmarks = landmarks[:limit]
        faces = []

        for landmark in landmarks:
            facial5points = [[landmark[j], landmark[j + 5]] for j in range(5)]
            warped_face, tfm = warp_and_crop_face(
                np.array(img), facial5points, self.refrence, crop_size=self.crop_size)
            faces.append(Image.fromarray(warped_face))
            tfm_flat = np.concatenate((tfm[:2, :2].flatten(), tfm[:, 2]))

        return boxes, faces, tfm
