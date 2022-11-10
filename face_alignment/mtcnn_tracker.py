from mtcnn import MTCNN


class MTCNNTracker(MTCNN):

    def __init__(self):
        super(MTCNNTracker, self).__init__()

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
        return boxes, faces, tfm
