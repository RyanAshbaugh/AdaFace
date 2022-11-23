import net
import torch
import pickle
import os
from os import path as osp
import cv2
from face_alignment import align
from face_alignment.kalmanFilter import kalmanFilter
import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd


'''
adaface_models = {
    'ir_50':"pretrained/adaface_ir50_ms1mv2.ckpt",
}
'''

adaface_models = {
    'ir_101': "pretrained/webface12m_adaface.ckpt",
}


def load_pretrained_model(architecture='ir_50'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {
        key[6:]: val for key, val in statedict.items()
        if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model


def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2, 0, 1)]).float()
    return tensor


def pad_dimension(pad_factor, start, length, image_dim_length):
    new_length = int(length * pad_factor)
    start -= int((new_length - length) / 2)
    end_point = start + new_length
    if end_point >= image_dim_length:
        start -= (end_point - image_dim_length) + 1

    if start < 0:
        start = 0

    assert (start >= 0) & (start + new_length < image_dim_length), \
        ("Start: " + str(start) + ", new_length: " + str(new_length) +
         ", image_dim_length: " + str(image_dim_length))
    return int(start), int(length)


def pad_bbox(pad_factor, x1, bbox_width, y1, bbox_height, width, height):
    x1, bbox_width = pad_dimension(pad_factor, x1, bbox_width, width)
    y1, bbox_height = pad_dimension(pad_factor, y1, bbox_height, height)

    return int(x1), int(bbox_width), int(y1), int(bbox_height)


def calculateStartDecrease(smaller, larger):
    extra_width_to_square = larger - smaller
    start_decrease_amount = int(extra_width_to_square / 2)

    return start_decrease_amount


def expandBBoxToBeSquare(bbox, height, width):
    x1 = bbox['min_x']
    y1 = bbox['min_y']
    bbox_width = bbox['width']
    bbox_height = bbox['height']

    if bbox['height'] > bbox['width']:
        x1 -= calculateStartDecrease(bbox['width'], bbox['height'])
        bbox_width = bbox['height']
    elif bbox['height'] < bbox['width']:
        y1 -= calculateStartDecrease(bbox['height'], bbox['width'])
        bbox_height = bbox['width']

    # check out of bounds
    if x1 < 0:
        x1 = 0
    if (x1 + bbox_width) > width:
        x1 = width - bbox_width
    if y1 < 0:
        y1 = 0
    if (y1 + bbox_height) > height:
        y1 = height - bbox_height

    return [x1, bbox_width, y1, bbox_height]


if __name__ == '__main__':

    model = load_pretrained_model('ir_101')

    pad_factor = 1.2
    save_output_images = True
    crop_size = (112, 112)

    detections_path = ('/research/iprobe-ashbau12/hpcc_briar/TRANSFER/'
                       'bts1.1_bts2_face_detect.csv.tgz')
    dataset_root = '/research/iprobe-ashbau12/hpcc_briar/ORNL/'
    video_fname = 'BGC2/BTS2/full/G02394/field/close_range/wb/G02394_set2_struct_PNP-9200RH_00091852D428_cae7200f.mp4'
    video_path = osp.join(dataset_root, video_fname)

    detections_df = pd.read_csv(
        detections_path,
        compression='gzip')

    output_root = 'output_kalman_nn/'
    if not osp.exists(output_root):
        os.makedirs(output_root)

    if save_output_images:
        image_folder = osp.join(output_root, 'images')
        kalman_image_folder = osp.join(output_root, 'images_kf')
        if not osp.exists(image_folder):
            os.makedirs(image_folder)
            os.makedirs(kalman_image_folder)

    video_rows = detections_df.media_path == video_fname

    labels = detections_df.loc[video_rows, [
        'frame_number', 'min_x', 'min_y', 'max_x', 'max_y', 'width', 'height']]
    labels.sort_values('frame_number', inplace=True)
    labels.reset_index(inplace=True)

    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    kalman_filter = kalmanFilter(num_frames,
                                       np.sqrt(0.01),
                                       np.sqrt(0.1))

    features = []
    frame_nums = []
    warp_params = []

    pickle.dump(labels.frame_number.to_numpy().astype(int),
                open(osp.join(output_root, 'detected_frame_numbers.pkl'), 'wb'))

    label_index = 0
    with tqdm(total=num_frames) as progress_bar:
        for ii in range(num_frames):
            status = cap.grab()
            assert status, f"No frame at {ii} from video reader: {video_path}"

            label_frame_num = int(labels.loc[label_index, 'frame_number'])

            if label_frame_num == ii:
                ret, image = cap.retrieve()

                roi = labels[labels.columns[1:8]].iloc[label_index]
                x1, bbox_width, y1, bbox_height = expandBBoxToBeSquare(roi,
                                                                       height,
                                                                       width)

                if pad_factor > 0:
                    x1, bbox_width, y1, bbox_height = pad_bbox(pad_factor,
                                                               x1,
                                                               bbox_width,
                                                               y1,
                                                               bbox_height,
                                                               width,
                                                               height)

                roi_image = image[y1:y1+bbox_height, x1:x1 + bbox_width, :]
                roi_image_pil = Image.fromarray(
                    roi_image[:, :, ::-1].astype('uint8'), 'RGB')

                # same original alignment step
                aligned_rgb_img, tfm = align.get_aligned_face('',
                                                              roi_image_pil)

                if tfm is not None:
                    tfm = np.concatenate((tfm[:2, :2].flatten(), tfm[:, 2]))
                kalman_filter.predict_and_estimate(tfm, ii)

                kalman_tfm = np.array(
                    ((*kalman_filter.estimated_signal[:2, ii],
                      kalman_filter.estimated_signal[4, ii]),
                     (*kalman_filter.estimated_signal[2:4, ii],
                      kalman_filter.estimated_signal[5, ii])))
                kalman_img = cv2.warpAffine(
                    roi_image,
                    kalman_tfm,
                    (crop_size[0], crop_size[1]))

                if save_output_images:
                    # also save kalman images
                    cv2.imwrite(osp.join(kalman_image_folder, f'{ii}.png'),
                                kalman_img)

                    if aligned_rgb_img is not None:
                        bgr_tensor_input = to_input(aligned_rgb_img)
                        feature, _ = model(bgr_tensor_input)
                        features.append(feature.detach())
                        frame_nums.append(ii)
                        warp_params.append(tfm)
                        aligned_bgr_img = np.asarray(
                            aligned_rgb_img,
                            dtype=np.uint8)[:, :, ::-1]
                        cv2.imwrite(osp.join(image_folder, f'{ii}.png'),
                                    aligned_bgr_img)

                label_index += 1
                if label_index == labels.shape[0]:
                    break
            progress_bar.update()

    progress_bar.close()

    pickle.dump(features,
                open(osp.join(output_root, 'features.pkl'), 'wb'))
    pickle.dump(frame_nums,
                open(osp.join(output_root, 'frame_nums.pkl'), 'wb'))
    pickle.dump(warp_params,
                open(osp.join(output_root, 'warp_params.pkl'), 'wb'))
    pickle.dump(kalman_filter,
                open(osp.join(output_root, 'kalman_filter.pkl'), 'wb'))
