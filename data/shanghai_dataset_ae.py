import copy

import numpy as np
import torch.utils.data
import glob
import os
import cv2


class ShanghaiDatasetAE(torch.utils.data.Dataset):
    def __init__(self, config,):
        self.config = config
        self.data = []
        self.true_labels = []
        self._read_data()

    def _read_data(self):
        for path in glob.glob(os.path.join(self.config['train_data_path'], "*")):
            imgs_path = glob.glob(os.path.join(path, f"*{self.config['extension']}"))
            imgs_path = sorted(imgs_path, key=lambda x: int(x.split('/')[-1].split('.')[0]))

            self.data += imgs_path

    def __getitem__(self, index):
        frame_no = int(self.data[index].split("/")[-1].split('.')[0])
        dir_path = "/".join(self.data[index].split("/")[:-1])
        len_frame_no = len(self.data[index].split("/")[-1].split('.')[0])

        current_img = cv2.imread(self.data[index]).astype(np.float)
        current_img = cv2.resize(current_img, (640, 360), interpolation=cv2.INTER_CUBIC)
        previous_img = self.read_prev_next_frame_if_exists(dir_path, frame_no, direction=-3, length=len_frame_no)
        previous_img = cv2.resize(previous_img, (640, 360), interpolation=cv2.INTER_CUBIC)
        next_img = self.read_prev_next_frame_if_exists(dir_path, frame_no, direction=3, length=len_frame_no)
        next_img = cv2.resize(next_img, (640, 360), interpolation=cv2.INTER_CUBIC)

        current_img = (current_img - 127.) / 128.
        previous_img = (previous_img - 127.) / 128.
        next_img = (next_img - 127.) / 128.
        label = copy.deepcopy(current_img)
        current_img = np.swapaxes(current_img, 0, -1)
        previous_img = np.swapaxes(previous_img, 0, -1)
        next_img = np.swapaxes(next_img, 0, -1)
        all_3_images = np.concatenate([current_img, previous_img, next_img], axis=0)

        label = cv2.resize(label, (320, 180), interpolation=cv2.INTER_CUBIC)
        label = np.swapaxes(label, 0, -1)
        return all_3_images, label

    def read_prev_next_frame_if_exists(self, dir_path, frame_no, direction=-3, length=1):
        frame_path = dir_path + "/" + str(frame_no + direction).zfill(length) + f"{self.config['extension']}"
        if os.path.exists(frame_path):
            return cv2.imread(frame_path)
        else:
            return cv2.imread(dir_path + "/" + str(frame_no).zfill(length) + f"{self.config['extension']}")

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return self.__class__.__name__
