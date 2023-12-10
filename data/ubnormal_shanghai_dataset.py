import numpy as np
import torch.utils.data
import glob
import os
import cv2


class UbnormalShanghaiDataset(torch.utils.data.Dataset):
    def __init__(self, config, train, train_and_eval=False):
        self.train = train
        self.train_and_eval = train_and_eval

        self.config = config
        self.data = []
        self.labels = {1:[],4:[],16:[]}
        self.true_labels = []

        self._read_data()

    def _read_data(self):
        if self.train is True:
            self.add_data_labels('ub_train_data_path', 'ub_train_labels_path', 'ub_train_labels_true_path')
            self.add_data_labels('sh_train_data_path', 'sh_train_labels_path', 'sh_train_labels_true_path', True)

        else:
            self.add_data_labels('sh_test_data_path', 'sh_test_labels_path', 'sh_test_labels_true_path', True)

    def add_data_labels(self, data_path, labels_path, true_labels_path=None, do_concatenate=False):
        for path in glob.glob(os.path.join(self.config[data_path], "*")):
            imgs_path = glob.glob(os.path.join(path, "*.jpg"))
            imgs_path = sorted(imgs_path, key=lambda x: int(x.split('/')[-1].split('.')[0]))
            self.data += imgs_path

            # Extract labels
            for res in self.labels.keys():
                lbls = glob.glob(os.path.join(self.config[labels_path], path.split('/')[-1],
                                              str(res) + "x" + str(res), "*.npy"))
                lbls = sorted(lbls, key=lambda x: int(x.split('/')[-1].split('.')[0]))
                self.labels[res] += lbls
            if true_labels_path is not None:
                lbls = np.loadtxt(os.path.join(self.config[true_labels_path], f"{path.split('/')[-1]}.txt"))
                self.true_labels.append(lbls)
        if do_concatenate:
            self.true_labels = np.concatenate(self.true_labels,dtype=np.float32)

    def __getitem__(self, index):
        frame_no = int(self.data[index].split("/")[-1].split('.')[0])
        dir_path = "/".join(self.data[index].split("/")[:-1])
        len_frame_no = len(self.data[index].split("/")[-1].split('.')[0])

        if self.train is True:
            current_img = cv2.imread(self.data[index]).astype(np.float)
            if current_img.shape[0] != 856:
                current_img = cv2.resize(current_img, (856, 480), interpolation=cv2.INTER_CUBIC)
            current_img = current_img.astype(np.float)
            previous_img = self.read_prev_next_frame_if_exists(dir_path, frame_no, direction=-3, length=len_frame_no)
            next_img = self.read_prev_next_frame_if_exists(dir_path, frame_no, direction=3, length=len_frame_no)
            labels = self.get_labels(index)
            current_img = (current_img - 127.) / 128.
            previous_img = (previous_img - 127.) / 128.
            next_img = (next_img - 127.) / 128.
            current_img = np.swapaxes(current_img, 0, -1)
            previous_img = np.swapaxes(previous_img, 0, -1)
            next_img = np.swapaxes(next_img, 0, -1)
            # label1 = np.swapaxes(label1, 0, 1)
            # label = 5 * np.expand_dims(label, 0)
            current_img = np.concatenate([current_img, previous_img, next_img], axis=0)
            return current_img, labels

        else:
            current_img = cv2.imread(self.data[index]).astype(np.float)
            if current_img.shape[0] != 856:
                current_img = cv2.resize(current_img, (856, 480), interpolation=cv2.INTER_CUBIC)
            current_img = current_img.astype(np.float)
            previous_img = self.read_prev_next_frame_if_exists(dir_path, frame_no, direction=-3, length=len_frame_no)
            next_img = self.read_prev_next_frame_if_exists(dir_path, frame_no, direction=3, length=len_frame_no)

            labels = self.get_labels(index)
            # lbl_s, lbl_m, lbl_l = self._get_labels(label)

            current_img = (current_img - 127.) / 128.
            previous_img = (previous_img - 127.) / 128.
            next_img = (next_img - 127.) / 128.
            current_img = np.swapaxes(current_img, 0, -1)
            previous_img = np.swapaxes(previous_img, 0, -1)
            next_img = np.swapaxes(next_img, 0, -1)
            # label1 = np.swapaxes(label1, 0, 1)
            # label = np.expand_dims(label, 0)
            current_img = np.concatenate([current_img, previous_img, next_img], axis=0)
            return current_img, labels, labels[-1], \
                   str(self.data[index].split('/')[-2]), \
                   int(self.data[index].split('/')[-1][:-4])

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return self.__class__.__name__

    def read_prev_next_frame_if_exists(self, dir_path, frame_no, direction=-3, length=1):
        frame_path_zfill = dir_path + "/" + str(frame_no + direction).zfill(length) + ".jpg"
        frame_path = dir_path + "/" + str(frame_no + direction) + ".jpg"
        if os.path.exists(frame_path_zfill):
            frame = cv2.imread(frame_path_zfill)
            if frame.shape[0] != 856:
                frame = cv2.resize(frame, (856, 480), interpolation=cv2.INTER_CUBIC)
            frame = frame.astype(np.float)

            return frame
        elif os.path.exists(frame_path):
            frame = cv2.imread(frame_path)
            if frame.shape[0] != 856:
                frame = cv2.resize(frame, (856, 480), interpolation=cv2.INTER_CUBIC)
            frame = frame.astype(np.float)
            return frame
        elif os.path.exists(dir_path + "/" + str(frame_no) + ".jpg"):
            frame = cv2.imread(dir_path + "/" + str(frame_no) + ".jpg")
            if frame.shape[0] != 856:
                frame = cv2.resize(frame, (856, 480), interpolation=cv2.INTER_CUBIC)
            frame = frame.astype(np.float)

            return frame
        else:
            frame = cv2.imread(dir_path + "/" + str(frame_no).zfill(length) + ".jpg")
            if frame.shape[0] != 856:
                frame = cv2.resize(frame, (856, 480), interpolation=cv2.INTER_CUBIC)
            frame = frame.astype(np.float)

            return frame

    def get_labels(self,index):
        label1 = np.load(self.labels[1][index])
        label4 = np.load(self.labels[4][index])
        label16 = np.load(self.labels[16][index])
        label4 = np.swapaxes(label4, 0, 1)
        label16 = np.swapaxes(label16, 0, 1)
        true_label = self.true_labels[index]

        return label1, label4, label16, true_label
