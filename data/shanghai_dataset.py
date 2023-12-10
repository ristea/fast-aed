import numpy as np
import torch.utils.data
import glob
import os
import cv2
import torch.nn.functional as F


class ShanghaiDataset(torch.utils.data.Dataset):
    def __init__(self, config, train, train_and_eval=False):
        self.train = train
        self.train_and_eval = train_and_eval

        self.config = config
        self.data = []
        self.labels = {}
        for teacher in self.config['teachers']:
            self.labels[teacher] = {"1x1": [], "4x4": [], "16x16": []}
        self.true_labels = []

        self._read_data()

    def _read_data(self):
        already_loaded = False
        for teacher in self.config['teachers']:
            train_data_path = self.config['sh_train_data_path']
            train_labels_path = self.config[f'sh_train_labels_path_{teacher}']
            test_data_path = self.config['sh_test_data_path']
            test_labels_path = self.config[f'sh_test_labels_path_{teacher}']
            test_labels_true_path = self.config['sh_test_labels_true_path']

            if self.train is True:
                for path in glob.glob(os.path.join(train_data_path, "*")):
                    imgs_path = glob.glob(os.path.join(path, f"*{self.config['extension']}"))
                    imgs_path = sorted(imgs_path, key=lambda x: int(x.split('/')[-1].split('.')[0]))

                    if not already_loaded:
                        self.data += imgs_path
                    # Extract labels
                    for res in self.labels[teacher].keys():
                        lbls = glob.glob(os.path.join(train_labels_path, path.split('/')[-1], res, "*.npy"))
                        lbls = sorted(lbls, key=lambda x: int(x.split('/')[-1].split('.')[0]))
                        self.labels[teacher][res] += lbls
            else:
                for path in glob.glob(os.path.join(test_data_path, "*")):
                    imgs_path = glob.glob(os.path.join(path, f"*{self.config['extension']}"))
                    imgs_path = sorted(imgs_path, key=lambda x: int(x.split('/')[-1].split('.')[0]))
                    if not already_loaded:
                        self.data += imgs_path

                    # Extract labels
                    for res in self.labels[teacher].keys():
                        lbls = glob.glob(os.path.join(test_labels_path, path.split('/')[-1], res, "*.npy"))
                        lbls = sorted(lbls, key=lambda x: int(x.split('/')[-1].split('.')[0]))
                        self.labels[teacher][res] += lbls

                    # Extract true labels
                    lbls = np.loadtxt(os.path.join(test_labels_true_path, f"{os.path.basename(path)}.txt"))

                    if not already_loaded:
                        self.true_labels.append(lbls)

                if not already_loaded:
                    self.true_labels = np.concatenate(self.true_labels)

            already_loaded = True

    def __getitem__(self, index):
        frame_no = int(self.data[index].split("/")[-1].split('.')[0])
        dir_path = "/".join(self.data[index].split("/")[:-1])
        len_frame_no = len(self.data[index].split("/")[-1].split('.')[0])
        if self.train is True or self.train_and_eval is True:
            current_img = cv2.imread(self.data[index]).astype(np.float)
            previous_img = self.read_prev_next_frame_if_exists(dir_path,frame_no,direction=-3,length=len_frame_no)
            next_img = self.read_prev_next_frame_if_exists(dir_path,frame_no,direction=3,length=len_frame_no)
            label = self.get_labels(index)

            current_img = np.concatenate([current_img,previous_img,next_img],axis=-1)

            current_img = (current_img - 127.) / 128.
            current_img = np.swapaxes(current_img, 0, -1)

            return current_img, label, current_img

        else:
            current_img = cv2.imread(self.data[index]).astype(np.float)
            previous_img = self.read_prev_next_frame_if_exists(dir_path, frame_no, direction=-3,length=len_frame_no)
            next_img = self.read_prev_next_frame_if_exists(dir_path, frame_no, direction=3,length=len_frame_no)
            label = self.get_labels(index)
            true_label = self.true_labels[index]

            # lbl_s, lbl_m, lbl_l = self._get_labels(label)
            current_img = np.concatenate([current_img, previous_img, next_img], axis=-1)

            current_img = (current_img - 127.) / 128.
            current_img = np.swapaxes(current_img, 0, -1)
            return current_img, label, true_label, \
                   str(self.data[index].split('/')[-2]), \
                   int(self.data[index].split('/')[-1][:-4]), \
                   current_img

    def __len__(self):
        return len(self.data)

    def get_labels(self, index):
        labels = {}
        for teacher in self.config['teachers']:
            label1 = np.load(self.labels[teacher][list(self.labels[teacher].keys())[0]][index])
            label4 = np.load(self.labels[teacher][list(self.labels[teacher].keys())[1]][index])
            label16 = np.load(self.labels[teacher][list(self.labels[teacher].keys())[2]][index])
            label4 = np.swapaxes(label4, 0, 1)
            label16 = np.swapaxes(label16, 0, 1)

            labels[teacher] = (np.expand_dims(label1, 0),
                               np.expand_dims(label4, 0),
                               np.expand_dims(label16, 0))

        return labels

    def read_prev_next_frame_if_exists(self, dir_path, frame_no,direction=-3,length=1):
        frame_path = dir_path+"/"+str(frame_no+direction).zfill(length)+f"{self.config['extension']}"
        if os.path.exists(frame_path):
            return cv2.imread(frame_path)
        else:
            return cv2.imread(dir_path+"/"+str(frame_no).zfill(length)+f"{self.config['extension']}")

    def __repr__(self):
        return self.__class__.__name__
