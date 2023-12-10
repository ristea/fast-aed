import math

import numpy as np
import sklearn.metrics as metrics
from scipy.ndimage import convolve


class StatsManager:

    def __init__(self, config):
        self.config = config
        self.data_w = config[config['data_set']]['w']
        self.data_h = config[config['data_set']]['h']

    def get_stats(self, predictions, labels, video, frame):
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        video = np.concatenate(video)
        frame = np.concatenate(frame)

        aucs = []
        filtered_preds = []
        filtered_labels = []
        for vid in np.unique(video):
            pred = predictions[video == vid]
            pred = self.filt(pred)
            filtered_preds.append(pred)

            lbl = labels[video == vid]
            filtered_labels.append(lbl)
            fpr, tpr, _ = metrics.roc_curve(lbl, pred)
            res = metrics.auc(fpr, tpr)

            aucs.append(res)

        macro_auc = np.mean(aucs)

        # Micro-AUC
        filtered_preds = np.concatenate(filtered_preds)
        filtered_labels = np.concatenate(filtered_labels)

        fpr, tpr, _ = metrics.roc_curve(filtered_labels, filtered_preds)
        micro_auc = metrics.auc(fpr, tpr)

        return micro_auc, macro_auc

    def filt(self, input, dim=9):
        filter_3d = np.ones((dim, dim, dim)) / (dim ** 3)
        filter_2d = self.gaussian_filter(np.arange(1, 302), 21)

        frame_scores = input #convolve(input, filter_3d)
        # frame_scores = frame_scores.max((1, 2))

        padding_size = len(filter_2d) // 2
        in_ = np.concatenate((np.zeros(padding_size), frame_scores, np.zeros(padding_size)))
        frame_scores = np.correlate(in_, filter_2d, 'valid')
        return frame_scores

    def gaussian_filter(self, support, sigma):
        mu = support[len(support) // 2 - 1]
        filter = 1.0 / (sigma * np.sqrt(2 * math.pi)) * np.exp(-0.5 * ((support - mu) / sigma) ** 2)
        return filter

    def get_stats_v2(self, predictions_raw, labels, video, frame, rbdc_tbdc=False):
        if rbdc_tbdc:
            predictions = np.concatenate(predictions_raw).max((-3, -2, -1))
        else:
            predictions = np.concatenate(predictions_raw)

        labels = np.concatenate(labels)
        # video = np.concatenate(video)
        video = [item for sublist in video for item in sublist]
        frame = np.concatenate(frame)

        aucs = []
        filtered_preds = []
        filtered_labels = []
        for vid in np.unique(video):
            pred = predictions[np.array(video) == vid]
            pred = self.filt(pred)
            filtered_preds.append(pred)

            lbl = labels[np.array(video) == vid]
            filtered_labels.append(lbl)
            fpr, tpr, _ = metrics.roc_curve(lbl, pred)
            res = metrics.auc(fpr, tpr)
            res = np.nan_to_num(res, nan=1.0)

            aucs.append(res)

        macro_auc = np.mean(aucs)

        # Micro-AUC
        filtered_preds = np.concatenate(filtered_preds)
        filtered_labels = np.concatenate(filtered_labels)

        fpr, tpr, _ = metrics.roc_curve(filtered_labels, filtered_preds)
        micro_auc = metrics.auc(fpr, tpr)
        micro_auc = np.nan_to_num(micro_auc, nan=1.0)

        if rbdc_tbdc:
            # rbdc, tbdc = self.get_rbdc_tbdc(predictions_raw, video, frame)
            # return micro_auc, macro_auc, rbdc, tbdc
            return micro_auc, macro_auc, 0, 0

        return micro_auc, macro_auc
