import os
import pickle
from enum import Enum

import cv2 as cv
import numpy as np
from sklearn import metrics


class TrackState(Enum):
    CREATED = "created"
    UPDATED = "updated"
    CLOSED = "closed"


class Track:

    def __init__(self, start_idx=0, end_idx=None, mask=0, video_name=""):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.bboxes = {}
        self.mask = mask
        self.state = TrackState.CREATED
        self.video_name = video_name

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


class AnomalyDetection:

    def __init__(self, frame_idx, bbox, score, video_name, track_id=-1):
        self.frame_idx = frame_idx
        self.bbox = bbox
        self.score = score
        self.video_name = video_name
        self.track_id = track_id

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def get_loc_v3(video_info_path):
    file_names = os.listdir(os.path.join(video_info_path, "meta_0.800"))
    video_loc_v3 = []
    for file_name in file_names:
        loc_v3 = np.loadtxt(os.path.join(video_info_path, "meta_0.800", file_name))

        video_loc_v3.append(loc_v3[:5])
    return video_loc_v3


def get_predicted_anomalies_per_video(output_path, video_name, size):
    """
    :param output_path
    :param video_name
    :param size = H, W
    """

    # compute anomaly detection from predicted heat map
    loc_v3 = np.load(os.path.join(args.output_folder_base, args.database_name, "test",
                                  video_name, "loc_v3_%f.npy" % args.lambda_))
    # locv3 format [[frame_idx, x_min, y_min, x_max, y_max]]

    ab_event = np.load(os.path.join(args.output_folder_base, args.database_name, "test",
                                    video_name, "ab_event3_%f.npy" % args.lambda_))

    ab_event_resized = []
    for i in range(ab_event.shape[2]):
        res = cv.resize(ab_event[:, :, i], (size[1], size[0]))
        ab_event_resized.append(res)

    pred_anomalies_detected = []
    for idx in range(len(loc_v3)):
        frame_idx = int(loc_v3[idx][0])
        bbox = loc_v3[idx][1:]
        bbox = [int(b) for b in bbox]
        crop_frame = ab_event_resized[frame_idx][bbox[1]: bbox[3], bbox[0]: bbox[2]]
        pred_anomalies_detected.append(AnomalyDetection(frame_idx, bbox, crop_frame.max(), video_name))

    return pred_anomalies_detected


def save_txt_predicted(preds, video_name):
    predictions = []
    for pred in preds:  # [frame_id, x_min, y_min, x_max, y_max, anomaly_score]
        predictions.append([pred.frame_idx] + pred.bbox + [pred.score])
    np.savetxt(f'avenue/det/{video_name}.txt', predictions, delimiter=',')


def get_all_predicted_anomalies(output_path):
    video_names = os.listdir(output_path)
    video_names.sort()
    pred_anomalies = []
    num_frames = 0
    for video_name in video_names:
        if os.path.isfile(os.path.join(output_path, video_name)):
            continue
        video_meta_data = pickle.load(open(os.path.join(output_path, video_name, "video_meta_data.pkl"), 'rb'))
        video_size = (video_meta_data['height'], video_meta_data['width'])  # h, w
        num_frames_video = video_meta_data['num_frames']
        pred = get_predicted_anomalies_per_video(output_path, video_name, video_size)
        # save_txt_predicted(pred, video_name)
        pred_anomalies += pred
        num_frames += num_frames_video

    return pred_anomalies, num_frames


def compute_iou(pred_anomaly, gt_anomalies_per_frame):
    max_iou = 0
    idx = -1
    for index, gt_anomaly in enumerate(gt_anomalies_per_frame):
        iou = bb_intersection_over_union(gt_anomaly.bbox, pred_anomaly.bbox)
        if max_iou < iou:
            max_iou = iou
            idx = index

    return max_iou, idx


def get_matching_gt_indices(pred_anomaly, gt_anomalies_per_frame, beta):
    indices = []
    for index, gt_anomaly in enumerate(gt_anomalies_per_frame):
        iou = bb_intersection_over_union(gt_anomaly.bbox, pred_anomaly.bbox)
        if iou >= beta:
            indices.append(index)

    return indices


def compute_tbdr(gt_tracks, num_matched_detections_per_track, alpha):
    percentages = np.array([x / len(y.bboxes) for x, y in zip(num_matched_detections_per_track, gt_tracks)])
    return np.sum(percentages >= alpha) / len(num_matched_detections_per_track)


def compute_fpr_rbdr(pred_anomalies_detected: [AnomalyDetection], gt_anomalies: [AnomalyDetection], all_gt_tracks,
                     num_frames, num_tracks, alpha=0.1, beta=0.1):
    num_matched_detections_per_track = [0] * num_tracks

    # TODO: add pixel level IOU
    num_detected_anomalies = len(pred_anomalies_detected)
    gt_anomaly_video_per_frame_dict = {}
    found_gt_anomaly_video_per_frame_dict = {}

    for anomaly in gt_anomalies:
        anomalies_per_frame = gt_anomaly_video_per_frame_dict.get((anomaly.video_name, anomaly.frame_idx), None)
        if anomalies_per_frame is None:
            gt_anomaly_video_per_frame_dict[(anomaly.video_name, anomaly.frame_idx)] = [anomaly]
            found_gt_anomaly_video_per_frame_dict[(anomaly.video_name, anomaly.frame_idx)] = [0]
        else:
            gt_anomaly_video_per_frame_dict[(anomaly.video_name, anomaly.frame_idx)].append(anomaly)
            found_gt_anomaly_video_per_frame_dict[(anomaly.video_name, anomaly.frame_idx)].append(0)

    tp = np.zeros(num_detected_anomalies)
    fp = np.zeros(num_detected_anomalies)
    tbdr = np.zeros(num_detected_anomalies)
    remove_idx = []
    pred_anomalies_detected.sort(key=lambda anomaly_detection: anomaly_detection.score, reverse=True)
    for idx, pred_anomaly in enumerate(pred_anomalies_detected):
        gt_anomalies_per_frame = gt_anomaly_video_per_frame_dict.get((pred_anomaly.video_name, pred_anomaly.frame_idx),
                                                                     None)

        if gt_anomalies_per_frame is None:
            fp[idx] = 1
        else:
            matching_gt_bboxes_indices = get_matching_gt_indices(pred_anomaly, gt_anomalies_per_frame, beta)
            if len(matching_gt_bboxes_indices) > 0:
                non_matched_indices = []
                for matched_ind in matching_gt_bboxes_indices:
                    if found_gt_anomaly_video_per_frame_dict.get((pred_anomaly.video_name,
                                                                  pred_anomaly.frame_idx))[matched_ind] == 0:
                        non_matched_indices.append(matched_ind)
                        found_gt_anomaly_video_per_frame_dict.get((pred_anomaly.video_name, pred_anomaly.frame_idx))[
                            matched_ind] = 1
                        num_matched_detections_per_track[gt_anomalies_per_frame[matched_ind].track_id] += 1

                tp[idx] = len(non_matched_indices)

            else:
                fp[idx] = 1

        tbdr[idx] = compute_tbdr(all_gt_tracks, num_matched_detections_per_track, alpha)

    cum_false_positive = np.cumsum(fp)
    cum_true_positive = np.cumsum(tp)
    # add the point (0, 0) for each vector
    cum_false_positive = np.concatenate(([0], cum_false_positive))
    cum_true_positive = np.concatenate(([0], cum_true_positive))
    tbdr = np.concatenate(([0], tbdr))

    rbdr = cum_true_positive / len(gt_anomalies)
    fpr = cum_false_positive / num_frames

    idx_1 = np.where(fpr <= 1)[0][-1] + 1

    if fpr[idx_1 - 1] != 1:
        print('fpr does not reach 1')
        rbdr = np.insert(rbdr, idx_1, rbdr[idx_1 - 1])
        tbdr = np.insert(tbdr, idx_1, tbdr[idx_1 - 1])
        fpr = np.insert(fpr, idx_1, 1)
        idx_1 += 1

    tbdc = metrics.auc(fpr[:idx_1], tbdr[:idx_1])
    rbdc = metrics.auc(fpr[:idx_1], rbdr[:idx_1])

    print('tbdc = ' + str(tbdc))
    print('rbdc = ' + str(rbdc))
    return rbdc, tbdc

    # print(tbdr[idx_1 - 1], rbdr[idx_1 - 1])
    # plt.plot(fpr, rbdr, '-')
    # plt.xlabel('FPR')
    # plt.ylabel('RBDR')
    # plt.show()


def save_tracks_as_txt(tracks, video_name):
    regions = []
    for track_id, track in enumerate(tracks):
        for frame_idx, bbox in track.bboxes.items():
            regions.append([track_id] + [frame_idx] + bbox)
    np.savetxt(f'avenue/tracks/{video_name}.txt', regions, delimiter=',')


def compute_rbdc_tbdc_func(pred_anomalies_detected, num_frames):
    data_set_path = 'test'
    output_path = 'test'

    all_gt_tracks = []
    num_tracks = 0

    video_names = os.listdir(output_path)
    video_names.sort()

    for video_name in video_names:
        if video_name.endswith(".csv"):
            continue
        tracks = pickle.load(open(os.path.join(data_set_path, 'tracks', video_name + '.pkl'), 'rb'))
        # save_tracks_as_txt(tracks, video_name)
        all_gt_tracks += tracks
        num_tracks += len(tracks)
    gt_anomalies = []

    for track_id, track in enumerate(all_gt_tracks):
        for frame_idx, bbox in track.bboxes.items():
            gt_anomalies.append(AnomalyDetection(frame_idx, bbox, 1, track.video_name, track_id=track_id))

    return compute_fpr_rbdr(pred_anomalies_detected, gt_anomalies, all_gt_tracks, num_frames, num_tracks)
