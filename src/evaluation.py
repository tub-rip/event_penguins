import json

import numpy as np
import pandas as pd
from absl import logging

try:
    from joblib import Parallel, delayed
    joblib_parallelization = True
except ImportError:
    joblib_parallelization = False


def segment_iou(target_segment, candidate_segments):
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    inter = (tt2 - tt1).clip(0)
    union = (
        (candidate_segments[:, 1] - candidate_segments[:, 0])
        + (target_segment[1] - target_segment[0])
        - inter
    )
    return inter.astype(float) / union


def interpolated_prec_rec(prec, rec):
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap


def compute_average_precision_detection(
    ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)
):
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        logging.warning("Evaluator returned all zeros due to empty predictions!")
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1
    sort_idx = prediction["score"].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    gt_by_video = ground_truth.groupby("video-id")

    for idx, pred in prediction.iterrows():
        vid = pred["video-id"]
        if vid not in gt_by_video.groups:
            fp[:, idx] = 1
            continue
        this_gt = gt_by_video.get_group(vid).reset_index(drop=True)
        tiou_arr = segment_iou(
            pred[["t-start", "t-end"]].values,
            this_gt[["t-start", "t-end"]].values,
        )
        order = tiou_arr.argsort()[::-1]
        for t, thr in enumerate(tiou_thresholds):
            assigned = False
            for j in order:
                if tiou_arr[j] < thr:
                    break
                if lock_gt[t, j] >= 0:
                    continue
                tp[t, idx] = 1
                lock_gt[t, j] = idx
                assigned = True
                break
            if not assigned:
                fp[t, idx] = 1

    tp_cum = np.cumsum(tp, axis=1).astype(float)
    fp_cum = np.cumsum(fp, axis=1).astype(float)
    rec = tp_cum / npos
    prec = tp_cum / (tp_cum + fp_cum)

    for i in range(len(tiou_thresholds)):
        ap[i] = interpolated_prec_rec(prec[i], rec[i])
    return ap


def compute_average_recall(
    ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10), max_pred=None
):
    if prediction.empty:
        logging.warning("Evaluator returned all zeros due to empty predictions!")
        return np.zeros(len(tiou_thresholds))

    gt_videos = ground_truth["video-id"].unique()
    filtered = []
    for vid in gt_videos:
        preds = prediction[prediction["video-id"] == vid]
        if preds.empty:
            continue
        preds = preds.sort_values("score", ascending=False)
        if max_pred:
            preds = preds.head(max_pred)
        filtered.append(preds)
    if not filtered:
        return np.zeros(len(tiou_thresholds))
    prediction = pd.concat(filtered, ignore_index=True)
    pred_by_video = prediction.groupby("video-id")

    tp = np.zeros(len(tiou_thresholds), dtype=int)
    for _, gt in ground_truth.iterrows():
        vid = gt["video-id"]
        if vid not in pred_by_video.groups:
            continue
        preds = pred_by_video.get_group(vid).reset_index(drop=True)
        tiou_arr = segment_iou(
            gt[["t-start", "t-end"]].values,
            preds[["t-start", "t-end"]].values,
        )
        max_tiou = tiou_arr.max() if len(tiou_arr) > 0 else 0
        tp += tiou_thresholds <= max_tiou

    return tp / len(ground_truth)


class DetectionsEvaluator:
    def __init__(
        self,
        ground_truth_filename,
        prediction_filename,
        tiou_thresholds=np.linspace(0.5, 0.95, 10),
        verbose=False,
        valid_sequences=None,
        valid_labels=None,
        min_duration: float = 0
    ):
        self.tiou_thresholds = tiou_thresholds
        self.verbose = verbose

        self.gt = self._import_ground_truth(
            ground_truth_filename, valid_sequences, valid_labels, min_duration
        )
        self.pred = self._import_prediction(
            prediction_filename, valid_sequences, valid_labels, min_duration
        )

    def _import_ground_truth(
        self, filename, valid_sequences, valid_labels, min_dur
    ):
        data = json.load(open(filename))
        gt_list = []
        for vid, info in data["database"].items():
            if valid_sequences and vid not in valid_sequences:
                continue
            for ann in info.get("annotations", []):
                lbl = ann["label"]
                if valid_labels and lbl not in valid_labels:
                    continue
                t0, t1 = ann["segment"]
                if (t1 - t0) < min_dur:
                    continue
                gt_list.append({
                    "video-id": vid,
                    "t-start": float(t0),
                    "t-end": float(t1),
                    "label": lbl
                })
        return pd.DataFrame(gt_list)

    def _import_prediction(
        self, filename, valid_sequences, valid_labels, min_dur
    ):
        data = json.load(open(filename))
        pred_list = []
        for vid, anns in data["results"].items():
            if valid_sequences and vid not in valid_sequences:
                continue
            for ann in anns:
                lbl = ann["label"]
                if valid_labels and lbl not in valid_labels:
                    continue
                t0, t1 = ann["segment"]
                if (t1 - t0) < min_dur:
                    continue
                pred_list.append({
                    "video-id": vid,
                    "t-start": float(t0),
                    "t-end": float(t1),
                    "label": lbl,
                    "score": float(ann.get("score", 1.0))
                })
        return pd.DataFrame(pred_list)

    def run(self):
        ap = compute_average_precision_detection(
            self.gt, self.pred, self.tiou_thresholds
        )
        mAP = ap.mean()
        if self.verbose:
            print(f"mAP: {mAP:.4f}")
        return mAP

    def evaluate_recall(self, max_pred=None):
        ar = compute_average_recall(
            self.gt, self.pred, self.tiou_thresholds, max_pred
        )
        return ar
