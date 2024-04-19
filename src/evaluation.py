import json

import numpy as np
import pandas as pd
from absl import logging

try:
    from joblib import Parallel, delayed

    joblib_parallelization = True
except:
    joblib_parallelization = False


class DetectionsEvaluator(object):
    GROUND_TRUTH_FIELDS = ["database", "version"]
    PREDICTION_FIELDS = ["results", "version"]

    def __init__(
        self,
        ground_truth_filename=None,
        prediction_filename=None,
        ground_truth_fields=GROUND_TRUTH_FIELDS,
        prediction_fields=PREDICTION_FIELDS,
        tiou_thresholds=np.linspace(0.5, 0.95, 10),
        verbose=False,
        valid_sequences=None,
        valid_roi_ids=None,
        valid_labels=None,
        min_duration: float = 2.0,
    ):
        if not ground_truth_filename:
            raise IOError("Please input a valid ground truth file.")
        if not prediction_filename:
            raise IOError("Please input a valid prediction file.")
        self.tiou_thresholds = tiou_thresholds
        self.verbose = verbose
        self.gt_fields = ground_truth_fields
        self.pred_fields = prediction_fields
        self.ap = None
        # Import ground truth and predictions.

        self.ground_truth, self.activity_index = self._import_ground_truth(
            ground_truth_filename,
            valid_sequences,
            valid_roi_ids,
            valid_labels,
            min_duration,
        )
        self.prediction = self._import_prediction(
            prediction_filename,
            valid_sequences,
            valid_roi_ids,
            valid_labels,
            min_duration,
        )

        if self.verbose:
            print("[INIT] Loaded annotations from {} subset.".format(subset))
            nr_gt = len(self.ground_truth)
            print("\tNumber of ground truth instances: {}".format(nr_gt))
            nr_pred = len(self.prediction)
            print("\tNumber of predictions: {}".format(nr_pred))
            print("\tFixed threshold for tiou score: {}".format(self.tiou_thresholds))

    def _import_ground_truth(
        self,
        ground_truth_filename,
        valid_sequences,
        valid_roi_ids,
        valid_labels,
        min_duration,
    ):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        with open(ground_truth_filename, "r") as fobj:
            data = json.load(fobj)
        # Checking format
        if not all([field in data.keys() for field in self.gt_fields]):
            raise IOError("Please input a valid ground truth file.")

        # Read ground truth data.
        activity_index, cidx = {}, 0
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
        for videoid, v in data["database"].items():
            if valid_sequences is not None and videoid not in valid_sequences:
                continue
            for roi_id, roi_annotations in v["annotations"].items():
                if roi_id == "null":
                    continue
                if valid_roi_ids is not None and int(roi_id) not in valid_roi_ids:
                    continue
                for ann in roi_annotations:
                    if valid_labels is not None and ann["label"] not in valid_labels:
                        continue
                    duration = ann["segment"][1] - ann["segment"][0]
                    if duration < min_duration:
                        continue
                    if ann["label"] not in activity_index:
                        activity_index[ann["label"]] = cidx
                        cidx += 1
                    video_lst.append(f"{videoid}_{roi_id}")
                    t_start_lst.append(float(ann["segment"][0]))
                    t_end_lst.append(float(ann["segment"][1]))
                    label_lst.append(activity_index[ann["label"]])

        ground_truth = pd.DataFrame(
            {
                "video-id": video_lst,
                "t-start": t_start_lst,
                "t-end": t_end_lst,
                "label": label_lst,
            }
        )
        return ground_truth, activity_index

    def _import_prediction(
        self,
        prediction_filename,
        valid_sequences,
        valid_roi_ids,
        valid_labels,
        min_duration,
    ):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.

        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.

        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        with open(prediction_filename, "r") as fobj:
            data = json.load(fobj)
        # Checking format...
        if not all([field in data.keys() for field in self.pred_fields]):
            raise IOError("Please input a valid prediction file.")

        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, score_lst = [], []
        for videoid, v in data["results"].items():
            if valid_sequences is not None and videoid not in valid_sequences:
                continue
            for roi_id, roi_annotation in v.items():
                if roi_id == "null":
                    continue
                if valid_roi_ids is not None and int(roi_id) not in valid_roi_ids:
                    continue
                for result in roi_annotation:
                    if valid_labels is not None and result["label"] not in valid_labels:
                        continue
                    duration = result["segment"][1] - result["segment"][0]
                    if duration < min_duration:
                        continue
                    label = self.activity_index[result["label"]]
                    video_lst.append(f"{videoid}_{roi_id}")
                    t_start_lst.append(float(result["segment"][0]))
                    t_end_lst.append(float(result["segment"][1]))
                    label_lst.append(label)
                    score_lst.append(result["score"])
        prediction = pd.DataFrame(
            {
                "video-id": video_lst,
                "t-start": t_start_lst,
                "t-end": t_end_lst,
                "label": label_lst,
                "score": score_lst,
            }
        )
        return prediction

    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label.
        """
        try:
            return prediction_by_label.get_group(cidx).reset_index(drop=True)
        except:
            print("Warning: No predictions of label '%s' were provdied." % label_name)
            return pd.DataFrame()

    def wrapper_compute_average_precision(self):
        """Computes average precision for each class in the subset."""
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby("label")
        prediction_by_label = self.prediction.groupby("label")

        if joblib_parallelization:
            results = Parallel(n_jobs=len(self.activity_index))(
                delayed(compute_average_precision_detection)(
                    ground_truth=ground_truth_by_label.get_group(cidx).reset_index(
                        drop=True
                    ),
                    prediction=self._get_predictions_with_label(
                        prediction_by_label, label_name, cidx
                    ),
                    tiou_thresholds=self.tiou_thresholds,
                )
                for label_name, cidx in self.activity_index.items()
            )
        else:
            results = []
            for label_name, cidx in self.activity_index.items():
                results.append(
                    compute_average_precision_detection(
                        ground_truth=ground_truth_by_label.get_group(cidx).reset_index(
                            drop=True
                        ),
                        prediction=self._get_predictions_with_label(
                            prediction_by_label, label_name, cidx
                        ),
                        tiou_thresholds=self.tiou_thresholds,
                    )
                )

        for i, cidx in enumerate(self.activity_index.values()):
            ap[:, cidx] = results[i]

        return ap

    def wrapper_compute_average_recall(self, max_pred=None):
        """Computes average recall for each class in the subset."""
        ar = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby("label")
        prediction_by_label = self.prediction.groupby("label")

        if joblib_parallelization:
            results = Parallel(n_jobs=len(self.activity_index))(
                delayed(compute_average_recall)(
                    ground_truth=ground_truth_by_label.get_group(cidx).reset_index(
                        drop=True
                    ),
                    prediction=self._get_predictions_with_label(
                        prediction_by_label, label_name, cidx
                    ),
                    tiou_thresholds=self.tiou_thresholds,
                    max_pred=max_pred,
                )
                for label_name, cidx in self.activity_index.items()
            )
        else:
            results = []
            for label_name, cidx in self.activity_index.items():
                results.append(
                    compute_average_recall(
                        ground_truth=ground_truth_by_label.get_group(cidx).reset_index(
                            drop=True
                        ),
                        prediction=self._get_predictions_with_label(
                            prediction_by_label, label_name, cidx
                        ),
                        tiou_thresholds=self.tiou_thresholds,
                        max_pred=max_pred,
                    )
                )

        for i, cidx in enumerate(self.activity_index.values()):
            ar[:, cidx] = results[i]

        return ar

    def run(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        self.ap = self.wrapper_compute_average_precision()

        self.mAP = self.ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()

        if self.verbose:
            print("[RESULTS] Performance on ActivityNet detection task.")
            print("\tAverage-mAP: {}".format(self.average_mAP))

        return self.average_mAP

    def evaluate_recall(self, max_pred=None):
        # shape: [num_tiou_thresholds, num_classes]
        self.ar = self.wrapper_compute_average_recall(max_pred=max_pred)
        labels = [label for label in self.activity_index]
        ar = pd.DataFrame(self.ar, columns=labels)
        ar.insert(0, "tiou_threshold", self.tiou_thresholds)
        return ar


def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.

    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (
        (candidate_segments[:, 1] - candidate_segments[:, 0])
        + (target_segment[1] - target_segment[0])
        - segments_intersection
    )
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011."""
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap


def compute_average_precision_detection(
    ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)
):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        logging.warning("Evaluator returned 0 due to empty predictions!")
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction["score"].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby("video-id")

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred["video-id"])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(
            this_pred[["t-start", "t-end"]].values, this_gt[["t-start", "t-end"]].values
        )
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]["index"]] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]["index"]] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(
            precision_cumsum[tidx, :], recall_cumsum[tidx, :]
        )

    return ap


def compute_average_recall(
    ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10), max_pred=None
):
    if prediction.empty:
        logging.warning("Evaluator returned 0 due to empty predictions!")
        return np.zeros(len(tiou_thresholds))

    # Filter videos as top x
    video_ids = ground_truth["video-id"].unique()
    filtered_prediction = None
    for video_id in video_ids:
        vid_prediction = prediction[prediction["video-id"] == video_id].reset_index(
            drop=True
        )
        sort_idx = vid_prediction["score"].values.argsort()[::-1]
        vid_prediction = vid_prediction.loc[sort_idx].reset_index(drop=True)
        vid_prediction = vid_prediction.head(max_pred).reset_index(drop=True)
        if filtered_prediction is None:
            filtered_prediction = vid_prediction
        else:
            filtered_prediction = pd.concat(
                [filtered_prediction, vid_prediction], ignore_index=True
            )

    prediction = filtered_prediction
    prediction_gbvn = prediction.groupby("video-id")

    tp = np.zeros(len(tiou_thresholds), dtype=int)

    for idx, this_gt in ground_truth.iterrows():
        if not this_gt["video-id"] in prediction_gbvn.groups.keys():
            logger.warning(
                f"No predictions for sequence {this_gt['video-id']}. Skipping."
            )
            continue
        pred_videoid = prediction_gbvn.get_group(this_gt["video-id"])
        this_pred = pred_videoid.reset_index()

        tiou_arr = segment_iou(
            this_gt[["t-start", "t-end"]].values, this_pred[["t-start", "t-end"]].values
        )

        max_tiou = np.max(tiou_arr)
        tp += tiou_thresholds <= max_tiou

    ar = tp / len(ground_truth)
    return ar
