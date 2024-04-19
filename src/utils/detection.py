import numpy as np


def temporal_iou(proposal_min, proposal_max, gt_min, gt_max):
    """
    Compute IoU score between a groundtruth bbox and the proposals.

    Args:
        proposal_min: List of temporal anchor min.
        proposal_max: List of temporal anchor max.
        gt_min: Groundtruth temporal box min.
        gt_max: Groundtruth temporal box max.

    Returns:
        list[float]: List of iou scores.
    """
    len_anchors = proposal_max - proposal_min
    int_tmin = np.maximum(proposal_min, gt_min)
    int_tmax = np.minimum(proposal_max, gt_max)
    inter_len = np.maximum(int_tmax - int_tmin, 0.0)
    union_len = len_anchors - inter_len + gt_max - gt_min
    jaccard = np.divide(inter_len, union_len)
    return jaccard


def temporal_nms(detections: np.ndarray, threshold: float) -> np.ndarray:
    """
    Perform 1D non-maximum suppression on n detections

    Args:
        detections: Detection results before NMS (n x 3).
                    Each detection has form (t_start, t_end, score)
        threshold: Threshold of NMS.

    Returns:
        Detection results after NMS.
    """
    starts = detections[:, 0]
    ends = detections[:, 1]
    scores = detections[:, 2]

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ious = temporal_iou(starts[order[1:]], ends[order[1:]], starts[i], ends[i])
        idxs = np.where(ious <= threshold)[0]
        order = order[idxs + 1]

    return detections[keep, :]
