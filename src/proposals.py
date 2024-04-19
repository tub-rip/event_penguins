import os
from multiprocessing import Pool
import pickle

import numpy as np
from absl import logging
import pandas as pd
import h5py

from .utils import temporal_nms


def get_event_rate(events, bin_width):
    t_min, t_max = events[0, 2], events[-1, 2]
    bin_num = int((t_max - t_min) / bin_width)
    counts, bins = np.histogram(events[:, 2], bins=bin_num)
    return counts, bins


def apply_robust_min_max(rate, percentile):
    rmin = np.percentile(rate.flat, 0.5 * percentile)
    rmax = np.percentile(rate.flat, 100 - 0.5 * percentile)
    rate[rate < rmin] = rmin
    rate[rate > rmax] = rmax
    return rate


def get_index_proposals_from_1d_score(
    score1d: np.ndarray, threshold: float
) -> np.ndarray:
    """
    Generates proposals from a 1D array by returning a (n x 2) array of intervals where
    "score1D" is consecutively above "threshold"

    Args:
        score1d: a 1D array of scores
        threshold: the threshold for the watershed algorithm

    Returns:
        (n x 2) array of n proposals defined by (start_index, end_index)
    """
    return np.where(np.diff(score1d > threshold, prepend=0, append=0))[0].reshape(-1, 2)


def check_merge_possible(proposal_1, proposal_2, basin_durations, threshold):
    basin_durations += proposal_2[1] - proposal_2[0]
    merged_duration = proposal_2[1] - proposal_1[0]

    if basin_durations / merged_duration > threshold:
        return True
    else:
        return False


def merge_proposals(unmerged, score, grouping_thres, times):
    merged = []
    current = None
    accumulated_basin_durations = 0

    for next_proposal in unmerged:
        if current is None:
            current = next_proposal
            accumulated_basin_durations += next_proposal[1] - next_proposal[0]
        else:
            do_merge = check_merge_possible(
                current, next_proposal, accumulated_basin_durations, grouping_thres
            )
            if do_merge:
                current[1] = next_proposal[1]
                accumulated_basin_durations += next_proposal[1] - next_proposal[0]

            if not do_merge or (next_proposal == unmerged[-1]).all():
                t_start = times[current[0]]
                t_end = times[current[1]]

                merged.append(
                    [
                        t_start,
                        t_end,
                        np.mean(score[current[0] : current[1]]),
                    ]
                )

                current = next_proposal
                accumulated_basin_durations = 0

    return merged


class ProposalGenerator:
    """
    Generates action proposals based on temporal actioness scores across
    multiple regions of interest (ROI) within recordings.

    Attributes:
        data_dir (str): Directory containing the preprocessed data.
        bin_width (float): Width of the bins used for event rate calculation [us].
        percentile (float): Percentile used for robust scaling of event rates.
        nms_threshold (float): Threshold for non-maximal suppression.
    """

    def __init__(self, data_path, bin_width, percentile, nms_threshold) -> None:
        self.data_path = data_path
        self.bin_width = bin_width * 1e6  # us
        self.percentile = percentile
        self.nms_threshold = nms_threshold
        self.actioness_thresholds = np.arange(0.05, 1, 0.05)
        self.grouping_thresholds = np.arange(0.05, 1, 0.05)

    def process_recording(self, rec):
        """
        Processes a single recording to generate action proposals for each ROI.

        Args:
            rec (str): The name of the recording to process.

        Returns:
            pd.DataFrame: A dataframe with proposals.
        """
        rec_proposal_data = {}

        with h5py.File(self.data_path, "r") as file:
            data = file[rec]

            roi_ids = data.keys()

            actioness_scores = {}

            for roi_id in roi_ids:
                events = np.array(data[roi_id]["events"])

                rate, bins = get_event_rate(events, self.bin_width)
                rate = apply_robust_min_max(rate, self.percentile)

                min_rate, max_rate = np.min(rate), np.max(rate)
                actioness = (rate - min_rate) / (max_rate - min_rate)

                actioness_scores[roi_id] = actioness

                proposals = []

                for at in self.actioness_thresholds:
                    for gt in self.grouping_thresholds:
                        unmerged = get_index_proposals_from_1d_score(actioness, at)
                        proposals += merge_proposals(unmerged, actioness, gt, bins)

                proposals = np.array(proposals)
                proposals = proposals[proposals[:, 1] - proposals[:, 0] > 2 * 1e6]
                proposals = temporal_nms(proposals, self.nms_threshold)
                rec_proposal_data[roi_id] = proposals

        return rec_proposal_data

    def run(self):
        """
        Processes all recordings in the data directory to generate action proposals.

        Returns:
            dict: Nested dictionary with recording names as keys and dictionaries
            (from process_recording) as values.
        """
        logging.info("Running Proposal Generator.")

        with h5py.File(self.data_path, "r") as f:
            recordings = [rec for rec in f.keys() if f[rec].attrs["split"] == "test"]

        proposal_data = {}

        with Pool(processes=16) as pool:
            results = pool.map(self.process_recording, recordings)

        for rec, rec_proposals in zip(recordings, results):
            rec_name = os.path.splitext(rec)[0]
            proposal_data[rec_name] = rec_proposals

        proposal_df = {
            "rec_name": [],
            "roi_id": [],
            "t_start": [],
            "t_end": [],
            "score": [],
        }

        for rec_name, rec_proposals in proposal_data.items():
            for roi_id, roi_proposals in rec_proposals.items():
                for proposal in roi_proposals:
                    proposal_df["rec_name"].append(rec_name)
                    proposal_df["roi_id"].append(roi_id)
                    proposal_df["t_start"].append(proposal[0])
                    proposal_df["t_end"].append(proposal[1])
                    proposal_df["score"].append(proposal[2])

        return pd.DataFrame(proposal_df)
