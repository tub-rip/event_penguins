# Modified ProposalGenerator class and its dependencies to work on full-frame histograms
# instead of ROI-specific event groups

import os
from multiprocessing import Pool
import pickle

import numpy as np
from absl import logging
import pandas as pd
import h5py

from .utils import temporal_nms  # Make sure this exists and works correctly


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


def get_index_proposals_from_1d_score(score1d: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(np.diff(score1d > threshold, prepend=0, append=0))[0].reshape(-1, 2)


def check_merge_possible(proposal_1, proposal_2, basin_durations, threshold):
    basin_durations += proposal_2[1] - proposal_2[0]
    merged_duration = proposal_2[1] - proposal_1[0]
    return (basin_durations / merged_duration) > threshold


def merge_proposals(unmerged, score, grouping_thres, times):
    merged = []
    current = None
    accumulated_basin_durations = 0

    for next_proposal in unmerged:
        if current is None:
            current = next_proposal
            accumulated_basin_durations += next_proposal[1] - next_proposal[0]
        else:
            do_merge = check_merge_possible(current, next_proposal, accumulated_basin_durations, grouping_thres)
            if do_merge:
                current[1] = next_proposal[1]
                accumulated_basin_durations += next_proposal[1] - next_proposal[0]

            if not do_merge or (next_proposal == unmerged[-1]).all():
                t_start = times[current[0]]
                t_end = times[current[1]]
                merged.append([t_start, t_end, np.mean(score[current[0]:current[1]])])
                current = next_proposal
                accumulated_basin_durations = 0

    return merged


class ProposalGenerator:
    def __init__(self, data_path, bin_width, percentile, nms_threshold) -> None:
        self.data_path = data_path
        self.bin_width = bin_width * 1e6  # microseconds
        self.percentile = percentile
        self.nms_threshold = nms_threshold
        self.actioness_thresholds = np.arange(0.05, 1, 0.05)
        self.grouping_thresholds = np.arange(0.05, 1, 0.05)

    def process_recording(self, rec):
        with h5py.File(self.data_path, "r") as file:
            if "events" not in file[rec]:
                print(f"No 'events' dataset found in recording {rec}. Skipping...")
                return []

            events = np.array(file[rec]["events"])

        rate, bins = get_event_rate(events, self.bin_width)
        rate = apply_robust_min_max(rate, self.percentile)
        actioness = (rate - np.min(rate)) / (np.max(rate) - np.min(rate))

        proposals = []
        for at in self.actioness_thresholds:
            for gt in self.grouping_thresholds:
                unmerged = get_index_proposals_from_1d_score(actioness, at)
                proposals += merge_proposals(unmerged, actioness, gt, bins)

        proposals = np.array(proposals)
        proposals = proposals[proposals[:, 1] - proposals[:, 0] > 2 * 1e6]  # duration > 2ms
        proposals = temporal_nms(proposals, self.nms_threshold)
        return proposals

    def run(self):
        logging.info("Running Proposal Generator (Full-frame).")

        with h5py.File(self.data_path, "r") as f:
            recordings = [rec for rec in f.keys() if f[rec].attrs["split"] == "test"]

        proposal_df = {
            "rec_name": [],
            "t_start": [],
            "t_end": [],
            "score": [],
        }

        with Pool(processes=16) as pool:
            results = pool.map(self.process_recording, recordings)

        for rec_name, proposals in zip(recordings, results):
            for proposal in proposals:
                proposal_df["rec_name"].append(rec_name)
                proposal_df["t_start"].append(proposal[0])
                proposal_df["t_end"].append(proposal[1])
                proposal_df["score"].append(proposal[2])

        return pd.DataFrame(proposal_df)
