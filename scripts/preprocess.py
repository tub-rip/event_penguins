# Create the data split with this script.
# The script crops the events according to the regions of interest,
# and saves them as .npz in folder according to the data split
import argparse
import os

import h5py
import yaml
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        help="Data path root (one-level above of the recordings)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory, where processed data is saved.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--recording_info_path",
        help="Path to csv file listing recording informations.",
        type=str,
        default="config/recording_info.csv",
    )
    args = parser.parse_args()
    return args


def crop_events_bbox(events: np.ndarray, bbox: tuple) -> np.ndarray:
    """Crop n events in a bounding box.

    Args:
        events: [n x 4]. [x, y, t, p].
        bbox: [x1, y1, x2, y2]

    Returns:
        cropped [m x 4]
    """
    x1, y1, x2, y2 = bbox
    mask = (
        (x1 <= events[..., 0])
        * (events[..., 0] < x2)
        * (y1 <= events[..., 1])
        * (events[..., 1] < y2)
    )
    cropped = events[mask]
    cropped[:, 0] -= x1
    cropped[:, 1] -= y1
    return cropped


def get_rois(roi_config_dir, roi_group_id):
    config_path = os.path.join(roi_config_dir, f"roi_{roi_group_id}.yaml")
    with open(config_path, "r") as f:
        roi_data = yaml.safe_load(f)
        roi_data.pop("Background", None)
    rois = {}
    for roi_id, value in roi_data.items():
        r = value["bbox"]
        rois[roi_id] = (r["xmin"], r["ymin"], r["xmax"], r["ymax"])
    return rois


def process_recording(
    h5f, data_root, recording_time, recording_id, split, roi_group_id
):
    print(f"Converting recording {recording_id}: {recording_time}")
    events_path = os.path.join(data_root, recording_time, "events.h5")

    if not os.path.exists(events_path):
        print(f"{events_path} does not exist, skipping...")
        return

    rois = get_rois("config/annotations/rois", roi_group_id)

    with h5py.File(events_path, "r") as f:
        events = np.stack((f["x"], f["y"], f["t"], f["p"])).T
        events = events[events[:, 2].argsort()]

        # Annotations are defined relative to time of first event
        # therefore, setting event times accordingly
        events[:, 2] = events[:, 2] - events[0, 2]

        # Recordings are 10min. DAVIS assigns sometimes wrong timestamp at beginning.
        # Delete these.
        events = events[events[:, 2] < 600 * 1e6]

        cropped = {}

        for roi_id, roi in rois.items():
            x1, y1, x2, y2 = roi
            cropped[roi_id] = {
                "events": crop_events_bbox(events, roi),
                "height": y2 - y1,
                "width": x2 - x1,
            }

    grp = h5f.create_group(recording_time)
    grp.attrs["split"] = split

    for roi_id, data in cropped.items():
        subgrp = grp.create_group(roi_id)
        subgrp.create_dataset("events", data=data["events"])
        subgrp.attrs["height"] = data["height"]
        subgrp.attrs["width"] = data["width"]


if __name__ == "__main__":
    args = parse_args()
    df = pd.read_csv(args.recording_info_path)
    print("Start conversion.")
    output_file_path = os.path.join(args.output_dir, "preprocessed.h5")

    with h5py.File(output_file_path, "w") as h5f:
        df.apply(
            lambda row: process_recording(
                h5f,
                args.data_root,
                row["timestamp"],
                row["recording_id"],
                row["split"],
                row["roi_group_id"],
            ),
            axis=1,
        )
    print("Done.")
