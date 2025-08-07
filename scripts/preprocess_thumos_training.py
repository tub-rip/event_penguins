import argparse
import os
import numpy as np
import h5py
import pandas as pd
import aedat  # Make sure aedat4 library is installed
import matplotlib.pyplot as plt

def read_aedat4_new(file_path):
    decoder = aedat.Decoder(file_path)

    all_events = []
    for packet in decoder:
        if "events" in packet:
            events = packet["events"]
            timestamps = events["t"]
            x_coords = events["x"]
            y_coords = events["y"]
            polarities = events["p"]

            stacked = np.rec.fromarrays(
                [x_coords, y_coords, timestamps, polarities],
                dtype=[("x", "int16"), ("y", "int16"), ("t", "int64"), ("p", "int8")]
            )
            all_events.append(stacked)

    if len(all_events) == 0:
        return np.empty((0,), dtype=[("x", "int16"), ("y", "int16"), ("t", "int64"), ("p", "int8")])

    all_events_array = np.concatenate(all_events)
    all_events_array.sort(order='t')
    return all_events_array


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Root dir with split folders")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save histograms")
    parser.add_argument("--recording_info_path", type=str, required=True, help="Path to all_labels.csv")
    return parser.parse_args()


def events_to_histogram(events, height=480, width=640, polarity_split=True):
    if polarity_split:
        hist = np.zeros((2, height, width), dtype=np.int32)
        for p in (0, 1):
            mask = events['p'] == p
            xs = events['x'][mask]
            ys = events['y'][mask]
            np.add.at(hist[p], (ys, xs), 1)
    else:
        hist = np.zeros((height, width), dtype=np.int32)
        xs = events['x']
        ys = events['y']
        np.add.at(hist, (ys, xs), 1)

    plt.imshow(hist[0])
    plt.show()

    return hist


def process_recording(h5f, data_root, file_name, label, split):
    file_path = os.path.join(data_root, split, file_name)
    print(f"Processing {file_path}...")

    if not os.path.exists(file_path):
        print(f"File {file_path} missing. Skipping...")
        return

    events = read_aedat4_new(file_path)
    if len(events) == 0:
        print(f"No events found in {file_path}. Skipping...")
        return

    print(f"Found {len(events)} events in {file_name}")
    events['t'] -= events['t'][0]  # Normalize time
    events = events[events['t'] < 600 * 1e6]  # Remove corrupted timestamps

    histogram = events_to_histogram(events)

    grp_name = os.path.splitext(file_name)[0]
    grp = h5f.create_group(grp_name)
    grp.attrs["label"] = label
    grp.attrs["split"] = split
    grp.create_dataset("histogram", data=histogram)


def main():
    args = parse_args()
    df = pd.read_csv(args.recording_info_path)
    df = df[df["split"].str.upper().str.contains("TRAIN")]

    os.makedirs(args.output_dir, exist_ok=True)
    output_file_path = os.path.join(args.output_dir, "preprocessed_train.h5")

    # Changed from 'w' to 'a' so existing file is not overwritten
    with h5py.File(output_file_path, "a") as h5f:
        for _, row in df.iterrows():
            file_name = row["file_name"]
            grp_name = os.path.splitext(file_name)[0]

            #  Skip if already processed
            if grp_name in h5f and "histogram" in h5f[grp_name]:
                print(f"Group '{grp_name}' already exists. Skipping...")
                continue

            #  Optional: avoid crash from 1 bad file
            try:
                process_recording(
                    h5f,
                    args.data_root,
                    file_name,
                    row["label"],
                    row["split"]
                )
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    print("Training split done!")


if __name__ == "__main__":
    main()
