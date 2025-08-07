import pandas as pd
import json
import glob
import os

# Replace with your actual directory path
csv_dir = "/home/gabriel.kofler/Project/ground_truth/csv"
output_json = "/home/gabriel.kofler/Project/ground_truth/ground_truth.json"

# Collect all CSV files
csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))

database = {}

for csv_file in csv_files:
    df = pd.read_csv(csv_file)

    for _, row in df.iterrows():
        video_name = row["video_name"]
        label = csv_file.split("/")[-1].split("_")[0]  # Extract label from filename
        print(f"Processing video: {video_name}, label: {label}")
        t_start = float(row["start_time"])
        t_end = float(row["end_time"])

        if video_name not in database:
            database[video_name] = {"annotations": []}

        database[video_name]["annotations"].append({
            "label": label,
            "segment": [t_start, t_end]
        })

# Final JSON structure
final_json = {
    "version": "VERSION 0.0",
    "database": database
}

# Write to JSON file
with open(output_json, "w") as f:
    json.dump(final_json, f, indent=2)

print(f"Saved ground truth JSON to: {output_json}")
