import os
import csv

# Root directory containing all splits
root_dir = "/mnt/ifi/iis/thumos14-dataset"

# Folders to process
splits = ["TH14_TRAIN_COMPRESSED", "TH14_VALID_COMPRESSED", "TH14_TEST_COMPRESSED"]

# Output CSV
output_csv = "all_labels.csv"

# Collect rows
rows = []

for split in splits:
    folder_path = os.path.join(root_dir, split)

    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
        continue
    sorted_files = sorted(os.listdir(folder_path))

    for file_name in sorted_files:
        if file_name.endswith(".aedat4"):
            # Extract label
            if split == "TH14_VALID_COMPRESSED": #example: video_validation_0000851.aedat4
                base = file_name.split(".aedat4")[0]  # 'video_validation_0000851'
                label = base[6:]


            else:
                base = file_name.split("_g")[0]  # 'v_ApplyEyeMakeup'
                label = base[2:]  # remove 'v_'

            # Append with split info
            rows.append([file_name, label, split])

print(f"Total files found: {len(rows)}")

# Write to CSV
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["file_name", "label", "split"])
    writer.writerows(rows)

print(f"CSV saved as {output_csv}")
