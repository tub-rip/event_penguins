import numpy as np
import h5py
import pandas as pd






import pandas as pd

def load_and_save_txt_to_csv(txt_file, csv_file):
    """
    Load data from a text file and save it to a CSV file.
    Args:
        txt_file (str): Path to the input text file.
        csv_file (str): Path to the output CSV file.
    """
    # Optional: Manual fallback parsing
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            video_name, start_time, end_time = parts
            data.append([video_name, float(start_time), float(end_time)])
        else:
            print(f"Invalid line skipped: {line}")

    # Load into DataFrame and save
    df = pd.DataFrame(data, columns=["video_name", "start_time", "end_time"])
    df.to_csv(csv_file, index=False)
    print(f"Data loaded from {txt_file} and saved to {csv_file}")



def main():
    list_txt_files = [
        "/home/gabriel.kofler/Project/ground_truth/annotation/LongJump_test.txt",
        "/home/gabriel.kofler/Project/ground_truth/annotation/Diving_test.txt",
        "/home/gabriel.kofler/Project/ground_truth/annotation/FrisbeeCatch_test.txt",
        "/home/gabriel.kofler/Project/ground_truth/annotation/ThrowDiscus_test.txt",
        "/home/gabriel.kofler/Project/ground_truth/annotation/HighJump_test.txt"
    ]

    
    for txt_file in list_txt_files:
        csv_file = txt_file.replace(".txt", ".csv")
        csv_file = csv_file.replace("annotation", "csv")
        load_and_save_txt_to_csv(txt_file, csv_file)


if __name__ == "__main__":
    main()
    print("Script executed successfully.")
    print("All text files have been processed and saved as CSV files.")
    