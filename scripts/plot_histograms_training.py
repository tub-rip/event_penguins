import h5py
import matplotlib.pyplot as plt

# Path to your HDF5 file
h5_file_path = "preprocessed_output_training/preprocessed_train.h5"

with h5py.File(h5_file_path, "r") as h5f:
    for group_name in h5f:
        group = h5f[group_name]
        
        # Check if histogram exists
        if "histogram" in group:
            histogram = group["histogram"][:]
            print(f"Processing group: {group_name}, Label: {group.attrs['label']}, Split: {group.attrs['split']}")
            print(f"Histogram shape: {histogram.shape}")
            positive_polarity_histogram = histogram[0]  # Assuming first dimension is positive polarity
            negative_polarity_histogram = histogram[1]  # Assuming second dimension is negative
            print(f"Positive Polarity Histogram: {positive_polarity_histogram}")
            print(f"Negative Polarity Histogram: {negative_polarity_histogram}")
            # Plot positive and negative polarity histograms
     
            plt.figure()
            plt.title(f"Histogram: {group_name}")
            plt.plot(positive_polarity_histogram, label='Positive Polarity', color='blue')
            plt.plot(negative_polarity_histogram, label='Negative Polarity', color='red')
            plt.xlabel("Bin")
            plt.ylabel("Value")
            plt.grid(True)
            plt.legend()
            plt.show()
            plt.close()

           
# I have 2 dimensional histograms, so we need to adjust the plotting accordingly
