import h5py
import torch
from torch.utils.data import DataLoader, Dataset
import os
import sys
import augmented_tsn
from augmented_tsn import AugmentedTsn
import h5py
import torchvision.transforms as T


CLASS_NAMES = [
    "LongJump",       # 0
    "ApplyMakeup",    # 1
    "Archery",        # 2
    "BabyCrawling",   # 3
    "BalanceBeam",    # 4
    "Nothing"         # 5 — no samples during training
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
transform = T.Compose([
    T.Resize((224, 224)),
])

class HistogramDataset(Dataset):
    def __init__(self, h5_path):
        self.h5 = h5py.File(h5_path, "r")
        self.keys = [
            k for k in self.h5.keys()
            if (
                "split" in self.h5[k].attrs and
                self.h5[k].attrs["split"].upper().startswith("TH14_TRAIN") and
                self.h5[k].attrs["label"] in CLASS_TO_IDX and
                self.h5[k].attrs["label"] != "Nothing"  # Skip samples of class "Nothing"
            )
        ]
        print(f"Loaded {len(self.keys)} samples from: {h5_path}")
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        grp = self.h5[self.keys[idx]]
        hist = grp["histogram"][:]
        label_str = grp.attrs["label"]
        label = CLASS_TO_IDX[label_str]
        hist_tensor = torch.tensor(hist, dtype=torch.float32)
        hist_tensor = torch.cat([hist_tensor, hist_tensor[0:1]], dim=0)  # still (3, H, W)
        #print(f"Loaded sample {idx}: label={label_str}, histogram shape={hist_tensor.shape}")
        hist_tensor = self.transform(hist_tensor)
        #print(f"Transformed histogram shape: {hist_tensor.shape}")
        return hist_tensor, torch.tensor(label, dtype=torch.long)

dataset = HistogramDataset("/home/gabriel.kofler/Project/event_penguins/scripts/preprocessed_output_training/preprocessed_train.h5")
print(f"Dataset length: {len(dataset)}")
loader = DataLoader(dataset, batch_size=8, shuffle=True)
model = AugmentedTsn(num_classes=6, num_tsn_samples=5, augment_factor=3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()
NUM_EPOCHS = 10

for epoch in range(NUM_EPOCHS):
    for imgs, labels in loader:
        imgs = imgs.unsqueeze(1).repeat(1, model.num_tsn_samples, 1, 1, 1)

        out = model(imgs)
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss = loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {running_loss}")
    print(f"Epoch {epoch}, Loss: {loss.item()}")


torch.save(model.state_dict(), "augmented_tsn_model.pth")

