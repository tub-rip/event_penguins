import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from absl import logging
from torch.utils.data import Dataset, DataLoader
import h5py

from .augmented_tsn import AugmentedTsn
from .utils import temporal_nms


def range_norm(matrix, new_max=255, lower=None, upper=None, dtype=None):
    if lower is None:
        lower = np.min(matrix)
    if upper is None:
        upper = np.max(matrix)

    matrix = np.clip(matrix, lower, upper)

    scaled = new_max * ((matrix - lower) / (upper - lower))
    if dtype is not None:
        scaled = scaled.astype(dtype)
    return scaled


def create_time_map(events, decay, height, width):
    time_map = np.zeros((height, width))
    time_map[events[:, 1], events[:, 0]] = events[:, 2]

    current_t = np.amax(events[:, 2]) if len(events) > 0 else 0
    time_map = np.exp(-decay * (current_t - time_map))

    polarity = np.copy(events[:, 3]).astype(int)
    polarity[np.where(events[:, 3] == 0)] = -1
    time_map[events[:, 1], events[:, 0]] *= polarity

    return time_map


def create_img_representation(events, decay, height, width, transforms=None):
    img = create_time_map(events, decay, height, width)
    img = range_norm(img, lower=-1, upper=1, dtype=np.uint8)
    img = np.repeat(img[..., None], 3, axis=2)
    img = Image.fromarray(img)
    img = img.resize((224, 224), resample=Image.BILINEAR)
    img = np.array(img)

    if transforms is not None:
        img = transforms(img)

    return img


class ProposalDataset(Dataset):
    def __init__(
        self,
        proposals,
        augment_fraction,
        data_path,
        num_tsn_samples,
        sample_duration,
        decay,
    ):
        self.proposals = proposals
        self.augment_fraction = augment_fraction
        self.data_path = data_path
        self.num_tsn_samples = num_tsn_samples
        self.sample_duration = sample_duration
        self.decay = decay

        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.proposals)

    def __getitem__(self, idx):
        t_start = self.proposals.loc[idx, "t_start"]
        t_end = self.proposals.loc[idx, "t_end"]
        rec_name = self.proposals.loc[idx, "rec_name"]
        roi_id = self.proposals.loc[idx, "roi_id"]

        with h5py.File(self.data_path, "r") as file:
            roi_events = np.array(file[rec_name][roi_id]["events"])
            height = file[rec_name][roi_id].attrs["height"]
            width = file[rec_name][roi_id].attrs["width"]

        # Augment
        t_delta = t_end - t_start
        t_start_aug = t_start - t_delta * self.augment_fraction
        t_end_aug = t_end + t_delta * self.augment_fraction

        # Determine times where image representation is build
        img_times = torch.linspace(t_start_aug, t_end_aug, self.num_tsn_samples)

        # Build image representations at those times
        t_imgs_start = img_times - 0.5 * self.sample_duration
        i_imgs_start = np.searchsorted(roi_events[:, 2], t_imgs_start)

        t_imgs_end = img_times + 0.5 * self.sample_duration
        i_imgs_end = np.searchsorted(roi_events[:, 2], t_imgs_end)

        imgs = []

        for i_start, i_end in zip(i_imgs_start, i_imgs_end):
            events = roi_events[i_start:i_end]
            imgs.append(
                create_img_representation(
                    events, self.decay, height, width, self.transforms
                )
            )

        imgs = torch.stack(imgs)
        return imgs, rec_name, roi_id, t_start, t_end


class ProposalClassifier:
    def __init__(
        self,
        device,
        model_path,
        num_tsn_samples,
        augment_factor,
        data_path,
        sample_duration,
        decay,
        nms_threshold,
        batch_size,
    ) -> None:
        self.device = device
        self.augment_fraction = 1 / augment_factor
        num_samples_augmented = np.ceil(self.augment_fraction * num_tsn_samples)

        # Proposal is augmented on each side and additional samples are considered
        # within the augmentated times
        self.num_tsn_samples = num_tsn_samples + int(2 * num_samples_augmented)

        self.data_path = data_path
        self.model = AugmentedTsn(2, num_tsn_samples, augment_factor)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device).eval()

        self.sample_duration = 1e6 * sample_duration  # [s] -> [us]
        self.decay = float(decay)

        self.nms_threshold = nms_threshold
        self.batch_size = batch_size

    def run(self, proposals):
        logging.info("Running Proposal Classifier.")

        # Data Preparation
        dataset = ProposalDataset(
            proposals,
            self.augment_fraction,
            self.data_path,
            self.num_tsn_samples,
            self.sample_duration,
            self.decay,
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=16
        )

        result = {}
        rec_names = proposals["rec_name"].unique()

        for rec in rec_names:
            result[rec] = {}
            rec_proposals = proposals[proposals["rec_name"] == rec]
            roi_ids = rec_proposals["roi_id"].unique()

            for roi_id in roi_ids:
                result[rec][roi_id] = []

        # Prediction
        with torch.no_grad():
            for batch in tqdm(loader):
                imgs, rec_names, roi_ids, start_times, end_times = batch
                outputs = self.model(imgs.to(self.device))

                _, preds = torch.max(outputs, 1)
                ed_scores = torch.nn.Softmax(dim=1)(outputs)[:, 1]

                for i, pred in enumerate(preds):
                    if pred.item():
                        rec_name, roi_id = rec_names[i], roi_ids[i]
                        result[rec_name][roi_id].append(
                            [
                                float(start_times[i]),
                                float(end_times[i]),
                                float(ed_scores[i]),
                            ]
                        )

        # Non-maximum Suppression
        nmsed_result = {}

        for rec_name, rec_results in result.items():
            nmsed_result[rec_name] = {}

            for roi_id, roi_result in rec_results.items():
                processed = (
                    temporal_nms(np.array(roi_result), self.nms_threshold)
                    if roi_result
                    else []
                )

                nmsed_result[rec_name][int(roi_id[1:])] = [
                    {
                        "label": "ed",
                        "segment": [action[0] / 1e6, action[1] / 1e6],
                        "score": action[2],
                    }
                    for action in processed
                ]

        return {"version": "VERSION 0.0", "results": nmsed_result}
