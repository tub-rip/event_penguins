import torch
from torch import nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np


class Consensus(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = 1

    def forward(self, x):
        return x.mean(dim=self.dim, keepdim=True)


class AugmentedTsn(nn.Module):
    def __init__(self, num_classes, num_tsn_samples=3, augment_factor=3) -> None:
        super().__init__()

        self.num_tsn_samples = num_tsn_samples
        self.num_augment = int(np.ceil(num_tsn_samples / augment_factor))
        backbone = models.resnet18()

        self.backbone = create_feature_extractor(
            backbone, return_nodes={"layer4.1.relu_1": "features"}
        )

        # Dry run to get number of channels
        with torch.no_grad():
            inp = torch.randn(6, 3, 224, 224)
            out = self.backbone(inp)["features"]
        in_channels = out.shape[1]

        self.num_classes = num_classes
        self.in_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.consensus = Consensus(dim=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc_cls = nn.Linear(3 * self.in_channels, self.num_classes)
        nn.init.xavier_uniform_(self.fc_cls.weight)

    def forward(self, x):
        # B x N x C x H x W -> B * N x C x H x W
        num_segs = x.shape[1]
        x = x.reshape((-1,) + x.shape[2:])
        x = self.backbone(x)["features"]
        # here we want feature map
        # e.g. (48 x 2048 x 7 x 7)
        x = self.avg_pool(x)
        x = x.reshape((-1, num_segs) + x.shape[1:])

        a = self.num_augment
        start_stage = x[:, 0:a]
        main_stage = x[:, a : (num_segs - a)]
        end_stage = x[:, num_segs - a : num_segs]

        start_stage = self.consensus(start_stage)
        start_stage = start_stage.squeeze(1)

        x = self.consensus(main_stage)
        x = x.squeeze(1)

        end_stage = self.consensus(end_stage)
        end_stage = end_stage.squeeze(1)

        x = torch.cat((start_stage, x, end_stage), dim=1)

        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score
