from torch import nn
import torch
import torch.nn.functional as F
import numpy as np


class SpliNetBackbone(nn.Module):
    """Backbone extracted from https://github.com/dros1986/neural_spline_enhancement/blob/ce99d4e9cb57f2524d8409b6a47b86b1e0b778e6/NeuralSpline.py#L17"""

    def __init__(
        self,
        n=10,
        nc=8,
        nexperts=1,
        downsample_strategy="avgpool",
        dropout=0.0,
        n_input_channels=3,
        n_output_channels=3,
    ):
        super().__init__()
        # define class params
        self.n = n
        self.dropout = dropout
        momentum = 0.01

        # define net layers
        self.c1 = nn.Conv2d(
            n_input_channels, nc, kernel_size=5, stride=4, padding=0
        )
        self.c2 = nn.Conv2d(nc, 2 * nc, kernel_size=3, stride=2, padding=0)
        self.b2 = nn.BatchNorm2d(2 * nc, momentum=momentum)
        self.c3 = nn.Conv2d(2 * nc, 4 * nc, kernel_size=3, stride=2, padding=0)
        self.b3 = nn.BatchNorm2d(4 * nc, momentum=momentum)
        self.c4 = nn.Conv2d(4 * nc, 8 * nc, kernel_size=3, stride=2, padding=0)
        self.b4 = nn.BatchNorm2d(8 * nc, momentum=momentum)
        # define downsample layers
        if downsample_strategy == "maxpool":
            self.downsample = nn.MaxPool2d(7, stride=1)
            self.fc = nn.ModuleList([])
            for i in range(nexperts):
                self.fc.append(
                    nn.Sequential(
                        nn.Linear(8 * nc, 8 * nc),
                        nn.ReLU(True),
                        nn.Linear(8 * nc, n_output_channels * n),
                    )
                )
        elif downsample_strategy == "avgpool":
            self.downsample = nn.AvgPool2d(7, stride=1)
            self.fc = nn.ModuleList([])
            for i in range(nexperts):
                self.fc.append(
                    nn.Sequential(
                        nn.Linear(8 * nc, 8 * nc),
                        nn.ReLU(True),
                        nn.Linear(8 * nc, n_output_channels * n),
                    )
                )
        elif downsample_strategy == "convs":
            self.downsample = nn.Sequential(
                nn.Conv2d(8 * nc, 16 * nc, kernel_size=3, stride=2, padding=0),
                nn.BatchNorm2d(16 * nc, momentum=momentum),
                nn.ReLU(True),
                nn.Conv2d(
                    16 * nc, 32 * nc, kernel_size=3, stride=2, padding=0
                ),
                nn.BatchNorm2d(32 * nc, momentum=momentum),
                nn.ReLU(True),
            )
            self.fc = nn.ModuleList([])
            for i in range(nexperts):
                self.fc.append(
                    nn.Sequential(
                        nn.Linear(32 * nc, 16 * nc),
                        nn.ReLU(True),
                        nn.Linear(16 * nc, n_output_channels * n),
                    )
                )
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(8 * nc, 8 * nc, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(8 * nc, momentum=momentum),
                nn.ReLU(True),
                nn.Conv2d(8 * nc, n_output_channels * n, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(3 * n, momentum=momentum),
                nn.ReLU(True),
                nn.AvgPool2d(7, stride=1),
            )
            self.fc = nn.ModuleList([])
            for i in range(nexperts):
                self.fc.append(lambda x: x)
        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.n_output_channels = n_output_channels

    def forward(self, batch):
        assert batch.size(2) == 256 and batch.size(3) == 256
        # get xs of the points with CNN
        ys = F.relu(self.c1(batch))
        ys = self.b2(F.relu(self.c2(ys)))
        ys = self.b3(F.relu(self.c3(ys)))
        ys = self.b4(F.relu(self.c4(ys)))
        ys = self.downsample(ys)
        ys = ys.view(ys.size(0), -1)
        if self.dropout > 0.0 and self.training:
            ys = F.dropout(ys, p=self.dropout, training=self.training)
        ys = torch.cat(
            [l(ys).view(-1, self.n_output_channels, self.n).unsqueeze(1) for l in self.fc], 1
        )  # nexp*(bs,3*n) -> (bs,nexp,3,n)
        return ys

class GammaBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones((1)))

    def forward(self, batch):
        return self.gamma
