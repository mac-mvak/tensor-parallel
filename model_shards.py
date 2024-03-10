import torch
import torch.nn as nn
import torch.nn.functional as F

from gathers import GatherGrad1, Gather1


class ModelShard(nn.Module):
    """
    A very simple model with minimal changes from the tutorial, used for the sake of simplicity.
    Feel free to replace it with EffNetV2-XL once you get comfortable injecting SyncBN into models programmatically.
    """

    def __init__(self, rank, size):
        super().__init__()
        self.seq1 = nn.Sequential(
            GatherGrad1(rank, size),
            nn.Conv2d(3, 32 // size, 3, 1),
            nn.ReLU(),
            Gather1(rank, size)
        )

        self.seq2 = nn.Sequential(
            GatherGrad1(rank, size),
            nn.Conv2d(32, 32 // size, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            Gather1(rank, size)
        )

        self.seq3 = nn.Sequential(
            GatherGrad1(rank, size),
            nn.Flatten(1),
            nn.Linear(6272, 128//size),
            nn.BatchNorm1d(128//size),
            nn.ReLU(),
            nn.Dropout(0.5),
            Gather1(rank, size)
        )

        self.seq4 = nn.Sequential(
            GatherGrad1(rank, size),
            nn.Linear(128, 100//size),
            Gather1(rank, size)
            )

    def forward(self, x):
        x = self.seq1(x)
        
        x = self.seq2(x)


        x = self.seq3(x)
        output = self.seq4(x)
        return output