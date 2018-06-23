import torch
import torch.nn as nn


class BinaryCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, logits, labels):
        loss = self.loss(logits, labels)
        return loss


class HuberLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.SmoothL1Loss()

    def forward(self, logits, labels):
        loss = self.loss(logits, labels)
        return loss
