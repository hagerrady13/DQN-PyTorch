import torch
import torch.nn as nn
import torch.nn.Functional as F

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

#loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
