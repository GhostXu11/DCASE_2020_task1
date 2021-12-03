import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np


class ClassifierModule10(nn.Module):
    def __init__(self, backbone):
        super(ClassifierModule10, self).__init__()
        self.net_low_freq = models.__dict__[backbone](pretrained=False)
        self.net_high_freq = models.__dict__[backbone](pretrained=False)
        self.fc_class_vec = nn.Linear(1000, 10)

    def forward(self, x):
        # pass through net
        n_freq = x.shape[2]
        x_low_freq = self.net_low_freq(x[:, :, :int(n_freq / 2), :])
        x_high_freq = self.net_high_freq(x[:, :, int(n_freq / 2):, :])
        x_low_freq = self.fc_class_vec(x_low_freq)
        x_high_freq = self.fc_class_vec(x_high_freq)
        # fc to 10 labels
        y = x_low_freq * 0.25 + x_high_freq * 0.75
        return y


# net = ClassifierModule10(backbone='resnet18')
