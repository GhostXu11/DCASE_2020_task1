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


class ClassifierModule10_2path(nn.Module):
    def __init__(self, backbone10, backbone3, fcnn):
        super(ClassifierModule10_2path, self).__init__()
        if not fcnn:
            self.net_low_freq = models.__dict__[backbone10](pretrained=False)
            self.net_high_freq = models.__dict__[backbone10](pretrained=False)
            self.class3 = models.__dict__[backbone3](pretrained=False)
        else:
            self.net_low_freq =



class FCNNModel(nn.Module):
    def __init__(self, channels, num_filters=8, output_features=10):
        super(FCNNModel, self).__init__()
        # layer 1
        self.conv_layer1 = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ZeroPad2d(padding=2),
            nn.Conv2d(in_channels=channels, out_channels=num_filters * channels,
                      kernel_size=5, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(num_filters * channels),
            nn.ReLU(),
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=num_filters * channels, out_channels=num_filters * channels, kernel_size=3, stride=1,
                      padding=0, bias=True),
            nn.BatchNorm2d(num_filters * channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0)
        )

        # layer 2
        self.conv_layer2 = nn.Sequential(
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=num_filters * channels, out_channels=num_filters * 2 * channels,
                      kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_filters * 2 * channels),
            nn.ReLU(),
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=num_filters * 2 * channels, out_channels=num_filters * 2 * channels, kernel_size=3,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(num_filters * 2 * channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0)
        )

        # layer 3
        self.conv_layer3 = nn.Sequential(
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=num_filters * 2 * channels, out_channels=num_filters * 4 * channels,
                      kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_filters * 4 * channels),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=num_filters * 4 * channels, out_channels=num_filters * 4 * channels, kernel_size=3,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(num_filters * 4 * channels),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=num_filters * 4 * channels, out_channels=num_filters * 4 * channels, kernel_size=3,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(num_filters * 4 * channels),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=num_filters * 4 * channels, out_channels=num_filters * 4 * channels, kernel_size=3,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(num_filters * 4 * channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0),
        )

        self.resnet_layer = nn.Sequential(
            nn.Conv2d(in_channels=num_filters * 4 * channels, out_channels=output_features, kernel_size=3, stride=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(output_features),
            nn.ReLU(),
            nn.BatchNorm2d(output_features)
        )

        self.dense1 = nn.Sequential(
            nn.Linear(in_features=output_features, out_features=int(np.ceil(output_features / 2)), bias=True),
            nn.ReLU()
        )

        self.dense2 = nn.Sequential(
            nn.Linear(in_features=int(np.ceil(output_features / 2)), out_features=output_features, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.resnet_layer(x)
        avg_pool = F.avg_pool2d(x, kernel_size=(x.shape[2], x.shape[3]))
        avg_pool = torch.reshape(avg_pool, (x.shape[0], 1, 1, x.shape[1]))
        avg_pool = self.dense1(avg_pool)
        avg_pool = self.dense2(avg_pool)

        max_pool = F.max_pool2d(x, kernel_size=(x.shape[2], x.shape[3]))
        max_pool = torch.reshape(max_pool, (x.shape[0], 1, 1, x.shape[1]))
        max_pool = self.dense1(max_pool)
        max_pool = self.dense2(max_pool)

        cbam_feature = torch.add(avg_pool, max_pool)





class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

