import torch
from torch import nn
import scipy
import numpy as np


def dct2(x):
    return scipy.fftpack.dct(scipy.fftpack.dct(x, axis=-1, norm='ortho'), axis=-2, norm='ortho')


class DCTLayer(nn.Module):
    def __init__(self, mean, std):
        super(DCTLayer, self).__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32).clone().detach()
        self.std = torch.tensor(std, dtype=torch.float32).clone().detach()

    def forward(self, inputs):
        x = inputs.detach().cpu().numpy()
        x = np.array([dct2(img) for img in x])
        x = torch.tensor(x, dtype=torch.float32, device=inputs.device)
        self.mean = self.mean.to(device=inputs.device)
        self.std = self.std.to(device=inputs.device)

        x = torch.abs(x)
        x += 1e-13
        x = torch.log(x)
        x = (x - self.mean) / self.std
        return x


class ClassificationHead(nn.Module):
    def __init__(self, num_classes=2):
        super(ClassificationHead, self).__init__()
        self.input, self.output = 64, 32
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.input, self.output, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.output),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.output, self.output, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.output),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(self.output, self.output, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.output),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(self.output, self.output, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.output),
            nn.ReLU()
        )
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.layer5 = nn.Sequential(
            nn.Conv2d(self.output, self.output, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.output),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(self.output, self.output, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.output),
            nn.ReLU()
        )
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.layer7 = nn.Sequential(
            nn.Conv2d(self.output, self.output, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.output),
            nn.ReLU()
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(self.output, self.output, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.output),
            nn.ReLU()
        )
        self.avg_pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.layer9 = nn.Sequential(
            nn.Conv2d(self.output, self.output, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.output),
            nn.ReLU()
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(self.output, self.output, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.output),
            nn.ReLU()
        )
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32, num_classes)
        # self.attention = SimpleCrossAttentionLayer(32)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool1(x)
        # [64, 32, 128, 128]

        x = self.layer5(x)
        x = self.layer6(x)
        x = self.avg_pool2(x)
        # [64, 32, 64, 64]
        # x = self.attention(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.avg_pool3(x)
        # [64, 32, 32, 32]
        # x = self.attention(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.adaptive_avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class FreWeightNet(nn.Module):
    def __init__(self, input_channel=3, output_channel=64, FreWeight=0, kernel_size=3, padding=1):
        super(FreWeightNet, self).__init__()
        assert FreWeight <= 1, 'Frequency weight must be less than 1'
        assert FreWeight >= 0.016, 'Frequency weight must be more than 0.016'
        self.spa_channel = int(output_channel * (1 - FreWeight))
        self.fre_channel = output_channel - self.spa_channel
        self.conv_spa = nn.Conv2d(input_channel, self.spa_channel,
                                  kernel_size=kernel_size, padding=padding)
        self.conv_fre = nn.Conv2d(input_channel, self.fre_channel,
                                  kernel_size=kernel_size, padding=padding)
        mean_2d = np.loadtxt('./Experiment_dataset/mean.txt')
        std_2d = np.loadtxt('./Experiment_dataset/std.txt')
        mean_3d = mean_2d.reshape((256, 256, 3))
        std_3d = std_2d.reshape((256, 256, 3))
        mean_3d = np.transpose(mean_3d, (2, 0, 1))
        std_3d = np.transpose(std_3d, (2, 0, 1))
        self.dct2d = DCTLayer(mean=mean_3d, std=std_3d)
        self.BN = nn.BatchNorm2d(output_channel)
        self.Relu = nn.ReLU()
        self.classifier = ClassificationHead()

    def forward(self, x):
        x_fre = self.dct2d(x)
        feature_spatial = self.conv_spa(x)
        feature_frequency = self.conv_fre(x_fre)
        feature = torch.cat((feature_spatial, feature_frequency), dim=1)
        feature_map = self.Relu(self.BN(feature))
        output = self.classifier(feature_map)
        return output, feature_map





