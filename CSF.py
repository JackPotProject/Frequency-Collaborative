import torch
from PIL import Image
from torch import nn
import numpy as np
import torch.nn.functional as F
import math


def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 6, 5, 5, 6, 4, 5, 4, 4, 3, 3, 3, 3, 6, 5, 4, 2, 2, 2, 2, 2, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 4]
        all_bot_indices_y = [6, 5, 6, 5, 4, 6, 4, 5, 4, 6, 5, 4, 3, 3, 3, 3, 6, 5, 4, 3, 2, 2, 2, 2, 2, 6, 5, 4, 3, 2, 1, 1]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='bot16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # learnable DCT init
        self.register_parameter('weight',
                                self.get_dct_filter(height, width, mapper_x, mapper_y, channel))


    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] \
                        = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
        return torch.nn.Parameter(dct_filter)


class HighPassFilterTransformSeparately(nn.Module):
    def __init__(self):
        super(HighPassFilterTransformSeparately, self).__init__()
        kernels = self.get_kernels()
        # Convert the list of kernels to torch tensors
        self.kernels = [torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0) for kernel in kernels]

    def forward(self, img_tensor):
        device = img_tensor.device
        for kernel in self.kernels:
            kernel = kernel.to(device)
            filtered_tensor = torch.zeros_like(img_tensor)
            for c in range(img_tensor.shape[1]):
                filtered_tensor[:, c:c + 1, :, :] = F.conv2d(img_tensor[:, c:c + 1, :, :], kernel, padding=2)
            filtered_tensor = filtered_tensor.squeeze(0)
        return filtered_tensor

    def rotate_kernel(self, kernel, angle):
        return np.array(Image.fromarray(np.array(kernel)).rotate(angle))

    def get_kernels(self):
        basic_kernels = [
            [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, -1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, -1, 0, 0], [0, 0, 3, 0, 0], [0, 0, -3, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, -2, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
            [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
        ]
        kernels = []
        # (a) and (b) with 8 rotations (45, 90, 135, 180, 225, 270, 315, 360)
        for i in range(8):
            angle = i * 45
            kernels.append(self.rotate_kernel(basic_kernels[0], angle))
            kernels.append(self.rotate_kernel(basic_kernels[1], angle))

        # (c) with 4 rotations (45, 90, 135, 180)
        for i in range(4):
            angle = i * 40
            kernels.append(self.rotate_kernel(basic_kernels[2], angle))

        # (d) and (e) with 4 rotations each (90, 180, 270, 360)
        for i in range(4):
            angle = i * 90
            kernels.append(self.rotate_kernel(basic_kernels[3], angle))
            kernels.append(self.rotate_kernel(basic_kernels[4], angle))

        # (f) and (g) with no rotation
        kernels.append(basic_kernels[5])
        kernels.append(basic_kernels[6])
        return kernels


class FcaBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes=3, planes=64, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16, ):
        global _mapper_x, _mapper_y
        super(FcaBasicBlock, self).__init__()
        # assert fea_h is not None
        # assert fea_w is not None
        c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7)])
        self.planes = planes
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.tanh = nn.Hardtanh()

        self.att1 = MultiSpectralAttentionLayer(planes, c2wh[planes], c2wh[planes], reduction=reduction,
                                                freq_sel_method='top16')

        self.att2 = MultiSpectralAttentionLayer(planes, c2wh[planes], c2wh[planes], reduction=reduction,
                                                freq_sel_method='bot16')

        self.HPF = HighPassFilterTransformSeparately()

    def forward(self, x):
        """
        concat_regions = np.concatenate((weak_texture, strong_texture), axis=1)
        """
        x = self.HPF(x)
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        b, c, h, w = x.shape
        weak_x = x[:, :, :h // 2, :]
        strong_x = x[:, :, h // 2:, :]

        weak_out = self.conv1(weak_x)
        weak_out = self.bn1(weak_out)
        weak_out = self.att1(weak_out)
        weak_out = self.tanh(weak_out)

        strong_out = self.conv2(strong_x)
        strong_out = self.bn2(strong_out)
        strong_out = self.att2(strong_out)
        strong_out = self.tanh(strong_out)

        feature_map = torch.cat((weak_out, strong_out), dim=2)
        return feature_map


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


class FCACoAttentionNet(nn.Module):
    def __init__(self, num_classes):
        super(FCACoAttentionNet, self).__init__()
        self.ExtractNet = FcaBasicBlock()
        self.num_classes = num_classes
        self.head = ClassificationHead(self.num_classes)

    def forward(self, x):
        x = self.ExtractNet(x)
        output = self.head(x)
        return output


