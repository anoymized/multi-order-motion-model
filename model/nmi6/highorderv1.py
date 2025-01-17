import torch
import torch.nn as nn
import torch.nn.functional as F
from networkfactory.convfactory import conv
import numpy as np
from model.nmi6.resnet3d import BasicBlock, get_inplanes, generate_model


def make_param(in_channels, values, requires_grad=True, dtype=None):
    if dtype is None:
        dtype = 'float32'
    values = np.require(values, dtype=dtype)
    n = in_channels * len(values)
    data = torch.from_numpy(values).view(1, -1)
    data = data.repeat(in_channels, 1)
    return torch.nn.Parameter(data=data, requires_grad=requires_grad)


class HigherOrderMotionDetector(nn.Module):
    def __init__(self, output_dim, norm_fn='instance', dropout=0.0, n_frames=16, kernel_size=7):
        super(HigherOrderMotionDetector, self).__init__()
        self.norm_fn = norm_fn

        self.norm_sigma = make_param(1, np.array([0.2]), requires_grad=True)
        self.norm_k = make_param(1, np.array([4.0]), requires_grad=True)
        self.basic_encoder = generate_model(34)
        self.re_projection = nn.Sequential(nn.Conv3d(output_dim, output_dim, kernel_size=3, padding=1), nn.Sigmoid())

    def normalize(self, x):  # TODO
        sum_activation = torch.mean(x) + torch.square(self.norm_sigma)
        x = self.norm_k.abs() * x / sum_activation
        return x

    def forward(self, x):
        feature_list = []
        assert isinstance(x, list) or isinstance(x, tuple)
        # batch inference
        # convert to tensor
        x = [x.unsqueeze(2) for x in x]
        # x[0].shape = torch.Size([1, 1, 3, 128, 128])
        x = torch.cat(x, dim=2)
        st_feature = self.basic_encoder(x)

        B, C, T, H, W = x.shape
        # reshape to B*T, C, H, W
        st_feature = self.re_projection(st_feature)
        st_feature = self.normalize(st_feature)
        # to T, B, C, H, W
        st_feature = st_feature.permute(2, 0, 1, 3, 4)
        return st_feature

    @staticmethod
    def demo():
        model = HigherOrderMotionDetector(output_dim=256, n_frames=15 + 1, kernel_size=6 + 1)
        x = [torch.randn(2, 3, 256, 256) for i in range(16)]
        y = model(x)
        print(y.shape)


if __name__ == '__main__':
    HigherOrderMotionDetector.demo()
