import numpy as np
import math
import torch

import numpy
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import os
import pandas as pd
import imageio
from torch.cuda.amp import autocast as autocast


def make_param(in_channels, values, requires_grad=True, dtype=None):
    if dtype is None:
        dtype = 'float32'
    values = np.require(values, dtype=dtype)
    n = in_channels * len(values)
    data = torch.from_numpy(values).view(1, -1)
    data = data.repeat(in_channels, 1)
    return torch.nn.Parameter(data=data, requires_grad=requires_grad)


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def inverse_sigmoid(p):
    return np.log(p / (1 - p))


def artanh(y):
    return 0.5 * np.log((1 + y) / (1 - y))


class V1SingleScale(nn.Module):
    def __init__(self, output_dim, norm_fn='instance', n_frames=8, kernel_size=7, kernel_radius=5):
        super(V1SingleScale, self).__init__()
        self.norm_fn = norm_fn
        self.norm_sigma = make_param(1, np.array([0.2]), requires_grad=True)
        self.norm_k = make_param(1, np.array([4.0]), requires_grad=True)
        self.spatial_filter = GaborFilters(kernel_radius=kernel_radius, num_units=output_dim, random=True)

        self.temporal_decay = 0.15
        self.spatial_decay = 0.15

        self.spatial_radius = kernel_radius
        self.spatial_kernel_size = kernel_radius * 2 + 1
        self.spatial_num = output_dim
        self.temporal_filter = TemporalFilter(num_ft=output_dim, kernel_size=kernel_size, random=True)
        self.temporal_pooling = make_param(output_dim, np.ones((n_frames - kernel_size + 1)), requires_grad=True)
        self.spontaneous_firing = make_param(1, np.array([0.3]), requires_grad=True)

    def normalize(self, x):  # TODO
        sum_activation = torch.mean(x, dim=[1], keepdim=True) + torch.square(self.norm_sigma)
        x = self.norm_k.abs() * x / sum_activation
        return x

    def motion_energy(self, x):  # x should be list of B,1,H,W
        # note: must ensure open the torch.cuda.cudnn_enabled = True otherwise the speed will be very slow
        assert torch.backends.cudnn.enabled, "Cudnn is not enabled"

        n, B, C, H, W = x.shape
        C_s = self.spatial_filter.n_channels_post_conv
        x = x.reshape(n * B, C, H, W)

        sy = x.size(2)
        sx = x.size(3)
        k_sin, k_cos = self.temporal_filter.make_temporal_filter()
        s_sin, s_cos = self.spatial_filter.make_gabor_filters(quadrature=True)

        gb_sin = s_sin.view(self.spatial_num, 1, self.spatial_kernel_size, self.spatial_kernel_size)
        gb_cos = s_cos.view(self.spatial_num, 1, self.spatial_kernel_size, self.spatial_kernel_size)

        # flip kernel
        gb_sin = torch.flip(gb_sin, dims=[-1, -2])
        gb_cos = torch.flip(gb_cos, dims=[-1, -2])

        res_sin = F.conv2d(input=x, weight=gb_sin,
                           padding=self.spatial_radius, groups=C)
        res_cos = F.conv2d(input=x, weight=gb_cos,
                           padding=self.spatial_radius, groups=C)

        res_sin = res_sin.view(B, -1, sy, sx)
        res_cos = res_cos.view(B, -1, sy, sx)
        g_asin_list = res_sin.reshape(n, B, -1, H, W)
        g_acos_list = res_cos.reshape(n, B, -1, H, W)
        # c,1,n

        # n,b,c,h,w ->bhw,c,n
        g_asin_list = g_asin_list.permute(1, 3, 4, 2, 0).reshape(B * H * W, -1, n)
        g_acos_list = g_acos_list.permute(1, 3, 4, 2, 0).reshape(B * H * W, -1, n)

        # reverse the impulse response
        k_sin = torch.flip(k_sin, dims=(-1,))
        k_cos = torch.flip(k_cos, dims=(-1,))

        # for g_asin_chunk, g_acos_chunk in zip(g_asin_list.chunk(4, dim=0), g_acos_list.chunk(4, dim=0)):
        a = F.conv1d(g_acos_list, k_sin, padding="valid", bias=None, groups=C_s)
        b = F.conv1d(g_asin_list, k_cos, padding="valid", bias=None, groups=C_s)
        g_o = a + b
        a = F.conv1d(g_acos_list, k_cos, padding="valid", bias=None, groups=C_s)
        b = F.conv1d(g_asin_list, k_sin, padding="valid", bias=None, groups=C_s)
        g_e = a - b
        energy_component = g_o ** 2 + g_e ** 2 + self.spontaneous_firing.square()
        energy_component = energy_component.reshape(B, H, W, C_s, energy_component.size(-1)).permute(4, 0, 3, 1, 2)

        pooling = self.temporal_pooling.transpose(0, 1).reshape(energy_component.size(0), 1, C_s, 1, 1)
        energy_component_pooling = abs(torch.mean(energy_component * pooling, dim=0, keepdim=False))

        return energy_component_pooling

    def forward(self, v1_energy):
        # to nb c h w
        n, B, C, H, W = v1_energy.shape
        v1_energy = self.motion_energy(v1_energy)
        v1_energy = self.normalize(v1_energy)
        return v1_energy

    @staticmethod
    def demo():
        model = V1SingleScale(output_dim=128, n_frames=8, kernel_size=7, kernel_radius=7).cuda()
        x = torch.randn(8, 2, 128, 64, 64).cuda()
        y = model(x)
        print(y.shape)


class V1(nn.Module):
    """each input includes 10 frame with 25 frame/sec sampling rate
    temporal window size = 5 frame(200ms)
    spatial window size = 5*2 + 1 = 11
    spatial filter is
    lambda is frequency of cos wave
    """

    def __init__(self, spatial_num=32, scale_num=8, scale_factor=16, kernel_radius=7, num_ft=32,
                 kernel_size=6, average_time=True, n_frames=10):
        super(V1, self).__init__()

        def make_param(in_channels, values, requires_grad=True, dtype=None):
            if dtype is None:
                dtype = 'float32'
            values = numpy.require(values, dtype=dtype)
            n = in_channels * len(values)
            data = torch.from_numpy(values).view(1, -1)
            data = data.repeat(in_channels, 1)
            return torch.nn.Parameter(data=data, requires_grad=requires_grad)

        assert spatial_num == num_ft
        scale_each_level = np.exp(1 / (scale_num - 1) * np.log(1 / scale_factor))
        self.scale_each_level = scale_each_level
        self.scale_num = scale_num
        self.cell_index = 0
        self.spatial_filter = nn.ModuleList(
            [GaborFilters(kernel_radius=kernel_radius, num_units=spatial_num, random=True)
             for i in range(scale_num)])
        self.temporal_decay = 0.2
        self.spatial_decay = 0.2

        self.spatial_radius = kernel_radius
        self.spatial_kernel_size = kernel_radius * 2 + 1
        self.spatial_num = spatial_num
        self.temporal_filter = nn.ModuleList([TemporalFilter(num_ft=num_ft, kernel_size=kernel_size, random=True)
                                              for i in range(scale_num)])  # 16 filter

        self.t_length = n_frames - kernel_size + 1
        self.n_frames = n_frames
        self._num_after_st = spatial_num * scale_num

        if not average_time:
            self._num_after_st = self._num_after_st * (self.n_frames - kernel_size + 1)
        if average_time:
            self.temporal_pooling = make_param(self._num_after_st, np.ones(self.t_length),
                                               requires_grad=True)
        self.norm_sigma = make_param(1, np.array([0.2]), requires_grad=True)
        self.spontaneous_firing = make_param(1, np.array([0.3]), requires_grad=True)
        self.norm_k = make_param(1, np.array([4.0]), requires_grad=True)
        self._average_time = average_time
        self.t_sin = None
        self.t_cos = None
        self.s_sin = None
        self.s_cos = None

    def infer_scale(self, x, scale):  # x should be list of B,1,H,W
        energy_list = []
        n = len(x)
        B, C, H, W = x[0].shape
        x = [img.unsqueeze(0) for img in x]
        x = torch.cat(x, dim=0).reshape(n * B, C, H, W)

        sy = x.size(2)
        sx = x.size(3)
        s_sin = self.s_sin
        s_cos = self.s_cos

        gb_sin = s_sin.view(self.spatial_num, 1, self.spatial_kernel_size, self.spatial_kernel_size)
        gb_cos = s_cos.view(self.spatial_num, 1, self.spatial_kernel_size, self.spatial_kernel_size)

        # flip kernel
        gb_sin = torch.flip(gb_sin, dims=[-1, -2])
        gb_cos = torch.flip(gb_cos, dims=[-1, -2])

        res_sin = F.conv2d(input=x, weight=gb_sin,
                           padding=self.spatial_radius, groups=1)
        res_cos = F.conv2d(input=x, weight=gb_cos,
                           padding=self.spatial_radius, groups=1)

        res_sin = res_sin.view(B, -1, sy, sx)
        res_cos = res_cos.view(B, -1, sy, sx)
        g_asin_list = res_sin.reshape(n, B, -1, H, W)
        g_acos_list = res_cos.reshape(n, B, -1, H, W)

        for channel in range(self.spatial_filter[0].n_channels_post_conv):
            k_sin = self.t_sin[channel, ...][None]
            k_cos = self.t_cos[channel, ...][None]
            # spatial filter
            g_asin, g_acos = g_asin_list[:, :, channel, :, :], g_acos_list[:, :, channel, :, :]  # n,b,h,w
            g_asin = g_asin.reshape(n, B * H * W, 1).permute(1, 2, 0)  # bhw,1,n
            g_acos = g_acos.reshape(n, B * H * W, 1).permute(1, 2, 0)

            # reverse the impulse response
            k_sin = torch.flip(k_sin, dims=(-1,))
            k_cos = torch.flip(k_cos, dims=(-1,))
            #
            a = F.conv1d(g_acos, k_sin, padding="valid", bias=None)
            b = F.conv1d(g_asin, k_cos, padding="valid", bias=None)
            g_o = a + b
            a = F.conv1d(g_acos, k_cos, padding="valid", bias=None)
            b = F.conv1d(g_asin, k_sin, padding="valid", bias=None)
            g_e = a - b
            energy_component = g_o ** 2 + g_e ** 2 + self.spontaneous_firing.square()
            energy_component = energy_component.reshape(B, H, W, a.size(-1)).permute(0, 3, 1, 2)
            if self._average_time:  # average motion energy across time
                total_channel = scale * self.spatial_num + channel
                pooling = self.temporal_pooling[total_channel][None, ..., None, None]
                energy_component = abs(torch.mean(energy_component * pooling, dim=1, keepdim=True))
            energy_list.append(energy_component)
        energy_list = torch.cat(energy_list, dim=1)
        return energy_list

    def infer_scale_fast(self, x, scale):  # x should be list of B,1,H,W
        # note: must ensure open the torch.cuda.cudnn_enabled = True otherwise the speed will be very slow
        assert torch.backends.cudnn.enabled, "Cudnn is not enabled"
        n = len(x)
        B, C, H, W = x[0].shape
        C_s = self.spatial_filter[0].n_channels_post_conv
        x = [img.unsqueeze(0) for img in x]
        x = torch.cat(x, dim=0).reshape(n * B, C, H, W)

        sy = x.size(2)
        sx = x.size(3)
        s_sin = self.s_sin
        s_cos = self.s_cos

        gb_sin = s_sin.view(self.spatial_num, 1, self.spatial_kernel_size, self.spatial_kernel_size)
        gb_cos = s_cos.view(self.spatial_num, 1, self.spatial_kernel_size, self.spatial_kernel_size)

        # flip kernel
        gb_sin = torch.flip(gb_sin, dims=[-1, -2])
        gb_cos = torch.flip(gb_cos, dims=[-1, -2])

        res_sin = F.conv2d(input=x, weight=gb_sin,
                           padding=self.spatial_radius, groups=1)
        res_cos = F.conv2d(input=x, weight=gb_cos,
                           padding=self.spatial_radius, groups=1)

        res_sin = res_sin.view(B, -1, sy, sx)
        res_cos = res_cos.view(B, -1, sy, sx)
        g_asin_list = res_sin.reshape(n, B, -1, H, W)
        g_acos_list = res_cos.reshape(n, B, -1, H, W)
        # c,1,n
        k_sin = self.t_sin
        k_cos = self.t_cos

        # n,b,c,h,w ->bhw,c,n
        g_asin_list = g_asin_list.permute(1, 3, 4, 2, 0).reshape(B * H * W, -1, n)
        g_acos_list = g_acos_list.permute(1, 3, 4, 2, 0).reshape(B * H * W, -1, n)

        # reverse the impulse response
        k_sin = torch.flip(k_sin, dims=(-1,))
        k_cos = torch.flip(k_cos, dims=(-1,))
        # chunk the input into 10 blocks for memory efficiency
        # a = F.conv1d(g_acos, k_sin, padding="valid", bias=None, groups=C_s)
        # b = F.conv1d(g_asin, k_cos, padding="valid", bias=None, groups=C_s)
        energy_component_list = []

        a = F.conv1d(g_acos_list, k_sin, padding="valid", bias=None, groups=C_s)
        b = F.conv1d(g_asin_list, k_cos, padding="valid", bias=None, groups=C_s)
        g_o = a + b
        a = F.conv1d(g_acos_list, k_cos, padding="valid", bias=None, groups=C_s)
        b = F.conv1d(g_asin_list, k_sin, padding="valid", bias=None, groups=C_s)
        g_e = a - b
        energy_component = g_o ** 2 + g_e ** 2 + self.spontaneous_firing.square()

        energy_component = energy_component.reshape(B, H, W, C_s, energy_component.size(-1)).permute(4, 0, 3, 1, 2)
        # save to mat

        if self._average_time:  # average motion energy across time
            # 32 per scale
            pooling = self.temporal_pooling[scale * self.spatial_num:(scale + 1)
                                                                     * self.spatial_num].transpose(0, 1).reshape(
                energy_component.size(0), 1, C_s, 1, 1)
            energy_component_pooling = abs(torch.mean(energy_component * pooling, dim=0, keepdim=False))
        else:
            energy_component_pooling = energy_component

        return energy_component_pooling, energy_component

    def forward(self, image_list):
        B, C, H, W = image_list[0].shape
        MT_size = (H // 8, W // 8)
        self.cell_index = 0
        with torch.no_grad():
            if image_list[0].max() > 10:
                image_list = [img / 255.0 for img in image_list]  # [B, 1, H, W]  0-1

        first_order = []
        second_order = []
        for scale in range(self.scale_num):
            image_list = [F.interpolate(img, scale_factor=self.scale_each_level, mode="bilinear") for img in image_list]
            self.t_sin, self.t_cos = self.temporal_filter[scale].make_temporal_filter()
            self.s_sin, self.s_cos = self.spatial_filter[scale].make_gabor_filters(quadrature=True)
            st_component, _ = self.infer_scale_fast(image_list, scale)
            st_component = F.interpolate(st_component, size=MT_size, mode="bilinear", align_corners=True)
            first_order.append(st_component)

        first_order = torch.cat(first_order, dim=1)
        first_order = self.normalize(first_order)

        return first_order

    def normalize(self, x):  # TODO
        sum_activation = torch.mean(x, dim=[1], keepdim=True) + torch.square(self.norm_sigma)
        x = self.norm_k.abs() * x / sum_activation
        return x

    def _get_v1_order(self):
        thetas = [gabor_scale.thetas for gabor_scale in self.spatial_filter]
        fss = [gabor_scale.fs for gabor_scale in self.spatial_filter]
        fts = [temporal_scale.ft for temporal_scale in self.temporal_filter]
        scale_each_level = self.scale_each_level

        scale_num = self.scale_num
        neural_representation = []
        index = 0
        for scale_idx in range(len(thetas)):
            theta_scale = thetas[scale_idx]
            theta_scale = torch.sigmoid(theta_scale) * 2 * torch.pi  # spatial orientation constrain to 0-pi
            fs_scale = fss[scale_idx]
            fs_scale = torch.sigmoid(fs_scale) * 0.25
            fs_scale = fs_scale * (scale_each_level ** scale_idx)

            ft_scale = fts[scale_idx]
            ft_scale = torch.sigmoid(ft_scale) * 0.25

            theta_scale = theta_scale.squeeze().cpu().detach().numpy()
            fs_scale = fs_scale.squeeze().cpu().detach().numpy()
            ft_scale = ft_scale.squeeze().cpu().detach().numpy()
            for gabor_idx in range(len(theta_scale)):
                speed = ft_scale[gabor_idx] / fs_scale[gabor_idx]
                assert speed >= 0
                angle = theta_scale[gabor_idx]
                a = {"theta": -angle + np.pi, "fs": fs_scale[gabor_idx], "ft": ft_scale[gabor_idx], "speed": speed,
                     "index": index}
                index = index + 1
                neural_representation.append(a)
        return neural_representation

    def visualize_activation(self, activation, if_log=True):
        neural_representation = self._get_v1_order()
        activation = activation[:, :, 14:-14, 14:-14]  # eliminate boundary
        activation = torch.mean(activation, dim=[2, 3], keepdim=False)[0]
        ax1 = plt.subplot(121, projection='polar')
        theta_list = []
        v_list = []
        energy_list = []
        for index in range(len(neural_representation)):
            v = neural_representation[index]["speed"]
            theta = neural_representation[index]["theta"]
            location = neural_representation[index]["index"]
            energy = activation.squeeze()[location].cpu().detach().numpy()
            theta_list.append(theta)
            v_list.append(v)
            energy_list.append(energy)
        v_list, theta_list, energy_list = np.array(v_list), np.array(theta_list), np.array(energy_list)
        x, y = pol2cart(v_list, theta_list)
        plt.scatter(theta_list, v_list, c=energy_list, cmap="rainbow", s=(energy_list + 20), alpha=0.5)
        plt.axis('on')
        if if_log:
            ax1.set_rscale('symlog')
        plt.colorbar()
        energy_list = np.expand_dims(energy_list, 0).repeat(len(theta_list), 0)
        plt.subplot(122, projection="polar")
        # plt.contourf(theta_list, v_list, energy_list)
        plt.show()

    @staticmethod
    def demo():
        input = [torch.ones(2, 1, 256, 256).cuda() for k in range(16)]
        model = V1(spatial_num=16, scale_num=16, scale_factor=16, kernel_radius=7, num_ft=16,
                   kernel_size=6, average_time=True).cuda()
        for i in range(100):
            import time
            start = time.time()
            with autocast(enabled=True):
                x = model(input)
                print(x.shape)
            torch.mean(x).backward()
            end = time.time()
            print(end - start)
            print("#================================++#")

    @property
    def num_after_st(self):
        return self._num_after_st


class TemporalFilter(nn.Module):
    def __init__(self, in_channels=1, num_ft=8, kernel_size=6, random=True):
        # 40ms per time unit, 200ms -> 5+1 frames
        # use exponential decay plus sin wave
        super().__init__()
        self.kernel_size = kernel_size

        def make_param(in_channels, values, requires_grad=True, dtype=None):
            if dtype is None:
                dtype = 'float32'
            values = numpy.require(values, dtype=dtype)
            n = in_channels * len(values)
            data = torch.from_numpy(values).view(1, -1)
            data = data.repeat(in_channels, 1)
            return torch.nn.Parameter(data=data, requires_grad=requires_grad)

        indices = torch.arange(kernel_size, dtype=torch.float32)
        self.register_buffer('indices', indices)
        if random:
            self.ft = make_param(in_channels, values=inverse_sigmoid(numpy.random.uniform(0.01, 0.99, num_ft)),
                                 requires_grad=True)
            self.tao = make_param(in_channels, values=numpy.arange(num_ft) / 2 + 1, requires_grad=True)
        else:  # evenly distributed
            self.ft = make_param(in_channels, values=inverse_sigmoid(numpy.linspace(0.01, 0.99, num_ft)),
                                 requires_grad=True)
            self.tao = make_param(in_channels, values=numpy.arange(num_ft) / 2 + 1, requires_grad=True)
        self.feat_dim = num_ft
        self.temporal_decay = 0.2

    def make_temporal_filter(self):
        fts = torch.sigmoid(self.ft) * 0.25
        tao = torch.sigmoid(self.tao) * (-self.kernel_size / np.log(self.temporal_decay))
        t = self.indices

        fts = fts.view(1, fts.shape[1], 1)
        tao = tao.view(1, tao.shape[1], 1)
        t = t.view(1, 1, t.shape[0])

        temporal_sin = torch.exp(-t / tao) * torch.sin(2 * torch.pi * fts * t)
        temporal_cos = torch.exp(-t / tao) * torch.cos(2 * torch.pi * fts * t)
        temporal_sin = temporal_sin.view(-1, self.kernel_size)
        temporal_cos = temporal_cos.view(-1, self.kernel_size)

        temporal_sin = temporal_sin.view(self.feat_dim, 1, self.kernel_size)
        temporal_cos = temporal_cos.view(self.feat_dim, 1, self.kernel_size)
        # temporal_sin = torch.chunk(temporal_sin, dim=0, chunks=self._feat_dim)
        # temporal_cos = torch.chunk(temporal_cos, dim=0, chunks=self._feat_dim)

        return temporal_sin, temporal_cos  # 1,kz

    def demo_temporal_filter(self, points=100):
        fts = torch.sigmoid(self.ft) * 0.25
        tao = torch.sigmoid(self.tao) * (-(self.kernel_size - 1) / np.log(self.temporal_decay))
        t = torch.linspace(self.indices[0], self.indices[-1], steps=points)

        fts = fts.view(1, fts.shape[1], 1)
        tao = tao.view(1, tao.shape[1], 1)
        t = t.view(1, 1, t.shape[0])
        print("ft:" + str(fts))
        print("tao:" + str(tao))

        temporal_sin = torch.exp(-t / tao) * torch.sin(2 * torch.pi * fts * t)
        temporal_cos = torch.exp(-t / tao) * torch.cos(2 * torch.pi * fts * t)
        temporal_sin = temporal_sin.view(-1, points)
        temporal_cos = temporal_cos.view(-1, points)

        temporal_sin = temporal_sin.view(self.feat_dim, 1, points)
        temporal_cos = temporal_cos.view(self.feat_dim, 1, points)
        # temporal_sin = torch.chunk(temporal_sin, dim=0, chunks=self._feat_dim)
        # temporal_cos = torch.chunk(temporal_cos, dim=0, chunks=self._feat_dim)

        return temporal_sin, temporal_cos  # 1,kz

    def forward(self, x_sin, x_cos):
        in_channels = x_sin.size(1)
        n = x_sin.size(2)
        # batch, c, sequence
        me = []
        t_sin, t_cos = self.make_temporal_filter()
        for n_t in range(self.feat_dim):
            k_sin = t_sin[n_t, ...].expand(in_channels, -1, -1)
            k_cos = t_cos[n_t, ...].expand(in_channels, -1, -1)

            a = F.conv1d(x_sin, weight=k_cos, padding="same", groups=in_channels, bias=None)
            b = F.conv1d(x_cos, weight=k_sin, padding="same", groups=in_channels, bias=None)
            g_o = a + b

            a = F.conv1d(x_sin, weight=k_sin, padding="same", groups=in_channels, bias=None)
            b = F.conv1d(x_cos, weight=k_cos, padding="same", groups=in_channels, bias=None)
            g_e = a - b

            energy_component = g_o ** 2 + g_e ** 2
            me.append(energy_component)

        return me


class GaborFilters(nn.Module):
    def __init__(self,
                 in_channels=1,
                 kernel_radius=7,
                 num_units=512,
                 random=True
                 ):
        # the total number of or units for each scale
        super().__init__()
        self.in_channels = in_channels
        kernel_size = kernel_radius * 2 + 1
        self.kernel_size = kernel_size
        self.kernel_radius = kernel_radius

        def make_param(in_channels, values, requires_grad=True, dtype=None):
            if dtype is None:
                dtype = 'float32'
            values = numpy.require(values, dtype=dtype)
            n = in_channels * len(values)
            data = torch.from_numpy(values).view(1, -1)
            data = data.repeat(in_channels, 1)
            return torch.nn.Parameter(data=data, requires_grad=requires_grad)

        # build all learnable parameters
        # random distribution
        if random:
            self.sigmas = make_param(in_channels, inverse_sigmoid(np.random.uniform(0.8, 0.99, num_units)))
            self.fs = make_param(in_channels, values=inverse_sigmoid(numpy.random.uniform(0.2, 0.8, num_units)))
            # maximun is 0.25 cycle/frame
            self.gammas = make_param(in_channels, numpy.ones(num_units))  # TODO: fix gamma or not
            self.psis = make_param(in_channels, np.zeros(num_units), requires_grad=False)  # fix phase
            self.thetas = make_param(in_channels, values=inverse_sigmoid(numpy.random.uniform(0.01, 0.99, num_units)),
                                     requires_grad=True)
        else:  # evenly distribution
            self.sigmas = make_param(in_channels, inverse_sigmoid(np.linspace(0.8, 0.99, num_units)))
            self.fs = make_param(in_channels, values=inverse_sigmoid(numpy.linspace(0.01, 0.99, num_units)))
            # maximun is 0.25 cycle/frame
            self.gammas = make_param(in_channels, numpy.ones(num_units))  # TODO: fix gamma or not
            self.psis = make_param(in_channels, np.zeros(num_units), requires_grad=False)  # fix phase
            self.thetas = make_param(in_channels, values=inverse_sigmoid(numpy.linspace(0, 1, num_units)),
                                     requires_grad=True)

        indices = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        self.register_buffer('indices', indices)
        self.spatial_decay = 0.5
        # number of channels after the conv
        self.n_channels_post_conv = num_units

    def make_gabor_filters(self, quadrature=True):
        sigmas = torch.sigmoid(self.sigmas) * np.sqrt(
            (self.kernel_radius - 1) ** 2 * 0.5 / np.log(
                1 / self.spatial_decay))  # std of gauss win decay to 0.2 by log(0.2)
        fs = torch.sigmoid(self.fs) * 0.25
        # frequency of cos and sine wave keep positive, must > 2 to avoid aliasing
        gammas = torch.abs(self.gammas)  # shape of gauss win, set as 1 by default
        psis = self.psis  # phase of cos wave
        thetas = torch.sigmoid(self.thetas) * 2 * torch.pi  # spatial orientation constrain to 0-2pi
        y = self.indices
        x = self.indices

        in_channels = sigmas.shape[0]
        assert in_channels == fs.shape[0]
        assert in_channels == gammas.shape[0]

        kernel_size = y.shape[0], x.shape[0]

        sigmas = sigmas.view(in_channels, sigmas.shape[1], 1, 1)
        fs = fs.view(in_channels, fs.shape[1], 1, 1)
        gammas = gammas.view(in_channels, gammas.shape[1], 1, 1)
        psis = psis.view(in_channels, psis.shape[1], 1, 1)
        thetas = thetas.view(in_channels, thetas.shape[1], 1, 1)
        y = y.view(1, 1, y.shape[0], 1)
        x = x.view(1, 1, 1, x.shape[0])

        sigma_x = sigmas
        sigma_y = sigmas / gammas

        sin_t = torch.sin(thetas)
        cos_t = torch.cos(thetas)
        y_theta = -x * sin_t + y * cos_t
        x_theta = x * cos_t + y * sin_t

        if quadrature:
            gb_cos = torch.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) \
                     * torch.cos(2.0 * math.pi * x_theta * fs + psis)
            gb_sin = torch.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) \
                     * torch.sin(2.0 * math.pi * x_theta * fs + psis)
            gb_cos = gb_cos.reshape(-1, 1, kernel_size[0], kernel_size[1])
            gb_sin = gb_sin.reshape(-1, 1, kernel_size[0], kernel_size[1])

            # remove DC
            gb_cos = gb_cos - torch.sum(gb_cos, dim=[-1, -2], keepdim=True) / (kernel_size[0] * kernel_size[1])
            gb_sin = gb_sin - torch.sum(gb_sin, dim=[-1, -2], keepdim=True) / (kernel_size[0] * kernel_size[1])

            return gb_sin, gb_cos

        else:
            gb = torch.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) \
                 * torch.cos(2.0 * math.pi * x_theta * fs + psis)

            gb = gb.view(-1, kernel_size[0], kernel_size[1])
            return gb

    def forward(self, x):
        batch_size = x.size(0)
        sy = x.size(2)
        sx = x.size(3)
        gb_sin, gb_cos = self.make_gabor_filters(quadrature=True)
        assert gb_sin.shape[0] == self.n_channels_post_conv
        assert gb_sin.shape[2] == self.kernel_size
        assert gb_sin.shape[3] == self.kernel_size
        gb_sin = gb_sin.view(self.n_channels_post_conv, 1, self.kernel_size, self.kernel_size)
        gb_cos = gb_cos.view(self.n_channels_post_conv, 1, self.kernel_size, self.kernel_size)

        # flip ke
        gb_sin = torch.flip(gb_sin, dims=[-1, -2])
        gb_cos = torch.flip(gb_cos, dims=[-1, -2])

        res_sin = F.conv2d(input=x, weight=gb_sin,
                           padding=self.kernel_radius, groups=self.in_channels)
        res_cos = F.conv2d(input=x, weight=gb_cos,
                           padding=self.kernel_radius, groups=self.in_channels)

        if self.rotation_invariant:
            res_sin = res_sin.view(batch_size, self.in_channels, -1, self.n_thetas, sy, sx)
            res_sin, _ = res_sin.max(dim=3)
            res_cos = res_cos.view(batch_size, self.in_channels, -1, self.n_thetas, sy, sx)
            res_cos, _ = res_cos.max(dim=3)

        res_sin = res_sin.view(batch_size, -1, sy, sx)
        res_cos = res_cos.view(batch_size, -1, sy, sx)

        return res_sin, res_cos

    def demo_gabor_filters(self, quadrature=True, points=100):

        sigmas = torch.sigmoid(self.sigmas) * np.sqrt(
            (self.kernel_radius - 1) ** 2 * 0.5 / np.log(
                1 / self.spatial_decay))  # std of gauss win decay to 0.2 by log(0.2)
        fs = torch.sigmoid(self.fs) * 0.25
        # frequency of cos and sine wave keep positive, must > 2 to avoid aliasing
        gammas = torch.abs(self.gammas)  # shape of gauss win, set as 1 by default
        thetas = torch.sigmoid(self.thetas) * 2 * torch.pi  # spatial orientation constrain to 0-2pi
        psis = self.psis  # phase of cos wave
        print("theta:" + str(thetas))
        print("fs:" + str(fs))

        x = torch.linspace(self.indices[0], self.indices[-1], points)
        y = torch.linspace(self.indices[0], self.indices[-1], points)

        in_channels = sigmas.shape[0]
        assert in_channels == fs.shape[0]
        assert in_channels == gammas.shape[0]
        kernel_size = y.shape[0], x.shape[0]

        sigmas = sigmas.view(in_channels, sigmas.shape[1], 1, 1)
        fs = fs.view(in_channels, fs.shape[1], 1, 1)
        gammas = gammas.view(in_channels, gammas.shape[1], 1, 1)
        psis = psis.view(in_channels, psis.shape[1], 1, 1)
        thetas = thetas.view(in_channels, thetas.shape[1], 1, 1)
        y = y.view(1, 1, y.shape[0], 1)
        x = x.view(1, 1, 1, x.shape[0])

        sigma_x = sigmas
        sigma_y = sigmas / gammas

        sin_t = torch.sin(thetas)
        cos_t = torch.cos(thetas)
        y_theta = -x * sin_t + y * cos_t
        x_theta = x * cos_t + y * sin_t

        if quadrature:
            gb_cos = torch.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) \
                     * torch.cos(2.0 * math.pi * x_theta * fs + psis)
            gb_sin = torch.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) \
                     * torch.sin(2.0 * math.pi * x_theta * fs + psis)
            gb_cos = gb_cos.reshape(-1, 1, points, points)
            gb_sin = gb_sin.reshape(-1, 1, points, points)

            # remove DC
            gb_cos = gb_cos - torch.sum(gb_cos, dim=[-1, -2], keepdim=True) / (points * points)
            gb_sin = gb_sin - torch.sum(gb_sin, dim=[-1, -2], keepdim=True) / (points * points)

            return gb_sin, gb_cos

        else:
            gb = torch.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) \
                 * torch.cos(2.0 * math.pi * x_theta * fs + psis)

            gb = gb.view(-1, kernel_size[0], kernel_size[1])
            return gb


def te_gabor_(num_units=48):
    s_point = 100
    s_kz = 7
    gb_sin, gb_cos = GaborFilters(num_units=num_units, kernel_radius=s_kz).demo_gabor_filters(points=s_point)
    gb = gb_sin ** 2 + gb_cos ** 2

    print(gb_sin.shape)

    for c in range(gb_sin.size(0)):
        plt.subplot(1, 3, 1)
        curve = gb_cos[c].detach().cpu().squeeze().numpy()
        plt.imshow(curve)
        plt.subplot(1, 3, 2)
        curve = gb_sin[c].detach().cpu().squeeze().numpy()
        plt.imshow(curve)

        plt.subplot(1, 3, 3)
        curve = gb[c].detach().cpu().squeeze().numpy()
        plt.imshow(curve)
        plt.show()


def te_spatial_temporal():
    t_point = 6 * 100
    s_point = 14 * 100
    s_kz = 7
    t_kz = 6
    filenames = []
    gb_sin_b, gb_cos_b = GaborFilters(num_units=48, kernel_radius=s_kz).demo_gabor_filters(points=s_point)
    temporal = TemporalFilter(num_ft=2, kernel_size=t_kz)
    t_sin, t_cos = temporal.demo_temporal_filter(points=t_point)
    x = np.linspace(0, t_kz, t_point)
    index = 0
    for i in range(gb_sin_b.size(0)):
        for j in range(t_sin.size(0)):
            plt.figure(figsize=(14, 9), dpi=80)
            plt.subplot(2, 3, 1)
            curve = gb_sin_b[i].squeeze().detach().numpy()
            plt.imshow(curve)
            plt.title("Gabor Sin")
            plt.subplot(2, 3, 2)
            curve = gb_cos_b[i].squeeze().detach().numpy()
            plt.imshow(curve)
            plt.title("Gabor Cos")

            plt.subplot(2, 3, 3)
            curve = t_sin[j].squeeze().detach().numpy()
            plt.plot(x, curve, label='sin')
            plt.title("Temporal Sin")

            curve = t_cos[j].squeeze().detach().numpy()
            plt.plot(x, curve, label='cos')
            plt.xlabel('Time (s)')
            plt.ylabel('Response to pulse at t=0')
            plt.legend()
            plt.title("Temporal filter")

            gb_sin = gb_sin_b[i].squeeze().detach()[5, :]
            gb_cos = gb_cos_b[i].squeeze().detach()[5, :]

            a = np.outer(t_cos[j].detach(), gb_sin)
            b = np.outer(t_sin[j].detach(), gb_cos)
            g_o = a + b

            a = np.outer(t_sin[j].detach(), gb_sin)
            b = np.outer(t_cos[j].detach(), gb_cos)
            g_e = a - b
            energy_component = g_o ** 2 + g_e ** 2

            plt.subplot(2, 3, 4)
            curve = g_o
            plt.imshow(curve, cmap="gray")
            plt.title("Spatial Temporal even")
            plt.subplot(2, 3, 5)
            curve = g_e
            plt.imshow(curve, cmap="gray")
            plt.title("Spatial Temporal odd")

            plt.subplot(2, 3, 6)
            curve = energy_component
            plt.imshow(curve, cmap="gray")
            plt.title("energy")
            plt.savefig('filter_%d.png' % (index))
            filenames.append('filter_%d.png' % (index))
            index += 1
            plt.show()
    # build gif
    with imageio.get_writer('filters_orientation.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Remove files
    for filename in set(filenames):
        os.remove(filename)


def te_temporal_():
    k_size = 6
    temporal = TemporalFilter(n_tao=2, num_ft=8, kernel_size=k_size)
    sin, cos = temporal.demo_temporal_filter()
    print(sin.shape)
    x = np.linspace(0, k_size, k_size * 100)

    # plot temporal filters to illustrate what they look like.
    for c in range(sin.size(0)):
        curve = cos[c].detach().cpu().squeeze().numpy()
        plt.plot(x, curve, label='cos')
        curve = sin[c].detach().cpu().squeeze().numpy()
        plt.plot(x, curve, label='sin')

        plt.xlabel('Time (s)')
        plt.ylabel('Response to pulse at t=0')
        plt.legend()
        plt.show()


def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x + np.pi) % (2 * np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins + 1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area / np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor='C0', fill=False, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches


def show_trained_model( file_name="[ path ]"):
    import utils.torch_utils as utils
    from model.nmi6.FFV1MT_MS import FFV1DNNV2
    model = FFV1DNNV2(num_scales=8,
                      # num_cells=256,
                      upsample_factor=8,
                      # feature_channels=256,
                      scale_factor=16,
                      num_layers=6, )
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(file_name),strict=True)
    model = model.module.ffv1
    t_point = 100
    s_point = 100
    t_kz = 6
    filenames = []
    x = np.arange(0, 6) * 40
    x = np.repeat(x[None], axis=0, repeats=256)
    temporal = model.temporal_pooling.data.cpu().squeeze().numpy()
    mean = np.mean(temporal, axis=0)
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    for idx in range(0, 256):
        plt.plot(x[idx], temporal[idx])
    plt.subplot(2, 1, 2)
    plt.plot(x[0], mean, label="mean")

    plt.xlabel("times (ms)")
    plt.ylabel("temporal pooling weight")
    plt.legend()
    plt.grid(True)
    plt.show()
    neural_representation = model._get_v1_order()

    fs = np.array([ne["fs"] for ne in neural_representation])
    ft = np.array([ne["ft"] for ne in neural_representation])

    ax1 = plt.subplot(131, projection='polar')
    theta_list = []
    v_list = []
    energy_list = []
    for index in range(len(neural_representation)):
        v = neural_representation[index]["speed"]
        theta = neural_representation[index]["theta"]
        theta_list.append(theta)
        v_list.append(v)

    v_list, theta_list = np.array(v_list), np.array(theta_list)
    x, y = pol2cart(v_list, theta_list)
    plt.scatter(theta_list, v_list, c=v_list, cmap="rainbow", s=(v_list + 20), alpha=0.8)
    plt.axis('on')
    # plt.colorbar()
    plt.grid(True)
    # plt.subplot(132, projection="polar")
    # plt.scatter(theta_list, np.ones_like(theta_list))
    plt.subplot(132, projection='polar')
    plt.scatter(theta_list, np.ones_like(v_list))
    lst = []
    for scale in range(8):
        lst += ["scale %d" % scale] * 32
    data = {"Spatial Frequency": fs, 'Temporal Frequency': ft, "Class": lst}
    df = pd.DataFrame(data=data)
    ax = plt.subplot(133, projection='polar')
    # theta_list = theta_list[v_list > (ft * v_list.mean())]
    print(len(theta_list))
    bins_number = 8  # the [0, 360) interval will be subdivided into this
    # number of equal bins
    zone = np.pi / 8
    theta_list[theta_list < (-np.pi + zone)] = theta_list[theta_list < (-np.pi + zone)] + np.pi * 2
    bins = np.linspace(-np.pi + zone, np.pi + zone, bins_number + 1)
    n, _, _ = plt.hist(theta_list, bins, edgecolor="black")
    # ax.set_theta_offset(-np.pi / 8 - np.pi)
    ax.set_yticklabels([])
    plt.grid(True)
    import seaborn as sns
    sns.jointplot(data=df, x="Spatial Frequency", y="Temporal Frequency", hue="Class", xlim=[0, 0.3], ylim=[0, 0.3])
    plt.grid(True)
    g = sns.jointplot(data=df, x="Spatial Frequency", y="Temporal Frequency", xlim=[0, 0.25], ylim=[0, 0.25])
    # g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)

    plt.grid(True)
    plt.show()

    # show spatial frequency preference and temporal frequency preference.

    x = np.linspace(0, t_kz, t_point)
    index = 0
    for scale in range(len(model.spatial_filter)):
        t_sin, t_cos = model.temporal_filter[scale].demo_temporal_filter(points=t_point)
        gb_sin_b, gb_cos_b = model.spatial_filter[scale].demo_gabor_filters(points=s_point)
        for i in range(gb_sin_b.size(0)):
            plt.figure(figsize=(14, 9), dpi=80)
            plt.subplot(2, 3, 1)
            curve = gb_sin_b[i].squeeze().detach().numpy()
            plt.imshow(curve)
            plt.title("Gabor Sin")
            plt.subplot(2, 3, 2)
            curve = gb_cos_b[i].squeeze().detach().numpy()
            plt.imshow(curve)
            plt.title("Gabor Cos")

            plt.subplot(2, 3, 3)
            curve = t_sin[i].squeeze().detach().numpy()
            plt.plot(x, curve, label='sin')
            plt.title("Temporal Sin")

            curve = t_cos[i].squeeze().detach().numpy()
            plt.plot(x, curve, label='cos')
            plt.xlabel('Time (s)')
            plt.ylabel('Response to pulse at t=0')
            plt.legend()
            plt.title("Temporal filter")

            gb_sin = gb_sin_b[i].squeeze().detach()[5, :]
            gb_cos = gb_cos_b[i].squeeze().detach()[5, :]

            a = np.outer(t_cos[i].detach(), gb_sin)
            b = np.outer(t_sin[i].detach(), gb_cos)
            g_o = a + b

            a = np.outer(t_sin[i].detach(), gb_sin)
            b = np.outer(t_cos[i].detach(), gb_cos)
            g_e = a - b
            energy_component = g_o ** 2 + g_e ** 2

            plt.subplot(2, 3, 4)
            curve = g_o
            plt.imshow(curve, cmap="gray")
            plt.title("Spatial Temporal even")
            plt.subplot(2, 3, 5)
            curve = g_e
            plt.imshow(curve, cmap="gray")
            plt.title("Spatial Temporal odd")

            plt.subplot(2, 3, 6)
            curve = energy_component
            plt.imshow(curve, cmap="gray")
            plt.title("energy")
            plt.savefig('filter_%d.png' % (index))
            filenames.append('filter_%d.png' % (index))
            index += 1
            # plt.show()

    # build gif
    with imageio.get_writer('filters_orientation.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Remove files
    for filename in set(filenames):
        os.remove(filename)


if __name__ == "__main__":
    # V1.demo()
    # draw_polar()
    # # V1.demo()
    # # draw_polar()
    show_trained_model()
    # te_spatial_temporal()
