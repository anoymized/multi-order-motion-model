import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.warp_utils import boundary_dilated_flow_warp
# from skimage.color import rgb2yuv
import cv2
from fast_slic.avx2 import SlicAvx2 as Slic
import torch.nn.functional as F
from copy import deepcopy
from texturize import api, commands
import PIL
import scipy.ndimage as ndimage

class WaterWave(object):
    def __init__(self, SeqLen=10, height=400, width=200, SF=3, SExp=0.2, TF=2, TExp=1, Length=1, random_shuffle=False):
        self.height = height
        self.width = width
        self.SF = SF
        self.SExp = SExp
        self.TF = TF
        self.TExp = TExp
        self.Length = Length
        self.SR = SeqLen
        self._kr = None
        self._iteration = 0
        self.random_shuffle = random_shuffle
        self.blur_c1 = GaussianBlurConv(channels=1).cuda()
        self.warp_ratio = 0.5

    def create_seq_buffer(self, mask_batch):
        # input: image_batch: [ C, H, W]
        # mask_batch: [ 1, H, W]
        # output: image_batch: [ C, H, W]
        kr_list = []
        iteration_list = []
        mask_batch = mask_batch[0]
        # mask_batch = torch.ones_like(image_batch[:, 0, :, :]).unsqueeze(dim=1)
        # mask_batch[mask != 0] = 0
        for mask in mask_batch:
            # mask_batch
            # H, W = mask.shape[1:]
            sfactor = np.random.uniform(2, 5)
            tfactor = np.random.uniform(0.5, 2)
            indices = (mask > 0.1).nonzero()
            most_left = indices[:, 2].min()
            most_right = indices[:, 2].max()
            most_top = indices[:, 1].min()
            most_bottom = indices[:, 1].max()
            # get the location of the top left corner
            location_topleft = [0, most_top, most_left]
            # get the location of the bottom right corner
            location_bottomright = [0, most_bottom, most_right]
            H = location_bottomright[1] - location_topleft[1]
            W = location_bottomright[2] - location_topleft[2]

            H = int(max(H, W) * 1.2)
            W = int(max(H, W) * 1.2)

            if H < 20 or W < 20:
                print('mask too small, skip this mask')
                kr_list.append(None)
                iteration_list.append(None)
                return False
                # continue  # skip this mask
            Param = {}
            Param['H'] = H  # pixel
            Param['W'] = W  # pixel
            Param['SR'] = self.SR  # Hz
            Param['Length'] = 1  # sec
            Param['Num'] = 100
            if self.random_shuffle:
                Param['Num'] = 2
            Param['MaxGrad'] = min(H, W)  # max pixel
            t = np.linspace(0, 2 * np.pi * Param['Length'], Param['Length'] * Param['SR'])
            Rx = np.linspace(-np.pi, np.pi, Param['W'])
            Ry = np.linspace(-np.pi, np.pi, Param['H'])
            rxx, ryy = np.meshgrid(Rx, Ry)
            Rrr = np.sqrt(rxx ** 2 + ryy ** 2)
            PosRandX = np.random.rand(Param['Num']) * (2 * np.pi) - np.pi
            PosRandY = np.random.rand(Param['Num']) * (2 * np.pi) - np.pi
            xyrand = np.random.rand(Param['Num']) * (2 * np.pi)
            tRand = np.random.rand(Param['Num']) * np.pi + 0.5
            TOri = np.tile(t, (Param['Num'], 1)) * tRand[:, np.newaxis]
            kernel = np.zeros((Param['H'], Param['W'], len(t)))
            SF = np.random.rand(Param['Num']) * 1 * sfactor
            TF = np.random.rand(Param['Num']) * 2 * tfactor
            TExp = np.random.rand(Param['Num']) / 2 + 0.5
            SExp = np.random.rand(Param['Num']) / 2

            for k in range(len(t)):
                kernal_cul = np.zeros((Param['H'], Param['W']))
                if np.random.binomial(1, 0.90):
                    RandNum = np.random.randint(1, Param['Num'])
                    for j in range(RandNum):
                        PosRandX = np.append(PosRandX, np.random.rand() * 2 * np.pi - np.pi)
                        PosRandY = np.append(PosRandY, np.random.rand() * 2 * np.pi - np.pi)
                        xyrand = np.append(xyrand, np.random.rand() * 2 * np.pi - np.pi)
                        tRand = np.append(tRand, np.random.rand() * np.pi + 0.5)
                        temp = (t - t[k]) * tRand[-1]
                        TOri = np.vstack([TOri, temp])
                        SF = np.append(SF, np.random.rand() * 1 * sfactor)
                        TF = np.append(TF, np.random.rand() * 2 * tfactor)
                        TExp = np.append(TExp, np.random.rand() / 2 + 0.5)
                        SExp = np.append(SExp, np.random.rand() / 2)
                for j in range(len(PosRandX)):
                    x = xyrand[j] * np.linspace(-np.pi, np.pi, Param['W']) + PosRandX[j]
                    y = xyrand[j] * np.linspace(-np.pi, np.pi, Param['H']) + PosRandY[j]
                    xx, yy = np.meshgrid(x, y)
                    rr = np.sqrt(xx ** 2 + yy ** 2)
                    wave = np.cos(SF[j] * rr) * np.exp(-SExp[j] * rr ** 2) * np.cos(
                        TF[j] * TOri[j, k]) * np.exp(-TExp[j] * TOri[j, k] ** 2)
                    wave = wave * np.exp(-Rrr ** 2 / (0.25 * np.pi ** 2)) * np.random.randn() * 2
                    # kernal[:, :, k, j]
                    kernal_cul = wave + kernal_cul
                kernel[:, :, k] = kernal_cul
                if self.random_shuffle:
                    # random generate a kernel of gaussain dist with the same size
                    keneral = np.random.randn(Param['H'], Param['W']) * 0.1
                    # bulr the kernel with gaussian filter using scipy

                    keneral = ndimage.filters.gaussian_filter(keneral, sigma=np.random.uniform(0.1, 5))
                    kernel[:, :, k] = keneral
                # Global Constraint

            kr_list.append(kernel)
            iteration_list.append(0)
        self._kr = kr_list
        self._iteration = iteration_list
        return True

    def clear_buffer(self):
        del self._kr
        del self._iteration
        self._kr = None
        self._iteration = 0

    def infer(self, image_batch, mask_batch, mask_idx=0):
        assert self._kr is not None
        warp_ratio = self.warp_ratio + np.random.uniform(-0.5 * self.warp_ratio, 0.5 * self.warp_ratio)

        kr = self._kr[mask_idx]
        iteration = self._iteration[mask_idx]

        if kr is None:
            return image_batch

        indices = (mask_batch > 0.1).nonzero()
        if len(indices) == 0:
            print("no mask")
            return None
        most_left = indices[:, 2].min()
        most_right = indices[:, 2].max()
        most_top = indices[:, 1].min()
        most_bottom = indices[:, 1].max()
        # get the location of the top left corner
        location_topleft = [0, most_top, most_left]
        # get the location of the bottom right corner
        location_bottomright = [0, most_bottom, most_right]
        ###########################################
        # import matplotlib.pyplot as plt
        # print(location_topleft)
        # print(location_bottomright)
        # plt.imshow(mask_batch.squeeze().detach().cpu().numpy())
        # plt.show()
        ###########################################
        H = location_bottomright[1] - location_topleft[1]
        W = location_bottomright[2] - location_topleft[2]
        flow_max = max(H, W) * warp_ratio

        start_point = torch.zeros(1, 2, 1, 1).type_as(image_batch).cuda()
        start_point[:, 0, 0, 0] = location_topleft[1]
        start_point[:, 1, 0, 0] = location_topleft[2]

        # gx = np.gradient(kr[:, :, iteration], axis=1)
        # gy = np.gradient(kr[:, :, iteration], axis=0)
        grad_x, grad_y = np.gradient(kr[:, :, iteration])
        grad_x = torch.from_numpy(grad_x).unsqueeze(0).float().cuda()
        grad_y = torch.from_numpy(grad_y).unsqueeze(0).float().cuda()

        flow = torch.cat([grad_x, grad_y], dim=0).unsqueeze(0)
        flow = (flow / flow.max()) * flow_max
        blur_mask = deepcopy(mask_batch)
        blur_mask = blur_mask.unsqueeze(0)
        for i in range(10):
            blur_mask = self.blur_c1(blur_mask)

        # normalize grid to [-1, 1]
        # warp image using grid
        # blur mask_bat
        img_copy = boundary_dilated_flow_warp(image_batch.unsqueeze(0), flow.cuda(), start_point)
        img_copy = img_copy * blur_mask.unsqueeze(0) + image_batch.unsqueeze(0) * (1 - blur_mask.unsqueeze(0))

        self._iteration[mask_idx] = self._iteration[mask_idx] + 1
        return img_copy.squeeze()


class SwirlWave(object):
    def __init__(self, SeqLen=10, height=400, width=200, SF=3, SExp=0.2, TF=2, TExp=1, Length=1, random_shuffle=False):
        self.height = height
        self.width = width
        self.SF = SF
        self.SExp = SExp
        self.TF = TF
        self.TExp = TExp
        self.Length = Length
        self.SR = SeqLen
        self._kr = None
        self._iteration = 0
        self.random_shuffle = random_shuffle
        self.blur_c1 = GaussianBlurConv(channels=1).cuda()
        self.warp_ratio = 0.5

    @staticmethod
    def vortex_flow(X, Y, t, strength, frequency, radius=1):
        # t = 1
        omega = 2 * np.pi
        t = 1
        # omega = omega % (2 * np.pi)

        X_rot = X * np.cos(omega * t) - Y * np.sin(omega * t)
        Y_rot = X * np.sin(omega * t) + Y * np.cos(omega * t)

        # u = strength / (2 * np.pi) * (Y / (X ** 2 + Y ** 2 + radius ** 2))
        # v = -strength / (2 * np.pi) * (X / (X ** 2 + Y ** 2 + radius ** 2))
        u = strength / (2 * np.pi) * (Y_rot / (X_rot ** 2 + Y_rot ** 2 + radius ** 2))
        v = -strength / (2 * np.pi) * (X_rot / (X_rot ** 2 + Y_rot ** 2 + radius ** 2))
        # decay with time

        return u, v

    def create_seq_buffer_swirl(self, mask_batch):
        # input: image_batch: [ C, H, W]
        # mask_batch: [ 1, H, W]
        # output: image_batch: [ C, H, W]
        kr_list = []
        iteration_list = []
        mask_batch = mask_batch[0]
        # mask_batch = torch.ones_like(image_batch[:, 0, :, :]).unsqueeze(dim=1)
        # mask_batch[mask != 0] = 0
        for mask in mask_batch:
            # mask_batch
            # H, W = mask.shape[1:]

            indices = (mask > 0.1).nonzero()
            most_left = indices[:, 2].min()
            most_right = indices[:, 2].max()
            most_top = indices[:, 1].min()
            most_bottom = indices[:, 1].max()
            # get the location of the top left corner
            location_topleft = [0, most_top, most_left]
            # get the location of the bottom right corner
            location_bottomright = [0, most_bottom, most_right]
            H = location_bottomright[1] - location_topleft[1]
            W = location_bottomright[2] - location_topleft[2]

            H = int(max(H, W) * 1.2)
            W = int(max(H, W) * 1.2)

            if H < 20 or W < 20:
                print('mask too small, skip this mask')
                kr_list.append(None)
                iteration_list.append(None)
                return False
                # continue  # skip this mask

            Param = {}
            Param['H'] = H  # pixel
            Param['W'] = W  # pixel
            Param['TF'] = np.random.uniform(2, 8)  # temporal frequency
            Param['TExp'] = 0.5
            Param['SR'] = self.SR  # Hz
            Param['Length'] = 1  # sec
            Param['Num'] = 200
            Param['MaxGrad'] = min(H, W)  # max pixel

            t = np.linspace(0, 2 * np.pi * Param['Length'], Param['Length'] * Param['SR'])
            Rx = np.linspace(-np.pi, np.pi, Param['W'])
            Ry = np.linspace(-np.pi, np.pi, Param['H'])
            rxx, ryy = np.meshgrid(Rx, Ry)
            Rrr = np.sqrt(rxx ** 2 + ryy ** 2)
            tRand = np.random.rand(Param['Num']) * np.pi + 0.5
            TOri = np.tile(t, (Param['Num'], 1)) * tRand[:, np.newaxis]
            xyrand = np.random.rand(Param['Num']) * (2 * np.pi)
            kernelX = np.zeros((Param['H'], Param['W'], len(t)))
            kernelY = np.zeros((Param['H'], Param['W'], len(t)))
            PosRandX = np.random.rand(Param['Num']) * (2 * np.pi) - np.pi
            PosRandY = np.random.rand(Param['Num']) * (2 * np.pi) - np.pi
            Radius = np.random.rand(Param['Num']) * 1.0 + 0.1
            Frequency = np.random.rand(Param['Num']) * 5 + 2
            Strength = np.random.rand(Param['Num']) * 5 + 1
            T = np.random.rand(Param['Num'])
            for k in range(len(t)):
                kernal_cul_x = np.zeros((Param['H'], Param['W']))
                kernal_cul_y = np.zeros((Param['H'], Param['W']))

                if np.random.binomial(1, 0.8):
                    RandNum = np.random.randint(1, Param['Num'] + 1)
                    for j in range(RandNum):
                        PosRandX = np.append(PosRandX, np.random.rand() * 2 * np.pi - np.pi)
                        PosRandY = np.append(PosRandY, np.random.rand() * 2 * np.pi - np.pi)
                        xyrand = np.append(xyrand, np.random.rand() * 2 * np.pi - np.pi)
                        tRand = np.append(tRand, np.random.rand() * np.pi + 0.5)
                        Radius = np.append(Radius, np.random.rand() * 1.5 + 0.05)
                        Frequency = np.append(Frequency, np.random.rand() * 5 + 2)
                        Strength = np.append(Strength, np.random.rand() * 5 * Radius)
                        T = np.append(T, np.random.rand())
                        temp = (t - t[k]) * tRand[-1]
                        TOri = np.vstack([TOri, temp])
                        TOri = np.vstack([TOri, temp])
                for j in range(len(PosRandX)):
                    x = xyrand[j] * np.linspace(-np.pi, np.pi, Param['W']) + PosRandX[j]
                    y = xyrand[j] * np.linspace(-np.pi, np.pi, Param['H']) + PosRandY[j]
                    xx, yy = np.meshgrid(x, y)
                    flowx, flowy = SwirlWave.vortex_flow(xx, yy, T[j], Strength[j], Frequency[j], Radius[j])
                    flowx = flowx * np.exp(-Param['TExp'] * TOri[j, k] ** 2)
                    flowy = flowy * np.exp(-Param['TExp'] * TOri[j, k] ** 2)

                    flowx = flowx * np.exp(-Rrr ** 2 / (0.5 * np.pi ** 2))
                    flowy = flowy * np.exp(-Rrr ** 2 / (0.5 * np.pi ** 2))
                    # kernal[:, :, k, j]
                    kernal_cul_x = kernal_cul_x + flowx
                    kernal_cul_y = kernal_cul_y + flowy

                kernelX[:, :, k] = kernal_cul_x
                kernelY[:, :, k] = kernal_cul_y
            kr_list.append([kernelX, kernelY])  # (H, W, T)
            iteration_list.append(0)
        self._kr = kr_list
        self._iteration = iteration_list
        return True

    def clear_buffer(self):
        del self._kr
        del self._iteration
        self._kr = None
        self._iteration = 0

    def infer(self, image_batch, mask_batch, mask_idx=0):
        assert self._kr is not None
        warp_ratio = self.warp_ratio + np.random.uniform(-0.5 * self.warp_ratio, 0.5 * self.warp_ratio)

        kr = self._kr[mask_idx]
        iteration = self._iteration[mask_idx]

        if kr is None:
            return image_batch

        indices = (mask_batch > 0.1).nonzero()
        if len(indices) == 0:
            print("no mask")
            return None
        most_left = indices[:, 2].min()
        most_right = indices[:, 2].max()
        most_top = indices[:, 1].min()
        most_bottom = indices[:, 1].max()
        # get the location of the top left corner
        location_topleft = [0, most_top, most_left]
        # get the location of the bottom right corner
        location_bottomright = [0, most_bottom, most_right]
        ###########################################
        # import matplotlib.pyplot as plt
        # print(location_topleft)
        # print(location_bottomright)
        # plt.imshow(mask_batch.squeeze().detach().cpu().numpy())
        # plt.show()
        ###########################################
        H = location_bottomright[1] - location_topleft[1]
        W = location_bottomright[2] - location_topleft[2]
        flow_max = max(H, W) * warp_ratio

        start_point = torch.zeros(1, 2, 1, 1).type_as(image_batch).cuda()
        start_point[:, 0, 0, 0] = location_topleft[1]
        start_point[:, 1, 0, 0] = location_topleft[2]

        # gx = np.gradient(kr[:, :, iteration], axis=1)
        # gy = np.gradient(kr[:, :, iteration], axis=0)
        grad_x, grad_y = kr[0][:, :, iteration], kr[1][:, :, iteration]
        grad_x = torch.from_numpy(grad_x).unsqueeze(0).float().cuda()
        grad_y = torch.from_numpy(grad_y).unsqueeze(0).float().cuda()

        flow = torch.cat([grad_x, grad_y], dim=0).unsqueeze(0)
        flow = (flow / flow.max()) * flow_max
        blur_mask = deepcopy(mask_batch)
        blur_mask = blur_mask.unsqueeze(0)
        for i in range(10):
            blur_mask = self.blur_c1(blur_mask)

        # normalize grid to [-1, 1]
        # warp image using grid
        # blur mask_bat
        img_copy = boundary_dilated_flow_warp(image_batch.unsqueeze(0), flow.cuda(), start_point)
        img_copy = img_copy * blur_mask.unsqueeze(0) + image_batch.unsqueeze(0) * (1 - blur_mask.unsqueeze(0))

        self._iteration[mask_idx] = self._iteration[mask_idx] + 1
        return img_copy.squeeze()


def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW
    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid


def denormalize_coords(flow):
    """ scale indices from [-1, 1] to [0, width/height] """
    _, c, h, w = flow.shape
    flow[:, 0, :, :] = 0.5 * (w - 1.0) * (flow[:, 0, :, :] + 1.0)
    flow[:, 1, :, :] = 0.5 * (h - 1.0) * (flow[:, 1, :, :] + 1.0)
    return flow


def local_process_image(image_batch, mask_batch, processor=None, min_size=128, div=32):
    indices = (mask_batch > 1e-6).nonzero()
    # get the location of the most left point
    most_left = indices[:, 2].min()
    most_right = indices[:, 2].max()
    most_top = indices[:, 1].min()
    most_bottom = indices[:, 1].max()
    # get the location of the top left corner
    location_topleft = [0, most_top, most_left]
    # get the location of the bottom right corner
    location_bottomright = [0, most_bottom, most_right]
    H = location_bottomright[1] - location_topleft[1]
    W = location_bottomright[2] - location_topleft[2]
    # get the square region that fully contains the mask
    if H > W:
        location_topleft[2] = max(0, location_topleft[2] - (H - W) // 2)
        location_bottomright[2] = min(image_batch.shape[2], location_bottomright[2] + (H - W) // 2)
    else:
        location_topleft[1] = max(0, location_topleft[1] - (W - H) // 2)
        location_bottomright[1] = min(image_batch.shape[1], location_bottomright[1] + (W - H) // 2)
    # avoid too small local region
    if H < min_size:
        location_topleft[1] = max(0, location_topleft[1] - (min_size - H) // 2)
        location_bottomright[1] = min(image_batch.shape[1], location_bottomright[1] + (min_size - H) // 2)
    if W < min_size:
        location_topleft[2] = max(0, location_topleft[2] - (min_size - W) // 2)
        location_bottomright[2] = min(image_batch.shape[2], location_bottomright[2] + (min_size - W) // 2)

    # adjust the image to be divisible by 64
    location_topleft[1] = (location_topleft[1] // div - 1) * div
    location_topleft[2] = (location_topleft[2] // div - 1) * div
    location_bottomright[1] = (location_bottomright[1] // div + 1) * div
    location_bottomright[2] = (location_bottomright[2] // div + 1) * div

    # avoid out of range
    location_topleft[1] = max(0, location_topleft[1])
    location_topleft[2] = max(0, location_topleft[2])
    location_bottomright[1] = min(image_batch.shape[1], location_bottomright[1])
    location_bottomright[2] = min(image_batch.shape[2], location_bottomright[2])

    # crop the image
    image_local = image_batch[:, location_topleft[1]:location_bottomright[1],
                  location_topleft[2]:location_bottomright[2]]
    # process the local image
    image_local_p = processor(deepcopy(image_local))
    # correct the mean luminance
    image_local_p = image_local_p * image_local.mean() / image_local_p.mean()
    # clamp the value
    image_local_p = torch.clamp(image_local_p, 0, 1)
    # paste the local image back
    image_batch[:, location_topleft[1]:location_bottomright[1],
    location_topleft[2]:location_bottomright[2]] = image_local_p * mask_batch[:,
                                                                   location_topleft[1]:location_bottomright[1],
                                                                   location_topleft[2]:location_bottomright[2]] + \
                                                   image_batch[:, location_topleft[1]:location_bottomright[1],
                                                   location_topleft[2]:location_bottomright[2]] * (
                                                           1 - mask_batch[:,
                                                               location_topleft[1]:location_bottomright[1],
                                                               location_topleft[2]:location_bottomright[2]])
    return image_batch


def water_ball_infer(image_batch, mask_batch):
    indices = (mask_batch > 0.1).nonzero()
    if len(indices) == 0:
        print("no mask")
        return None
    most_left = indices[:, 2].min()
    most_right = indices[:, 2].max()
    most_top = indices[:, 1].min()
    most_bottom = indices[:, 1].max()
    # get the location of the top left corner
    location_topleft = [0, most_top, most_left]
    # get the location of the bottom right corner
    location_bottomright = [0, most_bottom, most_right]
    ###########################################
    # import matplotlib.pyplot as plt
    # print(location_topleft)
    # print(location_bottomright)
    # plt.imshow(mask_batch.squeeze().detach().cpu().numpy())
    # plt.show()
    ###########################################
    H = location_bottomright[1] - location_topleft[1]
    W = location_bottomright[2] - location_topleft[2]
    flow_max = max(H, W) * 1.5

    start_point = torch.zeros(1, 2, 1, 1).type_as(image_batch).cuda()
    start_point[:, 0, 0, 0] = location_topleft[1]
    start_point[:, 1, 0, 0] = location_topleft[2]

    # blur the mask
    mask_blur = cv2.GaussianBlur(mask_batch.squeeze().detach().cpu().numpy(), (21, 21), 10)
    # get the gradient of the mask
    kr = mask_blur

    gx = np.gradient(kr, axis=1)
    gy = np.gradient(kr, axis=0)
    grad_x = torch.from_numpy(gx).unsqueeze(0).float().cuda()
    grad_y = torch.from_numpy(gy).unsqueeze(0).float().cuda()
    flow = torch.cat([grad_x, grad_y], dim=0).unsqueeze(0)
    flow = (flow / flow.max()) * flow_max
    # normalize grid to [-1, 1]
    # warp image using grid
    img_copy = boundary_dilated_flow_warp(image_batch.unsqueeze(0), flow.cuda(), start_point)
    return img_copy.squeeze()


def run_slic_seperate(img_batch, n_seg=200, compact=10, rd_select=(4, 8)):  # Nx1xHxW
    """

    :param img: Nx3xHxW 0~1 float32
    :param n_seg:
    :param compact:
    :return: Nx1xHxW float32
    """
    B = img_batch.size(0)
    dtype = img_batch.type()
    img_batch = np.split(
        img_batch.detach().cpu().numpy().transpose([0, 2, 3, 1]), B, axis=0)
    out_ins = []
    patch_instance = []
    n_seg = np.random.randint(int(n_seg * 0.5), int(n_seg * 1.5))
    fast_slic = Slic(num_components=n_seg, compactness=compact, min_size_factor=0.8)
    if rd_select[0] == rd_select[1]:
        n_select = rd_select[0]
    else:
        n_select = np.random.randint(rd_select[0], rd_select[1])
    for index, img in enumerate(img_batch):
        img = np.copy((img * 255).squeeze(0).astype(np.uint8), order='C')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        seg = fast_slic.iterate(img)

        select_list = np.random.choice(range(0, np.max(seg) + 1), n_select,
                                       replace=False)
        for seg_id in select_list:
            seg_msk = seg == seg_id
            patch_instance.append(seg_msk)
        out_ins.append(patch_instance)
    return out_ins


def random_texture_processor(temple):
    size = temple.shape[1:]
    # translate to PIL image
    image = PIL.Image.fromarray((temple.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    remix = commands.Remix(image)
    for result in api.process_octaves(remix, size=size[::-1], octaves=5):
        pass
    # The output can be saved in any PIL-supported format.
    temple = result.images.squeeze()
    return temple


class RandomNoiseProcessor(object):
    def __init__(self, noise_level=3):
        self.last_optical_flow = None
        self.noise_level = noise_level + np.random.uniform(-1, 1)
        self.noise_level = noise_level + np.random.uniform(-1, 1)
        self.last_state = None
        self.blur_level = np.random.randint(2, 5)
        self.factor = np.random.uniform(0.3, 0.6)

    def __call__(self, temple):
        noise = torch.randn_like(temple)
        noise = temple + noise * self.noise_level
        # gaussian blur
        noise = noise.permute(1, 2, 0).cpu().numpy()
        for i in range(self.blur_level):
            noise = cv2.GaussianBlur(noise, (11, 11), 5)

        noise = torch.from_numpy(noise).permute(2, 0, 1).float().cuda()
        noise = noise.squeeze()
        # generate random binary mask
        mask = torch.rand_like(temple)
        mask = (mask > np.random.uniform(0.3, 0.6)).float()
        # generate random noise
        temple = temple * mask + noise * (1 - mask)

        # clamp
        # temple = temple.clamp(0, 1)
        return temple


def random_fourier_phase_processor(temple):
    temple = temple.permute(1, 2, 0).cpu().numpy()

    # fourier transform of the image
    f = np.fft.fftn(temple)
    # extract the phase and amplitude
    magnitude_spectrum = np.abs(f)
    phase_spectrum = np.angle(f)
    # generate random phase while keeping the mean and std
    phase_spectrum = np.random.normal(np.mean(phase_spectrum), np.std(phase_spectrum), phase_spectrum.shape)
    # draw amplitude spectrum
    # plt.imshow(magnitude_spectrum[:,:,2])
    # plt.show()
    # combine the phase and amplitude
    f = magnitude_spectrum * np.exp(1j * phase_spectrum)
    # inverse fourier transform
    temple = np.fft.ifftn(f)
    temple = np.abs(temple)
    temple = torch.from_numpy(temple).permute(2, 0, 1).cuda()
    return temple


def random_shuffle_fourier_phase_processor(temple):
    temple = temple.permute(1, 2, 0).cpu().numpy()
    flow_max = min(temple.shape[0], temple.shape[1]) * 0.5

    # rgb to hsv
    temple = cv2.cvtColor(temple, cv2.COLOR_RGB2HSV)

    # fourier transform of the luminance
    f = np.fft.fftn(temple[:, :, 2])
    # unsqueeze
    f = f[:, :, np.newaxis]
    # fourier transform of the image
    # extract the phase and amplitude
    magnitude_spectrum = np.abs(f)
    phase_spectrum = np.angle(f)
    # to torch
    phase_spectrum = torch.from_numpy(phase_spectrum).permute(2, 0, 1).cuda()
    magnitude_spectrum = torch.from_numpy(magnitude_spectrum).permute(2, 0, 1).cuda()
    # generate random phase while keeping the mean and std
    shuffle_optical_flow_x = np.random.randn(temple.shape[0], temple.shape[1])
    shuffle_optical_flow_y = np.random.randn(temple.shape[0], temple.shape[1])
    # gaussian blur
    shuffle_optical_flow_x = cv2.GaussianBlur(shuffle_optical_flow_x, (11, 11), 5)
    shuffle_optical_flow_y = cv2.GaussianBlur(shuffle_optical_flow_y, (11, 11), 5)
    shuffle_optical_flow_y = cv2.GaussianBlur(shuffle_optical_flow_y, (11, 11), 5)
    # normalize

    grad_x = torch.from_numpy(shuffle_optical_flow_x).unsqueeze(0).float().cuda()
    grad_y = torch.from_numpy(shuffle_optical_flow_y).unsqueeze(0).float().cuda()
    flow = torch.cat([grad_x, grad_y], dim=0).unsqueeze(0)
    flow = (flow / flow.max()) * flow_max
    # normalize grid to [-1, 1]
    # warp image using grid
    start_point = torch.zeros(1, 2, 1, 1).type_as(phase_spectrum).cuda()
    start_point[:, 0, 0, 0] = 0
    start_point[:, 1, 0, 0] = 0
    # if np.random.rand() > 0.8:
    phase_spectrum = boundary_dilated_flow_warp(phase_spectrum.unsqueeze(0), flow.cuda(), start_point)[0]
    # magnitude_spectrum = boundary_dilated_flow_warp(magnitude_spectrum.unsqueeze(0), flow.cuda(), start_point)[0]

    # to numpy
    phase_spectrum = phase_spectrum.permute(1, 2, 0).cpu().numpy()
    magnitude_spectrum = magnitude_spectrum.permute(1, 2, 0).cpu().numpy()
    # keep the amplitude same

    f = magnitude_spectrum * np.exp(1j * phase_spectrum)
    # inverse fourier transform
    inv_f = np.fft.ifftn(f)
    # squeeze
    inv_f = inv_f[:, :, 0]
    temple[:, :, 2] = np.abs(inv_f)
    # hsv to rgb
    temple = cv2.cvtColor(temple, cv2.COLOR_HSV2RGB)
    # compress to [0, 1]
    temple = temple / temple.max()
    temple = torch.from_numpy(temple).permute(2, 0, 1).cuda()
    return temple


class CreateNonFourierData(object):
    def __init__(self, mean=15, sigma_i=4.0, sigma_v=0.5, rotate_range=0, sigma_rotate_v=0.0, sigma_zoom_v=0.0,
                 zoom_ini=0.0, compact=55, n_seg=50, SeqLen=25, rd_select=None, types=None, seed=None):
        super(CreateNonFourierData, self).__init__()
        self.type_list = None
        if rd_select is None:
            rd_select = [2, 5]
        self.mean = mean
        self.sigma_v = sigma_v
        self.sigma_i = sigma_i
        self.rotate_range = rotate_range
        self.key = 0
        self.zoom_ini = zoom_ini
        self.sigma_rotate_v = sigma_rotate_v
        self.sigma_zoom_v = sigma_zoom_v
        self.blur = GaussianBlurConv().cuda()
        self.blur_c1 = GaussianBlurConv(channels=1).cuda()
        self.compact = compact
        self.n_seg = n_seg
        self.if_avoid_overlap = False
        self.waterwave = WaterWave(SeqLen=SeqLen)
        self.SwirlWave = SwirlWave(SeqLen=SeqLen)
        self.occluded = False
        self.types = types
        self.random_noise_processor = RandomNoiseProcessor()
        self.rd_select = rd_select
        self.noise_blur_level = np.random.randint(1, 20)
        if seed is None:
            seed = np.random.randint(0, 10000)
        self.slic_seed = seed + 10
        self.traj_seed = seed + 20
        self.mask_blur_level = np.random.randint(40, 120)
        if types == 7:
            self.waterwave.random_shuffle = True
        else:
            self.waterwave.random_shuffle = False
        self.types = types

    def draw_img(self, mask_pixel, image_batch, types_ori=2):
        mask_batch = torch.ones_like(image_batch[:, 0, :, :]).unsqueeze(dim=1)
        mask_all = torch.zeros_like(image_batch[:, 0, :, :])
        image_batch_ori = deepcopy(image_batch)
        for index in range(image_batch.size(0)):
            for mask_idx, pixel in enumerate(mask_pixel[index]):
                if types_ori == 0:
                    types = self.type_list[mask_idx]
                else:
                    types = types_ori
                if types == 10:  # random select 1-9
                    types = np.random.randint(1, 10)
                mask = pixel[0].unsqueeze(dim=0)
                mask_batch[index][mask != 0] = 0
                mask = mask.unsqueeze(dim=0)
                for i in range(self.mask_blur_level):  # create gaussian blur
                    mask = self.to_0_1(self.blur_c1(mask))
                mask = mask.squeeze(dim=0)
                mask_all = mask_all + mask
                if types == 1:  # random noise
                    temple = local_process_image(image_batch[index], mask, processor=self.random_noise_processor)
                elif types == 2:  # second order with blur
                    temple = self.to_0_1(image_batch[index])
                    for i in range(self.noise_blur_level * 10 + 20):
                        temple = self.blur(temple.unsqueeze(dim=0)).squeeze(dim=0)
                    temple = temple * (image_batch[index].mean() / temple.mean())  # correct the luminance
                elif types == 3:  # second order with water wave
                    temple = self.waterwave.infer(image_batch[index], mask, mask_idx)
                    if temple is None:
                        temple = image_batch[index]
                        self.occluded = True
                elif types == 4:  # second order random texture
                    temple = local_process_image(image_batch[index], mask, processor=random_texture_processor)
                elif types == 5:  # luminance flip
                    temple = image_batch[index]
                    factor = np.random.rand() * 0.5 + 0.1
                    if np.random.rand() > 0.5:
                        temple = (temple * (1+factor)).clamp(0, 1)
                    else:
                        temple = (temple * (1-factor)).clamp(0, 1)
                elif types == 6:  # random optical flow shuffle fourier phase
                    temple = local_process_image(image_batch[index], mask,
                                                 processor=random_shuffle_fourier_phase_processor)
                elif types == 7:  # random flow shuffle
                    temple = self.waterwave.infer(image_batch[index], mask, mask_idx)
                    if temple is None:
                        temple = image_batch[index]
                        self.occluded = True
                elif types == 8:  # swirl
                    temple = self.SwirlWave.infer(image_batch[index], mask, mask_idx)
                    if temple is None:
                        temple = image_batch[index]
                        self.occluded = True
                elif types == 9:  # pure drift balanced motion
                    temple = torch.rand_like(image_batch[index])
                    temple = temple.unsqueeze(dim=0)
                    for i in range(self.noise_blur_level):
                        temple = self.blur(temple)
                    temple = temple.squeeze(dim=0)
                    temple = temple * (image_batch[index].mean() / temple.mean())
                    # clamp the luminance
                    temple = temple.clamp(0, 1)

                image_batch[index] = temple
            # mask_temp = 1 - mask_batch[index]
            # mask_temp = mask_temp.unsqueeze(dim=0)
            # for i in range(20):  # create gaussian blur
            #     mask_temp = self.to_0_1(self.blur_c1(mask_temp))
            mask_temp = mask_all.clamp(0, 1)
            image_batch[index] = (image_batch[index] * mask_temp + image_batch_ori[index] * (1 - mask_temp)).clamp(0, 1)
            # import matplotlib.pyplot as plt
            # plt.imshow(mask.cpu().numpy())
            # plt.show()
        return image_batch, mask_batch


    def pol2cart(self, rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)

    def to_0_1(self, x):
        return (x - x.min()) / (x.max() - x.min())


    @torch.no_grad()
    def forward_second_order(self, image_seq, types=1):
        img_start = image_seq[0]
        if img_start.shape[0] > 1:
            a, b = torch.chunk(img_start, chunks=2)
            img_start = torch.cat([b, a], dim=0)
        mask_pixel = []
        output = []
        output_mask = []
        flow_list = []
        flow_masked = []
        pixel_mask_list = []
        prev_s = None
        h, w = img_start.shape[-1], img_start.shape[-2]
        noise = torch.rand(img_start.size()).type_as(img_start)
        img_roll = img_start
        np.random.seed(self.slic_seed)
        mask_ins = run_slic_seperate(img_start, n_seg=self.n_seg, compact=self.compact, rd_select=self.rd_select)
        noise = self.to_0_1(noise)

        noise[noise == self.key] = self.key + 1e-4
        for index, img_ins in enumerate(mask_ins):
            mask_pixel.append(
                [torch.tensor(occ_ins, dtype=torch.int).type_as(img_start) * noise[index] for occ_ins in img_ins])
        flow_last = None

        blur_level = 20
        for i in range(blur_level):
            mask_pixel = [[self.blur(self.blur(mask_pixel.unsqueeze(0))).squeeze() for mask_pixel in mask_pixel[0]]]
        for idx, temp in enumerate(mask_pixel[0]):
            mask_pixel[0][idx][temp > 0.05] = 1.0
            mask_pixel[0][idx][temp != 1] = 0.0
        pixel_mask = deepcopy(mask_pixel)
        length = len(mask_pixel[0])
        self.type_list = np.random.randint(1, 10, length)
        image_seq.reverse()
        self.occluded = False
        if types == 10:
            if np.random.uniform(0, 1, 1) > 0.5:
                self.waterwave.random_shuffle = True
            else:
                self.waterwave.random_shuffle = False
            if not self.waterwave.create_seq_buffer(pixel_mask):
                return None, None, None
            if not self.SwirlWave.create_seq_buffer_swirl(pixel_mask):
                return None, None, None
        if types == 0:
            if np.random.uniform(0, 1, 1) > 0.5:
                self.waterwave.random_shuffle = True
            else:
                self.waterwave.random_shuffle = False
            if not self.waterwave.create_seq_buffer(pixel_mask):
                return None, None, None
            if not self.SwirlWave.create_seq_buffer_swirl(pixel_mask):
                return None, None, None

        if types == 7 or types == 3:
            if not self.waterwave.create_seq_buffer(pixel_mask):
                return None, None, None
        if types == 8:
            if not self.SwirlWave.create_seq_buffer_swirl(pixel_mask):
                return None, None, None
        if types == 9:  # pure drift balance noise
            ranodm_noise = torch.rand(image_seq[0].size()).type_as(image_seq[0])
            # blur the noise
            for i in range(self.noise_blur_level):
                ranodm_noise = self.blur(ranodm_noise)
            # use the noise to replace the image
            image_seq = [deepcopy(ranodm_noise) for i in range(len(image_seq))]

        for index, image in enumerate(image_seq):
            if self.occluded:
                self.waterwave.clear_buffer()
                return None, None, None
            image, mask = self.draw_img(pixel_mask, image, types_ori=types)
            output.append(image)
            output_mask.append(mask)
            pixel_mask_list.append(deepcopy(pixel_mask))
            if index == len(image_seq) - 1:
                break
            np.random.seed(self.traj_seed + index)

        self.waterwave.clear_buffer()
        pixel_mask_list.reverse()

        image_seq.reverse()
        output.reverse()
        output_mask.reverse()
        return output, output_mask


class GaussianBlurConv(torch.nn.Module):
    def __init__(self, channels=3):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
        kernel = np.array(kernel) / np.array(kernel).sum()
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = torch.nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        x = torch.nn.functional.conv2d(x, self.weight, padding='same', groups=self.channels)
        return x
