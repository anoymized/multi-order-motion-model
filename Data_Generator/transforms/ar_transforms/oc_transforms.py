import numpy as np
import torch
import torchvision
# from skimage.color import rgb2yuv
import cv2
from fast_slic.avx2 import SlicAvx2 as Slic
from skimage.segmentation import slic as sk_slic
import torch.nn.functional as F
from copy import deepcopy
from utils.flow_utils import evaluate_flow, InputPadder, flow_to_image
from torchvision import transforms as tf
from utils.flow_utils import InputPadder, flow_to_image, save_img_seq, viz_img_seq


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


class ColorJitter(tf.ColorJitter):
    def __call__(self, imgs, if_shadow):
        if if_shadow:
            transform = self.get_params([0.2, 0.9], self.contrast,
                                        self.saturation, self.hue)
        else:
            transform = self.get_params(self.brightness, self.contrast,
                                        self.saturation, self.hue)
        a = np.random.randint(-2, 3, 1)
        factor = np.random.uniform(0.97, 1.03, 1)
        delta = factor - 1
        imgs = ToPILImage()(imgs)
        imgs = [transform(im) for im in imgs]
        if a == 1:
            imgs = [self.get_params([1.0, 1.1], [1.0, 1.1],
                                    [1.0, 1.1], None)(im) for im in imgs]
        if a == -1:
            imgs = [self.get_params([0.9, 1.0], [0.9, 1.0],
                                    [0.9, 1.0], None)(im) for im in imgs]
        if abs(a) == 2:
            for index in range(len(imgs)):
                imgs[index] = self.get_params([factor, factor], [factor, factor],
                                              [factor, factor], None)(imgs[index])
                factor = delta + factor

        return ToTensor()(imgs)


class ToPILImage(tf.ToPILImage):
    def __call__(self, imgs):
        return [super(ToPILImage, self).__call__(im.cpu()) for im in imgs]


class ToTensor(tf.ToTensor):
    def __call__(self, imgs):
        return [super(ToTensor, self).__call__(im).cuda().unsqueeze(0) for im in imgs]


def run_slic_pt(img_batch, n_seg=200, compact=10, rd_select=(8, 16), fast=True):  # Nx1xHxW
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
    out = []
    if fast:
        fast_slic = Slic(num_components=n_seg, compactness=compact, min_size_factor=0.8)
    for img in img_batch:
        img = np.copy((img * 255).squeeze(0).astype(np.uint8), order='C')
        if fast:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            seg = fast_slic.iterate(img)
        else:
            seg = sk_slic(img, n_segments=200, compactness=10)

        if rd_select is not None:
            n_select = np.random.randint(rd_select[0], rd_select[1])
            select_list = np.random.choice(range(0, np.max(seg) + 1), n_select,
                                           replace=False)

            seg = np.bitwise_or.reduce([seg == seg_id for seg_id in select_list])
        out.append(seg)
    x_out = torch.tensor(np.stack(out)).type(dtype).unsqueeze(1)
    return x_out


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
        patch_instance = []
    return out_ins


class CreateOCCSeq(object):
    def __init__(self, mean=15, sigma_i=4.0, sigma_v=0.5, rotate_range=6, if_avoid_overlap=True):
        super(CreateOCCSeq, self).__init__()
        self.blur = GaussianBlurConv().cuda()
        self.blur_mask = GaussianBlurConv(1).cuda()
        self.if_avoid_overlap = if_avoid_overlap
        self.mean = mean
        self.sigma_v = sigma_v
        self.sigma_i = sigma_i
        self.rotate_range = rotate_range
        self.key = 0
        self.zoom_ini = 0.01
        self.sigma_rotate_v = 0.2
        self.sigma_zoom_v = 0.01
        self.mask = None

    def draw_img(self, mask_pixel, image_batch):
        mask_batch = torch.ones_like(image_batch[:, 0, :, :]).unsqueeze(dim=1)
        for index in range(image_batch.size(0)):
            for pixel in mask_pixel[index]:
                image_batch[index][pixel != self.key] = pixel[pixel != self.key]
                mask = pixel[0].unsqueeze(dim=0)
                mask_batch[index][mask != 0] = 0
        return image_batch, mask_batch

    def create_flow(self, image_batch, mask_pixel, flow_var):
        flow_bach = torch.zeros_like(image_batch[:, :2, :, :])
        for index in range(image_batch.size(0)):
            for pixel, flow in zip(mask_pixel[index], flow_var[index]):
                # image_batch[index][pixel != self.key] = pixel[pixel != self.key]  # de
                mask = pixel[0].unsqueeze(dim=0)
                flow_index = mask.expand(2, -1, -1)
                flow_bach[index][flow_index != self.key] = flow.squeeze()[flow_index != self.key]
        return flow_bach

    def pol2cart(self, rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)

    def affine_gird(self, image, angle, x, y, zoom, flow_last):
        w, h = image.shape[-1], image.shape[-2]
        x = x / w
        y = y / h
        angle = torch.tensor(-angle * torch.pi / 180, dtype=torch.float)
        theta = torch.tensor([
            [torch.cos(angle) * zoom, torch.sin(-angle) * zoom, float(x)],
            [torch.sin(angle) * zoom, torch.cos(angle) * zoom, float(y)]
        ], dtype=torch.float).type_as(image)

        image = image.unsqueeze(0)
        grid = F.affine_grid(theta.unsqueeze(0), size=image.shape, align_corners=True)
        image = F.grid_sample(image, grid, mode="bilinear", padding_mode='zeros', align_corners=True).squeeze()
        ini_gird = mesh_grid(1, h, w).type_as(image)
        grid = denormalize_coords(grid.permute(0, 3, 1, 2))
        if flow_last is None:
            flow_var = grid - ini_gird
        else:
            flow_var = (grid - ini_gird) - (flow_last - ini_gird)

        flow_last = deepcopy(grid)
        return image, flow_last, flow_var

    def occlusion_detection(self, images, image, x, y, r, r_count, x_c, y_c, index, recurrence_index=0):
        recurrence_index = recurrence_index + 1
        if recurrence_index > 5:  # max recurrent time
            return 0, 0, 0
        x_t = x + x_c
        y_t = y + y_c
        r_t = r + r_count
        images_det = deepcopy(images)
        images_det[index], _ = \
            self.affine_gird(image, r_t, x_t, y_t)
        for indx in range(len(images_det)):
            if indx == index:
                continue
            Flag = images_det[indx] * images_det[index]
            if (Flag != 0.0).any():
                if r_count == x_c == y_c == 0:
                    pre_v = np.random.randn(1) * self.sigma_i + self.mean
                    pre_d = np.random.uniform(0, 360, 1)
                    x, y = self.pol2cart(pre_v, pre_d)
                    r = np.random.uniform(-self.rotate_range, self.rotate_range, 1)
                else:
                    x = np.random.randn(1) * self.sigma_v * 0.5 + x * np.random.uniform(-0.5, 0.5, 1)
                    y = np.random.randn(1) * self.sigma_v * 0.5 + y * np.random.uniform(-0.5, 0.5, 1)
                x, y, r = self.occlusion_detection(images, image, x, y, r, r_count, x_c, y_c, index, recurrence_index)

        return x, y, r

    def to_0_1(self, x):
        return (x - x.min()) / (x.max() - x.min())

    def shift_ins(self, pixel_mask, mask_pixel, flow_last, prev_state=None):
        index_1 = 0
        flow_var = [None] * len(pixel_mask[0])
        flow_var = [flow_var] * len(pixel_mask)
        if prev_state is None:
            prev_state = [None] * len(pixel_mask[0])
            prev_state = [prev_state] * len(pixel_mask)
        if flow_last is None:
            flow_last = [None] * len(pixel_mask[0])
            flow_last = [flow_last] * len(pixel_mask)

        for maskpa, prev_a in zip(mask_pixel, prev_state):
            index_2 = 0
            for maskpb, prev_b in zip(maskpa, prev_a):
                if prev_b is None:
                    pre_v = np.random.randn(1) * self.sigma_i + self.mean
                    pre_d = np.random.uniform(0, 360, 1)
                    x, y = self.pol2cart(pre_v, pre_d)
                    r = np.random.uniform(-self.rotate_range, self.rotate_range, 1)
                    zoom = np.random.uniform(1 - self.zoom_ini, 1 + self.zoom_ini, 1)
                    r_count = 0
                    x_c = 0
                    y_c = 0
                    zoom_count = 1.0
                else:
                    x, y, r, r_count, x_c, y_c, zoom_count, zoom = *prev_b,
                    x = np.random.randn(1) * self.sigma_v + x
                    y = np.random.randn(1) * self.sigma_v + y
                    r = np.random.randn(1) * self.sigma_rotate_v + r
                    zoom = np.random.randn(1) * self.sigma_zoom_v + zoom
                # if self.if_avoid_overlap:
                # x, y, r = self.occlusion_detection(pixel_mask[index_1], maskpb, x, y, r, r_count, x_c, y_c, index_2)
                # if you wanna void overlap, use this function #
                r_count = r_count + r
                x_c = x_c + x
                y_c = y_c + y
                zoom_count = zoom_count * zoom
                pixel_mask[index_1][index_2], flow_last[index_1][index_2], flow_var[index_1][index_2] = \
                    self.affine_gird(maskpb, r_count, 2 * x_c, 2 * y_c, zoom_count, flow_last[index_1][index_2])
                prev_state[index_1][index_2] = (x, y, r, r_count, x_c, y_c, zoom_count, zoom)
                index_2 += 1
            index_1 += 1
        return pixel_mask, flow_last, flow_var, prev_state

    @torch.no_grad()
    def forward(self, image_seq):
        img_start = image_seq[0]
        if img_start.shape[0] > 1 and (np.random.randint(0, 2) == 1):
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
        img_roll = torch.roll(img_start, shifts=[int(np.random.uniform(0, h, 1)), int(np.random.uniform(0, w, 1))],
                              dims=[-1, -2])
        mask_ins = run_slic_seperate(img_roll, n_seg=175, compact=20, rd_select=[5, 10])
        noise = self.blur(self.blur(self.blur(noise)) / 2.5) + img_roll
        noise = self.to_0_1(noise)
        noise[noise == self.key] = self.key + 1e-4
        for index, img_ins in enumerate(mask_ins):
            mask_pixel.append(
                [torch.tensor(occ_ins, dtype=torch.int).type_as(img_start) * noise[index] for occ_ins in img_ins])
        flow_last = None
        pixel_mask = deepcopy(mask_pixel)
        image_seq.reverse()
        for index, image in enumerate(image_seq):
            image, mask = self.draw_img(pixel_mask, image)
            output.append(image)
            output_mask.append(mask)
            pixel_mask_list.append(deepcopy(pixel_mask))
            if index == len(image_seq) - 1:
                break
            pixel_mask, flow_last, flow_var, prev_s = self.shift_ins(pixel_mask, mask_pixel, flow_last, prev_s)
            flow_list.append(flow_var)
        pixel_mask_list.reverse()

        for flow_var, mask in zip(flow_list, pixel_mask_list):
            flow_masked.append(self.create_flow(image, mask, flow_var))
        self.mask = output_mask
        image_seq.reverse()
        output.reverse()
        output_mask.reverse()
        return output, output_mask, flow_masked

    @torch.no_grad()
    def apparent_variation(self, imgs, if_shadow=False, mask=None):
        if np.random.randint(0, 2) == 1:
            return imgs
        b, c, h, w = imgs[0].shape
        length = len(imgs)
        color_trans = ColorJitter(brightness=0.4, saturation=0.1, contrast=0.1, hue=0.01)
        if mask is None:
            mask = self.mask[::-1]
            mask = [self.blur_mask(self.blur_mask(
                F.interpolate(torch.cat(torch.chunk(mask, chunks=2)[::-1], dim=0), size=(h, w), mode="bilinear",
                              align_corners=True))) for mask in mask]  # make smoothness edge
            # if if_shadow:
            #     for ii in range(len(mask)):  # make smoothness edge
            #         mask[ii][:, :, :h // 2, :] = 1.0

        batch_list = [[None] * length for _ in range(b)]
        for bach_index in range(b):
            for time_index in range(length):
                batch_list[bach_index][time_index] = imgs[time_index][bach_index]
            # (1 - mask[index]) + image * mask[index]
        for index, list_imgs in enumerate(batch_list):
            batch_list[index] = color_trans(list_imgs, if_shadow)
        time_list = [[None] * b for _ in range(length)]

        for time_index in range(length):
            for batch_index in range(b):
                time_list[time_index][batch_index] = batch_list[batch_index][time_index]
            batch = torch.cat(time_list[time_index], dim=0)
            time_list[time_index] = mask[time_index] * imgs[time_index] + batch * (1 - mask[time_index])

        return time_list

    @torch.no_grad()
    def mask_add(self, image_seq):
        idx = np.random.randint(0, len(image_seq))
        texture = image_seq[idx]
        mask_pixel = []
        h, w = texture.shape[-1], texture.shape[-2]
        noise = torch.rand(texture.size()).type_as(texture)

        img_roll = torch.roll(texture, shifts=[int(np.random.uniform(0, h, 1)), int(np.random.uniform(0, w, 1))],
                              dims=[-1, -2])
        mask_ins = run_slic_seperate(img_roll, n_seg=100, compact=25, rd_select=[5, 10])
        noise = self.blur(self.blur(self.blur(noise)) * 1.2 + img_roll)

        noise = self.to_0_1(noise)

        noise[noise == self.key] = self.key + 1e-4
        for index, img_ins in enumerate(mask_ins):
            mask_pixel.append(
                [torch.tensor(occ_ins, dtype=torch.int).type_as(texture) * noise[index] for occ_ins in img_ins])
        image, mask = self.draw_img(mask_pixel, image_seq[-1])
        image_seq[-1] = image
        return image_seq, mask

    @staticmethod
    def demo():
        ...


def random_crop(img, flow, occ_mask=None, crop_sz=None):
    """
    :param img: Nx6xHxW
    :param flows: n * [Nx2xHxW]
    :param occ_masks: n * [Nx1xHxW]
    :param crop_sz:
    :return:
    """
    _, _, h, w = img[0].size()
    c_h, c_w = crop_sz

    if c_h == h and c_w == w:
        return img, flow

    x1 = np.random.randint(0, w - c_w)
    y1 = np.random.randint(0, h - c_h)
    img = [img[:, :, y1:y1 + c_h, x1: x1 + c_w] for img in img]
    flow = [flow[:, :, y1:y1 + c_h, x1: x1 + c_w] for flow in flow]
    if occ_mask is None:
        return img, flow
    else:
        occ_mask = [occ_mask[:, :, y1:y1 + c_h, x1: x1 + c_w] for occ_mask in occ_mask]
        return img, flow, occ_mask


class GaussianBlurConv(torch.nn.Module):
    def __init__(self, channels=3):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = torch.nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        x = torch.nn.functional.conv2d(x, self.weight, padding=2, groups=self.channels)
        return x
