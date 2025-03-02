import numpy as np
import torch
from torchvision import transforms as tf
from PIL import ImageFilter, Image
import cv2
from fast_slic.avx2 import SlicAvx2 as Slic
from copy import deepcopy

def get_ap_transforms(cfg):
    transforms = [ToPILImage()]
    if cfg.cj:
        transforms.append(ColorJitter(brightness=cfg.cj_bri,
                                      contrast=cfg.cj_con,
                                      saturation=cfg.cj_sat,
                                      hue=cfg.cj_hue))
    if cfg.gblur:
        # transforms.append(RandomGaussianBlur(0.5, 3))
        transforms.append(RandomBlur(r_blur=cfg.rblur))
    transforms.append(ToTensor())
    if cfg.gamma:
        transforms.append(RandomGamma(min_gamma=0.7, max_gamma=1.5, clip_image=True))
    return tf.Compose(transforms)


# from https://github.com/visinf/irr/blob/master/datasets/transforms.py
class ToPILImage(tf.ToPILImage):
    def __call__(self, imgs):
        return [super(ToPILImage, self).__call__(im) for im in imgs]


class ColorJitter(tf.ColorJitter):
    def __call__(self, imgs):
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        a = np.random.randint(-2, 3, 1)
        factor = np.random.uniform(0.96, 1.04, 1)
        delta = factor - 1
        imgs = [transform(im) for im in imgs]
        if a == 1:
            imgs = [self.get_params([1.0, 1.3], [1.0, 1.3],
                                    [1.0, 1.3], None)(im) for im in imgs]
        if a == -1:
            imgs = [self.get_params([0.7, 1.0], [0.7, 1.0],
                                    [0.7, 1.0], None)(im) for im in imgs]
        if abs(a) == 2:
            for index in range(len(imgs)):
                imgs[index] = self.get_params([factor, factor], [factor / 1.1, factor / 1.1],
                                              [factor / 1.1, factor / 1.1], None)(imgs[index])
                factor = delta + factor

        return imgs


def run_slic_pt(imgs, n_seg=70, compact=20, rd_select=(20, 40)):  # Nx1xHxW
    img = np.array(imgs[0])
    fast_slic = Slic(num_components=n_seg, compactness=compact, min_size_factor=0.8)

    seg = fast_slic.iterate(img)
    seg_list = []

    if rd_select is not None:
        for i in range(len(imgs)):
            n_select = np.random.randint(min(rd_select[0], np.max(seg) // 3), min(np.max(seg) // 1.5, rd_select[1]))
            select_list = np.random.choice(range(0, np.max(seg) + 1), n_select, replace=False)
            seg_ = np.bitwise_or.reduce([seg == seg_id for seg_id in select_list])
            seg_list.append(np.expand_dims(seg_, -1).astype(np.float))
    return seg_list


class RandomBlur(object):
    def __init__(self, r_blur=True):
        self.random_blur_generator =None
        self.random_gaussian = RandomGaussianBlur(0.5, 3)
        self.r_blur = r_blur

    def __call__(self, imgs):
        segs = run_slic_pt(imgs)
        imgs_ori = deepcopy(imgs)
        if self.r_blur == True:
            a = np.random.randint(0, 3, 1)
        else:
            a = 0
        if a == 1:
            imgs = [self.random_blur_generator.get_parm()(imgs) for imgs in imgs]
        if a == 2:
            bulr_filter = self.random_blur_generator.get_parm()
            imgs = [bulr_filter(imgs) for imgs in imgs]
        else:
            imgs = self.random_gaussian(imgs)

        a = np.random.randint(0, 3, 1)
        if a == 1:
            imgs = [Image.fromarray((np.array(img_ori) * (1 - seg) + np.array(img) * seg).astype(np.uint8))
                    for img_ori, img, seg in zip(imgs_ori, imgs, segs)]
            imgs = [im.filter(ImageFilter.GaussianBlur(3)) for im in imgs]
        return imgs


class ToTensor(tf.ToTensor):
    def __call__(self, imgs):
        return [super(ToTensor, self).__call__(im) for im in imgs]


class RandomGamma():
    def __init__(self, min_gamma=0.7, max_gamma=1.5, clip_image=True):
        self._min_gamma = min_gamma
        self._max_gamma = max_gamma
        self._clip_image = clip_image

    @staticmethod
    def get_params(min_gamma, max_gamma):
        return np.random.uniform(min_gamma, max_gamma)

    @staticmethod
    def adjust_gamma(image, gamma, clip_image):
        adjusted = torch.pow(image, gamma)
        if clip_image:
            adjusted.clamp_(0.0, 1.0)
        return adjusted

    def __call__(self, imgs):
        gamma = self.get_params(self._min_gamma, self._max_gamma)
        return [self.adjust_gamma(im, gamma, self._clip_image) for im in imgs]


class RandomGaussianBlur():
    def __init__(self, p, max_k_sz):
        self.p = p
        self.max_k_sz = max_k_sz

    def __call__(self, imgs):
        if np.random.random() < self.p:
            radius = np.random.uniform(0, self.max_k_sz)
            imgs = [im.filter(ImageFilter.GaussianBlur(radius)) for im in imgs]
        return imgs
