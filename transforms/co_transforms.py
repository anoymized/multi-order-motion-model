import numbers
import random
import numpy as np
# from scipy.misc import imresize
import torch


def get_co_transforms(aug_args):
    transforms = []
    if aug_args.hflip:
        transforms.append(RandomHorizontalFlip())
    if aug_args.crop:
        transforms.append(RandomCrop(aug_args.para_crop))
    if aug_args.vflip:
        transforms.append(RandomVerticalFlip())

    return Compose(transforms)


class Compose(object):
    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input, target):
        for t in self.co_transforms:
            input, target = t(input, target)
        return input, target


class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs, target):
        h, w, _ = inputs[0].shape
        # print('before crop')
        # print(inputs[0].shape)
        # print(target['flow'][0].shape)
        # print('===================')
        th, tw = self.size
        if w == tw and h == th:
            return inputs, target

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        start_point = torch.from_numpy(np.array([x1, y1]))
        # target["img_wo_crop"] = inputs  # for boundary dilated warp
        inputs = [img[y1: y1 + th, x1: x1 + tw] for img in inputs]
        if 'mask' in target:
            target['mask'] = [img[y1: y1 + th, x1: x1 + tw] for img in target['mask']]
        if 'flow' in target:
            target['flow'] = [img[y1: y1 + th, x1: x1 + tw] for img in target['flow']]
        if 'flowsec' in target:
            target['flowsec'] = [img[y1: y1 + th, x1: x1 + tw] for img in target['flowsec']]
        # target["start_point"] = start_point
        return inputs, target


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, inputs, target):
        if random.random() < 0.5:
            inputs = [np.copy(np.fliplr(im)) for im in inputs]
            if 'mask' in target:
                target['mask'] = [np.copy(np.fliplr(mask)) for mask in target['mask']]
            if 'flow' in target:
                for i, flo in enumerate(target['flow']):
                    flo = np.copy(np.fliplr(flo))
                    flo[:, :, 0] *= -1
                    target['flow'][i] = flo
            if 'flowsec' in target:
                for i, flo in enumerate(target['flowsec']):
                    flo = np.copy(np.fliplr(flo))
                    flo[:, :, 0] *= -1
                    target['flowsec'][i] = flo
        return inputs, target


class RandomVerticalFlip(object):
    """Randomly vertically flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, inputs, target):
        if random.random() < 0.5:
            inputs = [np.copy(np.flipud(im)) for im in inputs]
            if 'mask' in target:
                target['mask'] = [np.copy(np.flipud(mask)) for mask in target['mask']]
            if 'flow' in target:
                for i, flo in enumerate(target['flow']):
                    flo = np.copy(np.flipud(flo))
                    flo[:, :, 1] *= -1
                    target['flow'][i] = flo
            if 'flowsec' in target:
                for i, flo in enumerate(target['flowsec']):
                    flo = np.copy(np.flipud(flo))
                    flo[:, :, 1] *= -1
                    target['flowsec'][i] = flo
        return inputs, target