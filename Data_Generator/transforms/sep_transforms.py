import numpy as np
import torch
from skimage.color import rgb2gray as c2gray
from skimage.color import gray2rgb as gray2rgb
import cv2

cv2.ocl.setUseOpenCL(False)  # 设置opencv不使用多进程运行，但这句命令只在本作用域有效。
cv2.setNumThreads(0)  #


class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, array):
        assert (isinstance(array, np.ndarray))
        array = np.transpose(array, (2, 0, 1))
        # handle numpy array
        tensor = torch.from_numpy(array)
        # put it from HWC to CHW format
        return tensor.float()


def resize_flow(flow, shape):
    assert flow.shape[-1] == 2
    H, W = *shape,
    h, w = flow.shape[:2]
    flow = np.copy(flow)
    flow[:, :, 0] = flow[:, :, 0] / w * W
    flow[:, :, 1] = flow[:, :, 1] / h * H
    flow = cv2.resize(flow, (W, H), interpolation=cv2.INTER_LINEAR)
    return flow


class Zoom(object):
    def __init__(self, new_h, new_w, zoom_flow=False):
        self.new_h = new_h
        self.new_w = new_w
        self.zoom_flow = zoom_flow

    def __call__(self, inputs, target):
        if len(inputs[0].shape) == 3:
            inputs = [c2gray(image) for image in inputs]  # to gray
        h, w = inputs[0].shape
        if (h == self.new_h and w == self.new_w) or self.new_h == -1 or self.new_w == -1:
            inputs = [np.expand_dims(image, axis=-1) for image in inputs]
            return inputs, target
        inputs = [np.expand_dims(cv2.resize(image, (self.new_h, self.new_w), interpolation=cv2.INTER_CUBIC ), axis=-1) for image in
                  inputs]
        if "flow" in target:
            target["flow"] = [resize_flow(flow, (self.new_h, self.new_w)) for flow in target["flow"]]

        if "mask" in target:
            target["mask"] = [cv2.resize(flow, (self.new_h, self.new_w), interpolation=cv2.INTER_NEAREST) for flow in
                              target["mask"]]

        return inputs, target


class NonZoom(object):
    def __call__(self, inputs, target):
        if len(inputs[0].shape) == 3:
            inputs = [c2gray(image) for image in inputs]  # to gray

        if len(inputs[0].shape) == 2:
            inputs = [np.expand_dims(image, axis=-1) for image in inputs]
        return inputs, target


class ZoomSingle(object):
    def __init__(self, new_h, new_w, zoom_flow=False):
        self.new_h = new_h
        self.new_w = new_w
        self.zoom_flow = zoom_flow

    def __call__(self, inputs):
        if len(inputs.shape) == 3:
            inputs = c2gray(inputs)  # to gray
        h, w = inputs.shape
        if h == self.new_h and w == self.new_w:
            return np.expand_dims(inputs, axis=-1)
        inputs = np.expand_dims(cv2.resize(inputs, (int(self.new_h), int(self.new_w)), interpolation=cv2.INTER_CUBIC), axis=-1)
        # inputs = gray2rgb(inputs)
        return inputs
