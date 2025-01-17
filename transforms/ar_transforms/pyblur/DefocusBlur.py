# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
from skimage.draw import circle_perimeter as circle
import cv2
defocusKernelDims = [3, 5, 7, 9]


def DefocusBlur_random(img):
    kernelidx = np.random.randint(0, len(defocusKernelDims))
    kerneldim = defocusKernelDims[kernelidx]
    return DefocusBlur(img, kerneldim)


def DefocusBlur(img, dim):
    imgarray = np.array(img)
    imgarray = cv2.cvtColor(imgarray, cv2.COLOR_RGB2BGR)
    kernel = DiskKernel(dim)
    convolved = cv2.filter2D(imgarray, -1, kernel)
    convolved = cv2.cvtColor(convolved, cv2.COLOR_BGR2RGB)
    # convolved = convolve2d(imgarray, kernel, mode='same', fillvalue=255.0).astype("uint8")
    img = Image.fromarray(convolved)
    return img


def DiskKernel(dim):
    kernelwidth = dim
    kernel = np.zeros((kernelwidth, kernelwidth), dtype=np.float32)
    dim_ = dim-1
    circleCenterCoord = dim_ / 2
    circleRadius = circleCenterCoord+1

    rr, cc = circle(circleCenterCoord, circleCenterCoord, circleRadius)
    kernel[rr, cc] = 1

    if (dim == 3 or dim == 5):
        kernel = Adjust(kernel, dim)

    normalizationFactor = np.count_nonzero(kernel)
    kernel = kernel / normalizationFactor
    return kernel


def Adjust(kernel, kernelwidth):
    kernel[0, 0] = 0
    kernel[0, kernelwidth - 1] = 0
    kernel[kernelwidth - 1, 0] = 0
    kernel[kernelwidth - 1, kernelwidth - 1] = 0
    return kernel
