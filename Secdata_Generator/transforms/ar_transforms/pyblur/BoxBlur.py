import numpy as np
from PIL import Image
from scipy.signal import convolve2d
import cv2

boxKernelDims = [3, 5, 7, 9]


def BoxBlur_random(img):
    kernelidx = np.random.randint(0, len(boxKernelDims))
    kerneldim = boxKernelDims[kernelidx]
    return BoxBlur(img, kerneldim)


def BoxBlur(img, dim):
    imgarray = np.array(img)
    imgarray = cv2.cvtColor(imgarray, cv2.COLOR_RGB2BGR)
    kernel = BoxKernel(dim)
    convolved = cv2.filter2D(imgarray, -1, kernel)
    convolved = cv2.cvtColor(convolved, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(convolved)
    return img


def BoxKernel(dim):
    kernelwidth = dim
    kernel = np.ones((kernelwidth, kernelwidth), dtype=np.float32)
    normalizationFactor = np.count_nonzero(kernel)
    kernel = kernel / normalizationFactor
    return kernel
