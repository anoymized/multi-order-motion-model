# Copyright (c) Meta Platforms, Inc. and affiliates.
# modfied by Xudong Wang based on https://github.com/lucasb-eyer/pydensecrf/blob/master/pydensecrf/tests/test_dcrf.py and third_party/TokenCut

import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF

MAX_ITER = 20
POS_W = 7 
POS_XY_STD = 3
Bi_W = 10
Bi_XY_STD = 50 
Bi_RGB_STD = 5


def densecrf(image, mask):
    h, w = mask.shape
    mask = mask.reshape(1, h, w)
    fg = mask.astype(float)
    bg = 1 - fg
    output_logits = torch.from_numpy(np.concatenate((bg, fg), axis=0))

    H, W = image.shape[:2]
    image = np.ascontiguousarray(image)

    output_logits = F.interpolate(output_logits.unsqueeze(0), size=(H, W), mode="bilinear").squeeze()
    output_probs = F.softmax(output_logits, dim=0).cpu().numpy()

    # 增强对 mask 的置信度
    MASK_WEIGHT = 1.5  # 权重系数
    fg_confident_area = (fg > 0.8)  # 设定高置信度区域

    # 遍历类别并为每个类别更新 output_probs 的值
    # for c_idx in range(output_probs.shape[0]):
    #     if c_idx == 1:  # 前景类别的置信度
    #         output_probs[c_idx][fg_confident_area] = 0.99
    #     else:  # 背景类别的置信度
    #         output_probs[c_idx][fg_confident_area] = 0.01

    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U *= MASK_WEIGHT  # 增加原始 mask 的权重
    U = np.ascontiguousarray(U)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W )  # 增强平滑性
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=image, compat=Bi_W)  # 增强颜色相似性

    Q = d.inference(10)  # 减少迭代次数以保持与原始 mask 的接近性
    Q = np.array(Q).reshape((c, h, w))
    MAP = np.argmax(Q, axis=0).reshape((h, w)).astype(np.float32)
    return MAP


