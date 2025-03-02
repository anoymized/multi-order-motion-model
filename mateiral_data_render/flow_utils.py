import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib.colors import hsv_to_rgb
import torch.nn.functional as tf
from PIL import Image
from os.path import *
import json
from io import BytesIO
import pandas as pd
import cv2
from matplotlib.colors import Normalize
import matplotlib.cm as cm
TAG_CHAR = np.array([202021.25], np.float32)


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


# absolut color flow
def flow_to_image(flow, max_flow=32):
    if max_flow is not None:
        max_flow = max(max_flow, 1.)
    else:
        max_flow = np.max(flow)

    n = 8
    u, v = flow[:, :, 0], flow[:, :, 1]
    mag = np.sqrt(np.square(u) + np.square(v))
    angle = np.arctan2(v, u)
    im_h = np.mod(angle / (2 * np.pi) + 1, 1)
    im_s = np.clip(mag * n / max_flow, a_min=0, a_max=1)
    im_v = np.clip(n - im_s, a_min=0, a_max=1)
    im = hsv_to_rgb(np.stack([im_h, im_s, im_v], 2))
    return (im * 255).astype(np.uint8)


# relative color
def flow_to_image_relative(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def resize_flow(flow, new_shape):
    _, _, h, w = flow.shape
    new_h, new_w = new_shape
    flow = torch.nn.functional.interpolate(flow, (new_h, new_w),
                                           mode='bilinear', align_corners=True)
    scale_h, scale_w = h / float(new_h), w / float(new_w)
    flow[:, 0] /= scale_w
    flow[:, 1] /= scale_h
    return flow




class InputPadder:
    """ Pads images such that dimensions are divisible by 32 """

    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 16) + 1) * 16 - self.ht) % 16
        pad_wd = (((self.wd // 16) + 1) * 16 - self.wd) % 16
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, inputs):
        return [tf.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]

        return x[..., c[0]:c[1], c[2]:c[3]]





def writeFlow(filename, uv, v=None):
    """ Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert (uv.ndim == 3)
        assert (uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert (u.shape == v.shape)
    height, width = u.shape
    f = open(filename, 'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def flow_error_image_np(flow_pred, flow_gt, mask_occ, mask_noc=None, log_colors=True):
    """Visualize the error between two flows as 3-channel color image.
    Adapted from the KITTI C++ devkit.
    Args:
        flow_pred: prediction flow of shape [ height, width, 2].
        flow_gt: ground truth
        mask_occ: flow validity mask of shape [num_batch, height, width, 1].
            Equals 1 at (occluded and non-occluded) valid pixels.
        mask_noc: Is 1 only at valid pixels which are not occluded.
    """
    # mask_noc = tf.ones(tf.shape(mask_occ)) if mask_noc is None else mask_noc
    mask_noc = np.ones(mask_occ.shape) if mask_noc is None else mask_noc
    diff_sq = (flow_pred - flow_gt) ** 2
    # diff = tf.sqrt(tf.reduce_sum(diff_sq, [3], keep_dims=True))
    diff = np.sqrt(np.sum(diff_sq, axis=2, keepdims=True))
    if log_colors:
        height, width, _ = flow_pred.shape
        # num_batch, height, width, _ = tf.unstack(tf.shape(flow_1))
        colormap = [
            [0, 0.0625, 49, 54, 149],
            [0.0625, 0.125, 69, 117, 180],
            [0.125, 0.25, 116, 173, 209],
            [0.25, 0.5, 171, 217, 233],
            [0.5, 1, 224, 243, 248],
            [1, 2, 254, 224, 144],
            [2, 4, 253, 174, 97],
            [4, 8, 244, 109, 67],
            [8, 16, 215, 48, 39],
            [16, 1000000000.0, 165, 0, 38]]
        colormap = np.asarray(colormap, dtype=np.float32)
        colormap[:, 2:5] = colormap[:, 2:5] / 255
        # mag = tf.sqrt(tf.reduce_sum(tf.square(flow_2), 3, keep_dims=True))
        tempp = np.square(flow_gt)
        # temp = np.sum(tempp, axis=2, keep_dims=True)
        # mag = np.sqrt(temp)
        mag = np.sqrt(np.sum(tempp, axis=2, keepdims=True))
        # error = tf.minimum(diff / 3, 20 * diff / mag)
        error = np.minimum(diff / 3, 20 * diff / (mag + 1e-7))
        im = np.zeros([height, width, 3])
        for i in range(colormap.shape[0]):
            colors = colormap[i, :]
            cond = np.logical_and(np.greater_equal(error, colors[0]), np.less(error, colors[1]))
            # temp=np.tile(cond, [1, 1, 3])
            im = np.where(np.tile(cond, [1, 1, 3]), np.ones([height, width, 1]) * colors[2:5], im)
        # temp=np.cast(mask_noc, np.bool)
        # im = np.where(np.tile(np.cast(mask_noc, np.bool), [1, 1, 3]), im, im * 0.5)
        im = np.where(np.tile(mask_noc == 1, [1, 1, 3]), im, im * 0.5)
        im = im * mask_occ
    else:
        error = (np.minimum(diff, 5) / 5) * mask_occ
        im_r = error  # errors in occluded areas will be red
        im_g = error * mask_noc
        im_b = error * mask_noc
        im = np.concatenate([im_r, im_g, im_b], axis=2)
        # im = np.concatenate(axis=2, values=[im_r, im_g, im_b])
    return im[:, :, ::-1]


def viz_img_seq(img_list=[], flow_list=[], batch_index=0, if_debug=True):
    '''visulize image sequence from cuda'''
    if if_debug:

        assert len(img_list) != 0
        if len(img_list[0].shape) == 3:
            img_list = [np.expand_dims(img, axis=0) for img in img_list]
        elif img_list[0].shape[1] == 1:
            img_list = [img1[batch_index].detach().cpu().permute(1, 2, 0).numpy() for img1 in img_list]
            img_list = [cv2.cvtColor(flo * 255, cv2.COLOR_GRAY2BGR) for flo in img_list]
        elif img_list[0].shape[1] == 2:
            img_list = [img1[batch_index].detach().cpu().permute(1, 2, 0).numpy() for img1 in img_list]
            img_list = [flow_to_image_relative(flo) / 255.0 for flo in img_list]
        elif img_list[0].shape[1] == 4:
            # rgba to rgb
            img_list = [img1[batch_index].detach().cpu().permute(1, 2, 0).numpy() for img1 in img_list]
            img_list = [img[:, :, :3] for img in img_list]

        else:
            img_list = [img1[batch_index].detach().cpu().permute(1, 2, 0).numpy() for img1 in img_list]

        if len(flow_list) == 0:
            flow_list = [np.zeros_like(img) for img in img_list]
        elif len(flow_list[0].shape) == 3:
            flow_list = [np.expand_dims(img, axis=0) for img in flow_list]
        elif flow_list[0].shape[1] == 1:
            flow_list = [img1[batch_index].detach().cpu().permute(1, 2, 0).numpy() for img1 in flow_list]
            flow_list = [cv2.cvtColor(flo * 255, cv2.COLOR_GRAY2BGR) for flo in flow_list]
        elif flow_list[0].shape[1] == 2:
            flow_list = [img1[batch_index].detach().cpu().permute(1, 2, 0).numpy() for img1 in flow_list]
            flow_list = [flow_to_image_relative(flo) / 255.0 for flo in flow_list]
        else:
            flow_list = [img1[batch_index].detach().cpu().permute(1, 2, 0).numpy() for img1 in flow_list]

        if img_list[0].max() > 10:
            img_list = [img / 255.0 for img in img_list]
        if flow_list[0].max() > 10:
            flow_list = [img / 255.0 for img in flow_list]

        while len(img_list) > len(flow_list):
            flow_list.append(np.zeros_like(flow_list[-1]))
        while len(flow_list) > len(img_list):
            img_list.append(np.zeros_like(img_list[-1]))
        img_flo = np.concatenate([flow_list[0], img_list[0]], axis=0)
        # map flow to rgb image
        for i in range(1, len(img_list)):
            temp = np.concatenate([flow_list[i], img_list[i]], axis=0)
            img_flo = np.concatenate([img_flo, temp], axis=1)
        cv2.imshow('image', img_flo[:, :, [2, 1, 0]])
        cv2.waitKey()
    else:
        return


def plt_show_img_flow(img_list=[], flow_list=[], batch_index=0):
    assert len(img_list) != 0
    if len(img_list[0].shape) == 3:
        img_list = [np.expand_dims(img, axis=0) for img in img_list]
    elif img_list[0].shape[1] == 1:
        img_list = [img1[batch_index].detach().cpu().permute(1, 2, 0).numpy() for img1 in img_list]
        img_list = [cv2.cvtColor(flo * 255, cv2.COLOR_GRAY2BGR) for flo in img_list]
    elif img_list[0].shape[1] == 2:
        img_list = [img1[batch_index].detach().cpu().permute(1, 2, 0).numpy() for img1 in img_list]
        img_list = [flow_to_image_relative(flo) / 255.0 for flo in img_list]
    elif img_list[0].shape[1] == 4:
        # rgba to rgb
        img_list = [img1[batch_index].detach().cpu().permute(1, 2, 0).numpy() for img1 in img_list]
        img_list = [img[:, :, :3] for img in img_list]
    else:
        img_list = [img1[batch_index].detach().cpu().permute(1, 2, 0).numpy() for img1 in img_list]

    assert flow_list[0].shape[1] == 2
    flow_vec = [img[batch_index].detach().cpu().permute(1, 2, 0).numpy() for img in flow_list]
    flow_list = [flow_to_image_relative(flo) / 255.0 for flo in flow_vec]

    col = len(flow_list) // 2
    fig = plt.figure(figsize=(10, 5))
    for i in range(len(flow_list)):
        ax1 = fig.add_subplot(2, col, i + 1)
        plot_quiver(ax1, flow=flow_vec[i], mask=flow_list[i], spacing=(50 * flow_list[i].shape[0]) // 512)
        if i == len(flow_list) - 1:
            plt.title("Final Flow Result")
        else:
            plt.title("Flow from decoder (Layer %d)" % i)

    if img_list[0].max() > 10:
        img_list = [img / 255.0 for img in img_list]
    if flow_list[0].max() > 10:
        flow_list = [img / 255.0 for img in flow_list]

    while len(img_list) > len(flow_list):
        flow_list.append(np.zeros_like(flow_list[-1]))
    while len(flow_list) > len(img_list):
        img_list.append(np.zeros_like(img_list[-1]))
    img_flo = np.concatenate([flow_list[0], img_list[0]], axis=0)
    # map flow to rgb image
    for i in range(1, len(img_list)):
        temp = np.concatenate([flow_list[i], img_list[i]], axis=0)
        img_flo = np.concatenate([img_flo, temp], axis=1)
    fig2 = plt.figure(figsize=(10, 5))
    plt.imshow(img_flo)
    plt.show()


def plt_attention(attention, h, w):
    col = len(attention) // 2
    fig = plt.figure(figsize=(10, 5))

    for i in range(len(attention)):
        viz = attention[i][0, :, :, h, w].detach().cpu().numpy()
        # viz = viz[7:-7, 7:-7]
        if i == 0:
            viz_all = viz
        else:
            viz_all = viz_all + viz

        ax1 = fig.add_subplot(2, col + 1, i + 1)
        img = ax1.imshow(viz, cmap="rainbow", interpolation="bilinear")
        plt.colorbar(img, ax=ax1)
        ax1.scatter(h, w, color='red')
        plt.title("Attention of Iteration %d" % (i + 1))

    ax1 = fig.add_subplot(2, col + 1, 2 * (col + 1))
    img = ax1.imshow(viz_all, cmap="rainbow", interpolation="bilinear")
    plt.colorbar(img, ax=ax1)
    ax1.scatter(h, w, color='red')
    plt.title("Mean Attention")
    plt.show()


def plot_quiver(ax, flow, spacing, mask=None, show_win=None, margin=0, **kwargs):
    """Plots less dense quiver field.

    Args:
        ax: Matplotlib axis
        flow: motion vectors
        spacing: space (px) between each arrow in grid
        margin: width (px) of enclosing region without arrows
        kwargs: quiver kwargs (default: angles="xy", scale_units="xy")
    """
    h, w, *_ = flow.shape
    spacing = 60
    if show_win is None:
        nx = int((w - 2 * margin) / spacing)
        ny = int((h - 2 * margin) / spacing)
        x = np.linspace(margin, w - margin - 1, nx, dtype=np.int64)
        y = np.linspace(margin, h - margin - 1, ny, dtype=np.int64)
    else:
        h0, h1, w0, w1 = *show_win,
        h0 = int(h0 * h)
        h1 = int(h1 * h)
        w0 = int(w0 * w)
        w1 = int(w1 * w)
        num_h = (h1 - h0) // spacing
        num_w = (w1 - w0) // spacing
        y = np.linspace(h0, h1, num_h, dtype=np.int64)
        x = np.linspace(w0, w1, num_w, dtype=np.int64)

    flow = flow[np.ix_(y, x)]
    u = flow[:, :, 0]
    v = flow[:, :, 1] * -1  # ----------

    kwargs = {**dict(angles="xy", scale_units="xy"), **kwargs}
    if mask is not None:
        ax.imshow(mask)
    ax.quiver(x, y, u, v, color="black", scale=40, width=0.020, headwidth=5, minlength=0.5,alpha=0.5)  # bigger is short
    # ax.quiver(x, y, u, v, color="black")  # bigger is short
    x_gird, y_gird = np.meshgrid(x, y)
    ax.scatter(x_gird, y_gird, c="black", s=(h + w) // 150)
    ax.scatter(x_gird, y_gird, c="black", s=(h + w) // 250)
    ax.set_ylim(sorted(ax.get_ylim(), reverse=True))
    ax.set_aspect("equal")


def save_img_seq(img_list, batch_index=0, name='img', if_debug=False):
    if if_debug:
        temp = img_list[0]
        size = temp.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(name + '_flow.mp4', fourcc, 30, (size[-1], size[-2]))
        if img_list[0].shape[1] == 2:
            img_list = [img1[batch_index].detach().cpu().permute(1, 2, 0).numpy() for img1 in img_list]
            img_list = [flow_to_image_relative(flo) for flo in img_list]
        if img_list[0].shape[1] == 3:
            img_list = [img1[batch_index].detach().cpu().permute(1, 2, 0).numpy() * 255.0 for img1 in img_list]
        if img_list[0].shape[1] == 1:
            img_list = [img1[batch_index].detach().cpu().permute(1, 2, 0).numpy() for img1 in img_list]
            img_list = [cv2.cvtColor(flo * 255, cv2.COLOR_GRAY2BGR) for flo in img_list]

        for index, img in enumerate(img_list):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(name + '_%d.png' % index, img)
            out.write(img.astype(np.uint8))
        out.release()
    else:
        return


def check_tensor_all(flow_vec1, flow_vec2, flow_vec1_pre, flow_vec2_pre, image1, image2, mask, enable=False):
    # if batchsize is not 1, then only show the first one
    if flow_vec1.shape[0] != 1:
        flow_vec1 = flow_vec1[0]
        flow_vec2 = flow_vec2[0]
        flow_vec1_pre = flow_vec1_pre[0]
        flow_vec2_pre = flow_vec2_pre[0]
        image1 = image1[0]
        image2 = image2[0]
        mask = mask[0]

    flow_vec1 = flow_vec1.detach().cpu().squeeze().numpy().transpose([1, 2, 0])
    vis_flow_1 = flow_to_image(flow_vec1)
    flow_vec2 = flow_vec2.detach().cpu().squeeze().numpy().transpose([1, 2, 0])
    vis_flow_2 = flow_to_image(flow_vec2)
    flow_vec1_pre = flow_vec1_pre.detach().cpu().squeeze().numpy().transpose([1, 2, 0])
    vis_flow_1_pre = flow_to_image(flow_vec1_pre)
    flow_vec2_pre = flow_vec2_pre.detach().cpu().squeeze().numpy().transpose([1, 2, 0])
    vis_flow_2_pre = flow_to_image(flow_vec2_pre)
    image1 = image1.detach().cpu().squeeze().numpy().transpose([1, 2, 0])
    image2 = image2.detach().cpu().squeeze().numpy().transpose([1, 2, 0])

    if image1.max() > 10:
        image1 = image1 / 255.0
        image2 = image2 / 255.0
    mask = mask.detach().cpu().squeeze().numpy()
    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    ax1 = axes[0, 0]
    plot_quiver(ax1, flow=flow_vec1, mask=vis_flow_1, spacing=40)
    ax1.set_title('flow first')
    ax2 = axes[0, 1]
    plot_quiver(ax2, flow=flow_vec2, mask=vis_flow_2, spacing=40)
    ax2.set_title('flow second')
    ax3 = axes[0, 2]
    plot_quiver(ax3, flow=flow_vec1_pre, mask=vis_flow_1_pre, spacing=40)
    ax3.set_title('flow first pre')
    ax4 = axes[1, 0]
    plot_quiver(ax4, flow=flow_vec2_pre, mask=vis_flow_2_pre, spacing=40)
    ax4.set_title('flow second pre')
    ax5 = axes[1, 1]
    ax5.imshow(image1)
    ax5.set_title('image first')
    ax6 = axes[1, 2]
    ax6.imshow(mask)
    ax6.set_title('mask')

    if enable:
        plt.show()
    # concert the image to numpy array and return
    buffer_ = BytesIO()  # using buffer,great way!
    plt.savefig(buffer_, format='png')
    buffer_.seek(0)
    image = np.asarray(Image.open(buffer_))
    buffer_.close()
    plt.close()
    return image

def visualize_3d_flow(image, flow, occlusion, intrinsics, spacing=50, margin=10):
    """Visualizes combined 2D and 3D flow with circle rings."""
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    h, w, _ = flow.shape
    nx = int((w - 2 * margin) / spacing)
    ny = int((h - 2 * margin) / spacing)
    x = np.linspace(margin, w - margin - 1, nx, dtype=np.int64)
    y = np.linspace(margin, h - margin - 1, ny, dtype=np.int64)
    # Extract flow points
    flow = flow[np.ix_(y, x)]
    u = (flow[:, :, 0] * fx + (cx - w / 2))
    v = (flow[:, :, 1] * fy + (cy - h / 2))
    z = flow[:, :, 2]
    # Setup plot
    fig, ax = plt.subplots(figsize=(10, 5))
    # Handle occlusion by modifying alpha channel
    alpha_channel = 1 - occlusion / 255  # Normalize and invert
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)  # Convert to RGB
    image = np.dstack((image, alpha_channel))  # Add alpha channel
    ax.imshow(image)
    # Quiver plot for 2D flow, adjusting coordinates
    x_grid, y_grid = np.meshgrid(x, y)
    ax.quiver(x_grid.flatten(), y_grid.flatten(), u.flatten(), v.flatten(), color="black", alpha=0.3)
    # Normalize color based on Z value
    z_min, z_max = np.min(z), np.max(z)
    norm = Normalize(vmin=z_min, vmax=z_max)
    cmap = cm.ScalarMappable(norm=norm, cmap=cm.coolwarm)
    colors = cmap.to_rgba(z.flatten())  # Get RGBA values for each element in z
    # Scatter plot for 3D flow z component with rings
    sizes = np.abs(z.flatten()) * 700  # Scale sizes
    ax.scatter(x_grid.flatten(), y_grid.flatten(), c=colors, edgecolors='none', s=sizes)
    ax.set_ylim(sorted(ax.get_ylim(), reverse=True))
    ax.set_aspect('equal')
    ax.set_xlim([0, w])
    ax.set_ylim([0, h])
    # Create a colorbar
    cbar = plt.colorbar(cmap, ax=ax)
    cbar.set_label('Z-direction flow (red away, blue towards)')
    # Save figure to buffer
    buffer_ = BytesIO()
    plt.savefig(buffer_, format='png', bbox_inches='tight', dpi=800)
    plt.close()
    buffer_.seek(0)
    image = np.asarray(Image.open(buffer_))
    buffer_.close()
    return image

# Example of how to use the function
# flow = np.random.rand(100, 100, 3) * 2 - 1  # Random 3D flow example
# image = np.ones((100, 100, 3))/2  # Random image example
# visualize_3d_flow(image, flow, spacing=20, margin=5)