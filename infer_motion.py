from __future__ import print_function, division
import argparse
import os
import copy
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.nmi6.FFV1MT_MS import FFV1DNNV2
import matplotlib.pyplot as plt
import glob
import torchvision
import cv2
from utils.flow_utils import flow_to_image_relative, flow_to_image, resize_flow
import imageio
from io import BytesIO

flow_to_image = flow_to_image_relative
MAX_FLOW = 600
deepcopy = copy.deepcopy


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def viz(flo, flow_vec,
        flo1, flow_vec1,
        flo2, flow_vec2, image, gate, index):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=500)
    ax1 = axes[0]
    plot_quiver(ax1, flow=flow_vec, mask=flo, spacing=50, scale=20)
    ax1.set_title('flow all')

    ax1 = axes[1]
    ax1.imshow(image)
    ax1.set_title('image')

    plt.tight_layout()
    # eliminate the x and y-axis
    plt.axis('off')
    # save figure into a buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=200)
    buf.seek(0)
    # convert to numpy array
    im = np.array(Image.open(buf))
    buf.close()
    plt.close()
    return im


DEVICE = 'cuda'


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
    ax.quiver(x, y, u, v, color="black", scale=50, width=0.010, alpha=0.4)  # bigger is short
    # ax.quiver(x, y, u, v, color="black", width=0.010, alpha=0.4)  # bigger is short
    x_gird, y_gird = np.meshgrid(x, y)
    ax.scatter(x_gird, y_gird, c="black", s=(h + w) // 500, alpha=0.4)
    ax.scatter(x_gird, y_gird, c="maroon", s=(h + w) // 600, alpha=0.4)
    ax.set_ylim(sorted(ax.get_ylim(), reverse=True))
    ax.set_aspect("equal")
    # remove axis
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])


def load_image(imfile):
    img = Image.open(imfile)
    # if RGBA or gray, convert to RGB
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img).astype(np.float32)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def save_video(flo, img, writer):
    # map flow to rgb image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    flo = cv2.cvtColor(flo, cv2.COLOR_BGR2RGB)
    print(flo.shape)
    img_flo = np.concatenate([img, flo], axis=0).astype(np.uint8)
    writer.write(img_flo)


@torch.no_grad()
def demo(args):
    model = FFV1DNNV2(num_scales=8,
                      # num_cells=256,
                      upsample_factor=8,
                      # feature_channels=256,
                      scale_factor=16,
                      num_layers=6, )
    model = nn.DataParallel(model)
    print("Parameter Count: %d" % count_parameters(model))
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if args.model is not None:
        model_dict = torch.load(args.model)
        model.load_state_dict(model_dict, strict=True)

    model = model.module
    model.cuda()

    model.eval()

    img_list = []
    vis_flow_1 = []
    flow_vec_1 = []
    vis_flow_2 = []
    flow_vec_2 = []
    vis_flow_all = []
    flow_vec_all = []
    gate_list = []

    vis_flow_1_seq = []
    vis_flow_2_seq = []
    flow_vec_1_seq = []
    flow_vec_2_seq = []
    vis_flow_all_seq = []
    flow_vec_all_seq = []

    path_ = args.path
    save_dir = args.save_dir
    # mkdir for saving
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    images = glob.glob(os.path.join(path_, 'rgba*.png'))

    if len(images) == 0:
        images = glob.glob(os.path.join(path_, '*.png'))
    if len(images) == 0:
        images = glob.glob(os.path.join(path_, '*.jpg'))

    if "_" in images[0]:
        try:
            images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        except:
            images.sort()

    else:
        images.sort()
    # images.reverse()
    # load images
    images = [load_image(imfile) for imfile in images]
    H, W = images[0].shape[2:]
    H_8, W_8 = np.round(H / 8).astype(int), np.round(W / 8).astype(int)
    H_8 = H_8 * 8
    W_8 = W_8 * 8
    # resize images cubic
    images_re = [F.interpolate(im, size=(H_8, W_8), mode='bicubic', align_corners=True).clamp_(0, 255) for im in
                 images]

    n = 15
    # assert (len(imgs) - 1) % (n-1) == 0
    fac = (len(images) - 1) % (n - 1)
    gt_idx = int((n / 2 - 1) if n % 2 == 0 else (n - 1) / 2)
    images = [im.detach().cpu().squeeze().numpy().transpose([1, 2, 0]) / 255.0 for im in images]

    for i in range((len(images_re) - n) + 1):
        images_input = [images_re[i + j] for j in range(n)]
        length = len(images_input)
        # result_dict = model.forward_viz(images_input, layer=args.iters, channel=1) for more detailed visualization
        result_dict = model.forward(images_input, layer=args.iters)
        flow1_predictions = result_dict['flow_seq_1'][-1]

        flow_all = result_dict['flow_seq'][-1]

        flow_vec1_ = flow1_predictions.detach().cpu().squeeze().numpy().transpose([1, 2, 0])
        vis_flow_1_ = flow_to_image(flow_vec1_)

        flow_all_ = flow_all.detach().cpu().squeeze().numpy().transpose([1, 2, 0])
        vis_flow_all_ = flow_to_image(flow_all_)
        vis_flow_1_seq.append(vis_flow_1_)
        flow_vec_1_seq.append(flow_vec1_)

        vis_flow_all.append(vis_flow_all_)
        flow_vec_all.append(flow_all_)

    frames = []
    for index, flow_up in enumerate(vis_flow_all):
        i = index
        # flow = flow_up.transpose(1, 2, 0).astype(np.float64)
        im = viz(vis_flow_all[index], flow_vec_all[index],
                 [], [], [], [], images[gt_idx + i], [], i)
        frames.append(im)
    imageio.mimsave(os.path.join(save_dir, 'dual-channel.gif'), frames, 'GIF', duration=0.15, loop=0, palettesize=256)

    frames = []
    for index, flow_up in enumerate(vis_flow_1_seq):
        i = index
        im = viz(vis_flow_1_seq[index], flow_vec_1_seq[index],
                 [], [], [], [], images[gt_idx + i], [], i)
        frames.append(im)
    imageio.mimsave(os.path.join(save_dir, 'first-channel.gif'), frames, 'GIF', duration=0.1, loop=0,
                    palettesize=256)
    print("Done! The result is saved in %s" % save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint",
                        default=None)
    parser.add_argument('--path', help="data folder for evaluation",
                        default='demo/test-stimuli/second-order/circular')
    parser.add_argument('--iters', help="number of iterations", type=int,
                        default=8)  # should be odd number  when visualization
    parser.add_argument('--video', action='store_true', help='if save the video demo', default=True)
    parser.add_argument('--save_dir', help="save directory", default='result_v1mt')
    args = parser.parse_args()

    demo(args)
