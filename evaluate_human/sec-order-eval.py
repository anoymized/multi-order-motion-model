from __future__ import print_function, division
import argparse
import os
import copy
import numpy as np
from PIL import Image
import torch
import progressbar as pb
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import glob
import pandas as pd
import cv2
import path
from utils.flow_utils import flow_to_image_relative, flow_to_image, resize_flow
import imageio
from io import BytesIO
import re
from model.nmi6.FFV1MT_MS import FFV1DNNV2
deepcopy = copy.deepcopy


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def viz(flo, flow_vec, flo2, flow_vec2, image, index):
    fig, axes = plt.subplots(1, 3, figsize=(10, 5), dpi=500)
    ax1 = axes[0]
    plot_quiver(ax1, flow=flow_vec, mask=flo, spacing=60)
    ax1.set_title('flow first order')
    ax1 = axes[1]
    plot_quiver(ax1, flow=flow_vec2, mask=flo2, spacing=60)
    ax1.set_title('flow second order')
    ax1 = axes[2]
    ax1.imshow(image)
    ax1.set_title('image')
    # tight_layout automatically adjusts subplot params so that the subplot(s) fits in to the figure area.
    plt.tight_layout()
    # save figure into a buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    # convert to numpy array
    im = np.array(Image.open(buf))
    buf.close()
    plt.close()
    return im

    # plt.show()


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
    # ax.quiver(x, y, u, v, color="black", scale=1500, width=0.003, headwidth=5, minlength=0.5)  # bigger is short
    ax.quiver(x, y, u, v, color="black", width=0.006)  # bigger is short
    x_gird, y_gird = np.meshgrid(x, y)
    ax.scatter(x_gird, y_gird, c="black", s=(h + w) // 120)
    ax.scatter(x_gird, y_gird, c="maroon", s=(h + w) // 140)
    ax.set_ylim(sorted(ax.get_ylim(), reverse=True))
    ax.set_aspect("equal")


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
    # model = FFV1DNN(num_scales=8,
    #                 num_cells=256,
    #                 upsample_factor=8,
    #                 feature_channels=256,
    #                 scale_factor=16,
    #                 num_layers=6,
    #                 )
    #
    model = torch.nn.DataParallel(FFV1DNNV2())

    # model =torch.nn.DataParallel( raft_ori.RAFT.get_raft_model())
    modeldict = torch.load(args.model)
    if 'epoch' in modeldict.keys():
        modeldict = modeldict['state_dict']
    print("Parameter Count: %d" % count_parameters(model))
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if args.model is not None:
        model.load_state_dict(modeldict, strict=True)
    model.cuda()

    mean_theta_array = []
    std_theta_array = []
    img_list = []
    vis_flow_1 = []
    flow_vec_1 = []
    vis_flow_2 = []
    flow_vec_2 = []
    # model = model.module
    model.eval()
    types1, types2, types3, types4, types5, types6, types7, types8, types9, type10 = \
        [], [], [], [], [], [], [], [], [], []
    csv_x, csv_y = [], []
    gt_x, gt_y = [], []
    _root_dir_ = path.Path(args.path)
    _scene_dir_ = _root_dir_ / 'image'
    _data_dir_ = _root_dir_ / 'data.csv'
    _out_dir_ = _root_dir_ / 'result'

    if not _out_dir_.exists():
        _out_dir_.makedirs_p()
    assert _scene_dir_.exists() and _out_dir_.exists()
    scene_list = sorted(_scene_dir_.dirs())
    for scene_dir in scene_list:
        types = scene_dir.dirs()
        for type_ in types:
            if 'type_1' in str(type_) and 'type_10' not in str(type_):
                types1.append(type_)
            elif 'type_2' in str(type_):
                types2.append(type_)
            elif 'type_3' in str(type_):
                types3.append(type_)
            elif 'type_4' in str(type_):
                types4.append(type_)
            elif 'type_5' in str(type_):
                types5.append(type_)
            elif 'type_6' in str(type_):
                types6.append(type_)
            elif 'type_7' in str(type_):
                types7.append(type_)
    # sort files by types

    types1.sort(key=lambda x: int(x.split('Scene_')[1].split('/')[0]))
    types2.sort(key=lambda x: int(x.split('Scene_')[1].split('/')[0]))
    types3.sort(key=lambda x: int(x.split('Scene_')[1].split('/')[0]))
    types4.sort(key=lambda x: int(x.split('Scene_')[1].split('/')[0]))
    types5.sort(key=lambda x: int(x.split('Scene_')[1].split('/')[0]))
    types6.sort(key=lambda x: int(x.split('Scene_')[1].split('/')[0]))
    types7.sort(key=lambda x: int(x.split('Scene_')[1].split('/')[0]))

    types_list = [types1, types2, types3, types4, types5, types6, types7]
    for type_idx, types in enumerate(types_list):
        for scene_dir in types:

            images = scene_dir.files('*.png')
            if len(images) == 0:
                images = glob.glob(os.path.join(scene_dir, '*.png'))
            if len(images) == 0:
                images = glob.glob(os.path.join(scene_dir, '*.jpg'))
            print(scene_dir)
            # read csv
            # get upper direction
            scene_dir = path.Path(scene_dir)
            # read xlsx
            data = pd.read_excel(scene_dir.dirname() / 'NewConvention_Final.xlsx')
            # get ground truth

            # get picX and picY
            picW = data['picX'].values[0]
            picH = data['picY'].values[0]
            # get MoVI
            gt_u = data['flowU'].values[0]
            gt_v = data['flowV'].values[0]

            if "_" in images[0]:
                images.sort(key=lambda x: int(x.basename().split('_')[1].split('.')[0]))
            else:
                images.sort()

            # load images
            images = [load_image(imfile) for imfile in images]
            length = len(images)

            n = 15
            fac = (len(images) - 1) % (n - 1)
            gt_idx = int((n / 2 - 1) if n % 2 == 0 else (n - 1) / 2)

            if length > n:
                res = length - n
                idx_head = res // 2
                idx_tail = res - idx_head
                images = images[idx_head:-idx_tail]

            H, W = images[0].shape[2:]
            # resize images cubic
            images_re = [F.interpolate(im, size=args.infer_size, mode='bicubic', align_corners=True).clamp_(0, 255) for
                         im in images]

            # assert (len(imgs) - 1) % (n-1) == 0

            images = [im.detach().cpu().squeeze().numpy().transpose([1, 2, 0]) / 255.0 for im in images]

            # to tensor B,N,C,H,W
            images_input = images_re

            # images_input = torch.stack(images_re, dim=1)
            # result_dict = model(images_input, test = True)
            # result_dict = model(images_input[0], images_input[1], iters=args.iters)

            # result_dict = model(images_input, mix_enable=True, layer=args.iters)
            result_dict = model(images_input, layer=args.iters)
            flow2_predictions = result_dict['flow_seq'][-1]
            flow1_predictions = result_dict['flow_seq'][-1]
            # resize flow back to original size
            flow1_predictions = resize_flow(flow1_predictions, new_shape=(H, W))
            flow2_predictions = resize_flow(flow2_predictions, new_shape=(H, W))
            flow_vec1_ = flow1_predictions.detach().cpu().squeeze().numpy().transpose([1, 2, 0])
            vis_flow_1_ = flow_to_image(flow_vec1_)
            flow_vec_2_ = flow2_predictions.detach().cpu().squeeze().numpy().transpose([1, 2, 0])
            vis_flow_2_ = flow_to_image(flow_vec_2_)

            x = flow_vec_2_[picH, picW, 0]
            y = flow_vec_2_[picH, picW, 1]

            # get the average of surrounding 8 points
            x = (x + flow_vec_2_[picH - 1, picW, 0] + flow_vec_2_[picH + 1, picW, 0] + flow_vec_2_[picH, picW - 1, 0] +
                 flow_vec_2_[picH, picW + 1, 0] + flow_vec_2_[picH - 1, picW - 1, 0] + flow_vec_2_[
                     picH - 1, picW + 1, 0] +
                 flow_vec_2_[picH + 1, picW - 1, 0] + flow_vec_2_[picH + 1, picW + 1, 0]) / 9
            y = (y + flow_vec_2_[picH - 1, picW, 1] + flow_vec_2_[picH + 1, picW, 1] + flow_vec_2_[picH, picW - 1, 1] +
                 flow_vec_2_[picH, picW + 1, 1] + flow_vec_2_[picH - 1, picW - 1, 1] + flow_vec_2_[
                     picH - 1, picW + 1, 1] +
                 flow_vec_2_[picH + 1, picW - 1, 1] + flow_vec_2_[picH + 1, picW + 1, 1]) / 9
            csv_x.append(x)
            csv_y.append(y)
            gt_x.append(gt_u)
            gt_y.append(gt_v)
            assert gt_u != 0 or gt_v != 0
            print(x, y)

            # flow = flow_up.transpose(1, 2, 0).astype(np.float64)
            im = viz(vis_flow_1_, flow_vec1_, vis_flow_2_, flow_vec_2_, images[gt_idx], 0)
            im = Image.fromarray(im)
            im.save(os.path.join(scene_dir, 'type_{}.tif'.format(type_idx + 1)))
            # save image

    csv_x = np.array(csv_x)
    csv_y = np.array(csv_y)
    csv_x = csv_x.reshape(7, 40).T
    csv_y = csv_y.reshape(7, 40).T
    gt_x = np.array(gt_x).reshape(7, 40).T
    gt_y = np.array(gt_y).reshape(7, 40).T
    # save csv type1, type2, type3, type4, type5, type6, type7, type8, type9, type10
    df_csv_x = pd.DataFrame(csv_x,
                            columns=['type1', 'type2', 'type3', 'type4', 'type5', 'type6', 'type7'])
    df_csv_y = pd.DataFrame(csv_y,
                            columns=['type1', 'type2', 'type3', 'type4', 'type5', 'type6', 'type7'])
    df_csv_x.to_csv('csv_x.csv')
    df_csv_y.to_csv('csv_y.csv')

    # calculate pearson correlation coefficient between GT and csv of each type
    # concat csv_x and csv_y
    csv_xy = np.concatenate((csv_x, csv_y), axis=1)
    gt_xy = np.concatenate((gt_x, gt_y), axis=1)
    # calculate pearson correlation coefficient for each type
    corr = []
    for i in range(7):
        corr.append(np.corrcoef(csv_xy[:, i], gt_xy[:, i])[0, 1])
        print('type{}: {}'.format(i + 1, np.corrcoef(csv_xy[:, i], gt_xy[:, i])[0, 1]))
    print('average: {}'.format(np.mean(corr)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint",
                        default="[path to model]")
    parser.add_argument('--path', help="dataset for evaluation",
                        default='[path to human_static dataset]')
    parser.add_argument('--iters', help="number of iterations", type=int, default=6)
    parser.add_argument('--infer_size', help="infer size", type=list, default=[1024, 1024])
    parser.add_argument('--video', action='store_true', help='if save the video demo', default=True)
    parser.add_argument('--save_dir', help="save directory", default='result_v1mt')
    args = parser.parse_args()
    demo(args)
