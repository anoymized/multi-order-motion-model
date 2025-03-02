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
import re
import glob
import torchvision
import cv2
from utils.flow_utils import flow_to_image_relative, flow_to_image, resize_flow

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


DEVICE = 'cuda'
maindir = 'path to the selected kitti 2015 dataset'
datasetName = ["1_KITTI"]
# datasetName = ["1_KITTI"]

datasetN = len(datasetName)
sessionN = 12
movN = 2
frameN = 15


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imshow('image', img[:, :, [2, 1, 0]] / 255.0)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def save_video(flo, img, writer):
    # map flow to rgb image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    flo = cv2.cvtColor(flo, cv2.COLOR_BGR2RGB)
    print(flo.shape)
    img_flo = np.concatenate([img, flo], axis=0).astype(np.uint8)
    writer.write(img_flo)


def demo(args):
    model = FFV1DNNV2(num_scales=8,
                      # num_cells=256,
                      upsample_factor=8,
                      # feature_channels=256,
                      scale_factor=16,
                      num_layers=6, )
    model = nn.DataParallel(model)
    print("Parameter Count: %d" % count_parameters(model))
    if args.model is not None:
        model_dict = torch.load(args.model)
        model.load_state_dict(model_dict, strict=True)

    model = model.module
    model.cuda()
    with torch.no_grad():
        for dataset in range(datasetN):
            for session in range(1, sessionN + 1):
                destination_folder = os.path.join(maindir, datasetName[dataset], f'session{session:03d}')
                video_file = os.path.join(destination_folder, f'session{session:03d}.mp4')

                # 初始化视频写入器，帧率为12，分辨率根据第一帧图像确定
                out = None
                for file in glob.glob(os.path.join(destination_folder, 'flow_*.mat')):
                    os.remove(file)

                for mov in range(1, movN + 1):
                    image_list_ = glob.glob(os.path.join(destination_folder, f'Mov{mov}_F*.jpg'))

                    if len(image_list_) == 0:
                        image_list_ = glob.glob(os.path.join(destination_folder, f'Mov{mov}_F*.png'))
                    image_list_.sort(key=lambda x: int(re.sub('\D', '', x)))
                    vis_flow = []
                    flow_vec = []
                    print(image_list_)

                    temp = load_image(image_list_[0])
                    size = temp.shape
                    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                    out = cv2.VideoWriter(os.path.join(destination_folder, 'flow_%d.mp4' % mov), fourcc, 12,
                                          (size[3], size[2] * 2))


                    # load all images
                    image_list = [load_image(img) for img in image_list_]
                    # resize the image to that divisible by 8
                    image_size_ori = image_list[0].shape[-2:]
                    image_size = [(image_size_ori[0] // 8 + 1) * 8, (image_size_ori[1] // 8 + 1) * 8]
                    image_list_resize = [F.interpolate(img, size=image_size, mode='bicubic', align_corners=True) for img in image_list]

                    n = 15
                    for i in range((len(image_list) - n) + 1):
                        images_input = [image_list_resize[i + j] for j in range(n)]
                        length = len(images_input)
                        # result_dict = model.forward_viz(images_input, layer=args.iters, channel=1)
                        result_dict = model.forward(images_input, layer=args.iters)
                        flow_all = result_dict['flow_seq_1'][-1]
                        flow_all = resize_flow(flow_all, new_shape=image_size_ori)

                        flow_all = flow_all.detach().cpu().squeeze().numpy().transpose([1, 2, 0])
                        vis_flow += [flow_to_image(flow_all)]
                        flow_vec += [flow_all]

                    import scipy.io as sio
                    for index, flow_up in enumerate(vis_flow):
                        i = index
                        sio.savemat(os.path.join(destination_folder,
                                                 'flow_MOV%d_%d-%d.mat' % (mov, i + (n + 1) / 2, i + (n + 1) / 2 + 1)),
                                    {'flow': flow_vec[i]})
                        save_video(flow_up, load_image(image_list_[i])[0].permute(1, 2, 0).cpu().numpy(), out)

                    if out is not None:
                        out.release()

                cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint",
                        default="path to final ckpt")
    parser.add_argument('--path', help="dataset for evaluation",
                        default='path to kitti dataset')
    parser.add_argument('--iters', help="number of iterations", type=int, default=8)
    parser.add_argument('--video', action='store_true', help='if save the video demo', default=True)
    args = parser.parse_args()

    demo(args)
