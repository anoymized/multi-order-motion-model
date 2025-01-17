import imageio
import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import torch
from easydict import EasyDict
from torchvision import transforms
from transforms import sep_transforms
import PIL.Image as Image
from utils.flow_utils import resize_flow, flow_to_image_relative, plot_quiver, viz_img_seq
from utils.torch_utils import restore_model
from model.nmi6.FFV1MT_MS import FFV1DNNV2
# from model.raft.raft import RAFT
import os
import scipy.io as sio
from tqdm import tqdm
import copy

DEVICE = 'cuda'

probe = [[150, 275, 1350, 1475]
    , [350, 475, 1500, 1625]
    , [450, 575, 800, 925]
    , [275, 400, 1160, 1285]
    , [570, 695, 1700, 1825]]


# cut H 0-720, W 1200-1975

# 695+150=845 ->13->845
# 800-150=650
# 1825+150=1975 ->679->1975

def viz(img, flo, flow_vec):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    flo = flo
    img_flo = np.concatenate([img, flo], axis=0)
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.subplots()
    plot_quiver(ax1, flow=flow_vec, mask=flo, spacing=40)
    plt.show()
    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()
    #
    # cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)
    # cv2.waitKey()


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    # cv2.imshow('image', img[:, :, [2, 1, 0]] / 255.0)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


def divide_image(img, m=2, n=4):  # 分割成m行n列
    m = m + 1
    n = n + 1
    h, w = img.shape[-2], img.shape[-1]
    grid_h = int(h * 1.0 / (m - 1) + 0.5)  # 每个网格的高
    grid_w = int(w * 1.0 / (n - 1) + 0.5)  # 每个网格的宽-----------------------------------------------------------

    # 满足整除关系时的高、宽
    h = grid_h * (m - 1)
    w = grid_w * (n - 1)

    # 图像缩放
    img_re = torch.nn.functional.interpolate(img, size=(w, h), mode='bilinear', align_corners=True)

    # plt.imshow(img_re)
    gx, gy = np.meshgrid(np.linspace(0, w, n), np.linspace(0, h, m))
    gx = gx.astype(np.int)
    gy = gy.astype(np.int)

    divide_image = torch.zeros([m - 1, n - 1, grid_w, grid_h],
                               device='cuda')  # 这是一个五维的张量，前面两维表示分块后图像的位置（第m行，第n列），后面三维表示每个分块后的图像信息

    for i in range(m - 1):
        for j in range(n - 1):
            divide_image[i, j, ...] = img_re[:, :, gx[i][j]:gx[i + 1][j + 1], gy[i][j]:gy[i + 1][j + 1]]

    return divide_image


def image_concat(divide_image):
    m, n, grid_h, grid_w = [divide_image.shape[0], divide_image.shape[1],  # 每行，每列的图像块数
                            divide_image.shape[2], divide_image.shape[3]]  # 每个图像块的尺寸

    restore_image = np.zeros([m * grid_h, n * grid_w, 3], np.uint8)
    restore_image[0:grid_h, 0:]
    for i in range(m):
        for j in range(n):
            restore_image[i * grid_h:(i + 1) * grid_h, j * grid_w:(j + 1) * grid_w] = divide_image[i, j, :]
    return restore_image


def save_video(flo, img, writer):
    img = img
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    flo = cv2.cvtColor(flo, cv2.COLOR_RGB2BGR)
    # map flow to rgb image
    print(flo.shape)
    img_flo = np.concatenate([img, flo], axis=0).astype(np.uint8)
    writer.write(img_flo)


class TestHelper(object):
    def __init__(self, cfg):
        self.cfg = EasyDict(cfg)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
            "cpu")
        self.model = self.init_model()
        self.input_transform = transforms.Compose([
            sep_transforms.ZoomSingle(*self.cfg.test_shape, grey=False),
            sep_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0], std=[1]),
        ])

    def init_model(self):
        # model = FFV1DNN(num_layers=7, feature_channels=256)
        # model = RAFT()
        model = FFV1DNNV2(num_scales=8,
                          upsample_factor=8,
                          scale_factor=16,
                          num_layers=6)
        print('Number fo parameters: {}'.format(model.num_parameters()))
        model = model.to(self.device)
        # model = torch.nn.DataParallel(model)
        modeldict = torch.load(args.model)
        # stripe the module from the modeldict
        modeldict = {k.replace('module.', ''): v for k, v in modeldict.items()}
        model.load_state_dict(modeldict)

        model.cuda()

        model.eval()
        return model

    @torch.no_grad()
    def run(self, imgs, layer, probe=None):
        probe = None
        if probe is not None:
            probe_0, probe_2 = probe[0] - 150, probe[2] - 150
            probe_1, probe_3 = probe[1] + 150, probe[3] + 150
            reminderH = 16 - (probe_1 - probe_0) % 16
            reminderW = 16 - (probe_3 - probe_2) % 16

            imgs_ = copy.deepcopy([img[:, :, probe_0:probe_1 + reminderH, probe_2:probe_3 + reminderW] for img in imgs])

            print(imgs_[0].shape)
        else:
            # exchange the H,W
            imgs_ = [im.cuda() for im in imgs]



        flow = self.model.forward(imgs_, layer=layer)['flow_seq_1'][-1]
        if probe is not None:
            flow_ori = torch.zeros([flow.shape[0], flow.shape[1], imgs[0].shape[2], imgs[0].shape[3]], device='cuda')
            flow_ori[:, :, probe_0:probe_1 + reminderH, probe_2:probe_3 + reminderW] = flow
            flow = flow_ori

        return flow


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='modelckpt/dual_model_final.pth')
    parser.add_argument('-s', '--test_shape', default=[872, 2048], type=int, nargs=2)
    parser.add_argument('-i', '--path', type=str,
                        default='[path to the  dataset]')
    # please refer to "Psychophysical measurement of perceived motion flow of naturalistic scenes"
    # for prepare the sintel slow dataset
    parser.add_argument('-v', '--video', type=bool, default=True)
    parser.add_argument('-t', '--type', type=str, default='seq')
    parser.add_argument('-n', '--div_n', type=int, default=15)
    parser.add_argument('--save_mat', type=bool, default=True)

    args = parser.parse_args()

    cfg = {
        'upsample': True,
        'n_frames': 2,
        'reduce_dense': True,
        'type': "image",
        'model': {
            'upsample': True,
            'n_frames': 2,
            'reduce_dense': True,
            'type': "image"
        },
        'pretrained_model': args.model,
        'test_shape': args.test_shape,
        'train': {
            'mixed_precision': False},
        'mixed_precision': False,
        'with_bk': False

    }
    ts = TestHelper(cfg)
    layer_num = 9
    # mov 6 - 15
    # mv_list = ["mov6", "mov7", "mov8", "mov9", "mov10", "mov11", "mov12", "mov13", "mov14", "mov15"]
    # mov 1 - 5
    mv_list = ["movie01", "movie02", "movie03", "movie04", "movie05"]
    out_path = 'result-sintel'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    for mv_idx, mv in enumerate(mv_list):
        path = os.path.join(args.path, mv)
        images = glob.glob(os.path.join(path, 'HS*.png')) + \
                 glob.glob(os.path.join(path, 'HS*.jpg'))
        if "_" in images[0]:
            images.sort(key=lambda x: int(x.split('_')[-1][:-4]))
        else:
            images.sort()
        # extract odd frames
        # images = images[::2]
        out = None
        for layer in tqdm(range(layer_num)):
            temp = load_image(images[0])
            size = temp.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # TODO: change the image size here.
            out = cv2.VideoWriter(os.path.join(out_path, '%s_layer_%d.mp4' % (mv, layer)), fourcc, 10.0,
                                  (size[-1], size[-2] * 2))
            imgs = [imageio.imread(img).astype(np.float32) for img in images]
            h, w = imgs[0].shape[:2]
            vis_flow = []
            flow_vec = []
            n = args.div_n
            # assert (len(imgs) - 1) % (n-1) == 0
            fac = (len(imgs) - 1) % (n - 1)

            for i in range((len(imgs) - n) + 1):
                img_ = [imgs[i + j] for j in range(n)]
                # to tensor
                img_ = [torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) for img in img_]
                # resize the image args.test_shape
                img_ = [
                    torch.nn.functional.interpolate(img, size=(args.test_shape[0], args.test_shape[1]), mode='bicubic',
                                                    align_corners=True).clamp(0, 255) for img in img_]
                flow_12 = [ts.run(img_, layer, probe=None)]  # probe[mv_idx]
                # flow_12 = [resize_flow(flow_12, (h, w)).squeeze() for flow_12 in flow_12]
                flow_12 = [flow_12.squeeze().detach().cpu().numpy().transpose([1, 2, 0]) for flow_12 in flow_12]
                vis_flow += [flow_to_image_relative(flow_12) for flow_12 in flow_12]
                flow_vec += [flow_12]
            torch.cuda.empty_cache()
            i = 0
            # layer = 6
            # assert len(vis_flow) == len(imgs) - 1
            for index, flow_up in enumerate(vis_flow):
                i = index + 7
                flow = flow_vec[index][0]
                flow = flow.transpose(1, 2, 0).astype(np.float64)
                sio.savemat(os.path.join(out_path, '%s_layer%d_%d_%d.mat') % (mv, layer, i, i + 1), {'flow': flow})
                save_video(flow_up, imgs[i], writer=out)
            if out is not None:
                out.release()
