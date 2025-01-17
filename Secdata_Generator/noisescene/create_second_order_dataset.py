import easydict
import torch
from torchvision import transforms
from torch.utils.data import ConcatDataset
from co_transforms import get_co_transforms, get_co_transforms_s
from transforms import sep_transforms
from datasets.flow_datasets import SelfRender, Sintel
from transforms.ar_transforms.sp_transfroms import RandomAffineFlow
from non_fourier_data_creation import CreateNonFourierData
import numpy as np
import os
from utils.flow_utils import InputPadder, flow_to_image, save_img_seq, viz_img_seq, writeFlow
from copy import deepcopy
import cv2
import warnings


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def write_flow_scene(output_dir, flow_pr):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for frame, flow in enumerate(flow_pr):
        flow = flow.squeeze()
        output_file = os.path.join(output_dir, 'frame_%04d.flo' % (frame))
        writeFlow(output_file, flow)


def write_img_scene(output_dir, imgs):
    batch_index = 0
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_file = os.path.join(output_dir, 'video.mp4')
    if imgs[0].shape[1] == 3:
        imgs = [img1[batch_index].detach().cpu().permute(1, 2, 0).numpy() * 255.0 for img1 in imgs]
    if imgs[0].shape[1] == 1:
        imgs = [img1[batch_index].detach().cpu().permute(1, 2, 0).numpy() for img1 in imgs]
        imgs = [cv2.cvtColor(flo * 255, cv2.COLOR_GRAY2BGR) for flo in imgs]

    temp = imgs[0]
    size = temp.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_file, fourcc, 22, (size[-2], size[-3]))
    for frame, img in enumerate(imgs):
        output_file = os.path.join(output_dir, 'frame_%04d.png' % (frame))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_file, img)
        out.write(img.astype(np.uint8))
    out.release()


warnings.filterwarnings("ignore")
config_dict = {"data": {"at_cfg": {"cj": True,
                                   "cj_bri": 0.4,
                                   "cj_con": 0.4,
                                   "cj_hue": 0.05,
                                   "cj_sat": 0.4,
                                   "gamma": True,
                                   "rblur": True,
                                   "gblur": True},
                        "root_davis": "../../../opticalflowdataset/davis2016flow/",
                        "root_sintel": "../../../opticalflowdataset/MPI-Sintel-complete/",
                        "root_self_render": "../../../opticalflowdataset/sec/human/select",
                        "run_at": True,
                        "test_shape": [768, 768],
                        "train_shape": [768, 768],
                        "train_n_frames": 24,
                        "train_subsplit": "trainval",
                        "type": "Sintel_Special",
                        "val_n_frames": 2,
                        "val_subsplit": "trainval"},
               "data_aug": {"crop": True,
                            "hflip": True,
                            "para_crop": [768, 768],
                            "swap": False},
               "seed": 0,
               "train": {
                   "ot_size": [320, 704],
                   "st_cfg": {"add_noise": True,
                              "st_sm": True,
                              "hflip": True,
                              "rotate": [-0.2, 0.2, -0.015, 0.015],
                              "squeeze": [0.86, 1.16, 1.0, 1.0],
                              "trans": [0.2, 0.015],
                              "vflip": True,
                              "zoom": [1.0, 1.5, 0.985, 1.015]},
                   "workers": 8}}

cfg_all = easydict.EasyDict(config_dict)
cfg = cfg_all.data
cfg_train = cfg_all.train


def improve_contrast(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img.astype(np.uint8))

    return img


def improve_brigness(img):
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    return img


input_transform = transforms.Compose([
    # improve contrast
    # improve_contrast,
    # improve_brigness,
    sep_transforms.ArrayToTensor(),
    transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),  # normalize to [0，1]
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # normalize to [-1，1]
])
co_transform = get_co_transforms_s(aug_args=cfg_all.data_aug)
st_transorm = RandomAffineFlow(cfg_all.train.st_cfg).cuda()


def create_second_order_dataset():
    seq_len = 24
    # only translation motion
    train_set1 = Sintel(cfg.root_sintel, n_frames=seq_len,
                        split='training', subsplit=cfg.train_subsplit,
                        with_flow=True,
                        type='clean',
                        ap_transform=None,
                        transform=input_transform,
                        co_transform=co_transform,
                        target_transform={"flow": sep_transforms.ArrayToTensor()}
                        )
    train_set2 = Sintel(cfg.root_sintel, n_frames=seq_len,
                        split='training', subsplit=cfg.train_subsplit,
                        with_flow=True,
                        type='final',
                        ap_transform=None,
                        transform=input_transform,
                        co_transform=co_transform,
                        target_transform={"flow": sep_transforms.ArrayToTensor()}
                        )
    train_set3 = Sintel(cfg.root_sintel, n_frames=seq_len,
                        split='training', subsplit=cfg.train_subsplit,
                        with_flow=True,
                        type='albedo',
                        ap_transform=None,
                        transform=input_transform,
                        co_transform=co_transform,
                        target_transform={"flow": sep_transforms.ArrayToTensor()}
                        )
    train_set = torch.utils.data.ConcatDataset([train_set1, train_set2, train_set3])

    train_set = SelfRender(cfg.root_self_render, n_frames=cfg.train_n_frames,
                           split='train', subsplit=cfg.train_subsplit,
                           with_flow=True,
                           ap_transform=None,
                           transform=input_transform,
                           co_transform=co_transform,
                           target_transform={"flow": sep_transforms.ArrayToTensor()}
                           )
    sp_transform = RandomAffineFlow(cfg_train.st_cfg, addnoise=cfg_train.st_cfg.add_noise).cuda()

    output_path = "./"
    img_file = os.path.join(output_path, "image")
    flow_second = os.path.join(output_path, "flow2nd")
    flow_first = os.path.join(output_path, "flow1st")
    viz_file = os.path.join(output_path, "viz")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(img_file):
        os.makedirs(img_file)
    if not os.path.exists(flow_first):
        os.makedirs(flow_first)
    if not os.path.exists(viz_file):
        os.makedirs(viz_file)
    if not os.path.exists(flow_second):
        os.makedirs(flow_second)

    # random select a sample from the dataset
    file_idx = 58
    success_num = 7 * 20
    while file_idx < success_num:
        step = np.random.randint(0, len(train_set))
        data = train_set[step]
        data["img_list"] = [x.unsqueeze(0).cuda() for x in data["img_list"]]
        if len(data["img_list"]) < seq_len:
            print("skip this sample")
            continue

        # if you want to use the first frame as the reference frame, you can use the following code
        # data["img_list"] = [data["img_list"][0].clone() for i in range(seq_len)]

        # read data to device
        first_order_flow = [x.unsqueeze(0).cuda() for x in data['target']["flow"]]
        viz_img_seq(data['img_list'], if_debug=False)
        imgs_oc = None
        iter = 0

        while imgs_oc is None:
            iter += 1
            if iter > 2:
                break
            # type changes with the file_idx
            if file_idx < 20:
                types = 1
            elif 20 <= file_idx < 40:
                types = 2
            elif 40 <= file_idx < 60:
                types = 3
            elif 60 <= file_idx < 80:
                types = 5
            elif 80 <= file_idx < 100:
                types = 6
            elif 100 <= file_idx < 120:
                types = 7
            elif 120 <= file_idx < 140:
                types = 8
            else:
                raise ValueError("wrong file_idx")



            # if types == 1:  # random noise
            # elif types == 2:  # second order with blur
            # elif types == 3:  # second order with water wave
            # elif types == 4:  # second order random texture
            # elif types == 5:  # luminance flip
            # elif types == 6:  # random optical flow shuffle fourier phase
            # elif types == 7:  # random flow shuffle
            # elif types == 8:  # swirl
            # elif types == 9:  # pure drift balanced motion

            imgs_oc, occ_mask = CreateNonFourierData(mean=12,
                                                     sigma_i=3,
                                                     sigma_v=0.5,
                                                     sigma_zoom_v=0.00,
                                                     sigma_rotate_v=0.05,
                                                     zoom_ini=0.00,
                                                     rd_select=[4, 12],
                                                     rotate_range=0.5, compact=80, n_seg=60,
                                                     SeqLen=seq_len, types=types).forward_second_order(
                deepcopy(data['img_list']), types=types)  # occ is 0, nocc is 1

        if iter > 2:
            print("bad scene, skip this scene")
            continue

        file = os.path.join(img_file, "Scene_" + str(file_idx))
        write_img_scene(file, imgs_oc)

        file = os.path.join(flow_first, "Scene_" + str(file_idx))
        flow_temp = [flows.detach().cpu().numpy().transpose([0, 2, 3, 1]) for flows in first_order_flow]
        write_flow_scene(file, flow_temp)

        file = os.path.join(viz_file, "Scene_" + str(file_idx))
        if not os.path.exists(file):
            os.makedirs(file)
        save_img_seq(occ_mask, name=os.path.join(file, "second_mask"), if_debug=True)
        save_img_seq(first_order_flow, name=os.path.join(file, "first_flow"), if_debug=True)
        print("finish scene {}".format(file_idx))
        file_idx += 1


create_second_order_dataset()
