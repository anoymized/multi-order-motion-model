import copy
import random

import easydict
import torch
from datasets.get_dataset import get_dataset
from torch.utils.data import ConcatDataset
from transforms.co_transforms import get_co_transforms, get_co_transforms_s

from transforms.ar_transforms.sp_transfroms import RandomAffineFlow
from transforms.ar_transforms.oc_transforms import random_crop, run_slic_seperate, CreateOCCSeq
import numpy as np

from utils.flow_utils import InputPadder, flow_to_image, save_img_seq, viz_img_seq
from copy import deepcopy

import warnings

warnings.filterwarnings("ignore")
cconfig_dict = {"data": {"root_sintel": "../../opticalflowdataset/MPI-Sintel-complete/",
                         "root_nontex": "../../opticalflowdataset/flow_nontexture/",
                         "root_sintel_raw": "../../opticalflowdataset/Sintel_scene/scene",
                         "root_davis": "../../opticalflowdataset/davis2016flow/",
                         "run_at": False,
                         "test_shape": [448, 1024],
                         "train_n_frames": 25,
                         "type": "Sintel_extend",
                         "val_n_frames": 2,
                         "train_shape": [448, 1024],
                         "train_subsplit": "trainval",
                         "val_subsplit": "trainval"},
                "data_aug": {"crop": True,
                             "hflip": True,
                             "para_crop": [442, 900],
                             "swap": False},
                "loss": {"alpha": 30,
                         "occ_from_back": True,
                         "type": "unflow",
                         "w_l1": 0.15,
                         "w_smooth": 50.0,
                         "w_smooth_ac_frames": 0,
                         "w_ssim": 0.85,
                         "w_wssim": 0.0,
                         "smooth_1nd": True,
                         "smooth_2nd": False,
                         "w_ternary": 0.0,
                         "warp_pad": "border",
                         "with_bk": False},
                "model": {
                    "type": "Predictive Render"
                },
                "seed": 0,
                "train": {"batch_size": 2,
                          "val_batch_size": 8,
                          "beta": 0.999,
                          "bias_decay": 0,
                          "epoch_num": 150,
                          "st_cfg": {"add_noise": True,
                                     "hflip": False,
                                     "rotate": [0, 0, -0.015, 0.015],
                                     "squeeze": [1, 1, 1.0, 1.0],
                                     "trans": [0.2, 0.08],
                                     "st_sm": 0.0,
                                     "vflip": False,
                                     "zoom": [1.0, 1.7, 1.0, 1.0]},
                          "lr": 0.8e-4,
                          "momentum": 0.9,
                          "n_gpu": 5,
                          "eval_first": False,
                          "stage": "pre_train",
                          "loss_decay": True,
                          "optim": "adamw",
                          "mixed_precision": True,
                          "pretrained_model": None,
                          "print_freq": 5,
                          "record_freq": 20,
                          "save_iter": 1000,
                          'start_epoch': 0,
                          "val_epoch_size": 4,
                          "epoch_size": 0,
                          "valid_size": 0,
                          "weight_decay": 1e-05,
                          "workers": 16},
                "trainer": "Sintel"}
cfg_all = easydict.EasyDict(cconfig_dict)
train_set, val_set = get_dataset(cfg_all)

co_transform = get_co_transforms_s(aug_args=cfg_all.data_aug)
creat_occ_seq = CreateOCCSeq()


def a(ii):
    return ii


sp_transform = RandomAffineFlow(cfg_all.train.st_cfg, addnoise=cfg_all.train.st_cfg.add_noise).cuda()

for i_step, data in enumerate(val_set):
    i_step = i_step * 20
    data = train_set[i_step]
    imglist = [x.unsqueeze(0).cuda() for x in data["img_list"]]
    flow = [x.unsqueeze(0).cuda() for x in data["flow"]]

    # read data to device

    viz_img_seq(imglist, flow, if_debug=True)


    b_size, _, h_x1, w_x1 = imglist[0].size()
    # flow_ori = [torch.zeros(b_size, 2, h_x1, w_x1, dtype=torch.float32).cuda() for i in range(len(imgs_ph))]
    noc_ori = [torch.zeros(b_size, 1, h_x1, w_x1, dtype=torch.float32).cuda() for i in range(len(imglist))]
    #
    # imgs_crop, flow_t, noc_t = random_crop(deepcopy(data["img_list"]), flow_ori, noc_ori, cfg_train.ot_size)
    #
    # img_start = imgs_crop[0]
    # imgs_oc, occ_mask, flow = creat_occ_seq.forward(deepcopy(imgs_crop))  # occ is 0, nocc is 1
    # viz_img_seq(imgs_oc, flow, if_debug=True)
    # viz_img_seq(imgs_oc, occ_mask, if_debug=True)
    #
    # imgs_oc = creat_occ_seq.apparent_variation(imgs_crop)
    # viz_img_seq(imgs_oc, [], if_debug=True)
    # save_img_seq(imgs_oc, if_debug=True, name="4")
    # measure data loading time
    # imgs, flow_t, noc_t = random_crop(data["img_list"], flow_ori, noc_ori, cfg_train.ot_size)
    # imgs, occ_mask = creat_occ_seq.mask_add(imgs)  # occ is 0, nocc is 1
    # viz_img_seq(imgs, [], if_debug=True)
    #
    # # compute output
    #
    #
    save_img_seq(imglist,name="1",if_debug=True)
    save_img_seq(flow, name="2",if_debug=True)



    s = {'imgs': imglist, 'flows_f': flow, 'masks_f': noc_ori}
    st_res = sp_transform(copy.deepcopy(s))
    viz_img_seq(st_res['imgs'], st_res["flows_f"], if_debug=True)

    save_img_seq(st_res['imgs'], name="3",if_debug=True)
    save_img_seq(st_res["flows_f"], name="4",if_debug=True)

    # t the flow spatial trans
