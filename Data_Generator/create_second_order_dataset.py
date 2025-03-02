import easydict
from torchvision import transforms
from transforms.co_transforms import get_co_transforms, get_co_transforms_s
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
                        "root_sintel": "/home/4TSSD/opticalflowdataset/MPI-Sintel-complete/",
                        # Note: change the dataset address for your envs, here we select sintel dataset as an example.
                        # then this script will generate second-order effect  attached on the selected first-order dataset.
                        # note we dont use this version in the manuscript， but we think it could be helpful for others

                        "run_at": True,
                        "test_shape": [448, 1024],
                        "train_shape": [448, 1024],
                        "train_n_frames": 24,
                        "train_subsplit": "trainval",
                        "type": "Sintel_Special",
                        "val_n_frames": 2,
                        "val_subsplit": "trainval"},
               "data_aug": {"crop": True,
                            "hflip": True,
                            "para_crop": [416, 832],
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
    sep_transforms.ArrayToTensor(),
    transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),  # normalize to [0，1]
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # normalize to [-1，1]
])
co_transform = get_co_transforms_s(aug_args=cfg_all.data_aug)


def create_second_order_dataset():
    seq_len = 24
    train_set = Sintel(cfg.root_sintel, n_frames=seq_len,
                       split='training', subsplit=cfg.train_subsplit,
                       with_flow=True,
                       type='albedo',
                       ap_transform=None,
                       transform=input_transform,
                       co_transform=co_transform,
                       target_transform={"flow": sep_transforms.ArrayToTensor()}
                       )

    sp_transform = RandomAffineFlow(cfg_train.st_cfg, addnoise=cfg_train.st_cfg.add_noise).cuda()

    output_path = "./"
    img_file = os.path.join(output_path, "flow_data")

    # random select a sample from the dataset
    file_idx = 1
    success_num = 100 * 20
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
        # types1 = random noise
        # types2 = blur
        # types3 = water wave
        # types4 =  random texture
        # types5 =  LUMINANCE
        # types6 =  shuffle fourier phase
        # types7 =  flow shuffle
        # types8 =  swirl
        # types9 =  pure drift-balance
        while imgs_oc is None:
            iter += 1
            if iter > 2:
                break
            # type changes with the file_idx
            if file_idx < 20:
                types = 1
            elif 40 <= file_idx < 60:
                types = 2
            elif 60 <= file_idx < 80:
                types = 3
            elif 80 <= file_idx < 100:
                types = 4
            elif 100 <= file_idx < 120:
                types = 5
            elif 120 <= file_idx < 140:
                types = 6
            elif 140 <= file_idx < 160:
                types = 7
            elif 160 <= file_idx < 180:
                types = 8
            elif 180 <= file_idx < 200:
                types = 10

            print("type: ", types)

            try:
                imgs_oc, occ_mask, flow = CreateNonFourierData(mean=15,
                                                               sigma_i=3,
                                                               sigma_v=0.5,
                                                               sigma_zoom_v=0.00,
                                                               sigma_rotate_v=0.05,
                                                               zoom_ini=0.00,
                                                               rd_select=[2, 6],
                                                               rotate_range=0.5, compact=80, n_seg=80,
                                                               SeqLen=seq_len, types=types).forward_second_order(
                    deepcopy(data['img_list']), types=types)  # occ is 0, nocc is 1
            except:
                print("bad scene, skip this scene")
                continue

        if iter > 2:
            print("bad scene, skip this scene")
            continue

        scene_file = os.path.join(img_file, "Scene_" + str(file_idx))
        if not os.path.exists(scene_file):
            os.makedirs(scene_file)

        file = os.path.join(scene_file, "img")
        if not os.path.exists(file):
            os.makedirs(file)
        write_img_scene(file, imgs_oc)

        file = os.path.join(scene_file, 'flow_second')
        if not os.path.exists(file):
            os.makedirs(file)
        flow_temp = [flows.detach().cpu().numpy().transpose([0, 2, 3, 1]) for flows in flow]
        write_flow_scene(file, flow_temp)

        file = os.path.join(scene_file, 'flow_first')
        if not os.path.exists(file):
            os.makedirs(file)
        flow_temp = [flows.detach().cpu().numpy().transpose([0, 2, 3, 1]) for flows in first_order_flow]
        write_flow_scene(file, flow_temp)

        file = os.path.join(scene_file, 'viz')
        if not os.path.exists(file):
            os.makedirs(file)
        save_img_seq(flow, name=os.path.join(file, "flow"), if_debug=True)
        save_img_seq(occ_mask, name=os.path.join(file, "second_mask"), if_debug=True)
        save_img_seq(first_order_flow, name=os.path.join(file, "first_flow"), if_debug=True)
        print("finish scene {}".format(file_idx))
        file_idx += 1


create_second_order_dataset()
