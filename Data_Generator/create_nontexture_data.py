import easydict
from torchvision import transforms
from torch.utils.data import ConcatDataset
from transforms.co_transforms import get_co_transforms, get_co_transforms_s
from transforms import sep_transforms
from datasets.flow_datasets import Sintel, DAVIS, SelfRender
from Data_Generator.simple_motion_lib import CreateNonFourierData
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
                        "root_davis": "../../opticalflowdataset/davis2016flow/",
                        "run_at": True,
                        "train_n_frames": 24,
                        "train_subsplit": "trainval",
                        "type": "Sintel_Special",
                        "val_n_frames": 2,
                        "val_subsplit": "trainval"},
               "data_aug": {"crop": True,
                            "hflip": True,
                            "para_crop": [384, 512],
                            'resize_shape': [832, 832],
                            "swap": False},
               "seed": 0}

cfg_all = easydict.EasyDict(config_dict)
cfg = cfg_all.data


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

bais = 0 # indicate start index
def create_non_texture_dataset():
    creat_occ_seq = CreateNonFourierData(mean=0, sigma_i=10, sigma_v=0.5, sigma_zoom_v=0.005, sigma_rotate_v=0.5,
                                         zoom_ini=0.001,
                                         rotate_range=5, compact=150, n_seg=35)

    train_set = DAVIS(cfg.root_davis, n_frames=cfg.train_n_frames,
                      split='train', subsplit=cfg.train_subsplit,
                      with_flow=True,
                      ap_transform=None,
                      transform=input_transform,
                      co_transform=co_transform,
                      target_transform={"flow": sep_transforms.ArrayToTensor()}
                      )

    output_path = "./"
    img_file = os.path.join(output_path, "image")
    flow_file = os.path.join(output_path, "flow")
    viz_file = os.path.join(output_path, "viz")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(img_file):
        os.makedirs(img_file)
    if not os.path.exists(flow_file):
        os.makedirs(flow_file)
    if not os.path.exists(viz_file):
        os.makedirs(viz_file)

    for i_step, data in enumerate(train_set):
        step = i_step * 3
        i_step = i_step + bais
        data = train_set[step]
        data["img_list"] = [x.unsqueeze(0).cuda() for x in data["img_list"]]

        # read data to device

        viz_img_seq(data['img_list'], if_debug=False)

        imgs_oc, occ_mask, flow = creat_occ_seq.forward_first_order(deepcopy(data['img_list']),
                                                                    seq_len=25, bulr_level=100)  # occ is 0, nocc is 1
        data = {}

        file = os.path.join(img_file, "Scene_" + str(i_step))
        write_img_scene(file, imgs_oc)

        file = os.path.join(flow_file, "Scene_" + str(i_step))
        flow_temp = [flows.detach().cpu().numpy().transpose([0, 2, 3, 1]) for flows in flow]
        write_flow_scene(file, flow_temp)
        file = os.path.join(viz_file, "Scene_" + str(i_step))
        if not os.path.exists(file):
            os.makedirs(file)
        save_img_seq(flow, name=os.path.join(file, "flow"), if_debug=True)

        print("finish scene {}".format(i_step))

        # save_img_seq(imgs_oc, name="img", if_debug=True)
        # save_img_seq(flow, name="flow", if_debug=True)
        # viz_img_seq(imgs_oc, flow, if_debug=True)
        # viz_img_seq(imgs_oc, occ_mask, if_debug=True)


if __name__ == "__main__":
    create_non_texture_dataset()
