import easydict
import torch
from torchvision import transforms
from transforms.co_transforms import get_co_transforms, get_co_transforms_s
from transforms import sep_transforms
from datasets.flow_datasets import SelfRender, DAVIS
from transforms.ar_transforms.sp_transfroms import RandomAffineFlow
from non_fourier_data_creation import CreateNonFourierData
import numpy as np
import os
from utils.flow_utils import save_img_seq, viz_img_seq, writeFlow
from copy import deepcopy
import cv2
import warnings
import matplotlib.pyplot as plt

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
    out = cv2.VideoWriter(video_file, fourcc, 30, (size[-2], size[-3]))
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
                        "root_sintel": "/home/4TSSD/opticalflowdataset/MPI-Sintel-complete/",
                        "root_self_render": "../../opticalflowdataset/sec/human/select",
                        "run_at": True,
                        "test_shape": [1024, 1024],
                        "train_shape": [448, 832],
                        "train_n_frames": 30,
                        "train_subsplit": "trainval",
                        "type": "Sintel_Special",
                        "val_n_frames": 2,
                        "val_subsplit": "trainval"},
               "data_aug": {"crop": True,
                            "hflip": True,
                            "para_crop": [448, 832],
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
    seq_len = 16
    # only translation motion
    # train_set = SintelDualExtend(cfg.root_sintel, n_frames=cfg.train_n_frames,
    #                              split='training', subsplit=cfg.train_subsplit,
    #                              with_flow=True,
    #                              ap_transform=None,
    #                              transform=input_transform,
    #                              co_transform=co_transform)

    train_set = DAVIS(cfg.root_davis, n_frames=cfg.train_n_frames,
                      split='train', subsplit=cfg.train_subsplit,
                      with_flow=True,
                      ap_transform=None,
                      transform=input_transform,
                      co_transform=co_transform,
                      target_transform={"flow": sep_transforms.ArrayToTensor()}
                      )
    # using image folder
    import torchvision
    path = '/home/szt/下载/test_gt_samples_1000/test_gt/'
    image_set = torchvision.datasets.ImageFolder(path)
    # plt.imshow(image_set[0])
    # plt.show()

    output_path = "./"
    img_file = os.path.join(output_path, "image")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(img_file):
        os.makedirs(img_file)

    # random select a sample from the dataset
    file_idx =40
    success_num = 80

    # types1 = random noise
    # types2 = blur
    # types3 = water wave
    # types4 =  random texture
    # types5 =  LUMINANCE
    # types6 =  shuffle fourier phase
    # types7 =  flow shuffle
    # types8 =  swirl
    # types9 =  pure drift-balance

    while file_idx < success_num:
        step = np.random.randint(0, len(train_set))
        data = train_set[step]
        # data["img_list"] = [x.unsqueeze(0).cuda() for x in data["img_list"]]
        image = image_set[np.random.randint(0, len(image_set))][0]
        # to PIL RGB
        image = image.convert('RGB')
        image = np.array(image)

        print(image.shape)
        image = image.transpose([2, 0, 1])
        image = torch.from_numpy(image).float().unsqueeze(0).cuda()/255.0
        # idx = np.random.randint(0, len( data["img_list"]))
        # if you want to use the first frame as the reference frame, you can use the following code
        data["img_list"] = [image.clone() for i in range(seq_len)]

        # read data to device
        first_order_flow = [x.unsqueeze(0).cuda() for x in data['target']["flow"]]
        viz_img_seq(data['img_list'], if_debug=False)
        imgs_oc = None
        iter = 0
        speed_factor = np.random.uniform(8, 30)
        rotate_factor = np.random.uniform(0, 1)
        while imgs_oc is None:
            # type changes with the file_idx
            types =2
            # set numpy random seed
            seed = np.random.randint(0, 100000)
            np.random.seed(seed)
            # set torch random seed
            torch.manual_seed(seed)
            # set cuda random seed
            torch.cuda.manual_seed(seed)
            # set cudnn random seed

            print("type: ", types)
            imgs_oc, occ_mask, flow = CreateNonFourierData(mean=speed_factor,
                                                           sigma_i=1,
                                                           sigma_v=0.5,
                                                           sigma_zoom_v=0.00,
                                                           sigma_rotate_v=0.01,
                                                           zoom_ini=0.00, rd_select=[1, 1],
                                                           rotate_range=rotate_factor, compact=100, n_seg=140,
                                                           SeqLen=seq_len, types=types,
                                                           seed=seed).forward_second_order(
                deepcopy(data['img_list']),
                types=types)  # occ is 0, nocc is 1

        # check the quality of the generated data
        # occlude the boundary 300 pixels
        temptest = deepcopy(imgs_oc)
        # for i in range(len(temptest)):
        #     temptest[i][:, :, :, :100] = 0.5 * temptest[i][:, :, :, :100]
        #     temptest[i][:, :, :, -100:] = 0.5 * temptest[i][:, :, :, -100:]
        #     temptest[i][:, :, :100, :] = 0.5 * temptest[i][:, :, :100, :]
        #     temptest[i][:, :, -100:, :] = 0.5 * temptest[i][:, :, -100:, :]

        # show video of imgs_oc using plt
        save_img_seq(temptest, name="./temp/", if_debug=True)
        # get response from the keyboard
        response = input("Is the data qualified? (y/n)")
        if response == "y":
            # save the data
            save_img_seq(imgs_oc, name=img_file, if_debug=False)

            # refresh the numpy random seed
            for types in range(1, 9 ):
                if types == 4:
                    continue
                np.random.seed(seed)
                # set torch random seed
                torch.manual_seed(seed)
                # set cuda random seed
                torch.cuda.manual_seed(seed)
                # set cudnn random seed

                print("type: ", types)
                imgs_oc, occ_mask, flow = CreateNonFourierData(mean=speed_factor,
                                                               sigma_i=1,
                                                               sigma_v=0.5,
                                                               sigma_zoom_v=0.00,
                                                               sigma_rotate_v=0.01,
                                                               zoom_ini=0.00, rd_select=[1, 1],
                                                               rotate_range=rotate_factor, compact=100, n_seg=140,
                                                               SeqLen=seq_len, types=types,
                                                               seed=seed).forward_second_order(
                    deepcopy(data['img_list']),
                    types=types)  # occ is 0, nocc is 1
                file_root = os.path.join(img_file, "Scene_" + str(file_idx))
                file_root_type = os.path.join(file_root, "type_" + str(types))
                if not os.path.exists(file_root_type):
                    os.makedirs(file_root_type)
                write_img_scene(file_root_type, imgs_oc)

                if types == 1:
                    file = os.path.join(file_root_type, "sec flow")
                    if not os.path.exists(file):
                        os.makedirs(file)
                    flow_temp = [flows.detach().cpu().numpy().transpose([0, 2, 3, 1]) for flows in flow]
                    write_flow_scene(file, flow_temp)

                    file = os.path.join(file_root_type, "first flow")
                    if not os.path.exists(file):
                        os.makedirs(file)
                    flow_temp = [flows.detach().cpu().numpy().transpose([0, 2, 3, 1]) for flows in first_order_flow]
                    write_flow_scene(file, flow_temp)

                    file = os.path.join(file_root_type, "visualization")
                    if not os.path.exists(file):
                        os.makedirs(file)
                    save_img_seq(flow, name=os.path.join(file, "flow"), if_debug=True)
                    save_img_seq(occ_mask, name=os.path.join(file, "second_mask"), if_debug=True)
                    save_img_seq(first_order_flow, name=os.path.join(file, "first_flow"), if_debug=True)

            print("finish scene {}".format(file_idx))

            file_idx += 1
        else:
            continue


create_second_order_dataset()
