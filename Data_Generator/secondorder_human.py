import torch
from non_fourier_data_creation import CreateNonFourierData
import numpy as np
import os
from utils.flow_utils import save_img_seq, viz_img_seq, writeFlow
from copy import deepcopy
import cv2
import warnings
import torchvision

warnings.filterwarnings("ignore")


# This script is used for create second-order motion using natural static images
# We use the real image from:https://github.com/geomagical/lama-with-refiner/tree/refinement
# https://drive.google.com/drive/folders/1-1Ci9lFgLmEuHgvRhH3GIJanFDs8ce3-


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


def create_second_order_dataset():
    seq_len = 16
    path = './miniset'
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
    file_idx = 40
    success_num = 80

    # Different types of modulation
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

        # data["img_list"] = [x.unsqueeze(0).cuda() for x in data["img_list"]]
        image = image_set[np.random.randint(0, len(image_set))][0]
        # to PIL RGB
        image = image.convert('RGB')
        image = np.array(image)

        print(image.shape)
        image = image.transpose([2, 0, 1])
        image = torch.from_numpy(image).float().unsqueeze(0).cuda() / 255.0
        # idx = np.random.randint(0, len( data["img_list"]))
        # if you want to use the first frame as the reference frame, you can use the following code

        imgs_oc = None
        iter = 0
        speed_factor = np.random.uniform(8, 30)
        rotate_factor = np.random.uniform(0, 1)
        while imgs_oc is None:
            # type changes with the file_idx
            types = 2
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
                [deepcopy(image) for i in range(16)], types=types)  # occ is 0, nocc is 1

        # check the quality of the generated data
        # occlude the boundary 300 pixels
        temptest = deepcopy(imgs_oc)
        for i in range(len(temptest)):
            temptest[i][:, :, :, :100] = 0.5 * temptest[i][:, :, :, :100]
            temptest[i][:, :, :, -100:] = 0.5 * temptest[i][:, :, :, -100:]
            temptest[i][:, :, :100, :] = 0.5 * temptest[i][:, :, :100, :]
            temptest[i][:, :, -100:, :] = 0.5 * temptest[i][:, :, -100:, :]

        # show video of imgs_oc using plt
        save_img_seq(temptest, name="./temp/temp", if_debug=True)
        # get response from the keyboard
        response = input("Check the data in temp folder, Is the data qualified? (y/n)")
        if response == "y":
            # save the data
            save_img_seq(imgs_oc, name=img_file, if_debug=True)

            # refresh the numpy random seed
            for types in range(1, 9):
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
                    deepcopy([image] * 16),
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
                    file = os.path.join(file_root_type, "visualization")
                    if not os.path.exists(file):
                        os.makedirs(file)
                    save_img_seq(flow, name=os.path.join(file, "flow"), if_debug=True)
                    save_img_seq(occ_mask, name=os.path.join(file, "second_mask"), if_debug=True)

            print("finish scene {}".format(file_idx))

            file_idx += 1
        else:
            continue


create_second_order_dataset()
