import pandas as pd
import numpy as np
import os.path as path
import os
import matplotlib
import imageio

types = 7
Scene_id_start = 1
Scene_id_end = 40
file_root = '.\\human_static\\'
mid_frame_num = 7

print('Please predefine the location of the center of the aperture, using excel file')
def read_optical_flow(file):
    """
    read optical flow file
    :param file: optical flow file
    :return: optical flow matrix
    """
    f = open(file, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = int(np.fromfile(f, np.int32, count=1))
        h = int(np.fromfile(f, np.int32, count=1))
        print("Reading %d x %d flo file" % (h, w))
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h, w, 2))
    f.close()
    return data2d


for scene_id in range(Scene_id_start, Scene_id_end + 1):
    # read xlsx file
    root = file_root + 'Scene_%d\\' % scene_id
    # glob all folders
    folders = [folder for folder in os.listdir(root) if os.path.isdir(os.path.join(root, folder))]
    # sort folders
    folders.sort(key=lambda x: int(x.split('_')[-1]))
    # rename folders to type_%d
    for i, folder in enumerate(folders):
        os.rename(os.path.join(root, folder), os.path.join(root, 'type_%d' % (i + 1)))

    file_xlsx = os.path.join(root, 'NewConvention_Final.xlsx')

    pdSetupFile = pd.read_excel(file_xlsx)
    print("Reading %s" % file_xlsx)
    total_trial = types * 4
    MovI = pdSetupFile['MovI'].tolist()  # Scene number
    locationX = pdSetupFile['picX'].tolist()[0]  # this is GT center, not the presenting center
    locationY = pdSetupFile['picY'].tolist()[0]
    Angle = pdSetupFile['angle'].tolist()
    Radius = pdSetupFile['radius'].tolist()
    flowU = pdSetupFile['flowU'].tolist()
    flowV = pdSetupFile['flowV'].tolist()
    flipType = pdSetupFile['fliptype'].tolist()
    R = pdSetupFile['R'].tolist()
    G = pdSetupFile['G'].tolist()
    B = pdSetupFile['B'].tolist()
    Modulation = pdSetupFile['modulation'].tolist()

    # add image size to pdSetupFile
    pdSetupFile['W'] = [None] * len(MovI)
    pdSetupFile['H'] = [None] * len(MovI)

    # read optical flow 14th frame
    file_optical = os.path.join(root, 'type_1\\sec flow\\frame_%04d.flo' % mid_frame_num)
    optical = read_optical_flow(file_optical)
    vecX, vecY = optical[locationY, locationX, :]
    print(vecX, vecY)
    # assert non zero value
    assert vecX != 0 or vecY != 0, "Optical flow is zero, Please check the location of the image"
    # transform to polar coordinate with angle[0, 2pi]
    angle = np.arctan2(-vecY, vecX)

    if angle < 0:
        angle += 2 * np.pi
    # transform to polar coordinate with radius[0, 1]
    radius = np.sqrt(vecX ** 2 + vecY ** 2)
    radius = radius / np.sqrt(2)

    # read image rgb type 1 to type 10
    #
    # file1 = '.\\type_1\\frame_0014.png'
    # file2 = '.\\type_1\\frame_0015.png'
    index = 0
    for k in range(1,4+1):  # 4 flips
        for i in range(1, types + 1):
            file_name = os.path.join(root, 'type_%d' % i)  # '.\\type_' + str(i)
            # list all .png files in the folder
            file_list = [file for file in os.listdir(file_name) if file.endswith('.png')]
            # sort the file list
            file_list.sort()
            # get the file name
            imglist = [imageio.imread(file_name + '\\' + file_list[i]) for i in
                       range(mid_frame_num - 2, mid_frame_num + 2)]
            # read image
            image = sum(imglist) / len(imglist) / 255
            # convert RGB to HSV
            image = matplotlib.colors.rgb_to_hsv(image)
            # get the hue channel
            hue = image[:, :, 0][locationY, locationX]
            # get the luminance channel
            lum = image[:, :, 2][locationY, locationX]
            # reverse the luminance and hue
            lum = (lum + 0.5) % 1 if 0.3 < lum < 0.7 else 1 - lum
            hue = (hue + 0.5) % 1 if 0.3 < hue < 0.7 else 1 - hue
            # get the saturation channel
            sat = 0.95
            # hsv to rgb
            r, g, b = matplotlib.colors.hsv_to_rgb([hue, sat, lum])
            # according to the modulation type, wirte the rgb value to the xlsx file
            pdSetupFile['modulation'][index] = i
            # write flip type to xlsx file
            pdSetupFile['fliptype'][index] = k
            pdSetupFile['R'][index], pdSetupFile['G'][index], pdSetupFile['B'][index] = r, g, b
            index += 1

    # write angle and radius to xlsx file
    for i in range(len(Angle)):
        pdSetupFile['angle'][i] = angle
        pdSetupFile['radius'][i] = radius
    # write flowU and flowV to xlsx file
    for i in range(len(flowU)):
        pdSetupFile['flowU'][i] = vecX
        pdSetupFile['flowV'][i] = vecY
    # write image resolution to xlsx file
    for i in range(len(flowU)):
        pdSetupFile['W'][i] = image.shape[1]
        pdSetupFile['H'][i] = image.shape[0]

    # write MoVI to xlsx file
    for i in range(len(MovI)):
        pdSetupFile['MovI'][i] = scene_id

    # delete the index over the total trial
    pdSetupFile = pdSetupFile.drop(pdSetupFile.index[total_trial:])

    # modify the file = '.\\NewConvention' + '.xlsx'
    pdSetupFile.to_excel(file_xlsx, index=False)
