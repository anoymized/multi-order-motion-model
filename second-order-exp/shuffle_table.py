import pandas as pd
import numpy as np
import os.path as path
import os


# extract all scene and save into the shuffled list
Block_num = 40  # number if trails for each bock
types = 7
Scene_id_start = 1
Scene_id_end = 40
file_root = '.\\human_static\\'

MovI_all = []
locationX_all = []
locationY_all = []
Angle_all = []
Radius_all = []
flowU_all = []
flowV_all = []
flipType_all = []
R_all = []
G_all = []
B_all = []
Modulation_all = []
H_all = []
W_all = []




# combine all the xlsx files into one
for scene_id in range(Scene_id_start, Scene_id_end + 1):
    # read xlsx file
    root = file_root + 'Scene_%d\\' % scene_id

    file_xlsx = os.path.join(root, 'NewConvention_Final.xlsx')
    pdSetupFile = pd.read_excel(file_xlsx)
    print("Reading %s" % file_xlsx)
    # save excel file

    MovI = pdSetupFile['MovI'].tolist()  # Scene number
    locationX = pdSetupFile['picX'].tolist()  # this is GT center, not the presenting center
    locationY = pdSetupFile['picY'].tolist()
    Angle = pdSetupFile['angle'].tolist()
    Radius = pdSetupFile['radius'].tolist()
    flowU = pdSetupFile['flowU'].tolist()
    flowV = pdSetupFile['flowV'].tolist()
    flipType = pdSetupFile['fliptype'].tolist()
    R = pdSetupFile['R'].tolist()
    G = pdSetupFile['G'].tolist()
    B = pdSetupFile['B'].tolist()
    Modulation = pdSetupFile['modulation'].tolist()
    H = pdSetupFile['H'].tolist()
    W = pdSetupFile['W'].tolist()
    MovI_all.extend(MovI)
    locationX_all.extend(locationX)
    locationY_all.extend(locationY)
    Angle_all.extend(Angle)
    Radius_all.extend(Radius)
    flowU_all.extend(flowU)
    flowV_all.extend(flowV)
    flipType_all.extend(flipType)
    R_all.extend(R)
    G_all.extend(G)
    B_all.extend(B)
    Modulation_all.extend(Modulation)
    H_all.extend(H)
    W_all.extend(W)
# shuffle the list
index = np.arange(len(MovI_all))
np.random.shuffle(index)
MovI_all = np.array(MovI_all)[index]
locationX_all = np.array(locationX_all)[index]
locationY_all = np.array(locationY_all)[index]
Angle_all = np.array(Angle_all)[index]
Radius_all = np.array(Radius_all)[index]
flowU_all = np.array(flowU_all)[index]
flowV_all = np.array(flowV_all)[index]
flipType_all = np.array(flipType_all)[index]
R_all = np.array(R_all)[index]
G_all = np.array(G_all)[index]
B_all = np.array(B_all)[index]
Modulation_all = np.array(Modulation_all)[index]
H_all = np.array(H_all)[index]
W_all = np.array(W_all)[index]

# save the shuffled to xlsx file
pdSetupFile = pd.DataFrame()
pdSetupFile['MovI'] = MovI_all
pdSetupFile['picX'] = locationX_all
pdSetupFile['picY'] = locationY_all
pdSetupFile['angle'] = Angle_all
pdSetupFile['radius'] = Radius_all
pdSetupFile['flowU'] = flowU_all
pdSetupFile['flowV'] = flowV_all
pdSetupFile['fliptype'] = flipType_all
pdSetupFile['R'] = R_all
pdSetupFile['G'] = G_all
pdSetupFile['B'] = B_all
pdSetupFile['modulation'] = Modulation_all
pdSetupFile['H'] = H_all
pdSetupFile['W'] = W_all


Trail_Block = np.zeros(len(MovI_all))
assert len(MovI_all) % Block_num == 0
for i in range(len(MovI_all)):
    Trail_Block[i] = i // Block_num + 1

pdSetupFile['Trail_Block'] = Trail_Block
pdSetupFile.to_excel('.\\Static_AllTrails_Final_shuffled.xlsx', index=False)
