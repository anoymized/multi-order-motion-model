import os.path as path
import os
import matplotlib.pyplot as plt
import matplotlib
import imageio
import pandas as pd
import numpy as np


def polar2cartesian(angle, radius):  # angle in rad (0-2pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    x = radius * np.cos(angle)
    y = -radius * np.sin(angle)
    return x, y


def cartesian2polar(x, y):
    radius = np.sqrt(x ** 2 + y ** 2)
    angle = np.arctan2(-y, x)
    # if angle < 0:
    #     angle += 2 * np.pi
    angle = angle + (angle < 0) * 2 * np.pi
    return angle, radius


file_root = '.\\data\\szt'
# list all .csv files in the folder
file_list = [f for f in os.listdir(file_root) if f.endswith('.xlsx')]
# read csv file
Mod1_all, Mod2_all, Mod3_all, Mod4_all, Mod5_all, Mod6_all, Mod7_all, Mod8_all, Mod9_all, Mod10_all = \
    [], [], [], [], [], [], [], [], [], []
GT_v, GT_a, Res_v, Res_a, Modulation, FlipType, Mov = [], [], [], [], [], [], []

for file in file_list:
    # read xlsx file
    print("Reading file %s" % file)
    # pdSetupFile = pd.read_csv(path.join(file_root, file))
    pdSetupFile = pd.read_excel(path.join(file_root, file))
    GT_v += pdSetupFile['EXP_GTvelocity_raw'].tolist()[0:-4]
    # check float number
    GT_a += pdSetupFile['EXP_GTAngle_raw'].tolist()[0:-4]
    Res_v += pdSetupFile['EXP_Rvelocity_raw'].tolist()[0:-4]
    Res_a += pdSetupFile['EXP_RAngle_raw'].tolist()[0:-4]
    Modulation += pdSetupFile['Mod'].tolist()[0:-4]
    FlipType += pdSetupFile['FT'].tolist()[0:-4]
    Mov += pdSetupFile['mov'].tolist()[0:-4]
    # judge whether the data is float number
    GT_v = [float(i) if isinstance(i, float) else float(i[1:-1]) for i in GT_v]
    GT_a = [float(i) if isinstance(i, float) else float(i[1:-1]) for i in GT_a]
    Res_v = [float(i) if isinstance(i, float) else float(i[1:-1]) for i in Res_v]
    Res_a = [float(i) if isinstance(i, float) else float(i[1:-1]) for i in Res_a]

# select the data with modulation 1
assert len(GT_v) == len(GT_a) == len(Res_v) == len(Res_a) == len(Modulation) == len(FlipType) == len(Mov)
print('Total number of trials: ', len(GT_v))
for i in range(len(Mov)):
    if Modulation[i] == 1:
        Mod1_all.append({"GT_v": GT_v[i],
                         "GT_a": GT_a[i], "Res_v": Res_v[i], "Res_a": Res_a[i], "FlipType": FlipType[i], "Mov": Mov[i]})
    elif Modulation[i] == 2:
        Mod2_all.append({"GT_v": GT_v[i],
                         "GT_a": GT_a[i], "Res_v": Res_v[i], "Res_a": Res_a[i], "FlipType": FlipType[i], "Mov": Mov[i]})
    elif Modulation[i] == 3:
        Mod3_all.append({"GT_v": GT_v[i],
                         "GT_a": GT_a[i], "Res_v": Res_v[i], "Res_a": Res_a[i], "FlipType": FlipType[i], "Mov": Mov[i]})
    elif Modulation[i] == 4:
        Mod4_all.append(
            {"GT_v": GT_v[i], "GT_a": GT_a[i], "Res_v": Res_v[i], "Res_a": Res_a[i], "FlipType": FlipType[i],
             "Mov": Mov[i]})
    elif Modulation[i] == 5:
        Mod5_all.append({"GT_v": GT_v[i], "GT_a": GT_a[i], "Res_v": Res_v[i], "Res_a": Res_a[i],
                         "FlipType": FlipType[i], "Mov": Mov[i]})
    elif Modulation[i] == 6:
        Mod6_all.append(
            {"GT_v": GT_v[i], "GT_a": GT_a[i], "Res_v": Res_v[i], "Res_a": Res_a[i],
             "FlipType": FlipType[i], "Mov": Mov[i]})
    elif Modulation[i] == 7:
        Mod7_all.append(
            {"GT_v": GT_v[i], "GT_a": GT_a[i], "Res_v": Res_v[i], "Res_a": Res_a[i]
                , "Mov": Mov[i], "FlipType": FlipType[i]})
    elif Modulation[i] == 8:
        Mod8_all.append(
            {"GT_v": GT_v[i], "GT_a": GT_a[i], "Res_v": Res_v[i], "Res_a": Res_a[i], "Mov": Mov[i],
             "FlipType": FlipType[i]})
    elif Modulation[i] == 9:
        Mod9_all.append(
            {"GT_v": GT_v[i], "GT_a": GT_a[i], "Res_v": Res_v[i], "Res_a": Res_a[i], "Mov": Mov[i],
             "FlipType": FlipType[i]})
    elif Modulation[i] == 10:
        Mod10_all.append(
            {"GT_v": GT_v[i], "GT_a": GT_a[i], "Res_v": Res_v[i], "Res_a": Res_a[i], "Mov": Mov[i],
             "FlipType": FlipType[i]})


# sort the data by mov and flip type
def sort_by_mov(Mod_all):
    Mod_all = sorted(Mod_all, key=lambda x: (x["Mov"], x["FlipType"]))
    return Mod_all


Mod1_all = sort_by_mov(Mod1_all)
Mod2_all = sort_by_mov(Mod2_all)
Mod3_all = sort_by_mov(Mod3_all)
Mod4_all = sort_by_mov(Mod4_all)
Mod5_all = sort_by_mov(Mod5_all)
Mod6_all = sort_by_mov(Mod6_all)
Mod7_all = sort_by_mov(Mod7_all)
Mod8_all = sort_by_mov(Mod8_all)
Mod9_all = sort_by_mov(Mod9_all)
Mod10_all = sort_by_mov(Mod10_all)

# plot 10 scatter map for each Mov (not averaged)
GT_all_A, GT_all_V = [float(Mod["GT_a"]) for Mod in Mod1_all], [float(Mod["GT_v"]) for Mod in Mod1_all]

Mod1_all_A_res, Mod1_all_V_res = np.array([float(Mod["Res_a"]) for Mod in Mod1_all]), np.array(
    [float(Mod["Res_v"]) for Mod in Mod1_all])
Mod2_all_A_res, Mod2_all_V_res = np.array([float(Mod["Res_a"]) for Mod in Mod2_all]), np.array(
    [float(Mod["Res_v"]) for Mod in Mod2_all])
Mod3_all_A_res, Mod3_all_V_res = np.array([float(Mod["Res_a"]) for Mod in Mod3_all]), np.array(
    [float(Mod["Res_v"]) for Mod in Mod3_all])
Mod4_all_A_res, Mod4_all_V_res = np.array([float(Mod["Res_a"]) for Mod in Mod4_all]), np.array(
    [float(Mod["Res_v"]) for Mod in Mod4_all])
Mod5_all_A_res, Mod5_all_V_res = np.array([float(Mod["Res_a"]) for Mod in Mod5_all]), np.array(
    [float(Mod["Res_v"]) for Mod in Mod5_all])
Mod6_all_A_res, Mod6_all_V_res = np.array([float(Mod["Res_a"]) for Mod in Mod6_all]), np.array(
    [float(Mod["Res_v"]) for Mod in Mod6_all])
Mod7_all_A_res, Mod7_all_V_res = np.array([float(Mod["Res_a"]) for Mod in Mod7_all]), np.array(
    [float(Mod["Res_v"]) for Mod in Mod7_all])
Mod8_all_A_res, Mod8_all_V_res = np.array([float(Mod["Res_a"]) for Mod in Mod8_all]), np.array(
    [float(Mod["Res_v"]) for Mod in Mod8_all])
Mod9_all_A_res, Mod9_all_V_res = np.array([float(Mod["Res_a"]) for Mod in Mod9_all]), np.array(
    [float(Mod["Res_v"]) for Mod in Mod9_all])
Mod10_all_A_res, Mod10_all_V_res = np.array([float(Mod["Res_a"]) for Mod in Mod10_all]), np.array(
    [float(Mod["Res_v"]) for Mod in Mod10_all])

angle_diff = np.abs(Mod1_all_A_res - GT_all_A)
Mod1_all_A_res = Mod1_all_A_res + 360 * ((angle_diff > 180) & (Mod1_all_A_res < 180)) - 360 * ((angle_diff > 180) & (
        Mod1_all_A_res > 180))  # TODO: check if this is correct
angle_diff = np.abs(Mod2_all_A_res - GT_all_A)
Mod2_all_A_res = Mod2_all_A_res + 360 * ((angle_diff > 180) & (Mod2_all_A_res < 180)) - 360 * ((angle_diff > 180) & (
        Mod2_all_A_res > 180))  # TODO: check if this is correct
angle_diff = np.abs(Mod3_all_A_res - GT_all_A)
Mod3_all_A_res = Mod3_all_A_res + 360 * ((angle_diff > 180) & (Mod3_all_A_res < 180)) - 360 * ((angle_diff > 180) & (
        Mod3_all_A_res > 180))  # TODO: check if this is correct
angle_diff = np.abs(Mod4_all_A_res - GT_all_A)
Mod4_all_A_res = Mod4_all_A_res + 360 * ((angle_diff > 180) & (Mod4_all_A_res < 180)) - 360 * ((angle_diff > 180) & (
        Mod4_all_A_res > 180))  # TODO: check if this is correct
angle_diff = np.abs(Mod5_all_A_res - GT_all_A)
Mod5_all_A_res = Mod5_all_A_res + 360 * ((angle_diff > 180) & (Mod5_all_A_res < 180)) - 360 * ((angle_diff > 180) & (
        Mod5_all_A_res > 180))  # TODO: check if this is correct
angle_diff = np.abs(Mod6_all_A_res - GT_all_A)
Mod6_all_A_res = Mod6_all_A_res + 360 * ((angle_diff > 180) & (Mod6_all_A_res < 180)) - 360 * ((angle_diff > 180) & (
        Mod6_all_A_res > 180))  # TODO: check if this is correct
angle_diff = np.abs(Mod7_all_A_res - GT_all_A)
Mod7_all_A_res = Mod7_all_A_res + 360 * ((angle_diff > 180) & (Mod7_all_A_res < 180)) - 360 * ((angle_diff > 180) & (
        Mod7_all_A_res > 180))  # TODO: check if this is correct
angle_diff = np.abs(Mod8_all_A_res - GT_all_A)
Mod8_all_A_res = Mod8_all_A_res + 360 * ((angle_diff > 180) & (Mod8_all_A_res < 180)) - 360 * ((angle_diff > 180) & (
        Mod8_all_A_res > 180))  # TODO: check if this is correct
angle_diff = np.abs(Mod9_all_A_res - GT_all_A)
Mod9_all_A_res = Mod9_all_A_res + 360 * ((angle_diff > 180) & (Mod9_all_A_res < 180)) - 360 * ((angle_diff > 180) & (
        Mod9_all_A_res > 180))  # TODO: check if this is correct
angle_diff = np.abs(Mod10_all_A_res - GT_all_A)
Mod10_all_A_res = Mod10_all_A_res + 360 * ((angle_diff > 180) & (Mod10_all_A_res < 180)) - 360 * (
        (angle_diff > 180) & (Mod10_all_A_res > 180))  # TODO: check if this is correct

print(max(np.abs(Mod1_all_A_res - GT_all_A)))
print(max(np.abs(Mod2_all_A_res - GT_all_A)))
print(max(np.abs(Mod3_all_A_res - GT_all_A)))
print(max(np.abs(Mod4_all_A_res - GT_all_A)))
print(max(np.abs(Mod5_all_A_res - GT_all_A)))
print(max(np.abs(Mod6_all_A_res - GT_all_A)))
print(max(np.abs(Mod7_all_A_res - GT_all_A)))
print(max(np.abs(Mod8_all_A_res - GT_all_A)))
print(max(np.abs(Mod9_all_A_res - GT_all_A)))
print(max(np.abs(Mod10_all_A_res - GT_all_A)))

import numpy as np

# calculate the pearson correlation coefficient between Mod1_all_A and Res
corr_A_1 = np.corrcoef(GT_all_A, Mod1_all_A_res)[0, 1]
corr_V_1 = np.corrcoef(GT_all_V, Mod1_all_V_res)[0, 1]
corr_A_2 = np.corrcoef(GT_all_A, Mod2_all_A_res)[0, 1]
corr_V_2 = np.corrcoef(GT_all_V, Mod2_all_V_res)[0, 1]
corr_A_3 = np.corrcoef(GT_all_A, Mod3_all_A_res)[0, 1]
corr_V_3 = np.corrcoef(GT_all_V, Mod3_all_V_res)[0, 1]
corr_A_4 = np.corrcoef(GT_all_A, Mod4_all_A_res)[0, 1]
corr_V_4 = np.corrcoef(GT_all_V, Mod4_all_V_res)[0, 1]
corr_A_5 = np.corrcoef(GT_all_A, Mod5_all_A_res)[0, 1]
corr_V_5 = np.corrcoef(GT_all_V, Mod5_all_V_res)[0, 1]
corr_A_6 = np.corrcoef(GT_all_A, Mod6_all_A_res)[0, 1]
corr_V_6 = np.corrcoef(GT_all_V, Mod6_all_V_res)[0, 1]
corr_A_7 = np.corrcoef(GT_all_A, Mod7_all_A_res)[0, 1]
corr_V_7 = np.corrcoef(GT_all_V, Mod7_all_V_res)[0, 1]
corr_A_8 = np.corrcoef(GT_all_A, Mod8_all_A_res)[0, 1]
corr_V_8 = np.corrcoef(GT_all_V, Mod8_all_V_res)[0, 1]
corr_A_9 = np.corrcoef(GT_all_A, Mod9_all_A_res)[0, 1]
corr_V_9 = np.corrcoef(GT_all_V, Mod9_all_V_res)[0, 1]
corr_A_10 = np.corrcoef(GT_all_A, Mod10_all_A_res)[0, 1]
corr_V_10 = np.corrcoef(GT_all_V, Mod10_all_V_res)[0, 1]

plt.figure(figsize=(20, 10))
plt.subplot(2, 5, 1, projection='polar')
plt.scatter(GT_all_A, GT_all_V, c='r', marker='o', alpha=0.5, label="GT")
plt.scatter(Mod1_all_A_res, Mod1_all_V_res, c='b', marker='o', alpha=0.5, label="Res")
# set the range of the figure
plt.ylim(0, 15)
# show the correlation coefficient bold in the figure
plt.text(20, 20, "Corr_Angle: " + str(round(corr_A_1, 2)) + "\nCorr_Velocity: " + str(round(corr_V_1, 2)),
         fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
plt.legend()

plt.title("Modulation 1")
plt.subplot(2, 5, 2, projection='polar')
plt.scatter(GT_all_A, GT_all_V, c='r', marker='o', alpha=0.5, label="GT")
plt.scatter(Mod2_all_A_res, Mod2_all_V_res, c='b', marker='o', alpha=0.5, label="Res")
plt.ylim(0, 15)
plt.text(20, 20, "Corr_Angle: " + str(round(corr_A_2, 2)) + "\nCorr_Velocity: " + str(round(corr_V_2, 2)),
         fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
plt.legend()
plt.title("Modulation 2")

plt.subplot(2, 5, 3, projection='polar')
plt.scatter(GT_all_A, GT_all_V, c='r', marker='o', alpha=0.5, label="GT")
plt.scatter(Mod3_all_A_res, Mod3_all_V_res, c='b', marker='o', alpha=0.5, label="Res")
plt.ylim(0, 15)
plt.text(20, 20, "Corr_Angle: " + str(round(corr_A_3, 2)) + "\nCorr_Velocity: " + str(round(corr_V_3, 2)),
         fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
plt.legend()
plt.title("Modulation 3")

plt.subplot(2, 5, 4, projection='polar')
plt.scatter(GT_all_A, GT_all_V, c='r', marker='o', alpha=0.5, label="GT")
plt.scatter(Mod4_all_A_res, Mod4_all_V_res, c='b', marker='o', alpha=0.5, label="Res")
plt.ylim(0, 15)
plt.text(20, 20, "Corr_Angle: " + str(round(corr_A_4, 2)) + "\nCorr_Velocity: " + str(round(corr_V_4, 2)),
         fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
plt.legend()
plt.title("Modulation 4")

plt.subplot(2, 5, 5, projection='polar')
plt.scatter(GT_all_A, GT_all_V, c='r', marker='o', alpha=0.5, label="GT")
plt.scatter(Mod5_all_A_res, Mod5_all_V_res, c='b', marker='o', alpha=0.5, label="Res")
plt.ylim(0, 15)
plt.text(20, 20, "Corr_Angle: " + str(round(corr_A_5, 2)) + "\nCorr_Velocity: " + str(round(corr_V_5, 2)),
         fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
plt.legend()
plt.title("Modulation 5")

plt.subplot(2, 5, 6, projection='polar')
plt.scatter(GT_all_A, GT_all_V, c='r', marker='o', alpha=0.5, label="GT")
plt.scatter(Mod6_all_A_res, Mod6_all_V_res, c='b', marker='o', alpha=0.5, label="Res")
plt.ylim(0, 15)
plt.text(20, 20, "Corr_Angle: " + str(round(corr_A_6, 2)) + "\nCorr_Velocity: " + str(round(corr_V_6, 2)),
         fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
plt.legend()
plt.title("Modulation 6")

plt.subplot(2, 5, 7, projection='polar')
plt.scatter(GT_all_A, GT_all_V, c='r', marker='o', alpha=0.5, label="GT")
plt.scatter(Mod7_all_A_res, Mod7_all_V_res, c='b', marker='o', alpha=0.5, label="Res")
plt.ylim(0, 15)
plt.text(20, 20, "Corr_Angle: " + str(round(corr_A_7, 2)) + "\nCorr_Velocity: " + str(round(corr_V_7, 2)),
         fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
plt.legend()
plt.title("Modulation 7")

plt.subplot(2, 5, 8, projection='polar')
plt.scatter(GT_all_A, GT_all_V, c='r', marker='o', alpha=0.5, label="GT")
plt.scatter(Mod8_all_A_res, Mod8_all_V_res, c='b', marker='o', alpha=0.5, label="Res")
plt.ylim(0, 15)
plt.text(20, 20, "Corr_Angle: " + str(round(corr_A_8, 2)) + "\nCorr_Velocity: " + str(round(corr_V_8, 2)),
         fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
plt.legend()
plt.title("Modulation 8")

plt.subplot(2, 5, 9, projection='polar')
plt.scatter(GT_all_A, GT_all_V, c='r', marker='o', alpha=0.5, label="GT")
plt.scatter(Mod9_all_A_res, Mod9_all_V_res, c='b', marker='o', alpha=0.5, label="Res")
plt.ylim(0, 15)
plt.text(20, 20, "Corr_Angle: " + str(round(corr_A_9, 2)) + "\nCorr_Velocity: " + str(round(corr_V_9, 2)),
         fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
plt.legend()
plt.title("Modulation 9")

plt.subplot(2, 5, 10, projection='polar')
plt.scatter(GT_all_A, GT_all_V, c='r', marker='o', alpha=0.5, label="GT")
plt.scatter(Mod10_all_A_res, Mod10_all_V_res, c='b', marker='o', alpha=0.5, label="Res")
plt.ylim(0, 15)
plt.text(20, 20, "Corr_Angle: " + str(round(corr_A_10, 2)) + "\nCorr_Velocity: " + str(round(corr_V_10, 2)),
         fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
plt.legend()
plt.title("Modulation 10")
plt.tight_layout()
plt.show()

max_mov = 35
Avg_gt_X, Avg_gt_Y, = [], []

Mod1_avg_res_X, Mod2_avg_res_X, Mod3_avg_res_X, Mod4_avg_res_X, Mod5_avg_res_X, Mod6_avg_res_X, Mod7_avg_res_X, Mod8_avg_res_X, \
    Mod9_avg_res_X, Mod10_avg_res_X = [], [], [], [], [], [], [], [], [], []

Mod1_avg_res_Y, Mod2_avg_res_Y, Mod3_avg_res_Y, Mod4_avg_res_Y, Mod5_avg_res_Y, Mod6_avg_res_Y, Mod7_avg_res_Y, Mod8_avg_res_Y, \
    Mod9_avg_res_Y, Mod10_avg_res_Y = [], [], [], [], [], [], [], [], [], []

for i in range(max_mov):
    Avg_gt_X.append([])
    Avg_gt_Y.append([])
    Mod1_avg_res_X.append([]), Mod2_avg_res_X.append([]), Mod3_avg_res_X.append([]), Mod4_avg_res_X.append(
        []), Mod5_avg_res_X.append([]), \
        Mod6_avg_res_X.append([]), Mod7_avg_res_X.append([]), Mod8_avg_res_X.append([]), Mod9_avg_res_X.append(
        []), Mod10_avg_res_X.append([])
    Mod1_avg_res_Y.append([]), Mod2_avg_res_Y.append([]), Mod3_avg_res_Y.append([]), Mod4_avg_res_Y.append(
        []), Mod5_avg_res_Y.append([]), \
        Mod6_avg_res_Y.append([]), Mod7_avg_res_Y.append([]), Mod8_avg_res_Y.append([]), Mod9_avg_res_Y.append(
        []), Mod10_avg_res_Y.append([])

for i in range(len(Mod1_all)):
    for mov in range(max_mov):
        if Mod1_all[i]["Mov"] == mov:
            angle = np.deg2rad(Mod1_all[i]["GT_a"])
            x, y = polar2cartesian(angle, Mod1_all[i]["GT_v"])
            # rotation according to the flip type
            if Mod1_all[i]["FlipType"] == 1:
                # rotation according to the angle

                Avg_gt_X[mov].append(x)
                Avg_gt_Y[mov].append(y)
            else:
                if Mod1_all[i]["FlipType"] == 2:
                    x = -x
                elif Mod1_all[i]["FlipType"] == 3:
                    y = -y
                elif Mod1_all[i]["FlipType"] == 4:
                    x, y = -x, -y
                else:
                    raise ValueError("Flip type is not valid")
                Avg_gt_X[mov].append(x)
                Avg_gt_Y[mov].append(y)
        if Mod1_all[i]["Mov"] == mov:
            angle = np.deg2rad(Mod1_all[i]["Res_a"])
            x, y = polar2cartesian(angle, Mod1_all[i]["Res_v"])
            # rotation according to the flip type
            if Mod1_all[i]["FlipType"] == 1:
                # rotation according to the angle
                Mod1_avg_res_X[mov].append(x)
                Mod1_avg_res_Y[mov].append(y)
            else:
                if Mod1_all[i]["FlipType"] == 2:
                    x = -x
                elif Mod1_all[i]["FlipType"] == 3:
                    y = -y
                elif Mod1_all[i]["FlipType"] == 4:
                    x, y = -x, -y
                else:
                    raise ValueError("Flip type is not valid")
                Mod1_avg_res_X[mov].append(x)
                Mod1_avg_res_Y[mov].append(y)
        if Mod2_all[i]["Mov"] == mov:
            angle = np.deg2rad(Mod2_all[i]["Res_a"])
            x, y = polar2cartesian(angle, Mod2_all[i]["Res_v"])
            # rotation according to the flip type
            if Mod2_all[i]["FlipType"] == 1:
                # rotation according to the angle
                Mod2_avg_res_X[mov].append(x)
                Mod2_avg_res_Y[mov].append(y)
            else:
                if Mod2_all[i]["FlipType"] == 2:
                    x = -x
                elif Mod2_all[i]["FlipType"] == 3:
                    y = -y
                elif Mod2_all[i]["FlipType"] == 4:
                    x, y = -x, -y
                else:
                    raise ValueError("Flip type is not valid")
                Mod2_avg_res_X[mov].append(x)
                Mod2_avg_res_Y[mov].append(y)
        if Mod3_all[i]["Mov"] == mov:
            angle = np.deg2rad(Mod3_all[i]["Res_a"])
            x, y = polar2cartesian(angle, Mod3_all[i]["Res_v"])
            # rotation according to the flip type
            if Mod3_all[i]["FlipType"] == 1:
                # rotation according to the angle
                Mod3_avg_res_X[mov].append(x)
                Mod3_avg_res_Y[mov].append(y)
            else:
                if Mod3_all[i]["FlipType"] == 2:
                    x = -x
                elif Mod3_all[i]["FlipType"] == 3:
                    y = -y
                elif Mod3_all[i]["FlipType"] == 4:
                    x, y = -x, -y
                else:
                    raise ValueError("Flip type is not valid")
                Mod3_avg_res_X[mov].append(x)
                Mod3_avg_res_Y[mov].append(y)
        if Mod4_all[i]["Mov"] == mov:
            angle = np.deg2rad(Mod4_all[i]["Res_a"])
            x, y = polar2cartesian(angle, Mod4_all[i]["Res_v"])
            # rotation according to the flip type
            if Mod4_all[i]["FlipType"] == 1:
                # rotation according to the angle
                Mod4_avg_res_X[mov].append(x)
                Mod4_avg_res_Y[mov].append(y)
            else:
                if Mod4_all[i]["FlipType"] == 2:
                    x = -x
                elif Mod4_all[i]["FlipType"] == 3:
                    y = -y
                elif Mod4_all[i]["FlipType"] == 4:
                    x, y = -x, -y
                else:
                    raise ValueError("Flip type is not valid")
                Mod4_avg_res_X[mov].append(x)
                Mod4_avg_res_Y[mov].append(y)
        if Mod5_all[i]["Mov"] == mov:
            angle = np.deg2rad(Mod5_all[i]["Res_a"])
            x, y = polar2cartesian(angle, Mod5_all[i]["Res_v"])
            # rotation according to the flip type
            if Mod5_all[i]["FlipType"] == 1:
                # rotation according to the angle
                Mod5_avg_res_X[mov].append(x)
                Mod5_avg_res_Y[mov].append(y)
            else:
                if Mod5_all[i]["FlipType"] == 2:
                    x = -x
                elif Mod5_all[i]["FlipType"] == 3:
                    y = -y
                elif Mod5_all[i]["FlipType"] == 4:
                    x, y = -x, -y
                else:
                    raise ValueError("Flip type is not valid")
                Mod5_avg_res_X[mov].append(x)
                Mod5_avg_res_Y[mov].append(y)
        if Mod6_all[i]["Mov"] == mov:
            angle = np.deg2rad(Mod6_all[i]["Res_a"])
            x, y = polar2cartesian(angle, Mod6_all[i]["Res_v"])
            # rotation according to the flip type
            if Mod6_all[i]["FlipType"] == 1:
                # rotation according to the angle
                Mod6_avg_res_X[mov].append(x)
                Mod6_avg_res_Y[mov].append(y)
            else:
                if Mod6_all[i]["FlipType"] == 2:
                    x = -x
                elif Mod6_all[i]["FlipType"] == 3:
                    y = -y
                elif Mod6_all[i]["FlipType"] == 4:
                    x, y = -x, -y
                else:
                    raise ValueError("Flip type is not valid")
                Mod6_avg_res_X[mov].append(x)
                Mod6_avg_res_Y[mov].append(y)
        if Mod7_all[i]["Mov"] == mov:
            angle = np.deg2rad(Mod7_all[i]["Res_a"])
            x, y = polar2cartesian(angle, Mod7_all[i]["Res_v"])
            # rotation according to the flip type
            if Mod7_all[i]["FlipType"] == 1:
                # rotation according to the angle
                Mod7_avg_res_X[mov].append(x)
                Mod7_avg_res_Y[mov].append(y)
            else:
                if Mod7_all[i]["FlipType"] == 2:
                    x = -x
                elif Mod7_all[i]["FlipType"] == 3:
                    y = -y
                elif Mod7_all[i]["FlipType"] == 4:
                    x, y = -x, -y
                else:
                    raise ValueError("Flip type is not valid")
                Mod7_avg_res_X[mov].append(x)
                Mod7_avg_res_Y[mov].append(y)
        if Mod8_all[i]["Mov"] == mov:
            angle = np.deg2rad(Mod8_all[i]["Res_a"])
            x, y = polar2cartesian(angle, Mod8_all[i]["Res_v"])
            # rotation according to the flip type
            if Mod8_all[i]["FlipType"] == 1:
                # rotation according to the angle
                Mod8_avg_res_X[mov].append(x)
                Mod8_avg_res_Y[mov].append(y)
            else:
                if Mod8_all[i]["FlipType"] == 2:
                    x = -x
                elif Mod8_all[i]["FlipType"] == 3:
                    y = -y
                elif Mod8_all[i]["FlipType"] == 4:
                    x, y = -x, -y
                else:
                    raise ValueError("Flip type is not valid")
                Mod8_avg_res_X[mov].append(x)
                Mod8_avg_res_Y[mov].append(y)
        if Mod9_all[i]["Mov"] == mov:
            angle = np.deg2rad(Mod9_all[i]["Res_a"])
            x, y = polar2cartesian(angle, Mod9_all[i]["Res_v"])
            # rotation according to the flip type
            if Mod9_all[i]["FlipType"] == 1:
                # rotation according to the angle
                Mod9_avg_res_X[mov].append(x)
                Mod9_avg_res_Y[mov].append(y)
            else:
                if Mod9_all[i]["FlipType"] == 2:
                    x = -x
                elif Mod9_all[i]["FlipType"] == 3:
                    y = -y
                elif Mod9_all[i]["FlipType"] == 4:
                    x, y = -x, -y
                else:
                    raise ValueError("Flip type is not valid")
                Mod9_avg_res_X[mov].append(x)
                Mod9_avg_res_Y[mov].append(y)
        if Mod10_all[i]["Mov"] == mov:
            angle = np.deg2rad(Mod10_all[i]["Res_a"])
            x, y = polar2cartesian(angle, Mod10_all[i]["Res_v"])
            # rotation according to the flip type
            if Mod10_all[i]["FlipType"] == 1:
                # rotation according to the angle
                Mod10_avg_res_X[mov].append(x)
                Mod10_avg_res_Y[mov].append(y)
            else:
                if Mod10_all[i]["FlipType"] == 2:
                    x = -x
                elif Mod10_all[i]["FlipType"] == 3:
                    y = -y
                elif Mod10_all[i]["FlipType"] == 4:
                    x, y = -x, -y
                else:
                    raise ValueError("Flip type is not valid")
                Mod10_avg_res_X[mov].append(x)
                Mod10_avg_res_Y[mov].append(y)

for mov in range(max_mov):
    assert len(Mod1_avg_res_X[mov]) == len(Mod1_avg_res_Y[mov]) == len(Mod2_avg_res_X[mov]) == len(Mod2_avg_res_Y[mov]) \
           == len(Mod3_avg_res_X[mov]) == len(Mod3_avg_res_Y[mov]) == len(Mod4_avg_res_X[mov]) == len(
        Mod4_avg_res_Y[mov]) \
           == len(Mod5_avg_res_X[mov]) == len(Mod5_avg_res_Y[mov]) == len(Mod6_avg_res_X[mov]) == len(
        Mod6_avg_res_Y[mov]) \
           == len(Mod7_avg_res_X[mov]) == len(Mod7_avg_res_Y[mov]) == len(Mod8_avg_res_X[mov]) == len(
        Mod8_avg_res_Y[mov]) \
           == len(Mod9_avg_res_X[mov]) == len(Mod9_avg_res_Y[mov]) == len(Mod10_avg_res_X[mov]) == len(
        Mod10_avg_res_Y[mov]) == 4, \
        "The number of trials is not the same for all models"

Avg_gt_X, Avg_gt_Y = np.mean(np.array(Avg_gt_X), axis=1), np.mean(np.array(Avg_gt_Y), axis=1)
Mod1_avg_res_X, Mod1_avg_res_Y = np.mean(np.array(Mod1_avg_res_X), axis=1), np.mean(np.array(Mod1_avg_res_Y), axis=1)
Mod2_avg_res_X, Mod2_avg_res_Y = np.mean(np.array(Mod2_avg_res_X), axis=1), np.mean(np.array(Mod2_avg_res_Y), axis=1)
Mod3_avg_res_X, Mod3_avg_res_Y = np.mean(np.array(Mod3_avg_res_X), axis=1), np.mean(np.array(Mod3_avg_res_Y), axis=1)
Mod4_avg_res_X, Mod4_avg_res_Y = np.mean(np.array(Mod4_avg_res_X), axis=1), np.mean(np.array(Mod4_avg_res_Y), axis=1)
Mod5_avg_res_X, Mod5_avg_res_Y = np.mean(np.array(Mod5_avg_res_X), axis=1), np.mean(np.array(Mod5_avg_res_Y), axis=1)
Mod6_avg_res_X, Mod6_avg_res_Y = np.mean(np.array(Mod6_avg_res_X), axis=1), np.mean(np.array(Mod6_avg_res_Y), axis=1)
Mod7_avg_res_X, Mod7_avg_res_Y = np.mean(np.array(Mod7_avg_res_X), axis=1), np.mean(np.array(Mod7_avg_res_Y), axis=1)
Mod8_avg_res_X, Mod8_avg_res_Y = np.mean(np.array(Mod8_avg_res_X), axis=1), np.mean(np.array(Mod8_avg_res_Y), axis=1)
Mod9_avg_res_X, Mod9_avg_res_Y = np.mean(np.array(Mod9_avg_res_X), axis=1), np.mean(np.array(Mod9_avg_res_Y), axis=1)
Mod10_avg_res_X, Mod10_avg_res_Y = np.mean(np.array(Mod10_avg_res_X), axis=1), np.mean(np.array(Mod10_avg_res_Y),
                                                                                       axis=1)
avg_gt_xy = np.concatenate((Avg_gt_X, Avg_gt_Y))
avg_mod1_xy = np.concatenate((Mod1_avg_res_X, Mod1_avg_res_Y))
avg_mod2_xy = np.concatenate((Mod2_avg_res_X, Mod2_avg_res_Y))
avg_mod3_xy = np.concatenate((Mod3_avg_res_X, Mod3_avg_res_Y))
avg_mod4_xy = np.concatenate((Mod4_avg_res_X, Mod4_avg_res_Y))
avg_mod5_xy = np.concatenate((Mod5_avg_res_X, Mod5_avg_res_Y))
avg_mod6_xy = np.concatenate((Mod6_avg_res_X, Mod6_avg_res_Y))
avg_mod7_xy = np.concatenate((Mod7_avg_res_X, Mod7_avg_res_Y))
avg_mod8_xy = np.concatenate((Mod8_avg_res_X, Mod8_avg_res_Y))
avg_mod9_xy = np.concatenate((Mod9_avg_res_X, Mod9_avg_res_Y))
avg_mod10_xy = np.concatenate((Mod10_avg_res_X, Mod10_avg_res_Y))

corr_uv_mod1 = np.corrcoef(avg_gt_xy, avg_mod1_xy)[0, 1]
corr_uv_mod2 = np.corrcoef(avg_gt_xy, avg_mod2_xy)[0, 1]
corr_uv_mod3 = np.corrcoef(avg_gt_xy, avg_mod3_xy)[0, 1]
corr_uv_mod4 = np.corrcoef(avg_gt_xy, avg_mod4_xy)[0, 1]
corr_uv_mod5 = np.corrcoef(avg_gt_xy, avg_mod5_xy)[0, 1]
corr_uv_mod6 = np.corrcoef(avg_gt_xy, avg_mod6_xy)[0, 1]
corr_uv_mod7 = np.corrcoef(avg_gt_xy, avg_mod7_xy)[0, 1]
corr_uv_mod8 = np.corrcoef(avg_gt_xy, avg_mod8_xy)[0, 1]
corr_uv_mod9 = np.corrcoef(avg_gt_xy, avg_mod9_xy)[0, 1]
corr_uv_mod10 = np.corrcoef(avg_gt_xy, avg_mod10_xy)[0, 1]

# cartesian to polar
Avg_gt_a, Avg_gt_v = cartesian2polar(Avg_gt_X, Avg_gt_Y)
Avg_gt_a = np.rad2deg(Avg_gt_a)
Mod1_avg_res_a, Mod1_avg_res_v = cartesian2polar(Mod1_avg_res_X, Mod1_avg_res_Y)
Mod2_avg_res_a, Mod2_avg_res_v = cartesian2polar(Mod2_avg_res_X, Mod2_avg_res_Y)
Mod3_avg_res_a, Mod3_avg_res_v = cartesian2polar(Mod3_avg_res_X, Mod3_avg_res_Y)
Mod4_avg_res_a, Mod4_avg_res_v = cartesian2polar(Mod4_avg_res_X, Mod4_avg_res_Y)
Mod5_avg_res_a, Mod5_avg_res_v = cartesian2polar(Mod5_avg_res_X, Mod5_avg_res_Y)
Mod6_avg_res_a, Mod6_avg_res_v = cartesian2polar(Mod6_avg_res_X, Mod6_avg_res_Y)
Mod7_avg_res_a, Mod7_avg_res_v = cartesian2polar(Mod7_avg_res_X, Mod7_avg_res_Y)
Mod8_avg_res_a, Mod8_avg_res_v = cartesian2polar(Mod8_avg_res_X, Mod8_avg_res_Y)
Mod9_avg_res_a, Mod9_avg_res_v = cartesian2polar(Mod9_avg_res_X, Mod9_avg_res_Y)
Mod10_avg_res_a, Mod10_avg_res_v = cartesian2polar(Mod10_avg_res_X, Mod10_avg_res_Y)
Mod1_avg_res_a, Mod2_avg_res_a, Mod3_avg_res_a, Mod4_avg_res_a, Mod5_avg_res_a, Mod6_avg_res_a, Mod7_avg_res_a, \
    Mod8_avg_res_a, Mod9_avg_res_a, Mod10_avg_res_a = np.rad2deg(Mod1_avg_res_a), np.rad2deg(Mod2_avg_res_a), \
    np.rad2deg(Mod3_avg_res_a), np.rad2deg(Mod4_avg_res_a), np.rad2deg(Mod5_avg_res_a), np.rad2deg(Mod6_avg_res_a), \
    np.rad2deg(Mod7_avg_res_a), np.rad2deg(Mod8_avg_res_a), np.rad2deg(Mod9_avg_res_a), np.rad2deg(Mod10_avg_res_a)

angle_diff = np.abs(Mod1_avg_res_a - Avg_gt_a)
Mod1_avg_res_a_ = Mod1_avg_res_a + 360 * ((angle_diff > 180) & (Mod1_avg_res_a < 180)) - 360 * ((angle_diff > 180) & (
        Mod1_avg_res_a > 180))  # TODO: check if this is correct
angle_diff = np.abs(Mod2_avg_res_a - Avg_gt_a)
Mod2_avg_res_a_ = Mod2_avg_res_a + 360 * ((angle_diff > 180) & (Mod2_avg_res_a < 180)) - 360 * ((angle_diff > 180) & (
        Mod2_avg_res_a > 180))  # TODO: check if this is correct
angle_diff = np.abs(Mod3_avg_res_a - Avg_gt_a)
Mod3_avg_res_a_ = Mod3_avg_res_a + 360 * ((angle_diff > 180) & (Mod3_avg_res_a < 180)) - 360 * ((angle_diff > 180) & (
        Mod3_avg_res_a > 180))  # TODO: check if this is correct
angle_diff = np.abs(Mod4_avg_res_a - Avg_gt_a)
Mod4_avg_res_a_ = Mod4_avg_res_a + 360 * ((angle_diff > 180) & (Mod4_avg_res_a < 180)) - 360 * ((angle_diff > 180) & (
        Mod4_avg_res_a > 180))  # TODO: check if this is correct
angle_diff = np.abs(Mod5_avg_res_a - Avg_gt_a)
Mod5_avg_res_a_ = Mod5_avg_res_a + 360 * ((angle_diff > 180) & (Mod5_avg_res_a < 180)) - 360 * ((angle_diff > 180) & (
        Mod5_avg_res_a > 180))  # TODO: check if this is correct
angle_diff = np.abs(Mod6_avg_res_a - Avg_gt_a)
Mod6_avg_res_a_ = Mod6_avg_res_a + 360 * ((angle_diff > 180) & (Mod6_avg_res_a < 180)) - 360 * ((angle_diff > 180) & (
        Mod6_avg_res_a > 180))  # TODO: check if this is correct
angle_diff = np.abs(Mod7_avg_res_a - Avg_gt_a)
Mod7_avg_res_a_ = Mod7_avg_res_a + 360 * ((angle_diff > 180) & (Mod7_avg_res_a < 180)) - 360 * ((angle_diff > 180) & (
        Mod7_avg_res_a > 180))  # TODO: check if this is correct
angle_diff = np.abs(Mod8_avg_res_a - Avg_gt_a)
Mod8_avg_res_a_ = Mod8_avg_res_a + 360 * ((angle_diff > 180) & (Mod8_avg_res_a < 180)) - 360 * ((angle_diff > 180) & (
        Mod8_avg_res_a > 180))  # TODO: check if this is correct
angle_diff = np.abs(Mod9_avg_res_a - Avg_gt_a)
Mod9_avg_res_a_ = Mod9_avg_res_a + 360 * ((angle_diff > 180) & (Mod9_avg_res_a < 180)) - 360 * ((angle_diff > 180) & (
        Mod9_avg_res_a > 180))  # TODO: check if this is correct
angle_diff = np.abs(Mod10_avg_res_a - Avg_gt_a)
Mod10_avg_res_a_ = Mod10_avg_res_a + 360 * ((angle_diff > 180) & (Mod10_avg_res_a < 180)) - 360 * (
        (angle_diff > 180) & (Mod10_avg_res_a > 180))  # TODO: check if this is correct

import scipy.stats as stats
# calculate the pearson correlation coefficient
corr_A_1 = stats.pearsonr(Avg_gt_a, Mod1_avg_res_a_)[0]
corr_A_2 = stats.pearsonr(Avg_gt_a, Mod2_avg_res_a_)[0]
corr_A_3 = stats.pearsonr(Avg_gt_a, Mod3_avg_res_a_)[0]
corr_A_4 = stats.pearsonr(Avg_gt_a, Mod4_avg_res_a_)[0]
corr_A_5 = stats.pearsonr(Avg_gt_a, Mod5_avg_res_a_)[0]
corr_A_6 = stats.pearsonr(Avg_gt_a, Mod6_avg_res_a_)[0]
corr_A_7 = stats.pearsonr(Avg_gt_a, Mod7_avg_res_a_)[0]
corr_A_8 = stats.pearsonr(Avg_gt_a, Mod8_avg_res_a_)[0]
corr_A_9 = stats.pearsonr(Avg_gt_a, Mod9_avg_res_a_)[0]
corr_A_10 = stats.pearsonr(Avg_gt_a, Mod10_avg_res_a_)[0]

# calculate the pearson correlation coefficient
corr_V_1 = stats.pearsonr(Avg_gt_v, Mod1_avg_res_v)[0]
corr_V_2 = stats.pearsonr(Avg_gt_v, Mod2_avg_res_v)[0]
corr_V_3 = stats.pearsonr(Avg_gt_v, Mod3_avg_res_v)[0]
corr_V_4 = stats.pearsonr(Avg_gt_v, Mod4_avg_res_v)[0]
corr_V_5 = stats.pearsonr(Avg_gt_v, Mod5_avg_res_v)[0]
corr_V_6 = stats.pearsonr(Avg_gt_v, Mod6_avg_res_v)[0]
corr_V_7 = stats.pearsonr(Avg_gt_v, Mod7_avg_res_v)[0]
corr_V_8 = stats.pearsonr(Avg_gt_v, Mod8_avg_res_v)[0]
corr_V_9 = stats.pearsonr(Avg_gt_v, Mod9_avg_res_v)[0]
corr_V_10 = stats.pearsonr(Avg_gt_v, Mod10_avg_res_v)[0]

# draw the polar plot

plt.figure(figsize=(20, 10))
plt.subplot(2, 5, 1, projection='polar')
plt.scatter(Avg_gt_a, Avg_gt_v, c='r', marker='o', alpha=0.5, label="GT")
plt.scatter(Mod1_avg_res_a, Mod1_avg_res_v, c='b', marker='o', alpha=0.5, label="Res")
# set the range of the figure
plt.ylim(0, 15)
# show the correlation and corr_uv_ coefficient bold in the figure
plt.text(20, 20, "Corr_Angle: " + str(round(corr_A_1, 3)) + "\nCorr_Velocity: " + str(round(corr_V_1, 3)) +
         "\nCorr_UV: " + str(round(corr_uv_mod1, 3)),
         fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
plt.legend()
plt.title("Modulation 1")
plt.legend()

plt.subplot(2, 5, 2, projection='polar')
plt.scatter(Avg_gt_a, Avg_gt_v, c='r', marker='o', alpha=0.5, label="GT")
plt.scatter(Mod2_avg_res_a, Mod2_avg_res_v, c='b', marker='o', alpha=0.5, label="Res")
# set the range of the figure
plt.ylim(0, 15)
# show the correlation and corr_uv_ coefficient bold in the figure
plt.text(20, 20, "Corr_Angle: " + str(round(corr_A_2, 3)) + "\nCorr_Velocity: " + str(round(corr_V_2, 3)) +
         "\nCorr_UV: " + str(round(corr_uv_mod2, 3)),
         fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
plt.legend()
plt.title("Modulation 2")
plt.legend()

plt.subplot(2, 5, 3, projection='polar')
plt.scatter(Avg_gt_a, Avg_gt_v, c='r', marker='o', alpha=0.5, label="GT")
plt.scatter(Mod3_avg_res_a, Mod3_avg_res_v, c='b', marker='o', alpha=0.5, label="Res")
# set the range of the figure
plt.ylim(0, 15)
# show the correlation and corr_uv_ coefficient bold in the figure
plt.text(20, 20, "Corr_Angle: " + str(round(corr_A_3, 3)) + "\nCorr_Velocity: " + str(round(corr_V_3, 3)) +
         "\nCorr_UV: " + str(round(corr_uv_mod3, 3)),
         fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
plt.legend()
plt.title("Modulation 3")

plt.subplot(2, 5, 4, projection='polar')
plt.scatter(Avg_gt_a, Avg_gt_v, c='r', marker='o', alpha=0.5, label="GT")
plt.scatter(Mod4_avg_res_a, Mod4_avg_res_v, c='b', marker='o', alpha=0.5, label="Res")
# set the range of the figure
plt.ylim(0, 15)
# show the correlation and corr_uv_ coefficient bold in the figure
plt.text(20, 20, "Corr_Angle: " + str(round(corr_A_4, 3)) + "\nCorr_Velocity: " + str(round(corr_V_4, 3)) +
         "\nCorr_UV: " + str(round(corr_uv_mod4, 3)),
         fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
plt.legend()
plt.title("Modulation 4")

plt.subplot(2, 5, 5, projection='polar')
plt.scatter(Avg_gt_a, Avg_gt_v, c='r', marker='o', alpha=0.5, label="GT")
plt.scatter(Mod5_avg_res_a, Mod5_avg_res_v, c='b', marker='o', alpha=0.5, label="Res")
# set the range of the figure
plt.ylim(0, 15)
# show the correlation and corr_uv_ coefficient bold in the figure
plt.text(20, 20, "Corr_Angle: " + str(round(corr_A_5, 3)) + "\nCorr_Velocity: " + str(round(corr_V_5, 3)) +

         "\nCorr_UV: " + str(round(corr_uv_mod5, 3)),
         fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
plt.legend()
plt.title("Modulation 5")

plt.subplot(2, 5, 6, projection='polar')
plt.scatter(Avg_gt_a, Avg_gt_v, c='r', marker='o', alpha=0.5, label="GT")
plt.scatter(Mod6_avg_res_a, Mod6_avg_res_v, c='b', marker='o', alpha=0.5, label="Res")
# set the range of the figure
plt.ylim(0, 15)
# show the correlation and corr_uv_ coefficient bold in the figure
plt.text(20, 20, "Corr_Angle: " + str(round(corr_A_6, 3)) + "\nCorr_Velocity: " + str(round(corr_V_6, 3)) +
         "\nCorr_UV: " + str(round(corr_uv_mod6, 3)),
         fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
plt.legend()
plt.title("Modulation 6")

plt.subplot(2, 5, 7, projection='polar')
plt.scatter(Avg_gt_a, Avg_gt_v, c='r', marker='o', alpha=0.5, label="GT")
plt.scatter(Mod7_avg_res_a, Mod7_avg_res_v, c='b', marker='o', alpha=0.5, label="Res")
# set the range of the figure
plt.ylim(0, 15)
# show the correlation and corr_uv_ coefficient bold in the figure
plt.text(20, 20, "Corr_Angle: " + str(round(corr_A_7, 3)) + "\nCorr_Velocity: " + str(round(corr_V_7, 3)) +
         "\nCorr_UV: " + str(round(corr_uv_mod7, 3)),
         fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
plt.legend()
plt.title("Modulation 7")

plt.subplot(2, 5, 8, projection='polar')
plt.scatter(Avg_gt_a, Avg_gt_v, c='r', marker='o', alpha=0.5, label="GT")
plt.scatter(Mod8_avg_res_a, Mod8_avg_res_v, c='b', marker='o', alpha=0.5, label="Res")
# set the range of the figure
plt.ylim(0, 15)
# show the correlation and corr_uv_ coefficient bold in the figure
plt.text(20, 20, "Corr_Angle: " + str(round(corr_A_8, 3)) + "\nCorr_Velocity: " + str(round(corr_V_8, 3)) +
         "\nCorr_UV: " + str(round(corr_uv_mod8, 3)),
         fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
plt.legend()
plt.title("Modulation 8")

plt.subplot(2, 5, 9, projection='polar')
plt.scatter(Avg_gt_a, Avg_gt_v, c='r', marker='o', alpha=0.5, label="GT")
plt.scatter(Mod9_avg_res_a, Mod9_avg_res_v, c='b', marker='o', alpha=0.5, label="Res")
# set the range of the figure
plt.ylim(0, 15)
# show the correlation and corr_uv_ coefficient bold in the figure
plt.text(20, 20, "Corr_Angle: " + str(round(corr_A_9, 3)) + "\nCorr_Velocity: " + str(round(corr_V_9, 3)) +
         "\nCorr_UV: " + str(round(corr_uv_mod9, 3)),
         fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
plt.legend()
plt.title("Modulation 9")

plt.subplot(2, 5, 10, projection='polar')
plt.scatter(Avg_gt_a, Avg_gt_v, c='r', marker='o', alpha=0.5, label="GT")
plt.scatter(Mod10_avg_res_a, Mod10_avg_res_v, c='b', marker='o', alpha=0.5, label="Res")
# set the range of the figure
plt.ylim(0, 15)
# show the correlation and corr_uv_ coefficient bold in the figure
plt.text(20, 20, "Corr_Angle: " + str(round(corr_A_10, 3)) + "\nCorr_Velocity: " + str(round(corr_V_10, 3)) +
         "\nCorr_UV: " + str(round(corr_uv_mod10, 3)),
         fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
plt.legend()
plt.title("Modulation 10")

plt.show()
