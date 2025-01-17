import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from copy import deepcopy
from matplotlib import cm as cmx
import seaborn as sns
from PIL import Image
from os.path import *
import torchvision.transforms.functional as fn
import os
from matplotlib.backends.backend_pdf import PdfPages


def plot_polar_one_cell(x, index, layer, if_show=True):
    fold_name = "layer%2d_activation" % layer
    if not os.path.exists(fold_name):
        os.makedirs(fold_name)
    rainbow = plt.get_cmap('rainbow')
    cNorm = colors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=rainbow)
    gamma_factor = 0.8
    plt.figure(figsize=(10, 10), clear=True)
    ax = plt.subplot(211, polar=True)
    max_speed = max([dist["tf"] for dist in x]) / min([dist["sf"] for dist in x])
    corrmax = max([np.power((dist["tf"] / dist["sf"]) / max_speed, gamma_factor) for dist in x])
    # norm = max([max(x["act_dist"]) for x in x])

    for dist in x:
        theta = dist["theta"]
        sf = dist["sf"]
        tf = dist["tf"]

        color = np.power((tf / sf) / max_speed, gamma_factor) / corrmax

        # act = [x / norm for x in dist["act_dist"]]
        act = [x for x in dist["act_dist"]]
        ax.plot(theta, act, linewidth=1, label='sf:%.2e_tf:%.2f' % (sf, tf), marker='o', color=scalarMap.to_rgba(color))
        ax.set_rmax(1)
        ax.set_rmin(0)

    ax.legend(loc=2, bbox_to_anchor=(1.08, 1.0), borderaxespad=0.)
    ax.grid(True)
    ax.set_title("cell_%04d" % index, va='bottom')

    cmap = plt.cm.get_cmap("rainbow")
    c = cmap(np.arange(cmap.N))
    ax = plt.subplot(212)
    ax.imshow([c], extent=[0, 1, 0, 0.05])
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.set_title("Normalized speed color bar (After gamma correction)")
    plt.tight_layout()

    if if_show:
        plt.show()
    plt.savefig(os.path.join(fold_name, "cell_%04d.png" % index))


def plot_polar_global_pdf(cell_combination, layer):
    row = 16
    column = 24
    gamma_factor = 0.7
    div_factor = 4
    fold_name = "layer%2d_activation" % layer
    if not os.path.exists(fold_name):
        os.makedirs(fold_name)
    rainbow = plt.get_cmap('rainbow')
    cNorm = colors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=rainbow)
    max_speed = max([dist["tf"] for dist in cell_combination[0]]) / min([dist["sf"] for dist in cell_combination[0]])
    corrmax = max([np.power((dist["tf"] / dist["sf"]) / max_speed, gamma_factor) for dist in cell_combination[0]])
    step = len(cell_combination) // div_factor

    for index in range(div_factor):
        num = index * step
        fig = plt.figure(figsize=(15 * row // 2, 15 * column // 2), clear=True)
        for cell_index, cell in enumerate(cell_combination[num:num + step]):
            # plot_one_cell(cell, cell_index, layer, if_show=False)
            for dist in cell:
                ax = plt.subplot(column // 2, row // 2, cell_index + 1, polar=True)
                theta = dist["theta"]
                sf = dist["sf"]
                tf = dist["tf"]
                color = np.power((tf / sf) / max_speed, gamma_factor) / corrmax
                # act = [x / norm for x in dist["act_dist"]]
                act = [x for x in dist["act_dist"]]
                ax.plot(theta, act, linewidth=2, label='sf:%.2e_tf:%.2f' % (sf, tf), marker='o',
                        color=scalarMap.to_rgba(color))
                ax.set_rmax(1.0)
                ax.set_rmin(0)
                ax.set_title("cell_%04d" % (cell_index + num + 1), va='bottom', fontsize=100)
        plt.tight_layout()
        plt.savefig(os.path.join(fold_name, "subgroup_%d.pdf" % index), dpi=5000)
        plt.close('all')


def plot_one_cell_spectral_rf(x, index, layer, if_show=False):
    fold_name = "layer%2d_receptiveRF" % layer
    if not os.path.exists(fold_name):
        os.makedirs(fold_name)
    # plt.figure(figsize=(20, 10), clear=True)
    activation_accumulate = np.zeros(len(x[0]["act_dist"]))
    sf_list = []
    tf_list = []
    # norm = max([max(x["act_dist"]) for x in x])
    for dist in x:
        theta = dist["theta"]
        sf = dist["sf"]
        sf_list.append(sf)
        tf = dist["tf"]
        tf_list.append(tf)
        # act = [x / norm for x in dist["act_dist"]]
        act = [x for x in dist["act_dist"]]
        for idx in range(len(activation_accumulate)):
            activation_accumulate[idx] += act[idx]
    sf_list = list(set(sf_list))
    sf_list.sort()
    tf_list = list(set(tf_list))
    tf_list.sort()
    tf_list.reverse()
    act_mat = np.zeros([len(sf_list), len(tf_list)])
    mx_index = np.argmax(activation_accumulate)  # find the theta index of max activation
    for dist in x:
        sf = dist["sf"]
        index_sf = sf_list.index(sf)
        tf = dist["tf"]
        index_tf = tf_list.index(tf)
        act = dist["act_dist"][mx_index]
        act_mat[index_sf, index_tf] = act
    act_mat = act_mat.transpose([1, 0])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    sf_listshow = ["%8.3g" % val if hasattr(val, "__float__") else val for val in sf_list]
    tf_listshow = ["%8.3g" % val if hasattr(val, "__float__") else val for val in tf_list]
    plt.sca(ax1)
    sns.heatmap(act_mat, cmap="YlGnBu", xticklabels=sf_listshow, yticklabels=tf_listshow)
    ax1.set_title("RF_cell_%04d" % index, va='bottom')
    ax1.set_xlabel('Spatial frequency (per pixel)')
    ax1.set_ylabel('Temporal frequency (per frame)')

    X, Y = np.meshgrid(sf_list, tf_list)
    cp = ax2.contourf(X, Y, act_mat, linestyles="solid", linewidths=2)
    plt.colorbar(cp)
    ax2.set_title('Contour')
    ax2.set_xlabel('Spatial frequency (per pixel)')
    ax2.set_ylabel('Temporal frequency (per frame)')
    plt.tight_layout()
    if if_show:
        plt.show()
    plt.savefig(os.path.join(fold_name, "RF_cell_%04d.png" % index))


def plot_one_layer(activation, layer, norm):
    # input list->dic["layer_0"]...["layer_n"]
    # "cfg"->tf,sf,theta, flow_vecx, flow_vecy
    # ""
    sf_list = [value["cfg"]["sf"] for value in activation]
    tf_list = [value["cfg"]["tf"] for value in activation]
    theta_list = [value["cfg"]["theta"] for value in activation]
    sf_set = sorted(set(sf_list))
    tf_set = sorted(set(tf_list))
    theta_set = sorted(set(theta_list))

    combination = []
    for sf in sf_set:
        for tf in tf_set:
            combination.append({"act_dist": [], "theta": [], "sf": sf, "tf": tf})
    assert len(combination) == len(activation) / len(theta_set)
    cell_combination = [deepcopy(combination) for i in range(len(activation[0]["activation"]))]

    for dict in activation:
        act = dict["activation"]
        cfg = dict["cfg"]
        sf = cfg["sf"]
        tf = cfg["tf"]
        theta = cfg["theta"]
        index_com = [i for i in range(len(combination)) if combination[i]["sf"] == sf and combination[i]["tf"] == tf]
        assert len(index_com) == 1
        index_com = index_com[0]

        for index_cell, cell_act in enumerate(act):
            cell_combination[index_cell][index_com]["theta"].append(-theta * np.pi + np.pi)
            cell_combination[index_cell][index_com]["act_dist"].append(cell_act / norm)
            # normalization of all cells

    # for cell_index, cell in enumerate(cell_combination):
    #     plot_one_cell_spectral_rf(cell, cell_index, layer, if_show=False)

    for cell_index, cell in enumerate(cell_combination):
        plot_polar_one_cell(cell, cell_index, layer, if_show=False)
