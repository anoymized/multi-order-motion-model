import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
from utils import warp_utils
import torch.nn as nn
import torch
import imageio

def plot_quiver(flow, spacing=60):
    """Plots less dense quiver field.

    Args:
        ax: Matplotlib axis
        flow: motion vectors
        spacing: space (px) between each arrow in grid
        margin: width (px) of enclosing region without arrows
        kwargs: quiver kwargs (default: angles="xy", scale_units="xy")
    """
    h, w, *_ = flow.shape

    nx = int((w - 2 * 0) / spacing)
    ny = int((h - 2 * 0) / spacing)
    x = np.linspace(0, w - 0 - 1, nx, dtype=np.int64)
    y = np.linspace(0, h - 0 - 1, ny, dtype=np.int64)

    flow = flow[np.ix_(y, x)]
    u = flow[:, :, 0]
    v = flow[:, :, 1]   # ----------

    kwargs = {**dict(angles="xy", scale_units="xy")}
    # ax.quiver(x, y, u, v, color="black", scale=10, width=0.010, headwidth=5, minlength=0.5)  # bigger is short
    plt.quiver(x, y, u, v, color="black")  # bigger is short

    plt.show()


def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid


def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2


def boundary_dilated_flow_warp(x, flo, start_point=None):
    """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        start_point: [B,2,1,1]
        Note: sample down the image will create huge error, please use on original scale
        """
    _, _, Hx, Wx = x.size()
    B, C, H, W = flo.size()
    # mesh grid
    if start_point is None:
        start_point = torch.zeros(B, 2, 1, 1).type_as(x)
    start_points = torch.zeros(B, 2, 1, 1).type_as(x)
    start_points[:, 0, 0, 0] = start_point[:, 1, 0, 0]
    start_points[:, 1, 0, 0] = start_point[:, 0, 0, 0]
    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW
    v_grid = base_grid + flo + start_points

    v_grid[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / max(Wx - 1, 1) - 1.0
    v_grid[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / max(Hx - 1, 1) - 1.0

    v_grid = v_grid.permute(0, 2, 3, 1)  # B H,W,C
    # BHW2
    output = nn.functional.grid_sample(x, v_grid, mode="bilinear", padding_mode='reflection', align_corners=True)
    x_t = x.clone()
    if start_point[:, 0, 0, 0].long() > Hx or start_point[:, 1, 0, 0].long() > Wx:
        return x_t
    if (start_point[:, 0, 0, 0].long() + H > Hx) and (start_point[:, 1, 0, 0].long() + W > Wx):
        x_t[:, :, start_point[:, 0, 0, 0].long():, start_point[:, 1, 0, 0].long():] = output[:, :,
                                                                                      :Hx - start_point[:, 0, 0,
                                                                                            0].long(),
                                                                                      :Wx - start_point[:, 1, 0,
                                                                                            0].long()]
    elif start_point[:, 1, 0, 0].long() + W > Wx:
        x_t[:, :, start_point[:, 0, 0, 0].long(): start_point[:, 0, 0, 0].long() + H,
        start_point[:, 1, 0, 0].long():] = output[:, :, :, :Wx - start_point[:, 1, 0, 0].long()]
    elif start_point[:, 0, 0, 0].long() + H > Hx:
        x_t[:, :, start_point[:, 0, 0, 0].long():,
        start_point[:, 1, 0, 0].long(): start_point[:, 1, 0, 0].long() + W] = output[:, :,
                                                                              :Hx - start_point[:, 0, 0, 0].long(), :]
    else:
        x_t[:, :, start_point[:, 0, 0, 0].long(): start_point[:, 0, 0, 0].long() + H,
        start_point[:, 1, 0, 0].long(): start_point[:, 1, 0, 0].long() + W] = output
    # here align corners must be true

    return x_t


Param = {}
Param['H'] = 200  # pixel
Param['W'] = 200  # pixel
Param['SF'] = 6  # spatial frequency
Param['SExp'] = 0
Param['TF'] = 8  # temporal frequency
Param['TExp'] = 1
Param['SR'] = 60  # Hz
Param['Length'] = 1  # sec
Param['Num'] = 50
Param['MaxGrad'] = 50  # max pixel
t = np.linspace(0, 2 * np.pi * Param['Length'], Param['Length'] * Param['SR'])
Rx = np.linspace(-np.pi, np.pi, Param['W'])
Ry = np.linspace(-np.pi, np.pi, Param['H'])
rxx, ryy = np.meshgrid(Rx, Ry)
Rrr = np.sqrt(rxx ** 2 + ryy ** 2)
PosRandX = np.random.rand(Param['Num']) * (2 * np.pi) - np.pi
PosRandY = np.random.rand(Param['Num']) * (2 * np.pi) - np.pi
xyrand = np.random.rand(Param['Num']) * (2 * np.pi)
tRand = np.random.rand(Param['Num']) * np.pi + 0.5
TOri = np.tile(t, (Param['Num'], 1)) * tRand[:, np.newaxis]
kernal = np.zeros((Param['H'], Param['W'], len(t), len(t) * Param['Num'] + Param['Num']))
for k in range(len(t)):
    if np.random.binomial(1, 0.8):
        RandNum = np.random.randint(1, Param['Num'] + 1)
        for j in range(RandNum):
            PosRandX = np.append(PosRandX, np.random.rand() * 2 * np.pi - np.pi)
            PosRandY = np.append(PosRandY, np.random.rand() * 2 * np.pi - np.pi)
            xyrand = np.append(xyrand, np.random.rand() * 2 * np.pi - np.pi)
            tRand = np.append(tRand, np.random.rand() * np.pi + 0.5)
            temp = (t - t[k]) * tRand[-1]
            TOri = np.vstack([TOri, temp])
    for j in range(len(PosRandX)):
        x = xyrand[j] * np.linspace(-np.pi, np.pi, Param['W']) + PosRandX[j]
        y = xyrand[j] * np.linspace(-np.pi, np.pi, Param['H']) + PosRandY[j]
        xx, yy = np.meshgrid(x, y)
        rr = np.sqrt(xx ** 2 + yy ** 2)
        kernal[:, :, k, j] = np.cos(Param['SF'] * rr) * np.exp(-Param['SExp'] * rr ** 2) * np.cos(
            Param['TF'] * TOri[j, k]) * np.exp(-Param['TExp'] * TOri[j, k] ** 2)
        kernal[:, :, k, j] = - kernal[:, :, k, j] * np.exp(-Rrr ** 2 / (0.25 * np.pi ** 2))  # Global Constraint

kernal = np.sum(kernal, axis=3)


def update(frame):
    # Clear the current plot
    ax.clear()
    # Plot the current frame
    gx, gy = np.gradient(kernal[:, :, frame])
    v = gx ** 2 + gy ** 2
    ax.imshow(v)
    # Set the title and colorbar label
    ax.set_title('Frame {}'.format(frame))


# # Create the animation
# fig, ax = plt.subplots()
# anim = FuncAnimation(fig, update, frames=range(Param['SR'] * Param['Length']))
# # Show the animation
# plt.show()
#
# kernal_grad_x, kernal_grad_y, kernal_grad_t = np.gradient(kernal)
# norm_x = np.max(np.abs(kernal_grad_x))
# norm_y = np.max(np.abs(kernal_grad_y))
# kernal_grad_x = (kernal_grad_x / norm_x * Param['MaxGrad'])
# kernal_grad_y = (kernal_grad_x / norm_x * Param['MaxGrad'])
#


# input image (HxWx3)
img = torch.rand(1, 3, 480, 840)
img = torch.from_numpy(np.array(Image.open('testimg/00014.jpg').convert('RGB'))).permute(2, 0, 1).unsqueeze(
    0).float() / 255
img_copy = img.clone()
# flow field (HxWx2)
start_point = torch.zeros(1, 2, 1, 1).type_as(img)
image_list = []
for i in range(60):
    # plot the kernal
    # plt.imshow(kernal[:, :, i])
    # plt.show()
    gx = np.gradient(kernal[:, :, i], axis=1)
    gy = np.gradient(kernal[:, :, i], axis=0)
    # xpand last dimension
    gx_show = gx[:, :, np.newaxis]
    gy_show = gy[:, :, np.newaxis]
    # plot arrow of 2d gradient gx,gy
    # sample the local of the gradient
    # plot_quiver(np.concatenate([gx_show, gy_show], axis=-1),spacing=10)

    grad_x = torch.from_numpy(gx).unsqueeze(0).float()
    grad_y = torch.from_numpy(gy).unsqueeze(0).float()
    flow = torch.cat([grad_x, grad_y], dim=0).unsqueeze(0)
    # generate random flow field follow beta distribution
    # flow = np.random.beta(0.5, 0.5, [1, 2, 480, 840]) * 100
    # flow = torch.from_numpy(flow).float()

    flow = flow / flow.max() * 100
    start_point[:, 0, 0, 0] = 150
    start_point[:, 1, 0, 0] += 0

    img_copy = warp_utils.boundary_dilated_flow_warp(img, flow, start_point)
    image_list.append(img_copy)


def update2(frame):
    # Clear the current plot
    ax.clear()
    # Plot the current frame
    ax.imshow(image_list[frame].numpy().squeeze(0).transpose(1, 2, 0))
    # Set the title and colorbar label
    ax.set_title('Frame {}'.format(frame))


# Create the animation
fig, ax = plt.subplots()
anim = FuncAnimation(fig, update2, frames=60, repeat_delay=10)
anim.save('test.gif', writer='imagemagick', fps=20)
# Show the animation
plt.show()
anim.save('test.gif', writer='imagemagick', fps=20)

# save image list frame_*。png
for i in range(len(image_list)):
    imageio.imwrite('frame_%d.png' % i, image_list[i].numpy().squeeze(0).transpose(1, 2, 0))

