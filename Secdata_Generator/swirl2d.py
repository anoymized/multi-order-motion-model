import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 设置网格大小和漩涡流场参数
size = 60
x = np.linspace(-10, 10, size) - 5
y = np.linspace(-10, 10, size)
X, Y = np.meshgrid(x, y)


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
    v = flow[:, :, 1]  # ----------

    kwargs = {**dict(angles="xy", scale_units="xy")}
    # ax.quiver(x, y, u, v, color="black", scale=10, width=0.010, headwidth=5, minlength=0.5)  # bigger is short
    plt.quiver(x, y, u, v, color="black")  # bigger is short

    # plt.show()


# 定义漩涡流场的参数
def vortex_flow(t, strength, frequency):
    radius = 1
    omega = 2 * np.pi * frequency

    # rotate x and y
    X_rot = X * np.cos(omega * t) - Y * np.sin(omega * t)
    Y_rot = X * np.sin(omega * t) + Y * np.cos(omega * t)

    # u = strength / (2 * np.pi) * (Y / (X ** 2 + Y ** 2 + radius ** 2))
    # v = -strength / (2 * np.pi) * (X / (X ** 2 + Y ** 2 + radius ** 2))
    u = strength / (2 * np.pi) * (Y_rot / (X_rot ** 2 + Y_rot ** 2 + radius ** 2))
    v = -strength / (2 * np.pi) * (X_rot / (X_rot ** 2 + Y_rot ** 2 + radius ** 2))
    return u, v


def poisson_reconstruct(dx, dy, num_iters=500):
    # 初始化重建图像为0
    height, width = dx.shape
    reconstruction = np.zeros((height, width))
    # 迭代更新重建图像
    for _ in range(num_iters):
        # 计算梯度
        grad_x = np.roll(reconstruction, -1, axis=1) - reconstruction
        grad_y = np.roll(reconstruction, -1, axis=0) - reconstruction
        # 更新重建图像
        reconstruction = (dx + grad_x + dy + grad_y) / 4
    return reconstruction


# 创建画布和子图
for t in range(40):
    u, v = vortex_flow(t, strength=10, frequency=0.1)  # 调整漩涡流场参数
    flow = np.stack([u, v], axis=2)
    plot_quiver(flow, spacing=2)
    # plt video
    plt.savefig('swirl2d' + str(t) + '.png')
    plt.close()

# save gif
images = []
import imageio
for t in range(40):
    filename = 'swirl2d' + str(t) + '.png'
    images.append(imageio.imread(filename))
imageio.mimsave('swirl2d.gif', images, fps=15)




    # 根据导数求积分
    # ke = poisson_reconstruct(u, v)
    # # compute the gradient
    #
    # ke_x, ke_y = np.gradient(ke)
    # # plot the gradient
    # flow = np.stack([ke_x, ke_y], axis=2)
    # plot_quiver(flow, spacing=2)
    #
    # plt.imshow(ke, cmap='gray')
    # plt.show()

# 显示动画
