import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.warp_utils import boundary_dilated_flow_warp
# from skimage.color import rgb2yuv
import cv2
from Data_Generator.basic_frame import draw_shape_on_mask
from fast_slic.avx2 import SlicAvx2 as Slic
import torch.nn.functional as F
from copy import deepcopy
from texturize import api, commands
import PIL


def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW
    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid


def denormalize_coords(flow):
    """ scale indices from [-1, 1] to [0, width/height] """
    _, c, h, w = flow.shape
    flow[:, 0, :, :] = 0.5 * (w - 1.0) * (flow[:, 0, :, :] + 1.0)
    flow[:, 1, :, :] = 0.5 * (h - 1.0) * (flow[:, 1, :, :] + 1.0)
    return flow


def local_process_image(image_batch, mask_batch, processor=None, min_size=128, div=32):
    indices = (mask_batch > 1e-6).nonzero()
    # get the location of the most left point
    most_left = indices[:, 2].min()
    most_right = indices[:, 2].max()
    most_top = indices[:, 1].min()
    most_bottom = indices[:, 1].max()
    # get the location of the top left corner
    location_topleft = [0, most_top, most_left]
    # get the location of the bottom right corner
    location_bottomright = [0, most_bottom, most_right]
    H = location_bottomright[1] - location_topleft[1]
    W = location_bottomright[2] - location_topleft[2]
    # get the square region that fully contains the mask
    if H > W:
        location_topleft[2] = max(0, location_topleft[2] - (H - W) // 2)
        location_bottomright[2] = min(image_batch.shape[2], location_bottomright[2] + (H - W) // 2)
    else:
        location_topleft[1] = max(0, location_topleft[1] - (W - H) // 2)
        location_bottomright[1] = min(image_batch.shape[1], location_bottomright[1] + (W - H) // 2)
    # avoid too small local region
    if H < min_size:
        location_topleft[1] = max(0, location_topleft[1] - (min_size - H) // 2)
        location_bottomright[1] = min(image_batch.shape[1], location_bottomright[1] + (min_size - H) // 2)
    if W < min_size:
        location_topleft[2] = max(0, location_topleft[2] - (min_size - W) // 2)
        location_bottomright[2] = min(image_batch.shape[2], location_bottomright[2] + (min_size - W) // 2)

    # adjust the image to be divisible by 64
    location_topleft[1] = (location_topleft[1] // div - 1) * div
    location_topleft[2] = (location_topleft[2] // div - 1) * div
    location_bottomright[1] = (location_bottomright[1] // div + 1) * div
    location_bottomright[2] = (location_bottomright[2] // div + 1) * div

    # avoid out of range
    location_topleft[1] = max(0, location_topleft[1])
    location_topleft[2] = max(0, location_topleft[2])
    location_bottomright[1] = min(image_batch.shape[1], location_bottomright[1])
    location_bottomright[2] = min(image_batch.shape[2], location_bottomright[2])

    # crop the image
    image_local = image_batch[:, location_topleft[1]:location_bottomright[1],
                  location_topleft[2]:location_bottomright[2]]
    # process the local image
    image_local_p = processor(deepcopy(image_local))
    # correct the mean luminance
    image_local_p = image_local_p * image_local.mean() / image_local_p.mean()
    # clamp the value
    image_local_p = torch.clamp(image_local_p, 0, 1)
    # paste the local image back
    image_batch[:, location_topleft[1]:location_bottomright[1],
    location_topleft[2]:location_bottomright[2]] = image_local_p
    return image_batch


def water_ball_infer(image_batch, mask_batch):
    indices = (mask_batch > 0.1).nonzero()
    if len(indices) == 0:
        print("no mask")
        return None
    most_left = indices[:, 2].min()
    most_right = indices[:, 2].max()
    most_top = indices[:, 1].min()
    most_bottom = indices[:, 1].max()
    # get the location of the top left corner
    location_topleft = [0, most_top, most_left]
    # get the location of the bottom right corner
    location_bottomright = [0, most_bottom, most_right]
    ###########################################
    # import matplotlib.pyplot as plt
    # print(location_topleft)
    # print(location_bottomright)
    # plt.imshow(mask_batch.squeeze().detach().cpu().numpy())
    # plt.show()
    ###########################################
    H = location_bottomright[1] - location_topleft[1]
    W = location_bottomright[2] - location_topleft[2]
    flow_max = max(H, W) * 1000

    start_point = torch.zeros(1, 2, 1, 1).type_as(image_batch).cuda()
    start_point[:, 0, 0, 0] = location_topleft[1]
    start_point[:, 1, 0, 0] = location_topleft[2]

    # blur the mask
    mask_blur = cv2.GaussianBlur(mask_batch.squeeze().detach().cpu().numpy(), (21, 21), 10)
    # get the gradient of the mask
    kr = mask_blur

    gx = np.gradient(kr, axis=1)
    gy = np.gradient(kr, axis=0)
    grad_x = torch.from_numpy(gx).unsqueeze(0).float().cuda()
    grad_y = torch.from_numpy(gy).unsqueeze(0).float().cuda()
    flow = torch.cat([grad_x, grad_y], dim=0).unsqueeze(0)
    flow = (flow / flow.max()) * flow_max
    # normalize grid to [-1, 1]
    # warp image using grid
    img_copy = boundary_dilated_flow_warp(image_batch.unsqueeze(0), flow.cuda(), start_point)
    return img_copy.squeeze()


def run_slic_seperate(img_batch, n_seg=200, compact=10, rd_select=(4, 8)):  # Nx1xHxW
    """

    :param img: Nx3xHxW 0~1 float32
    :param n_seg:
    :param compact:
    :return: Nx1xHxW float32
    """
    B = img_batch.size(0)
    dtype = img_batch.type()
    img_batch = np.split(
        img_batch.detach().cpu().numpy().transpose([0, 2, 3, 1]), B, axis=0)
    out_ins = []
    patch_instance = []
    n_seg = np.random.randint(int(n_seg * 0.5), int(n_seg * 1.5))
    fast_slic = Slic(num_components=n_seg, compactness=compact, min_size_factor=0.8)
    n_select = np.random.randint(rd_select[0], rd_select[1])
    for index, img in enumerate(img_batch):
        img = np.copy((img * 255).squeeze(0).astype(np.uint8), order='C')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        seg = fast_slic.iterate(img)

        select_list = np.random.choice(range(0, np.max(seg) + 1), n_select,
                                       replace=False)
        for seg_id in select_list:
            seg_msk = seg == seg_id
            patch_instance.append(seg_msk)
        out_ins.append(patch_instance)
        patch_instance = []
        length = len(out_ins[0])
    for idx, mask in enumerate(out_ins[0]):
        # find the center of the mask
        mask = np.array(mask).astype(np.float32)

        # indices = np.where(mask == 1)
        # if len(indices[0]) == 0 or len(indices[1]) == 0 or np.random.binomial(1, 0.5) == 0:
        #     continue

        if idx < (length // 2 + np.random.randn()):
            H, W = mask.shape
            mask_ = draw_shape_on_mask(H, W)
            mask_ = (mask_ > 0.5).astype(np.float32)
            out_ins[0][idx] = mask_
        else:
            if np.random.binomial(1, 0.5) == 1:
                indices = np.where(mask == 1)
                most_left = indices[0].min()
                most_right = indices[0].max()
                most_top = indices[1].min()
                most_bottom = indices[1].max()
                # get the location of the top left corner
                location_topleft = [most_top, most_left]
                # get the location of the bottom right corner
                location_bottomright = [most_bottom, most_right]
                # get the square region that fully contains the mask
                H = location_bottomright[0] - location_topleft[0]
                W = location_bottomright[1] - location_topleft[1]

                # factor = np.random.uniform(1.8, 2.2)
                factor = 2
                mid_point = [H // factor + location_topleft[0], W // factor + location_topleft[1]]

                mask_small = cv2.resize(mask.astype(np.float32),
                                        (int(mask.shape[1] // factor), int(mask[0].shape[0] // factor)))
                mask_small = (mask_small > 0.5).astype(np.float32)
                # find indices of the small mask
                indices = np.where(mask_small > 0.5)
                # back to the original size
                mid_point.reverse()
                indices = np.array(indices) + np.round((np.array(mid_point)[:, np.newaxis] / factor)).astype(np.int)

                # mask the indices of original mask to be 0
                mask[indices[0], indices[1]] = 0
                out_ins[0][idx] = mask
        # shuffle the mask

        # plt.imshow(mask)
        # plt.show()
    # np.random.shuffle(out_ins[0])
    return out_ins


def random_texture_processor(temple):
    size = temple.shape[1:]
    # translate to PIL image
    image = PIL.Image.fromarray((temple.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    remix = commands.Remix(image)
    for result in api.process_octaves(remix, size=size[::-1], octaves=4):
        pass
    # The output can be saved in any PIL-supported format.
    temple = result.images.squeeze()
    return temple


def random_noise_processor(temple):
    noise = torch.rand_like(temple)
    temple = temple * noise
    # gaussian blur
    temple = temple.permute(1, 2, 0).cpu().numpy()
    temple = cv2.GaussianBlur(temple, (11, 11), 5)
    temple = torch.from_numpy(temple).permute(2, 0, 1).float().cuda()

    noise = torch.rand_like(temple)
    temple = temple + noise
    # gaussian blur
    temple = temple.permute(1, 2, 0).cpu().numpy()
    temple = cv2.GaussianBlur(temple, (11, 11), 5)
    temple = torch.from_numpy(temple).permute(2, 0, 1).float().cuda()

    # clamp
    temple = temple.clamp(0, 1)

    return temple


def random_fourier_phase_processor(temple):
    temple = temple.permute(1, 2, 0).cpu().numpy()

    # fourier transform of the image
    f = np.fft.fftn(temple)
    # extract the phase and amplitude
    magnitude_spectrum = np.abs(f)
    phase_spectrum = np.angle(f)
    # generate random phase while keeping the mean and std
    phase_spectrum = np.random.normal(np.mean(phase_spectrum), np.std(phase_spectrum), phase_spectrum.shape)
    # draw amplitude spectrum
    # plt.imshow(magnitude_spectrum[:,:,2])
    # plt.show()
    # combine the phase and amplitude
    f = magnitude_spectrum * np.exp(1j * phase_spectrum)
    # inverse fourier transform
    temple = np.fft.ifftn(f)
    temple = np.abs(temple)
    temple = torch.from_numpy(temple).permute(2, 0, 1).cuda()
    return temple


class CreateNonFourierData(object):
    def __init__(self, mean=15, sigma_i=4.0, sigma_v=0.5, rotate_range=0, sigma_rotate_v=0.0, sigma_zoom_v=0.0,
                 zoom_ini=0.0, compact=55, n_seg=50, SeqLen=25):
        super(CreateNonFourierData, self).__init__()
        self.mean = mean
        self.sigma_v = sigma_v
        self.sigma_i = sigma_i
        self.rotate_range = rotate_range
        self.key = 0
        self.zoom_ini = zoom_ini
        self.sigma_rotate_v = sigma_rotate_v
        self.sigma_zoom_v = sigma_zoom_v
        self.blur = GaussianBlurConv().cuda()
        self.blur_c1 = GaussianBlurConv(channels=1).cuda()
        self.compact = compact
        self.n_seg = n_seg
        self.if_avoid_overlap = False
        self.occluded = False

    def occlusion_detection(self, images, image, x, y, r, r_count, x_c, y_c, zoom, zoom_count, index,
                            recurrence_index=0):
        recurrence_index = recurrence_index + 1
        if recurrence_index > 10:
            self.occluded = True
            return 0, 0, 0
        x_t = x + x_c
        y_t = y + y_c
        angle = r + r_count
        zoom = zoom_count * zoom
        images_det = deepcopy(images)
        boundary_mat = torch.ones_like(images_det[index])
        boundary_mat[:, 5:-5, 5:-5] = 0
        w, h = image.shape[-1], image.shape[-2]
        x_t = x_t / w
        y_t = y_t / h
        angle = torch.tensor(-angle * torch.pi / 180, dtype=torch.float)
        theta = torch.tensor([
            [torch.cos(angle) * zoom, torch.sin(-angle) * zoom, float(x_t)],
            [torch.sin(angle) * zoom, torch.cos(angle) * zoom, float(y_t)]
        ], dtype=torch.float).type_as(image)
        image = image.unsqueeze(0)
        grid = F.affine_grid(theta.unsqueeze(0), size=image.shape, align_corners=True)
        image = F.grid_sample(image, grid, mode="bilinear", padding_mode='zeros', align_corners=True).squeeze()
        images_det[index] = image
        Flag2 = images_det[index] * boundary_mat
        for indx in range(len(images_det)):
            if indx == index:
                continue
            Flag1 = images_det[indx] * images_det[index]
            if (Flag1 != 0.0).any() or (Flag2 != 0.0).any():
                if r_count == x_c == y_c == 0:
                    pre_v = np.random.randn(1) * self.sigma_i + self.mean
                    pre_d = np.random.uniform(0, 360, 1)
                    x, y = self.pol2cart(pre_v, pre_d)
                    r = 0
                else:
                    x = np.random.randn(1) * self.sigma_v * 0.5 + x * np.random.uniform(-0.5, 0.5, 1)
                    y = np.random.randn(1) * self.sigma_v * 0.5 + y * np.random.uniform(-0.5, 0.5, 1)
                x, y, r = self.occlusion_detection(images, image, x, y, r, r_count, x_c, y_c, zoom, zoom_count, index,
                                                   recurrence_index)

        return x, y, r

    def draw_img(self, mask_pixel, image_batch, type=2):
        mask_batch = torch.ones_like(image_batch[:, 0, :, :]).unsqueeze(dim=1)
        image_batch_ori = deepcopy(image_batch)
        for index in range(image_batch.size(0)):
            for mask_idx, pixel in enumerate(mask_pixel[index]):
                mask = pixel[0].unsqueeze(dim=0)
                mask_batch[index][mask != 0] = 0
                if type == 0:  # first order
                    temple = pixel
                elif type == 1:  # pure random noise
                    temple = local_process_image(image_batch[index], mask, processor=random_noise_processor)

                elif type == 2:  # second order with blur
                    temple = self.to_0_1(image_batch[index])
                    for i in range(100):
                        temple = self.blur(temple)
                    temple = temple * (image_batch[index].mean() / temple.mean())  # correct the luminance
                elif type == 3:  # second order with water wave
                    temple = self.waterwave.infer(image_batch[index], mask, mask_idx)
                    if temple is None:
                        temple = image_batch[index]
                        self.occluded = True
                elif type == 4:  # second order random texture
                    temple = local_process_image(image_batch[index], mask, processor=random_texture_processor)
                elif type == 5:  # luminance change
                    temple = image_batch[index]
                    if np.random.rand() > 0.5:
                        temple = (temple * 1.5).clamp(0, 1)
                    else:
                        temple = (temple * 0.5).clamp(0, 1)
                elif type == 6:  # random fourier phase
                    temple = local_process_image(image_batch[index], mask, processor=random_fourier_phase_processor)
                # temple = image_batch[index] * 2

                image_batch[index][pixel != self.key] = temple[pixel != self.key]

                # image_batch[index] = temple
            mask_temp = 1 - mask_batch[index]
            for i in range(20):  # create gaussian blur
                mask_temp = self.to_0_1(self.blur_c1(mask_temp.unsqueeze(dim=0)).squeeze())
            # t the mask
            # plt.imshow(mask_temp.cpu().numpy())
            # plt.show()
            image_batch[index] = (image_batch[index] * mask_temp + image_batch_ori[index] * (1 - mask_temp)).clamp(0, 1)

        return image_batch, mask_batch

    def create_flow(self, image_batch, mask_pixel, flow_var):
        flow_bach = torch.zeros_like(image_batch[:, :2, :, :])
        for index in range(image_batch.size(0)):
            for pixel, flow in zip(mask_pixel[index], flow_var[index]):
                # image_batch[index][pixel != self.key] = pixel[pixel != self.key]  # de
                mask = pixel[0].unsqueeze(dim=0)
                flow_index = mask.expand(2, -1, -1)
                flow_bach[index][flow_index != self.key] = flow.squeeze()[flow_index != self.key]
        return flow_bach

    def pol2cart(self, rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)

    def affine_gird(self, image, angle, x, y, zoom, flow_last):
        w, h = image.shape[-1], image.shape[-2]
        x = x / w
        y = y / h
        angle = torch.tensor(-angle * torch.pi / 180, dtype=torch.float)
        theta = torch.tensor([
            [torch.cos(angle) * zoom, torch.sin(-angle) * zoom, float(x)],
            [torch.sin(angle) * zoom, torch.cos(angle) * zoom, float(y)]
        ], dtype=torch.float).type_as(image)

        image = image.unsqueeze(0)
        grid = F.affine_grid(theta.unsqueeze(0), size=image.shape, align_corners=True)
        image = F.grid_sample(image, grid, mode="bilinear", padding_mode='zeros', align_corners=True).squeeze()
        ini_gird = mesh_grid(1, h, w).type_as(image)
        grid = denormalize_coords(grid.permute(0, 3, 1, 2))
        if flow_last is None:
            flow_var = grid - ini_gird
        else:
            flow_var = (grid - ini_gird) - (flow_last - ini_gird)

        flow_last = deepcopy(grid)
        return image, flow_last, flow_var

    def to_0_1(self, x):
        return (x - x.min()) / (x.max() - x.min())

    def shift_ins_random(self, pixel_mask, mask_pixel, flow_last, prev_state=None):
        index_1 = 0
        flow_var = [None] * len(pixel_mask[0])
        flow_var = [flow_var] * len(pixel_mask)
        if prev_state is None:
            prev_state = [None] * len(pixel_mask[0])
            prev_state = [prev_state] * len(pixel_mask)
        if flow_last is None:
            flow_last = [None] * len(pixel_mask[0])
            flow_last = [flow_last] * len(pixel_mask)

        for maskpa, prev_a in zip(mask_pixel, prev_state):
            index_2 = 0
            for maskpb, prev_b in zip(maskpa, prev_a):
                if prev_b is None:
                    pre_v = np.random.randn(1) * self.sigma_i + self.mean
                    pre_d = np.random.uniform(0, 360, 1)
                    x, y = self.pol2cart(pre_v, pre_d)
                    r = np.random.uniform(-self.rotate_range, self.rotate_range, 1)
                    zoom = np.random.uniform(1 - self.zoom_ini, 1 + self.zoom_ini, 1)
                    r_count = 0
                    x_c = 0
                    y_c = 0
                    zoom_count = 1.0
                else:
                    x, y, r, r_count, x_c, y_c, zoom_count, zoom = *prev_b,
                    x = np.random.randn(1) * self.sigma_v + x
                    y = np.random.randn(1) * self.sigma_v + y
                    r = np.random.randn(1) * self.sigma_rotate_v + r
                    zoom = np.random.randn(1) * self.sigma_zoom_v + zoom
                if self.if_avoid_overlap:
                    x, y, r = self.occlusion_detection(pixel_mask[index_1], maskpb, x, y, r, r_count, x_c, y_c, zoom,
                                                       zoom_count, index_2)
                # if you wanna void overlap, use this function #
                r_count = r_count + r
                x_c = x_c + x
                y_c = y_c + y
                zoom_count = zoom_count * zoom
                pixel_mask[index_1][index_2], flow_last[index_1][index_2], flow_var[index_1][index_2] = \
                    self.affine_gird(maskpb, 0, x_c, y_c, 1, flow_last[index_1][index_2])
                prev_state[index_1][index_2] = (x, y, r, r_count, x_c, y_c, zoom_count, zoom)

                angle = np.arctan2(y, x) * 180 / np.pi
                # print('======ranodm =========')
                # print('angle given', angle)
                # print('U, V', x, y)
                # x__ = flow_var[index_1][index_2][:, 0, :, :].mean().cpu().numpy()
                # y__ = flow_var[index_1][index_2][:, 1, :, :].mean().cpu().numpy()
                # angle_ = np.arctan2(y__, x__) * 180 / np.pi
                # print('angle flow', angle_)
                # print('U, V', x__, y__)
                # print('===============')

                index_2 += 1
            index_1 += 1
        return pixel_mask, flow_last, flow_var, prev_state

    def shift_ins_circular(self, pixel_mask, mask_pixel, flow_last, prev_state=None):
        index_1 = 0
        flow_var = [None] * len(pixel_mask[0])
        flow_var = [flow_var] * len(pixel_mask)
        if prev_state is None:
            prev_state = [None] * len(pixel_mask[0])
            prev_state = [prev_state] * len(pixel_mask)
        if flow_last is None:
            flow_last = [None] * len(pixel_mask[0])
            flow_last = [flow_last] * len(pixel_mask)

        for maskpa, prev_a in zip(mask_pixel, prev_state):
            index_2 = 0
            for maskpb, prev_b in zip(maskpa, prev_a):
                if prev_b is None:
                    t_start = 0
                    t_step = np.random.uniform(0, np.pi * 2 / 30)
                    direction = np.random.choice([-1, 1], 2)
                    theta1 = 1
                    theta2 = 1 * np.random.uniform(0.7, 1.3, 1)
                    radius = np.random.uniform(20, 55, 1)
                    phi = np.random.uniform(0, np.pi * 2, 1)
                    x_c = 0
                    y_c = 0
                else:
                    t_start, t_step, theta1, theta2, radius, direction, phi, x_c, y_c = *prev_b,

                x_ = radius * np.cos(t_start * theta1 + phi) * direction[0]
                y_ = radius * np.sin(t_start * theta1 + phi) * direction[1]
                x_c = x_c + x_
                y_c = y_c + y_
                t_start = t_start + t_step
                pixel_mask[index_1][index_2], flow_last[index_1][index_2], flow_var[index_1][index_2] = \
                    self.affine_gird(maskpb, 0, x_c, y_c, 1, flow_last[index_1][index_2])
                prev_state[index_1][index_2] = (t_start, t_step, theta1, theta2, radius, direction, phi, x_c, y_c)
                # flow_var[index_1][index_2][:, 1, :, :] = flow_var[index_1][index_2][:, 1, :, :] * -1
                # flow_var[index_1][index_2][:, 0, :, :] = flow_var[index_1][index_2][:, 0, :, :] * -1
                # TODO correct Y flow, this is a bug
                index_2 += 1
            index_1 += 1
        return pixel_mask, flow_last, flow_var, prev_state

    def shift_ins_line(self, pixel_mask, mask_pixel, flow_last, prev_state=None):
        index_1 = 0
        flow_var = [None] * len(pixel_mask[0])
        flow_var = [flow_var] * len(pixel_mask)
        if prev_state is None:
            prev_state = [None] * len(pixel_mask[0])
            prev_state = [prev_state] * len(pixel_mask)
        if flow_last is None:
            flow_last = [None] * len(pixel_mask[0])
            flow_last = [flow_last] * len(pixel_mask)

        for maskpa, prev_a in zip(mask_pixel, prev_state):
            index_2 = 0
            for maskpb, prev_b in zip(maskpa, prev_a):
                if prev_b is None:
                    t_start = 0
                    t_step = np.random.uniform(0, np.pi * 2 / 30)
                    direction = np.random.choice([-1, 1], 2)
                    theta1 = 1
                    theta2 = 1 * np.random.uniform(0.7, 1.3, 1)
                    radius = np.random.uniform(20, 55, 1)
                    phi = np.random.uniform(0, np.pi * 2, 1)
                    x_c = 0
                    y_c = 0
                else:
                    t_start, t_step, theta1, theta2, radius, direction, phi, x_c, y_c = *prev_b,

                x_ = radius * np.cos(t_start * theta1 + phi) * direction[0] / 40
                y_ = radius * np.sin(t_start * theta1 + phi) * direction[1]
                x_c = x_c + x_
                y_c = y_c + y_
                t_start = t_start + t_step
                pixel_mask[index_1][index_2], flow_last[index_1][index_2], flow_var[index_1][index_2] = \
                    self.affine_gird(maskpb, 0, x_c, y_c, 1, flow_last[index_1][index_2])
                prev_state[index_1][index_2] = (t_start, t_step, theta1, theta2, radius, direction, phi, x_c, y_c)
                # flow_var[index_1][index_2][:, 1, :, :] = flow_var[index_1][index_2][:, 1, :, :] * -1
                # flow_var[index_1][index_2][:, 0, :, :] = flow_var[index_1][index_2][:, 0, :, :] * -1
                # TODO correct Y flow, this is a bug
                index_2 += 1
            index_1 += 1
        return pixel_mask, flow_last, flow_var, prev_state

    @torch.no_grad()
    def forward_first_order(self, image_seq, seq_len=25, bulr_level=5):
        blur_level = np.random.randint(bulr_level // 2, bulr_level * 2)
        img_start = image_seq[0]
        if img_start.shape[0] > 1 and (np.random.randint(0, 2) == 1):
            a, b = torch.chunk(img_start, chunks=2)
            img_start = torch.cat([b, a], dim=0)
        mask_pixel = []
        output = []
        output_mask = []
        flow_list = []
        flow_list_1 = []
        flow_list_2 = []
        flow_list_3 = []
        flow_masked = []
        pixel_mask_list = []
        prev_s1 = None
        prev_s2 = None
        prev_s3 = None
        prev_s = None
        h, w = img_start.shape[-1], img_start.shape[-2]
        noise = torch.rand(img_start.size()).type_as(img_start)
        img_roll = torch.roll(img_start, shifts=[int(np.random.uniform(0, h, 1)), int(np.random.uniform(0, w, 1))],
                              dims=[-1, -2])
        max_attempts = 5
        attempt = 0

        while attempt < max_attempts:
            try:
                mask_ins = run_slic_seperate(img_roll, n_seg=self.n_seg, compact=self.compact, rd_select=[4, 15])
                break
            except Exception as e:
                print(f"Attempt {attempt + 1}: Error occurred - {e}")
                attempt += 1

        if attempt == max_attempts:
            print("Failed after multiple attempts. Please check the input parameters or function implementation.")

        noise = self.blur(self.blur(self.blur(noise)) / 5) + img_roll

        for i in range(blur_level):
            noise = self.blur(noise)
        noise = self.to_0_1(noise)

        noise[noise == self.key] = self.key + 1e-4
        # for index, img_ins in enumerate(mask_ins):
        #     mask_pixel.append(
        #         [(torch.tensor(occ_ins, dtype=torch.int).type_as(img_start) *
        #           noise[index]
        #           # *2* torch.rand(size=( 3, 1, 1)).type_as(
        #           #           img_start)
        #           ).clamp(0.1, 0.9) for occ_ins in img_ins])

        common_color = torch.rand(size=(3, 1, 1)).type_as(img_start)
        var_scale = torch.rand(1).type_as(img_start)

        if np.random.rand() > 0.2:
            for index, img_ins in enumerate(mask_ins):
                mask_pixel.append(
                    [torch.tensor(occ_ins, dtype=torch.int).type_as(img_start) * (
                            torch.zeros_like(noise[index]).type_as(img_start) + common_color + (
                            torch.rand(size=(3, 1, 1)).type_as(img_start) * 4 - 2) * var_scale.type_as(
                        img_start)).clamp(
                        0.1, 0.9) for occ_ins in img_ins])
        else:
            for index, img_ins in enumerate(mask_ins):
                mask_pixel.append(
                    [(torch.tensor(occ_ins, dtype=torch.int).type_as(img_start) *
                      noise[index]) for occ_ins in img_ins])

        flow_last1 = None
        flow_last2 = None
        flow_last = None
        pixel_mask = deepcopy(mask_pixel)
        pixel_mask = pixel_mask[0]
        # split the pixel_mask into three parts using rate 2:1:1
        pixel_mask_1 = [pixel_mask[:len(pixel_mask) // 2]]
        pixel_mask_2 = [pixel_mask[len(pixel_mask) // 2: len(pixel_mask) // 2 + len(pixel_mask) // 4]]
        pixel_mask_3 = [pixel_mask[len(pixel_mask) // 2 + len(pixel_mask) // 4:]]

        pixel_mask = [pixel_mask]

        mask_pixel = mask_pixel[0]
        mask_pixel_1 = [mask_pixel[:len(mask_pixel) // 2]]
        mask_pixel_2 = [mask_pixel[len(mask_pixel) // 2: len(mask_pixel) // 2 + len(mask_pixel) // 4]]
        mask_pixel_3 = [mask_pixel[len(mask_pixel) // 2 + len(mask_pixel) // 4:]]
        mask_pixel = [mask_pixel]

        # uniform bkgd
        bkgd = torch.rand(1).type_as(img_start).clamp(0.1, 0.9)
        image_seq = [torch.ones_like(img_start) * bkgd for i in range(seq_len)]
        image_seq.reverse()
        for index, image in enumerate(image_seq):
            image, mask = self.draw_img(pixel_mask, image, type=0)
            output.append(image)
            output_mask.append(mask)
            pixel_mask_list.append(deepcopy(pixel_mask))
            if index == len(image_seq) - 1:
                break
            pixel_mask_1, flow_last1, flow_var1, prev_s1 = self.shift_ins_circular(pixel_mask_1, mask_pixel_1,
                                                                                   flow_last1, prev_s1)
            pixel_mask_2, flow_last2, flow_var2, prev_s2 = self.shift_ins_line(pixel_mask_2, mask_pixel_2,
                                                                               flow_last2,
                                                                               prev_s2)
            pixel_mask_3, flow_last, flow_var3, prev_s3 = self.shift_ins_random(pixel_mask_3, mask_pixel_3, flow_last,
                                                                                prev_s3)
            flow_list_1.append(flow_var1)
            flow_list_2.append(flow_var2)
            flow_list_3.append(flow_var3)
            pixel_mask = [pixel_mask_1[0] + pixel_mask_2[0] + pixel_mask_3[0]]

        pixel_mask_list.reverse()
        flow_list_3.reverse()
        flow_list_2.reverse()
        flow_list_1.reverse()
        for idx in range(len(flow_list_1)):
            flow_list.append([flow_list_1[idx][0] + flow_list_2[idx][0] + flow_list_3[idx][0]])
        for flow_var, mask in zip(flow_list, pixel_mask_list):
            flow_masked.append(self.create_flow(image, mask, flow_var))
        self.mask = output_mask
        image_seq.reverse()
        output.reverse()
        output_mask.reverse()
        return output, output_mask, flow_masked


class GaussianBlurConv(torch.nn.Module):
    def __init__(self, channels=3):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
        kernel = np.array(kernel) / np.array(kernel).sum()
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = torch.nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        x = torch.nn.functional.conv2d(x, self.weight, padding='same', groups=self.channels)
        return x
