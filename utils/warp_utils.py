import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect


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


def get_corresponding_map(data):
    """

    :param data: unnormalized coordinates Bx2xHxW
    :return: Bx1xHxW
    """
    B, _, H, W = data.size()

    # x = data[:, 0, :, :].view(B, -1).clamp(0, W - 1)  # BxN (N=H*W)
    # y = data[:, 1, :, :].view(B, -1).clamp(0, H - 1)

    x = data[:, 0, :, :].view(B, -1)  # BxN (N=H*W)
    y = data[:, 1, :, :].view(B, -1)

    # invalid = (x < 0) | (x > W - 1) | (y < 0) | (y > H - 1)   # BxN
    # invalid = invalid.repeat([1, 4])

    x1 = torch.floor(x)
    x_floor = x1.clamp(0, W - 1)
    y1 = torch.floor(y)
    y_floor = y1.clamp(0, H - 1)
    x0 = x1 + 1
    x_ceil = x0.clamp(0, W - 1)
    y0 = y1 + 1
    y_ceil = y0.clamp(0, H - 1)

    x_ceil_out = x0 != x_ceil
    y_ceil_out = y0 != y_ceil
    x_floor_out = x1 != x_floor
    y_floor_out = y1 != y_floor
    invalid = torch.cat([x_ceil_out | y_ceil_out,
                         x_ceil_out | y_floor_out,
                         x_floor_out | y_ceil_out,
                         x_floor_out | y_floor_out], dim=1)

    # encode coordinates, since the scatter function can only index along one axis
    corresponding_map = torch.zeros(B, H * W).type_as(data)
    indices = torch.cat([x_ceil + y_ceil * W,
                         x_ceil + y_floor * W,
                         x_floor + y_ceil * W,
                         x_floor + y_floor * W], 1).long()  # BxN   (N=4*H*W)
    values = torch.cat([(1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_floor)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_floor))],
                       1)
    # values = torch.ones_like(values)

    values[invalid] = 0

    corresponding_map.scatter_add_(1, indices, values)
    # decode coordinates
    corresponding_map = corresponding_map.view(B, H, W)

    return corresponding_map.unsqueeze(1)


def flow_warp(x, flow12, pad='border', mode='bilinear'):
    B, _, H, W = x.size()
    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW
    v_grid = norm_grid(base_grid + flow12)  # BHW2
    if 'align_corners' in inspect.getfullargspec(torch.nn.functional.grid_sample).args:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad, align_corners=True)
    else:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad)
    return im1_recons


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
        x_t[:, :, start_point[:, 0, 0, 0].long():, start_point[:, 1, 0, 0].long():] = output[:, :, :Hx - start_point[:, 0, 0, 0].long(),:Wx - start_point[:, 1, 0, 0].long()]
    elif start_point[:, 1, 0, 0].long() + W > Wx:
        x_t[:, :, start_point[:, 0, 0, 0].long(): start_point[:, 0, 0, 0].long() + H,
        start_point[:, 1, 0, 0].long():] = output[:, :, :, :Wx - start_point[:, 1, 0, 0].long()]
    elif start_point[:, 0, 0, 0].long() + H > Hx:
        x_t[:, :, start_point[:, 0, 0, 0].long():,
        start_point[:, 1, 0, 0].long(): start_point[:, 1, 0, 0].long() + W] = output[:, :, :Hx - start_point[:, 0, 0, 0].long(), :]
    else:
        x_t[:, :, start_point[:, 0, 0, 0].long(): start_point[:, 0, 0, 0].long() + H,
        start_point[:, 1, 0, 0].long(): start_point[:, 1, 0, 0].long() + W] = output
    # here align corners must be true

    return x_t


@torch.no_grad()
def occ_mask_expand(occ_ori, flow, location, image_full):
    otb_occ = out_boundary_det(flow)
    otb_occ_dilated = 1.0 - out_boundary_det(flow, location, image_full)
    occ_mask_dilated = (occ_ori + otb_occ) * otb_occ_dilated
    occ_mask_dilated = torch.clamp(occ_mask_dilated, min=0.0, max=1.0)
    occ_mask_dilated = occ_mask_dilated.detach()
    return occ_mask_dilated


def check_boundary_dilated_flow_warp(x, flo, start_point, x_):
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
    flo = 0
    start_point = start_point.unsqueeze(-1).unsqueeze(-1)
    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW
    v_grid = base_grid + flo + start_point

    v_grid[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / max(Wx - 1, 1) - 1.0
    v_grid[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / max(Hx - 1, 1) - 1.0

    v_grid = v_grid.permute(0, 2, 3, 1)  # B H,W,C
    # BHW2
    output = nn.functional.grid_sample(x, v_grid, mode="bilinear", padding_mode='border', align_corners=True)
    # error print out
    print(torch.mean(output - x_))

    return output


@torch.no_grad()
def out_boundary_det(flow, start_point=None, dilated_image=None):
    # detect which location is out of boundary 1：OCC
    if start_point is None:
        B, C, H, W = flow.shape
        base_grid = mesh_grid(B, H, W).type_as(flow)
        ot_boundary = torch.zeros_like(base_grid)[:, 0, :, :]
        ot_boundary[(base_grid + flow)[:, 1, :, :] > H] = 1
        ot_boundary[(base_grid + flow)[:, 1, :, :] < 0] = 1
        ot_boundary[(base_grid + flow)[:, 0, :, :] > W] = 1
        ot_boundary[(base_grid + flow)[:, 0, :, :] < 0] = 1
    else:
        _, _, H_d, W_d = dilated_image.shape
        start_point = start_point.unsqueeze(-1).unsqueeze(-1)
        B, C, H, W = flow.shape
        base_grid = mesh_grid(B, H, W).type_as(flow)
        base_grid = base_grid + start_point
        ot_boundary = torch.zeros_like(base_grid)[:, 0, :, :]
        ot_boundary[(base_grid + flow)[:, 1, :, :] > H_d] = 1
        ot_boundary[(base_grid + flow)[:, 1, :, :] < 0] = 1
        ot_boundary[(base_grid + flow)[:, 0, :, :] > W_d] = 1
        ot_boundary[(base_grid + flow)[:, 0, :, :] < 0] = 1

    return ot_boundary.unsqueeze(dim=1).float()


@torch.no_grad()
def get_occu_mask_bidirection(flow12, flow21, scale=0.01, bias=0.5):
    flow21_warped = flow_warp(flow21, flow12, pad='zeros')
    flow12_diff = flow12 + flow21_warped
    mag = (flow12 * flow12).sum(1, keepdim=True) + \
          (flow21_warped * flow21_warped).sum(1, keepdim=True)
    occ_thresh = scale * mag + bias
    occ = (flow12_diff * flow12_diff).sum(1, keepdim=True) > occ_thresh
    return occ.float()


def confidence_calculation(flow12, flow21):
    """ compute occ confidence"""
    # 2B,2,H,W
    flow21_warped = flow_warp(flow21, flow12)
    flow12_diff = flow12 + flow21_warped
    mag = (flow12 * flow12).sum(1, keepdim=True) + \
          (flow21_warped * flow21_warped).sum(1, keepdim=True)
    mag = mag * 0.01 + 0.5
    occ_conf = (flow12_diff * flow12_diff).sum(1, keepdim=True) / mag
    occ_mask = occ_conf > 1
    return occ_conf, occ_mask.float()


@torch.no_grad()
def get_occu_mask_backward(flow21, th=0.3):
    B, _, H, W = flow21.size()
    base_grid = mesh_grid(B, H, W).type_as(flow21)  # B2HW

    corr_map = get_corresponding_map(base_grid + flow21)  # BHW
    occu_mask = corr_map.clamp(min=0., max=1.) < th
    return occu_mask.float()


# from upflow
def upsample2d_as(inputs, target_as, mode="bilinear"):
    _, _, h, w = target_as.size()
    return F.interpolate(inputs, [h, w], mode=mode, align_corners=True)


def upsample2d_image_as(inputs, target_as, mode="bicubic"):
    _, _, h, w = target_as.size()
    res = F.interpolate(inputs, [h, w], mode=mode, align_corners=True)

    return res


def upsample_flow(inputs, target_size=None, target_flow=None, mode="bilinear"):
    if target_size is not None:
        h, w = target_size
    elif target_flow is not None:
        _, _, h, w = target_flow.size()
    else:
        raise ValueError('wrong input')
    _, _, h_, w_ = inputs.size()
    res = F.interpolate(inputs, [h, w], mode=mode, align_corners=True)
    res[:, 0, :, :] *= (w / w_)
    res[:, 1, :, :] *= (h / h_)
    return res


def rescale_flow(flow, div_flow, width_im, height_im, to_local=True):
    if to_local:
        u_scale = float(flow.size(3) / width_im / div_flow)
        v_scale = float(flow.size(2) / height_im / div_flow)
    else:
        u_scale = float(width_im * div_flow / flow.size(3))
        v_scale = float(height_im * div_flow / flow.size(2))

    u, v = flow.chunk(2, dim=1)
    u *= u_scale
    v *= v_scale

    return torch.cat([u, v], dim=1)


def get_grid(x):
    grid_H = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
    grid_V = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
    grid = torch.cat([grid_H, grid_V], 1)
    if x.is_cuda:
        grids_cuda = grid.float().requires_grad_(False).cuda()
    else:
        grids_cuda = grid.float().requires_grad_(False)  # .cuda()
    return grids_cuda


class WarpingLayer(nn.Module):

    def __init__(self):
        super(WarpingLayer, self).__init__()

    def forward(self, x, flow, height_im, width_im, div_flow):
        flo_list = []
        flo_w = flow[:, 0] * 2 / max(width_im - 1, 1) / div_flow
        flo_h = flow[:, 1] * 2 / max(height_im - 1, 1) / div_flow
        flo_list.append(flo_w)
        flo_list.append(flo_h)
        flow_for_grid = torch.stack(flo_list).transpose(0, 1)
        grid = torch.add(get_grid(x), flow_for_grid).transpose(1, 2).transpose(2, 3)
        x_warp = F.grid_sample(x, grid, align_corners=True)
        if x.is_cuda:
            mask = torch.ones(x.size(), requires_grad=False).cuda()
        else:
            mask = torch.ones(x.size(), requires_grad=False)  # .cuda()
        mask = F.grid_sample(mask, grid, align_corners=True)
        mask = (mask >= 1.0).float()
        return x_warp * mask


class WarpingLayer_no_div(nn.Module):

    def __init__(self):
        super(WarpingLayer_no_div, self).__init__()

    def forward(self, x, flow):
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        if x.is_cuda:
            grid = grid.cuda()
        # print(grid.shape,flo.shape,'...')
        vgrid = grid + flow
        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)  # B H,W,C
        x_warp = F.grid_sample(x, vgrid, padding_mode='zeros', align_corners=True)
        if x.is_cuda:
            mask = torch.ones(x.size(), requires_grad=False).cuda()
        else:
            mask = torch.ones(x.size(), requires_grad=False)  # .cuda()
        mask = F.grid_sample(mask, vgrid, align_corners=True)
        mask = (mask >= 1.0).float()
        return x_warp * mask
