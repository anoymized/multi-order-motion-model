import torch


def multi_frame_sequence_loss(flow_preds_list, flow_gt_list, valid_list=None, gamma=0.8, gamma_seq=1.0, max_flow=400,
                              smooth_term=0):
    """ Loss function defined over sequence of flow predictions """
    # flow_preds: list of flow predictions, each with shape [B, 2, H, W]
    # flow_gt: ground truth flow with shape [B, 2, H, W]
    # valid: valid mask with shape [B, 1, H, W]
    # smooth_term: whether to add smoothness for mask region
    # convert everything to float32
    if valid_list is None:
        valid_list = [torch.ones_like(flow_gt_list[0][:, 0:1]) for _ in range(len(flow_preds_list))]
    loss_all = []
    metrics_all = {}
    assert len(flow_preds_list) == len(flow_gt_list) == len(valid_list)
    length = len(flow_preds_list)

    for t_i in range(len(flow_preds_list)):
        n_predictions = len(flow_preds_list[t_i])
        flow_gt = flow_gt_list[t_i]
        valid = valid_list[t_i]
        flow_pred = flow_preds_list[t_i]
        flow_pred = [flow_pred.float() for flow_pred in flow_pred]

        flow_loss = 0.0
        mag = torch.sum(flow_gt ** 2, dim=1).sqrt()[:, None, :, :]
        valid = ((valid >= 0.5) & (mag < max_flow)).float()
        for s_i in range(n_predictions):
            i_weight = gamma ** (n_predictions - s_i - 1)
            i_loss = (flow_pred[s_i] - flow_gt).abs()
            flow_loss += i_weight * (valid * i_loss).sum() / (valid.sum() + 1e-4)
            if smooth_term > 0:
                flow_loss += smooth_term * i_weight * simple_flow_smoothness(flow_pred[s_i], 1 - valid)

        epe = torch.sum((flow_pred[-1] - flow_gt) ** 2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1) > 0.5]
        metrics = epe.mean().item()
        loss_all.append((flow_loss / len(flow_pred)) * (gamma_seq ** (length - t_i - 1)))
        metrics_all.update({'EPE_seq_%d' % t_i: metrics})

    # compute all loss
    loss_all = sum(loss_all) / len(loss_all)
    metrics_all.update({'EPE': sum(metrics_all.values()) / len(metrics_all.values())})
    return loss_all, metrics_all


def sequence_loss(flow_preds, flow_gt, valid=None, gamma=0.8, max_flow=400, smooth_term=0):
    """ Loss function defined over sequence of flow predictions """
    # 如果未给定 valid，则默认全部像素有效
    if valid is None:
        valid = torch.ones(flow_gt.size(0), 1, flow_gt.size(2), flow_gt.size(3)).to(flow_gt.device)

    flow_gt = flow_gt.float()
    valid = valid.float()
    flow_preds = [flow_pred.float() for flow_pred in flow_preds]
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # 计算 flow_gt 的幅值
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt().unsqueeze(1)
    # nan_mask：对于 flow_gt 的任一通道为 nan，则认为该像素无效
    nan_mask = torch.isnan(flow_gt).any(dim=1, keepdim=True)
    # 提取有效区域：原始 valid, 位移小于 max_flow 且不含 nan
    valid_mask = (valid >= 0.5) & (mag < max_flow) & (~nan_mask)
    # 计算有效像素个数（转换为 float 累加）
    valid_sum = valid_mask.float().sum() + 1e-4

    # 计算所有预测的权重和（考虑 gamma 衰减）
    total_weight = sum([gamma ** (n_predictions - i - 1) for i in range(n_predictions)])

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        # 计算当前预测与 ground truth 的绝对误差
        i_loss = (flow_preds[i] - flow_gt).abs()
        # 对无效位置用 0 替代，避免 0*nan 产生 nan
        masked_loss = torch.where(valid_mask, i_loss, torch.zeros_like(i_loss))
        flow_loss += i_weight * masked_loss.sum() / (2 * valid_sum)
        if smooth_term > 0:
            flow_loss += smooth_term * i_weight * simple_flow_smoothness(flow_preds[i], 1 - valid_mask.float())

    # 根据总权重进行归一化
    flow_loss = flow_loss / total_weight

    # 计算 end-point error (EPE) 时，仅对有效区域进行统计
    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt().unsqueeze(1)
    epe = epe.view(-1)[valid_mask.view(-1)]

    metrics = {
        'EPE': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy


def gradient_loss(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    loss_x = D_dx.abs() / 2.
    loss_y = D_dy.abs() / 2
    return loss_x.mean() / 2. + loss_y.mean() / 2.


def seq_smooth_grad_1st(flo, image, alpha):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)
    dx, dy = gradient(flo)

    loss_x = weights_x * dx.abs() / 2.
    loss_y = weights_y * dy.abs() / 2

    return loss_x.mean() / 2. + loss_y.mean() / 2.


def simple_flow_smoothness(flo, mask):
    loss = torch.sqrt(torch.sum(flo ** 2, dim=1) + 1e-6)
    loss = torch.sum(loss * mask) / (torch.sum(mask) + 1e-6)
    return loss
