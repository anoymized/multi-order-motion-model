from __future__ import print_function, division
import argparse
import os
import easydict
import numpy as np
from configdict import config_dict
import torch
import progressbar as pb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.nmi6.FFV1MT_MS import FFV1DNNV2
from datasets.get_dataset import get_dataset
from torch.cuda.amp import GradScaler
import datetime
from utils.misc_utils import AverageMeter
from utils.loss import sequence_loss

# from signal import signal, SIGPIPE, SIG_DFL, SIG_IGN
# signal(SIGPIPE, SIG_IGN)
MAX_FLOW = 600


@torch.no_grad()
def validate(model, dataloader):
    print_freq = 100
    model.eval()
    # model = model.module
    epe_all = 0
    n_step = 0
    result_dict = {}
    if type(dataloader) is not list:
        dataloader = [dataloader]

    for i_set, loader in enumerate(dataloader):
        epe_list = []

        for i_step, data in enumerate(loader):
            # currently only evalute the 1/10 of the train set
            if i_step % print_freq == 0 or i_step == len(loader) - 1:
                print("val step %d" % i_step)
            if i_step > 1000:
                break
            # only select 500 of the data
            imagelist, traget = data['img_list'], data['target']
            length = len(imagelist)
            gt_idx = int((length / 2 - 1) if length % 2 == 0 else (length - 1) / 2)
            flow1 = traget['flow'][gt_idx].cuda()
            flow2 = traget['flowsec'][gt_idx].cuda()

            for i in range(len(imagelist)):
                imagelist[i] = imagelist[i].cuda()
            # mask: first order is 1, second order is 0

            result_dict = model(imagelist, mix_enable=args.mixed_precision, layer=args.iters)
            flow1_predictions = result_dict['flow_seq'][-1]

            # mask_first[flow1[:, 0, :, :] < 1e-4 & flow1[:, 1, :, :] < 1e-4] = 0 # static area

            epe = torch.sum((flow1_predictions - flow1) ** 2, dim=1).sqrt().squeeze().mean()
            epe_list.append(epe)

            # measure elapsed time

        n_step += len(loader)
        epe_mean = torch.mean(torch.stack(epe_list, dim=0), dim=0).data
        epe_all += epe_mean
        print("Flow validation EPE: %s" % epe_mean)
        result_dict.update({'Subset %d' % i_set: {'EPE': epe_mean}})

    model.train()
    # release the memory
    torch.cuda.empty_cache()

    return result_dict, epe_all


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    all_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            all_params[name] = param

    # assgin MT parameters with a 10 times smaller learning rate
    params = []
    for name, param in all_params.items():
        if 'MT' in name:
            # print('Small learning rate for MT: ', name)
            params.append({'params': param, 'lr': args.lr})
        else:
            params.append({'params': param, 'lr': args.lr})
            # print('Normal learning rate for: ', name)
    optimizer = optim.AdamW(params, weight_decay=args.wdecay, eps=args.epsilon)

    # print the trainable parameters and modules name
    print("Trainable parameters: %d" % count_parameters(model))
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


# linearly increase the smooth term from 1e-4 to 0.1
def smooth_term(step, total_steps):
    return 1e-4 + (0.1 - 1e-4) * step / total_steps


def train(args, cfg):

    model = nn.DataParallel(FFV1DNNV2(num_scales=8,
                                      upsample_factor=8,
                                      scale_factor=16,
                                      num_layers=8, ))

    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        # model.load_state_dict(torch.load(args.restore_ckpt), strict=False)
        # only update the parameters with the same shape
        pretrained_dict = torch.load(args.restore_ckpt)
        # model_dict = model.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if
        #                    k in model_dict and v.shape == model_dict[k].shape}
        #
        #
        # model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=True)

    model.cuda()
    model.train()
    Average = AverageMeter(i=5, names=['loss', 'loss_1', 'loss_2', 'EPE_1', 'EPE_2'])

    # open
    train_set, test_set = get_dataset(cfg)



    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    # get the training data number
    print("Training data number: %d" % len(train_set))
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    should_keep_training = True
    # wandb.watch(model, log='all', log_freq=cfg.train.record_freq)
    widgets = ['train iter: ', pb.Counter(), '/', str(args.num_steps), ' ',
               pb.Percentage(), ' ', pb.ETA(), ' ', pb.Bar(marker='=')]
    pbar = pb.ProgressBar(widgets=widgets, maxval=args.num_steps).start()
    while should_keep_training:
        for i_batch, data in enumerate(train_loader):
            optimizer.zero_grad()
            imagelist, traget = data['img_list'], data['target']
            length = len(imagelist)
            assert len(traget['flow']) == length - 1
            gt_idx = int((length / 2 - 1) if length % 2 == 0 else (length - 1) / 2)
            for i in range(len(imagelist)):
                imagelist[i] = imagelist[i].cuda()
            flow1 = traget['flow'][gt_idx].cuda()
            flow2 = traget['flowsec'][gt_idx].cuda()
            mask = traget['mask']
            mask = mask[gt_idx].cuda()
            # mask: first order is 0, second order is 1
            # model.module.freeze_first_order()

            result_dict = model(imagelist, mix_enable=args.mixed_precision, layer=args.iters)
            flowall = result_dict['flow_seq']
            flowallbi = result_dict['flow_seq_bi']

            flow1_pre = result_dict['flow_seq_1']
            flow1_pre_bi = result_dict['flow_seq_1_bi']

            if total_steps % (cfg.train.record_freq * 10) == 0:
                print("Mean flow1: %f, flow2: %f" % (flow1.abs().mean(), flow2.abs().mean()))


            loss1, metrics1 = sequence_loss(flowall, flow1, valid=None, gamma=args.gamma, smooth_term=0)
            loss2 = sequence_loss(flowallbi, flow1, valid=None, gamma=args.gamma, smooth_term=0)[0]

            loss1_pre, metrics1_pre = sequence_loss(flow1_pre, flow1, valid=None, gamma=args.gamma, smooth_term=0)
            loss2_pre = sequence_loss(flow1_pre_bi, flow1, valid=None, gamma=args.gamma, smooth_term=0)[0]

            # loss2, metrics2 = sequence_loss(flow1_bi, flow1, valid=None, gamma=args.gamma, smooth_term=0)
            Average.update_single('EPE_1', metrics1['EPE'])
            Average.update_single('loss_1', loss1.item())

            loss = (loss1 + loss2) / 2 + (loss1_pre + loss2_pre) / 4
                   # flowallbi.abs().mean() * 1e-5 + flow1_pre_bi.abs().mean() * 1e-5 # bayesian prior
            # if nan, skip this iteration
            if torch.isnan(loss):
                print('Warning! nan loss detected')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            if total_steps % cfg.train.print_freq == cfg.train.print_freq - 1:
                print('total steps', total_steps)
                print(Average)
                Average.reset(5)
            if total_steps % cfg.train.save_iter == cfg.train.save_iter - 1:
                resultdict, epe1 = validate(model, test_loader)
                PATH = '%d_%s_epe_%.2f_.pth' % (total_steps + 1, args.name, epe1)
                torch.save(model.state_dict(), os.path.join(cfg.local_dir, PATH))
                model.train()


            total_steps += 1
            pbar.update(total_steps)
            if total_steps > args.num_steps:
                should_keep_training = False
                break
    torch.save(model.state_dict(), os.path.join(cfg.local_dir, 'final.pth'))
    pbar.finish()

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help="name your experiment", default='train_demo')
    parser.add_argument('--restore_ckpt', help="restore checkpoint",
                        default='/home/2TSSD/experiment/secmotion-shared/modelckpt/dual_model_final.pth')
    # os.environ['WANDB_MODE'] = 'dryrun' # for debug

    parser.add_argument('--lr', type=float, default=0.8e-4)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision', default=True)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--iters', type=int, default=6)
    parser.add_argument('--wdecay', type=float, default=1e-6)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.85, help='exponential weighting')
    args = parser.parse_args()
    # add new properties from the config file
    config = easydict.EasyDict(config_dict)
    print(config)
    torch.manual_seed(1234)
    np.random.seed(1234)
    # open the cudnn backend for faster training

    # merge the config and args in one dict
    config.update(vars(args))
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # get current time to make project name unique
    now = datetime.datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    args.name = args.name + '_' + dt_string
    # create a folder to save checkpoints
    os.makedirs(os.path.join('checkpoints', args.name), exist_ok=True)
    config.update({'local_dir': os.path.join('checkpoints', args.name)})
    # save the config to the checkpoint folder
    train(args, config)
