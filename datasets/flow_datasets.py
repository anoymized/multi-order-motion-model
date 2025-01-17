import imageio
import matplotlib.pyplot as plt
import numpy as np
import random
from path import Path
from abc import abstractmethod, ABCMeta
from torch.utils.data import Dataset
from utils.flow_utils import load_flow
from utils.flow_utils import flow_to_image, flow_to_image_relative
from utils.flow_utils import InputPadder, flow_to_image, save_img_seq, viz_img_seq
import cv2
import os
import torch
import torch.multiprocessing as multiprocessing
from glob import glob
import os.path as osp

cv2.ocl.setUseOpenCL(False)  # 设置opencv不使用多进程运行，但这句命令只在本作用域有效。
cv2.setNumThreads(0)


class ImgSeqDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, root, n_frames, input_transform=None, co_transform=None,
                 target_transform=None, ap_transform=None, with_flow=False):
        self.root = Path(root)
        self.n_frames = n_frames
        self.input_transform = input_transform
        self.co_transform = co_transform
        self.target_transform = target_transform
        self.samples = self.collect_samples()
        self.init_seed = False
        self.with_flow = with_flow

    @abstractmethod
    def collect_samples(self):
        pass

    def _load_sample(self, s):
        images = s['imgs']
        images = [imageio.imread(self.root / p).astype(np.float32) for p in images]
        # if is rbga, convert to rgb
        if len(images[0].shape) == 3 and images[0].shape[-1] == 4:
            images = [cv2.cvtColor(img, cv2.COLOR_RGBA2RGB) for img in images]  # HxWx3
        # if is gray, convert to rgb
        if len(images[0].shape) == 2:
            images = [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in images]  # HxWx3
        H, W, _ = images[0].shape
        target = {}
        if 'sinwave' in s:
            if s['sinwave'] == 2:
                target['flow'] = [np.zeros((H, W, 2), dtype=np.float32)] * (len(images) - 1)
                target['flowsec'] = [load_flow(self.root / p) for p in s['flow']]
                # 0~255 HxWx1
                mask = [np.zeros((H, W), dtype=np.float32)] * (len(images) - 1)
                target['mask'] = mask
                return images, target

        if 'flow' in s:
            target['flow'] = [load_flow(self.root / p) for p in s['flow']]
        if 'flowsec' in s:
            target['flowsec'] = [load_flow(self.root / p) for p in s['flowsec']]
        if 'mask' in s:
            # 0~255 HxWx1
            mask = [imageio.imread(self.root / p).astype(np.float32) for p in s['mask']]
            if mask[0].max() > 1.5:
                mask = [m / 255. for m in mask]

            if len(mask[0].shape) == 3:
                mask = [m[:, :, 0] for m in mask]
            target['mask'] = [np.expand_dims(m, -1) for m in mask]
        return images, target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True
        images, target = self._load_sample(self.samples[idx])
        # print(self.samples[idx])  # for debug
        data = {}
        if self.co_transform is not None:
            if self.with_flow:
                images, target = self.co_transform(images, target)
            # In unsupervised learning, there is no need to change target with image
            else:
                images, target = self.co_transform(images, {})

        if self.input_transform is not None:
            images = [self.input_transform(i) for i in images]

        data['img_list'] = images

        if "flowsec" not in target.keys():
            target["flowsec"] = [np.zeros_like(target['flow'][0]) for _ in range(self.n_frames - 1)]
            target["mask"] = [np.expand_dims(np.ones_like(target['flow'][0][:, :, 0]), -1) for _ in
                              range(self.n_frames - 1)]
            # print('after cotransform')
            # print(data['img_list'][0].shape, data['img_list'][1].shape, data['img_list'][2].shape)
            # print(target["flowsec"][0].shape, target["mask"][0].shape, target['flow'][0].shape)
            # print("===================")
            # second order regions = 0

        if self.target_transform is not None:
            for key in self.target_transform.keys():
                target[key] = [self.target_transform[key](i) for i in target[key]]
        data['target'] = target
        # viz_img_seq(data['img_list'], data['img_ph_list'])  # just for debugging

        return data

    def __mul__(self, v):

        if v > 1:
            print("Augmenting dataset with factor %d" % v)
            self.samples = v * self.samples
        if v < 1:
            print("Reducing dataset with factor %d" % v)
            number = int(len(self.samples) * v)
            self.samples = np.random.choice(self.samples, number, replace=False)
        return self


class Sintel(ImgSeqDataset):
    def __init__(self, root, n_frames=2, type='clean', split='training',
                 subsplit='trainval', with_flow=True, ap_transform=None,
                 transform=None, target_transform=None, co_transform=None):
        self.dataset_type = type
        self.with_flow = with_flow

        self.split = split
        self.subsplit = subsplit
        self.training_scene = ['alley_1', 'ambush_4', 'ambush_6', 'ambush_7', 'bamboo_2',
                               'bandage_2', 'cave_2', 'market_2', 'market_5', 'shaman_2',
                               'sleeping_2', 'temple_3']  # Unofficial train-val split

        root = Path(root) / split
        super(Sintel, self).__init__(root, n_frames, input_transform=transform,
                                     target_transform=target_transform,
                                     co_transform=co_transform, ap_transform=ap_transform, with_flow=with_flow)

    def collect_samples(self):
        img_dir = self.root / Path(self.dataset_type)
        flow_dir = self.root / 'flow'
        scene_list = flow_dir.dirs()
        scene_list = [os.path.split(scene)[1] for scene in scene_list]
        assert img_dir.isdir() and flow_dir.isdir()

        samples = []
        for scene in sorted(scene_list):
            if self.split == 'training' and self.subsplit != 'trainval':
                if self.subsplit == 'train' and scene not in self.training_scene:
                    continue
                if self.subsplit == 'val' and scene in self.training_scene:
                    continue
            img_scene = img_dir / scene
            img_list = img_scene.files('*.png')
            img_list.sort()
            f_dir = flow_dir / scene
            flo_list = f_dir.files('*.flo')
            flo_list.sort()

            try:
                assert all([p.isfile() for p in flo_list])
                assert all([p.isfile() for p in img_list])
            except AssertionError:
                print('Incomplete sample in file:' + img_scene)
            for st in range(0, len(img_list) - self.n_frames + 1):
                seq = img_list[st:st + self.n_frames]
                img_sample = [self.root.relpathto(file) for file in seq]
                if self.with_flow:  # evaluated a sequence
                    seq = flo_list[st:st + self.n_frames - 1]
                    flow_sample = [self.root.relpathto(file) for file in seq]
                    samples.append({'imgs': img_sample, 'flow': flow_sample})
                else:
                    samples.append({'imgs': img_sample})
        return samples


class SintelSlow(ImgSeqDataset):
    def __init__(self, root, n_frames=2, split='train',
                 subsplit='trainval', with_flow=True, ap_transform=None,
                 transform=None, target_transform=None, co_transform=None):
        self.with_flow = with_flow

        self.split = split
        self.subsplit = subsplit

        root = Path(root)
        super(SintelSlow, self).__init__(root, n_frames, input_transform=transform,
                                         target_transform=target_transform,
                                         co_transform=co_transform, ap_transform=ap_transform, with_flow=with_flow)

    def collect_samples(self):
        img_dir = self.root / "image"
        flow_dir = self.root / 'flow'
        scene_list = img_dir.dirs()
        scene_list = [os.path.split(scene)[1] for scene in scene_list]
        assert img_dir.isdir() and flow_dir.isdir()

        samples = []
        for scene in sorted(scene_list):
            img_scene = img_dir / scene
            img_list = img_scene.files('*.png')
            img_list.sort()
            f_dir = flow_dir / scene
            flo_list = f_dir.files('*.flo')
            flo_list.sort()
            try:
                assert all([p.isfile() for p in flo_list])
                assert all([p.isfile() for p in img_list])
            except AssertionError:
                print('Incomplete sample in file:' + img_scene)
            for st in range(0, len(img_list) - self.n_frames + 1):
                seq = img_list[st:st + self.n_frames]
                img_sample = [self.root.relpathto(file) for file in seq]
                if self.with_flow:  # evaluated a sequence
                    seq = flo_list[st:st + self.n_frames - 1]
                    flow_sample = [self.root.relpathto(file) for file in seq]
                    samples.append({'imgs': img_sample, 'flow': flow_sample})
                else:
                    samples.append({'imgs': img_sample})
        return samples


class NonTex(ImgSeqDataset):
    def __init__(self, root, n_frames=2, split='train',
                 subsplit='trainval', with_flow=True, ap_transform=None,
                 transform=None, target_transform=None, co_transform=None):
        self.with_flow = with_flow

        self.split = split
        self.subsplit = subsplit

        root = Path(root) / split
        super(NonTex, self).__init__(root, n_frames, input_transform=transform,
                                     target_transform=target_transform,
                                     co_transform=co_transform, ap_transform=ap_transform, with_flow=with_flow)

    def collect_samples(self):
        img_dir = self.root / "image"
        flow_dir = self.root / 'flow'
        scene_list = img_dir.dirs()
        scene_list = [os.path.split(scene)[1] for scene in scene_list]
        assert img_dir.isdir() and flow_dir.isdir()

        samples = []
        for scene in sorted(scene_list):
            img_scene = img_dir / scene
            img_list = img_scene.files('*.png')
            img_list.sort()
            f_dir = flow_dir / scene
            flo_list = f_dir.files('*.flo')
            flo_list.sort()
            try:
                assert all([p.isfile() for p in flo_list])
                assert all([p.isfile() for p in img_list])
            except AssertionError:
                print('Incomplete sample in file:' + img_scene)
            for st in range(0, len(img_list) - self.n_frames + 1):
                seq = img_list[st:st + self.n_frames]
                img_sample = [self.root.relpathto(file) for file in seq]
                if self.with_flow:  # evaluated a sequence
                    seq = flo_list[st:st + self.n_frames - 1]
                    flow_sample = [self.root.relpathto(file) for file in seq]
                    samples.append({'imgs': img_sample, 'flow': flow_sample})
                else:
                    samples.append({'imgs': img_sample})
        print('Found {} samples in {} scenes'.format(len(samples), len(scene_list)) + self.root)
        return samples


class DAVIS(ImgSeqDataset):
    def __init__(self, root, n_frames=2, split='train',
                 subsplit='trainval', with_flow=True, ap_transform=None,
                 transform=None, target_transform=None, co_transform=None):
        self.with_flow = with_flow

        self.split = split
        self.subsplit = subsplit

        root = Path(root) / split
        super(DAVIS, self).__init__(root, n_frames, input_transform=transform,
                                    target_transform=target_transform,
                                    co_transform=co_transform, ap_transform=ap_transform, with_flow=with_flow)

    def collect_samples(self):
        img_dir = self.root / "image"
        flow_dir = self.root / 'flow'
        scene_list = flow_dir.dirs()
        scene_list = [os.path.split(scene)[1] for scene in scene_list]
        assert img_dir.isdir() and flow_dir.isdir()

        samples = []
        for scene in sorted(scene_list):
            img_scene = img_dir / scene
            img_list = img_scene.files('*.png')
            img_list.sort()
            f_dir = flow_dir / scene
            flo_list = f_dir.files('*.flo')
            flo_list.sort()
            try:
                assert all([p.isfile() for p in flo_list])
                assert all([p.isfile() for p in img_list])
            except AssertionError:
                print('Incomplete sample in file:' + img_scene)
            for st in range(0, len(img_list) - self.n_frames + 1):
                seq = img_list[st:st + self.n_frames]
                img_sample = [self.root.relpathto(file) for file in seq]
                if self.with_flow:  # evaluated a sequence
                    seq = flo_list[st:st + self.n_frames - 1]
                    flow_sample = [self.root.relpathto(file) for file in seq]
                    samples.append({'imgs': img_sample, 'flow': flow_sample})
                else:
                    samples.append({'imgs': img_sample})
        return samples


class SelfRender(ImgSeqDataset):
    def __init__(self, root, n_frames=2, split='train',
                 subsplit='trainval', with_flow=True, ap_transform=None,
                 transform=None, target_transform=None, co_transform=None):
        self.with_flow = with_flow

        self.split = split
        self.subsplit = subsplit

        root = Path(root)
        super(SelfRender, self).__init__(root, n_frames, input_transform=transform,
                                         target_transform=target_transform,
                                         co_transform=co_transform, ap_transform=ap_transform, with_flow=with_flow)

    def collect_samples(self):
        img_dir = self.root

        scene_list = img_dir.dirs()
        scene_list = [os.path.split(scene)[1] for scene in scene_list]
        assert img_dir.isdir()
        samples = []
        for scene in sorted(scene_list):
            img_scene = img_dir / scene
            img_list = img_scene.files('rgba_*.png')
            img_list.sort()
            flo_list = img_scene.files('forward_*.png')
            flo_list.sort()
            try:
                assert all([p.isfile() for p in flo_list])
                assert all([p.isfile() for p in img_list])
            except AssertionError:
                print('Incomplete sample in file:' + img_scene)
            for st in range(0, len(img_list) - self.n_frames + 1):
                seq = img_list[st:st + self.n_frames]
                img_sample = [self.root.relpathto(file) for file in seq]
                if self.with_flow:  # evaluated a sequence
                    seq = flo_list[st:st + self.n_frames - 1]
                    flow_sample = [self.root.relpathto(file) for file in seq]
                    samples.append({'imgs': img_sample, 'flow': flow_sample})
                else:
                    samples.append({'imgs': img_sample})
        return samples


class SecondOrderMotion(ImgSeqDataset):
    def __init__(self, root, n_frames=2, split='secondorderv1', with_flow=True, ap_transform=None,
                 transform=None, target_transform=None, co_transform=None, test=False, **kwargs):
        self.with_flow = with_flow
        self.split = split
        self.if_test = test
        if split is not None:
            root = Path(root) / split
        super(SecondOrderMotion, self).__init__(root, n_frames, input_transform=transform,
                                                target_transform=target_transform,
                                                co_transform=co_transform, ap_transform=ap_transform,
                                                with_flow=with_flow)

    def collect_samples(self):
        img_dir = self.root / "image"
        flow1_dir = self.root / 'flow1st'
        flow2_dir = self.root / 'flow2nd'
        mask_dir = self.root / 'viz'
        if self.if_test:
            temp = Path(os.path.split(self.root)[0])
            mask_dir = temp / 'viz'
        scene_list = img_dir.dirs()
        scene_list = [os.path.split(scene)[1] for scene in scene_list]
        assert img_dir.isdir() and flow1_dir.isdir() and flow2_dir.isdir() and mask_dir.isdir()

        samples = []
        for scene in sorted(scene_list):
            img_scene = img_dir / scene
            img_list = img_scene.files('*.png')
            img_list.sort()

            f_dir = flow1_dir / scene
            flo1_list = f_dir.files('*.flo')
            flo1_list.sort()

            f_dir = flow2_dir / scene
            flo2_list = f_dir.files('*.flo')
            flo2_list.sort()

            m_dir = mask_dir / scene
            mask_list = m_dir.files('*_mask_*.png')
            # sort the mask list according to the frame number
            mask_list = sorted(mask_list, key=lambda x: int(x.name.split('_')[-1].split('.')[0]))
            try:
                assert all([p.isfile() for p in flo2_list])
                assert all([p.isfile() for p in img_list])
                assert all([p.isfile() for p in flo1_list])
                assert all([p.isfile() for p in mask_list])
            except AssertionError:
                print('Incomplete sample in file:' + img_scene)
            for st in range(0, len(img_list) - self.n_frames + 1):
                seq = img_list[st:st + self.n_frames]
                img_sample = [self.root.relpathto(file) for file in seq]
                if self.with_flow:  # evaluated a sequence
                    seq = flo1_list[st:st + self.n_frames - 1]
                    flow_sample_1 = [self.root.relpathto(file) for file in seq]

                    seq = flo2_list[st:st + self.n_frames - 1]
                    flow_sample_2 = [self.root.relpathto(file) for file in seq]

                    seq = mask_list[st:st + self.n_frames - 1]
                    mask_sample = [self.root.relpathto(file) for file in seq]

                    samples.append(
                        {'imgs': img_sample, 'flow': flow_sample_1, 'flowsec': flow_sample_2, 'mask': mask_sample})
                else:
                    samples.append({'imgs': img_sample})
        print('Found {} samples in {} scenes'.format(len(samples), len(scene_list)) + self.root)
        return samples


class SecondOrderSintel(ImgSeqDataset):
    def __init__(self, root, n_frames=2, split='secondorderv1', with_flow=True, ap_transform=None,
                 transform=None, target_transform=None, co_transform=None, test=False, **kwargs):
        self.with_flow = with_flow
        self.split = split
        self.if_test = test
        if split is not None:
            root = Path(root) / split
        super(SecondOrderSintel, self).__init__(root, n_frames, input_transform=transform,
                                                target_transform=target_transform,
                                                co_transform=co_transform, ap_transform=ap_transform,
                                                with_flow=with_flow)

    def collect_samples(self):
        scene_list = self.root.dirs()
        scene_list = [os.path.split(scene)[1] for scene in scene_list]
        img_dir = "img"
        flow1_dir = 'flow_first'
        flow2_dir = 'flow_second'
        mask_dir = 'viz'
        if self.if_test:
            temp = Path(os.path.split(self.root)[0])
            mask_dir = temp / 'viz'
        assert all([self.root / scene / img_dir for scene in scene_list])

        samples = []
        for scene in sorted(scene_list):
            img_scene = self.root / scene / img_dir
            img_list = img_scene.files('*.png')
            img_list.sort()

            f_dir = self.root  / scene / flow1_dir
            flo1_list = f_dir.files('*.flo')
            flo1_list.sort()

            f_dir = self.root / scene / flow2_dir
            flo2_list = f_dir.files('*.flo')
            flo2_list.sort()

            m_dir = self.root / scene / mask_dir
            mask_list = m_dir.files('*_mask_*.png')
            # sort the mask list according to the frame number
            mask_list = sorted(mask_list, key=lambda x: int(x.name.split('_')[-1].split('.')[0]))
            try:
                assert all([p.isfile() for p in flo2_list])
                assert all([p.isfile() for p in img_list])
                assert all([p.isfile() for p in flo1_list])
                assert all([p.isfile() for p in mask_list])
            except AssertionError:
                print('Incomplete sample in file:' + img_scene)
            for st in range(0, len(img_list) - self.n_frames + 1):
                seq = img_list[st:st + self.n_frames]
                img_sample = [self.root.relpathto(file) for file in seq]
                if self.with_flow:  # evaluated a sequence
                    seq = flo1_list[st:st + self.n_frames - 1]
                    flow_sample_1 = [self.root.relpathto(file) for file in seq]

                    seq = flo2_list[st:st + self.n_frames - 1]
                    flow_sample_2 = [self.root.relpathto(file) for file in seq]

                    seq = mask_list[st:st + self.n_frames - 1]
                    mask_sample = [self.root.relpathto(file) for file in seq]

                    samples.append(
                        {'imgs': img_sample, 'flow': flow_sample_1, 'flowsec': flow_sample_2, 'mask': mask_sample})
                else:
                    samples.append({'imgs': img_sample})
        print('Found {} samples in {} scenes'.format(len(samples), len(scene_list)) + self.root)
        return samples


class Sinewave(ImgSeqDataset):
    def __init__(self, root, n_frames=2, split='train',
                 subsplit='trainval', with_flow=True, ap_transform=None,
                 transform=None, target_transform=None, co_transform=None, dataset_idx=1):
        self.with_flow = with_flow
        self.split = split
        self.subsplit = subsplit
        self.dataset_idx = dataset_idx  # 1 or 2
        assert self.dataset_idx in [1, 2]

        root = Path(root)
        super(Sinewave, self).__init__(root, n_frames, input_transform=transform,
                                       target_transform=target_transform,
                                       co_transform=co_transform, ap_transform=ap_transform, with_flow=with_flow)

    def collect_samples(self):
        img_dir = self.root
        flow_dir = self.root
        scene_list = flow_dir.dirs()
        scene_list = [os.path.split(scene)[1] for scene in scene_list]
        assert img_dir.isdir() and flow_dir.isdir()

        samples = []
        for scene in sorted(scene_list):
            img_scene = img_dir / scene
            img_list = img_scene.files('*.png')
            img_list.sort()
            f_dir = flow_dir / scene
            flo_list = f_dir.files('*.csv')
            if len(flo_list) == 1:
                flo_list = flo_list * (len(img_list) - 1)
            assert len(flo_list) == len(img_list) - 1, 'flow list length {} not equal to image list length {}'.format(
                len(flo_list), len(img_list))
            flo_list.sort()
            try:
                assert all([p.isfile() for p in flo_list])
                assert all([p.isfile() for p in img_list])
            except AssertionError:
                print('Incomplete sample in file:' + img_scene)
            for st in range(0, len(img_list) - self.n_frames + 1):
                seq = img_list[st:st + self.n_frames]
                img_sample = [self.root.relpathto(file) for file in seq]
                if self.with_flow:  # evaluated a sequence
                    seq = flo_list[st:st + self.n_frames - 1]
                    flow_sample = [self.root.relpathto(file) for file in seq]
                    samples.append({'imgs': img_sample, 'flow': flow_sample, 'sinwave': self.dataset_idx})
                else:
                    samples.append({'imgs': img_sample})
        return samples


class GeneralDataLoader(ImgSeqDataset):
    def __init__(self, root, n_frames=2, split='train',
                 subsplit='trainval', with_flow=True, ap_transform=None,
                 transform=None, target_transform=None, co_transform=None):
        self.with_flow = with_flow

        self.split = split
        self.subsplit = subsplit

        root = Path(root)
        super(GeneralDataLoader, self).__init__(root, n_frames, input_transform=transform,
                                                target_transform=target_transform,
                                                co_transform=co_transform, ap_transform=ap_transform,
                                                with_flow=with_flow)

    def collect_samples(self):
        img_dir = self.root / "image"
        flow_dir = self.root / 'flow'
        scene_list = flow_dir.dirs()
        scene_list = [os.path.split(scene)[1] for scene in scene_list]
        assert img_dir.isdir() and flow_dir.isdir()

        samples = []
        for scene in sorted(scene_list):
            img_scene = img_dir / scene
            img_list = img_scene.files('*.png')
            img_list.sort()
            f_dir = flow_dir / scene
            flo_list = f_dir.files('*.flo')
            flo_list.sort()
            try:
                assert all([p.isfile() for p in flo_list])
                assert all([p.isfile() for p in img_list])
            except AssertionError:
                print('Incomplete sample in file:' + img_scene)
            for st in range(0, len(img_list) - self.n_frames + 1):
                seq = img_list[st:st + self.n_frames]
                img_sample = [self.root.relpathto(file) for file in seq]
                if self.with_flow:  # evaluated a sequence
                    seq = flo_list[st:st + self.n_frames - 1]
                    flow_sample = [self.root.relpathto(file) for file in seq]
                    samples.append({'imgs': img_sample, 'flow': flow_sample})
                else:
                    samples.append({'imgs': img_sample})
        return samples


class FlyingChairs(ImgSeqDataset):
    def __init__(self, root, n_frames=2, split='training',
                 subsplit='trainval', with_flow=True, ap_transform=None,
                 transform=None, target_transform=None, co_transform=None):
        if n_frames != 2:
            print('Incomplete setting for Flyingchair dataset! Forcibly set n_frame = 2')

        self.with_flow = with_flow
        self.split = split
        self.subsplit = subsplit
        self.training_scene = []  # Unofficial train-val split, use for filter

        root = Path(root)
        super(FlyingChairs, self).__init__(root, 2, input_transform=transform,
                                           target_transform=target_transform,
                                           co_transform=co_transform, ap_transform=ap_transform)

    def collect_samples(self):
        img_dir = self.root / 'data'
        flow_dir = self.root / 'data'

        assert img_dir.isdir() and flow_dir.isdir()
        samples = []
        img_list = img_dir.files('*.ppm')
        img_list.sort()
        flo_list = flow_dir.files('*.flo')
        flo_list.sort()
        assert (len(img_list) // 2 == len(flo_list))
        try:
            assert all([p.isfile() for p in flo_list])
            assert all([p.isfile() for p in img_list])
        except AssertionError:
            print('Incomplete sample in file:' + img_dir)

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(img_list) // 2):
            xid = split_list[i]
            if xid == 2:
                image = [img_list[2 * i], img_list[2 * i + 1]]
                image = [self.root.relpathto(file) for file in image]
                flow = [flo_list[i]]
                flow = [self.root.relpathto(file) for file in flow]
                sample = {'imgs': image, 'flow': flow}
                samples.append(sample)
        return samples

    def __getitem__(self, idx):
        images, target = self._load_sample(self.samples[idx])
        images = [self.input_transform(image) for image in images]
        target = target["flow"]
        target = [self.target_transform(target) for target in target]
        return images, target


class ChairSDHom(ImgSeqDataset):

    def __init__(self, root='../opticalflowdataset/ChairsSDHom', n_frames=2, split='test',
                 subsplit='trainval', with_flow=True, ap_transform=None,
                 transform=None, target_transform=None, co_transform=None):
        if n_frames != 2:
            print('Incomplete setting for Flyingchair dataset! Forcibly set n_frame = 2')

        self.with_flow = with_flow
        self.split = split
        self.subsplit = subsplit
        self.training_scene = []  # Unofficial train-val split, use for filter

        root = Path(root)
        super(ChairSDHom, self).__init__(root, 2, input_transform=transform,
                                         target_transform=target_transform,
                                         co_transform=co_transform, ap_transform=ap_transform)

    def collect_samples(self):
        img_dir = self.root / 'data'
        image = os.path.join(img_dir, self.split)
        image1 = os.path.join(image, 't0')
        image2 = os.path.join(image, 't1')
        flow = os.path.join(image, 'flow')
        assert Path(image1).isdir() and Path(flow).isdir()
        samples = []
        images_1 = sorted(glob(osp.join(image1, '*.png')))
        images_2 = [os.path.join(image2, os.path.basename(img)) for img in images_1]

        flows = sorted(glob(osp.join(flow, '*.pfm')))
        assert len(images_1) == len(flows)

        for i in range(len(flows)):
            flow_list = [flows[i]]
            image_list = [images_1[i], images_2[i]]
            image_list = [self.root.relpathto(file) for file in image_list]
            flow_list = [self.root.relpathto(file) for file in flow_list]

            sample = {'imgs': image_list, 'flow': flow_list}
            samples.append(sample)
        return samples

    def __getitem__(self, idx):
        images, target = self._load_sample(self.samples[idx])
        images = [self.input_transform(image) for image in images]
        target = target["flow"]
        target = [self.target_transform(target) for target in target]
        return images, target
