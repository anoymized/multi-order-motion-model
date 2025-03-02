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
import torch.multiprocessing
from glob import glob
import os.path as osp

torch.multiprocessing.set_sharing_strategy('file_system')


class ImgSeqDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, root, n_frames, input_transform=None, co_transform=None,
                 target_transform=None, ap_transform=None, with_flow=False):
        self.root = Path(root)
        self.n_frames = n_frames
        self.input_transform = input_transform
        self.co_transform = co_transform
        self.ap_transform = ap_transform
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

        target = {}
        if 'flow' in s:
            target['flow'] = [load_flow(self.root / p) for p in s['flow']]
        if 'mask' in s:
            # 0~255 HxWx1
            mask = imageio.imread(s['mask']).astype(np.float32) / 255.
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            target['mask'] = np.expand_dims(mask, -1)
        return images, target

    def __len__(self):
        return len(self.samples)

    def __rmul__(self, v):
        self.samples = v * self.samples
        return self

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
            if "img_wo_crop" in target:
                target["img_wo_crop"] = [self.input_transform(i) for i in target["img_wo_crop"]]
                data['crop_parm'] = (target["img_wo_crop"], target["start_point"])

        data['img_list'] = images
        if self.ap_transform is not None:
            imgs_ph = self.ap_transform(
                [i.clone() for i in data['img_list']])
            data['img_ph_list'] = imgs_ph

        if self.target_transform is not None:
            for key in self.target_transform.keys():
                target[key] = [self.target_transform[key](i) for i in target[key]]
        data['flow'] = target["flow"]
        # viz_img_seq(data['img_list'], data['img_ph_list'])  # just for debugging

        return data


class ImgSeqDatasetSpec(Dataset, metaclass=ABCMeta):
    # specific dataset for sintel ar unsupervised training
    def __init__(self, root, n_frames, input_transform=None, co_transform=None,
                 target_transform=None, ap_transform=None):
        self.root = Path(root)
        self.n_frames = n_frames
        self.input_transform = input_transform
        self.co_transform = co_transform
        self.ap_transform = ap_transform
        self.target_transform = target_transform
        self.samples = self.collect_samples()
        self.init_seed = False

    @abstractmethod
    def collect_samples(self):
        pass

    def _load_sample(self, s):
        images_c = s['imgs_c']
        images_f = s['imgs_f']
        images_c = [imageio.imread(self.root / p).astype(np.float32) for p in images_c]
        images_f = [imageio.imread(self.root / p).astype(np.float32) for p in images_f]

        target = {}
        if 'flow' in s:
            target['flow'] = [load_flow(self.root / p) for p in s['flow']]
        if 'mask' in s:
            # 0~255 HxWx1
            mask = imageio.imread(s['mask']).astype(np.float32) / 255.
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            target['mask'] = np.expand_dims(mask, -1)
        return images_c, images_f, target

    def __len__(self):
        return len(self.samples)

    def __rmul__(self, v):
        self.samples = v * self.samples
        return self

    def __getitem__(self, idx):

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        images_c, images_f, target = self._load_sample(self.samples[idx])
        # print(self.samples[idx])  # for debug
        data = {}
        if self.co_transform is not None:
            # In unsupervised learning, there is no need to change target with image
            images, crop_parm = self.co_transform(images_c + images_f, {})
            images_c = images[:self.n_frames]
            images_f = images[self.n_frames:]
            assert len(images_c) == len(images_f) == self.n_frames
        if self.input_transform is not None:
            images_c = [self.input_transform(i) for i in images_c]
            images_f = [self.input_transform(i) for i in images_f]
            crop_parm["img_wo_crop"] = [self.input_transform(i) for i in crop_parm["img_wo_crop"][:self.n_frames]]
        data['img_list'] = images_c
        data['crop_parm'] = crop_parm

        if self.ap_transform is not None:
            imgs_ph = self.ap_transform(
                [i.clone() for i in images_f])

            data['img_ph_list'] = imgs_ph
        if self.target_transform is not None:
            for key in self.target_transform.keys():
                target[key] = [self.target_transform[key](i) for i in target[key]]
        data['target'] = target
        # viz_img_seq(data['img_list'], data['img_ph_list'])  # just for debugging

        return data


class SintelRaw(ImgSeqDataset):
    def __init__(self, root, n_frames=2, transform=None, co_transform=None):
        super(SintelRaw, self).__init__(root, n_frames, input_transform=transform,
                                        co_transform=co_transform)

    def collect_samples(self):
        scene_list = self.root.dirs()
        samples = []
        for scene in scene_list:
            img_list = scene.files('*.png')
            img_list.sort()

            for st in range(0, len(img_list) - self.n_frames + 1):
                seq = img_list[st:st + self.n_frames]
                sample = {'imgs': [self.root.relpathto(file) for file in seq]}
                samples.append(sample)
        return samples


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


class Sinewave(ImgSeqDataset):
    def __init__(self, root, n_frames=2, split='train',
                 subsplit='trainval', with_flow=True, ap_transform=None,
                 transform=None, target_transform=None, co_transform=None):
        self.with_flow = with_flow
        self.split = split
        self.subsplit = subsplit

        root = Path(root) / split
        super(Sinewave, self).__init__(root, n_frames, input_transform=transform,
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


class SintelDual(ImgSeqDatasetSpec):
    def __init__(self, root, n_frames=2, split='training',
                 subsplit='trainval', with_flow=True, ap_transform=None,
                 transform=None, target_transform=None, co_transform=None, ):
        self.with_flow = with_flow

        self.split = split
        self.subsplit = subsplit
        self.training_scene = ['alley_1', 'ambush_4', 'ambush_6', 'ambush_7', 'bamboo_2',
                               'bandage_2', 'cave_2', 'market_2', 'market_5', 'shaman_2',
                               'sleeping_2', 'temple_3']  # Unofficial train-val split

        root = Path(root) / split
        super(SintelDual, self).__init__(root, n_frames, input_transform=transform,
                                         target_transform=target_transform,
                                         co_transform=co_transform, ap_transform=ap_transform)

    def collect_samples(self):
        img_dir_clean = self.root / Path("clean")
        img_dir_final = self.root / Path("final")
        flow_dir = self.root / 'flow'
        scene_list = flow_dir.dirs()
        scene_list = [os.path.split(scene)[1] for scene in scene_list]
        assert img_dir_clean.isdir() and flow_dir.isdir() and img_dir_final.isdir()

        samples = []
        for scene in sorted(scene_list):
            if self.split == 'training' and self.subsplit != 'trainval':
                if self.subsplit == 'train' and scene not in self.training_scene:
                    continue
                if self.subsplit == 'val' and scene in self.training_scene:
                    continue
            img_scene = img_dir_clean / scene
            img_list_clean = img_scene.files('*.png')
            img_list_clean.sort()

            img_scene = img_dir_final / scene
            img_list_final = img_scene.files('*.png')
            img_list_final.sort()

            f_dir = flow_dir / scene
            flo_list = f_dir.files('*.flo')
            flo_list.sort()

            try:
                assert all([p.isfile() for p in flo_list])
                assert all([p.isfile() for p in img_list_final])
                assert all([p.isfile() for p in img_list_clean])
            except AssertionError:
                print('Incomplete sample in file:' + img_scene)
            for st in range(0, len(img_list_final) - self.n_frames + 1):
                seq_final = img_list_final[st:st + self.n_frames]
                img_sample_final = [self.root.relpathto(file) for file in seq_final]

                seq_clean = img_list_clean[st:st + self.n_frames]
                img_sample_clean = [self.root.relpathto(file) for file in seq_clean]
                if self.with_flow:  # evaluated a sequence
                    seq = flo_list[st:st + self.n_frames - 1]
                    flow_sample = [self.root.relpathto(file) for file in seq]
                    samples.append({'imgs_c': img_sample_clean, 'flow': flow_sample, "imgs_f": img_sample_final})
                else:
                    samples.append({'imgs_c': img_sample_clean, "imgs_f": img_sample_final})
        return samples


class SintelDualExtend(ImgSeqDatasetSpec):
    def __init__(self, root, n_frames=2, split='training',
                 subsplit='trainval', with_flow=False, ap_transform=None,
                 transform=None, target_transform=None, co_transform=None, ):
        self.with_flow = with_flow

        self.split = split
        self.subsplit = subsplit
        self.training_scene = ['alley_1', 'ambush_4', 'ambush_6', 'ambush_7', 'bamboo_2',
                               'bandage_2', 'cave_2', 'market_2', 'market_5', 'shaman_2',
                               'sleeping_2', 'temple_3']  # Unofficial train-val split

        root = Path(root) / split
        super(SintelDualExtend, self).__init__(root, n_frames, input_transform=transform,
                                               target_transform=target_transform,
                                               co_transform=co_transform, ap_transform=ap_transform)

    def collect_samples(self):
        img_dir_clean = self.root / Path("clean")
        img_dir_final = self.root / Path("final")
        img_dir_albedo = self.root / Path("albedo")
        flow_dir = self.root / 'flow'
        scene_list = img_dir_clean.dirs()
        scene_list = [os.path.split(scene)[1] for scene in scene_list]
        assert img_dir_clean.isdir() and flow_dir.isdir() and img_dir_final.isdir()

        samples = []
        for scene in sorted(scene_list):
            # if self.split == 'training' and self.subsplit != 'trainval':
            #     if self.subsplit == 'train' and scene not in self.training_scene:
            #         continue
            #     if self.subsplit == 'val' and scene in self.training_scene:
            #         continue
            img_scene = img_dir_clean / scene
            img_list_clean = img_scene.files('*.png')
            img_list_clean.sort()

            img_scene = img_dir_final / scene
            img_list_final = img_scene.files('*.png')
            img_list_final.sort()

            img_scene = img_dir_albedo / scene
            img_list_albedo = img_scene.files('*.png')
            img_list_albedo.sort()

            # f_dir = flow_dir / scene
            # flo_list = f_dir.files('*.flo')
            # flo_list.sort()

            try:
                # assert all([p.isfile() for p in flo_list])
                assert all([p.isfile() for p in img_list_final])
                assert all([p.isfile() for p in img_list_albedo])
                assert all([p.isfile() for p in img_list_clean])
            except AssertionError:
                print('Incomplete sample in file:' + img_scene)
            for st in range(0, len(img_list_final) - self.n_frames + 1):
                seq_final = img_list_final[st:st + self.n_frames]
                img_sample_final = [self.root.relpathto(file) for file in seq_final]

                seq_clean = img_list_clean[st:st + self.n_frames]
                img_sample_clean = [self.root.relpathto(file) for file in seq_clean]
                if self.with_flow:  # evaluated a sequence
                    seq = flo_list[st:st + self.n_frames - 1]
                    flow_sample = [self.root.relpathto(file) for file in seq]
                    samples.append({'imgs_c': img_sample_clean, 'flow': flow_sample, "imgs_f": img_sample_final})
                else:
                    samples.append({'imgs_c': img_sample_clean, "imgs_f": img_sample_final})

            for st in range(0, len(img_list_final) - self.n_frames + 1):
                seq_final = img_list_final[st:st + self.n_frames]
                img_sample_final = [self.root.relpathto(file) for file in seq_final]

                seq_clean = img_list_albedo[st:st + self.n_frames]
                img_sample_clean = [self.root.relpathto(file) for file in seq_clean]
                if self.with_flow:  # evaluated a sequence
                    seq = flo_list[st:st + self.n_frames - 1]
                    flow_sample = [self.root.relpathto(file) for file in seq]
                    samples.append({'imgs_c': img_sample_clean, 'flow': flow_sample, "imgs_f": img_sample_final})
                else:
                    samples.append({'imgs_c': img_sample_clean, "imgs_f": img_sample_final})

            for st in range(0, len(img_list_clean) - self.n_frames + 1):
                seq_final = img_list_clean[st:st + self.n_frames]
                img_sample_final = [self.root.relpathto(file) for file in seq_final]

                seq_clean = img_list_albedo[st:st + self.n_frames]
                img_sample_clean = [self.root.relpathto(file) for file in seq_clean]
                if self.with_flow:  # evaluated a sequence
                    seq = flo_list[st:st + self.n_frames - 1]
                    flow_sample = [self.root.relpathto(file) for file in seq]
                    samples.append({'imgs_c': img_sample_clean, 'flow': flow_sample, "imgs_f": img_sample_final})
                else:
                    samples.append({'imgs_c': img_sample_clean, "imgs_f": img_sample_final})
        return samples


class KITTIFlowMV(ImgSeqDataset):
    """
    This dataset is used for unsupervised training only
    """

    def __init__(self, root, n_frames=2,
                 transform=None, co_transform=None, ap_transform=None, ):
        super(KITTIFlowMV, self).__init__(root, n_frames,
                                          input_transform=transform,
                                          co_transform=co_transform,
                                          ap_transform=ap_transform)

    def collect_samples(self):
        flow_occ_dir = 'flow_' + 'occ'
        assert (self.root / flow_occ_dir).isdir()

        img_l_dir, img_r_dir = 'image_2', 'image_3'
        assert (self.root / img_l_dir).isdir() and (self.root / img_r_dir).isdir()

        samples = []
        for flow_map in sorted((self.root / flow_occ_dir).glob('*.png')):
            flow_map = flow_map.basename()
            root_filename = flow_map[:-7]

            for img_dir in [img_l_dir, img_r_dir]:
                img_list = (self.root / img_dir).files('*{}*.png'.format(root_filename))
                img_list.sort()

                for st in range(0, len(img_list) - self.n_frames + 1):
                    seq = img_list[st:st + self.n_frames]
                    sample = {}
                    sample['imgs'] = []
                    for i, file in enumerate(seq):
                        """ not use """
                        # frame_id = int(file[-6:-4])
                        # if 12 >= frame_id >= 9:  # why??
                        #     break
                        """         """
                        sample['imgs'].append(self.root.relpathto(file))
                    if len(sample['imgs']) == self.n_frames:
                        samples.append(sample)
        return samples


class KITTIFlow(ImgSeqDataset):
    """
    This dataset is used for validation only, so all files about target are stored as
    file filepath and there is no transform about target.
    """

    def __init__(self, root, n_frames=2, type="2012", transform=None):
        self.type = type
        super(KITTIFlow, self).__init__(root, n_frames, input_transform=transform)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # img 1 2 for 2 frames, img 0 1 2 for 3 frames.
        imgs = s["img_list"]
        inputs = [imageio.imread(self.root / p).astype(np.float32) for p in imgs]

        raw_size = inputs[0].shape[:2]
        data = {
            'flow_occ': self.root / s['flow_occ'],
            'flow_noc': self.root / s['flow_noc'],
        }

        data.update({  # for test set
            'im_shape': raw_size
        })

        if self.input_transform is not None:
            inputs = [self.input_transform(i) for i in inputs]
        # viz_img_seq(inputs, [])
        data['img_list'] = inputs
        return data

    def collect_samples(self):
        '''Will search in training folder for folders 'flow_noc' or 'flow_occ'
               and 'colored_0' (KITTI 2012) or 'image_2' (KITTI 2015) '''
        flow_occ_dir = 'flow_' + 'occ'
        flow_noc_dir = 'flow_' + 'noc'
        assert (self.root / flow_occ_dir).isdir()

        img_dir = 'image_2'
        assert (self.root / img_dir).isdir()

        samples = []
        for flow_map in sorted((self.root / flow_occ_dir).glob('*.png')):
            flow_map = flow_map.basename()
            root_filename = flow_map[:-7]
            if self.type == "2012" and (root_filename == "000031" or root_filename == "000082"):
                # incomplete data sequence, ignored in evaluation
                continue

            flow_occ_map = flow_occ_dir + '/' + flow_map
            flow_noc_map = flow_noc_dir + '/' + flow_map
            s = {'flow_occ': flow_occ_map, 'flow_noc': flow_noc_map}

            img_list = [img_dir + '/' + root_filename + "_%02d.png" % i for i in range(0, 11 + 1)]
            for img in img_list:
                assert (self.root / img).isfile()
            # for first image to image 11
            s.update({'img_list': img_list})
            samples.append(s)
        return samples

    @staticmethod
    def get_data_for_test(root, type='2012', test_size=[256, 832]):
        from torchvision import transforms
        from transforms import sep_transforms
        input_transform = transforms.Compose([
            sep_transforms.Zoom(*test_size),
            sep_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),  # normalize to [0，1]
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # normalize to [-1，1]
        ])
        return KITTIFlow(root, type=type, transform=input_transform)


class KITTITest(ImgSeqDataset):
    """
    This dataset is used for validation only, so all files about target are stored as
    file filepath and there is no transform about target.
    """

    def __init__(self, root, n_frames=2, type="2012", test_size=[256, 832]):
        self.type = type
        super(KITTITest, self).__init__(root, n_frames, input_transform=None)
        from torchvision import transforms
        from transforms import sep_transforms
        self.input_transform = transforms.Compose([
            sep_transforms.Zoom(*test_size),
            sep_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),  # normalize to [0，1]
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # normalize to [-1，1]
        ])

    def __getitem__(self, idx):

        s = self.samples[idx]
        # img 1 2 for 2 frames, img 0 1 2 for 3 frames.
        imgs = s["img_list"]
        inputs = [imageio.imread(self.root / p).astype(np.float32) for p in imgs]

        raw_size = inputs[0].shape[:2]
        data = {}

        data.update({  # for test set
            'im_shape': raw_size
        })
        inputs = [self.input_transform(i) for i in inputs]
        # viz_img_seq(inputs, [])
        data['img_list'] = inputs
        data['img_dir'] = s['img_dir']
        data['frame_id'] = s['frame_id']
        return data

    def collect_samples(self):
        '''Will search in training folder for folders 'flow_noc' or 'flow_occ'
               and 'colored_0' (KITTI 2012) or 'image_2' (KITTI 2015) '''

        img_dirs = ["image_2"]

        samples = []
        for img_dir in img_dirs:
            assert (self.root / img_dir).isdir()
            images1 = sorted(glob(self.root / img_dir / '*_10.png'))

            for img1 in images1:
                s = {}
                frame_id = osp.basename(img1)[0:6]
                if self.type == "2012" and (frame_id == '000127' or frame_id == '000182'):
                    img_list = [img_dir + '/' + frame_id + "_%02d.png" % i for i in range(5, 11 + 1)]
                else:
                    img_list = [img_dir + '/' + frame_id + "_%02d.png" % i for i in range(0, 11 + 1)]
                assert any([(self.root / img).isfile() for img in img_list])
                # for first image to image 11
                s.update({'img_list': img_list})
                s.update({'img_dir': img_dir})
                s.update({'frame_id': frame_id})
                samples.append(s)
        return samples


class FlyingChairsRaw(ImgSeqDataset):
    def __init__(self, root, n_frames=2, transform=None, co_transform=None, ap_transform=None):
        super(FlyingChairsRaw, self).__init__(root, 2, input_transform=transform,
                                              co_transform=co_transform, ap_transform=ap_transform)
        if n_frames != 2:
            print('Incomplete setting for Flyingchair dataset! Forcibly set n_frame = 2')

    def collect_samples(self):
        scene = self.root / 'data'
        samples = []
        img_list = scene.files('*.ppm')

        img_list.sort()
        split_list = np.loadtxt('./chairs_split.txt', dtype=np.int32)
        for i in range(len(img_list) // 2):
            xid = split_list[i]
            if xid == 1:
                image_list = [img_list[2 * i], img_list[2 * i + 1]]
                image_list = [self.root.relpathto(file) for file in image_list]
                sample = {'imgs': image_list}
                samples.append(sample)
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


class ChairSDHomUn(ImgSeqDataset):

    def __init__(self, root='../opticalflowdataset/ChairsSDHom', n_frames=2, split='train',
                 subsplit='trainval', with_flow=True, ap_transform=None,
                 transform=None, target_transform=None, co_transform=None):
        if n_frames != 2:
            print('Incomplete setting for Flyingchair dataset! Forcibly set n_frame = 2')

        self.with_flow = with_flow
        self.split = split
        self.subsplit = subsplit
        self.training_scene = []  # Unofficial train-val split, use for filter

        root = Path(root)
        super(ChairSDHomUn, self).__init__(root, 2, input_transform=transform,
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
            image_list = [images_1[i], images_2[i]]
            image_list = [self.root.relpathto(file) for file in image_list]
            sample = {'imgs': image_list}
            samples.append(sample)
        return samples


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
