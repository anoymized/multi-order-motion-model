import copy
from torchvision import transforms
from torch.utils.data import ConcatDataset
from transforms.co_transforms import get_co_transforms
from transforms import sep_transforms
from datasets.flow_datasets import Sintel, NonTex, DAVIS, GeneralDataLoader, Sinewave, SecondOrderMotion, SelfRender, \
    SintelSlow, SecondOrderSintel
from datasets.flow_datasets import FlyingChairs
from utils import flow_utils
import copy


def get_dataset(all_cfg):
    cfg = all_cfg.data

    input_transform = transforms.Compose([
        sep_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0], std=[255]),  # normalize to [0，1]
    ])
    co_transform = get_co_transforms(aug_args=all_cfg.data_aug)

    if cfg.type == 'Demo':
        co_transform.co_transforms.insert(0, sep_transforms.Zoom(*cfg.train_shape['general'], gray=False))

        train_set_1 = SelfRender(cfg.root_mini_diffuse, n_frames=cfg.train_n_frames,
                                 split='train',
                                 with_flow=True,
                                 transform=input_transform,
                                 co_transform=co_transform,
                                 target_transform={"flow": sep_transforms.ArrayToTensor(),
                                                   "flowsec": sep_transforms.ArrayToTensor(),
                                                   "mask": sep_transforms.ArrayToTensor()}
                                 )

        train_set_2 = SelfRender(cfg.root_mini_nondiffuse, n_frames=cfg.train_n_frames,
                                 split='train',
                                 with_flow=True,
                                 transform=input_transform,
                                 co_transform=co_transform,
                                 target_transform={"flow": sep_transforms.ArrayToTensor(),
                                                   "flowsec": sep_transforms.ArrayToTensor(),
                                                   "mask": sep_transforms.ArrayToTensor()}
                                 )

        val1 = SelfRender(cfg.root_mini_diffuse, n_frames=cfg.train_n_frames,
                          split='train',
                          with_flow=True,
                          transform=input_transform,
                          co_transform=None,
                          target_transform={"flow": sep_transforms.ArrayToTensor(),
                                            "flowsec": sep_transforms.ArrayToTensor(),
                                            "mask": sep_transforms.ArrayToTensor()}
                          )

        val2 = SelfRender(cfg.root_mini_nondiffuse, n_frames=cfg.train_n_frames,
                          split='train',
                          with_flow=True,
                          transform=input_transform,
                          co_transform=None,
                          target_transform={"flow": sep_transforms.ArrayToTensor(),
                                            "flowsec": sep_transforms.ArrayToTensor(),
                                            "mask": sep_transforms.ArrayToTensor()}
                          )

        train_set = ConcatDataset([train_set_1, train_set_2])
        valid_set = [val1, val2]

        print("Diffuse number %d" % len(train_set_1))
        print("Specular number %d" % len(train_set_2))

    elif cfg.type == 'Dual-final':
        co_transform_gray = copy.deepcopy(co_transform)
        co_transform_sintel = copy.deepcopy(co_transform)
        co_transform_sintel.co_transforms.insert(0, sep_transforms.Zoom(*cfg.train_shape['sintel'], gray=False))
        co_transform_sintel2k = copy.deepcopy(co_transform)
        co_transform_sintel2k.co_transforms.insert(0, sep_transforms.Zoom(*cfg.train_shape['sintel_2K'], gray=False))
        co_transform.co_transforms.insert(0, sep_transforms.Zoom(*cfg.train_shape['general'], gray=False))

        train_set_1 = Sintel(cfg.root_sintel, n_frames=cfg.train_n_frames, type='clean',
                             split='training', subsplit='training',
                             with_flow=True,
                             transform=input_transform,
                             co_transform=co_transform_sintel,
                             target_transform={"flow": sep_transforms.ArrayToTensor(),
                                               "flowsec": sep_transforms.ArrayToTensor(),
                                               "mask": sep_transforms.ArrayToTensor()}
                             ) * 0.2
        train_set_2 = Sintel(cfg.root_sintel, n_frames=cfg.train_n_frames, type='final',
                             split='training', subsplit='training',
                             with_flow=True,
                             transform=input_transform,
                             co_transform=co_transform_sintel,
                             target_transform={"flow": sep_transforms.ArrayToTensor(),
                                               "flowsec": sep_transforms.ArrayToTensor(),
                                               "mask": sep_transforms.ArrayToTensor()}
                             ) * 0.3

        train_set_3 = NonTex(cfg.root_nontex, n_frames=cfg.train_n_frames,
                             split='train',
                             with_flow=True,
                             transform=input_transform,
                             co_transform=co_transform,
                             target_transform={"flow": sep_transforms.ArrayToTensor(),
                                               "flowsec": sep_transforms.ArrayToTensor(),
                                               "mask": sep_transforms.ArrayToTensor()}
                             ) * 1.0

        train_set_4 = SelfRender(cfg.root_glass, n_frames=cfg.train_n_frames,
                                 split='train',
                                 with_flow=True,
                                 transform=input_transform,
                                 co_transform=co_transform,
                                 target_transform={"flow": sep_transforms.ArrayToTensor(),
                                                   "flowsec": sep_transforms.ArrayToTensor(),
                                                   "mask": sep_transforms.ArrayToTensor()}
                                 ) * 2
        train_set_5 = SelfRender(cfg.root_specular, n_frames=cfg.train_n_frames,
                                 split='train',
                                 with_flow=True,
                                 transform=input_transform,
                                 co_transform=co_transform,
                                 target_transform={"flow": sep_transforms.ArrayToTensor(),
                                                   "flowsec": sep_transforms.ArrayToTensor(),
                                                   "mask": sep_transforms.ArrayToTensor()}
                                 ) * 2

        train_set_6 = SelfRender(cfg.root_glass_dynamic, n_frames=cfg.train_n_frames,
                                 split='train',
                                 with_flow=True,
                                 transform=input_transform,
                                 co_transform=co_transform,
                                 target_transform={"flow": sep_transforms.ArrayToTensor(),
                                                   "flowsec": sep_transforms.ArrayToTensor(),
                                                   "mask": sep_transforms.ArrayToTensor()}
                                 ) * 2

        train_set_7 = SelfRender(cfg.root_frosted_glass, n_frames=cfg.train_n_frames,
                                 split='train',
                                 with_flow=True,
                                 transform=input_transform,
                                 co_transform=co_transform,
                                 target_transform={"flow": sep_transforms.ArrayToTensor(),
                                                   "flowsec": sep_transforms.ArrayToTensor(),
                                                   "mask": sep_transforms.ArrayToTensor()}
                                 ) * 2

        train_set_8 = SintelSlow(cfg.root_sintel_slow, n_frames=cfg.train_n_frames,
                                 split='training', subsplit='training',
                                 with_flow=True,
                                 transform=input_transform,
                                 co_transform=co_transform_sintel2k,
                                 target_transform={"flow": sep_transforms.ArrayToTensor(),
                                                   "flowsec": sep_transforms.ArrayToTensor(),
                                                   "mask": sep_transforms.ArrayToTensor()}
                                 ) * 0.2

        print("Sintel clean number %d" % len(train_set_1))
        print("Sintel final number %d" % len(train_set_2))
        print("Non texture number %d" % len(train_set_3))
        print("Glass number %d" % len(train_set_4))
        print("Specular number %d" % len(train_set_5))
        print("Glass dynamic number %d" % len(train_set_6))
        print("Frosted glass number %d" % len(train_set_7))
        print("Sintel slow number %d" % len(train_set_8))

        train_set = ConcatDataset([train_set_1, train_set_2, train_set_3, train_set_4, train_set_5, train_set_6,
                                   train_set_7, train_set_8])

        valid_set_1 = Sintel(cfg.root_sintel, n_frames=cfg.train_n_frames, type='clean',
                             split='training',
                             with_flow=True,
                             transform=input_transform,
                             co_transform=sep_transforms.Zoom(*cfg.test_shape, gray=False),
                             target_transform={"flow": sep_transforms.ArrayToTensor()}
                             )
        valid_set_2 = Sintel(cfg.root_sintel, n_frames=cfg.train_n_frames, type='final',
                             split='training',
                             with_flow=True,
                             transform=input_transform,
                             co_transform=sep_transforms.Zoom(*cfg.test_shape, gray=False),
                             target_transform={"flow": sep_transforms.ArrayToTensor()}
                             )

        valid_set = ConcatDataset([valid_set_1, valid_set_2])

    elif cfg.type == 'Specular':
        # co_transform_gray = copy.deepcopy(co_transform)
        # co_transform_sintel = copy.deepcopy(co_transform)
        co_transform.co_transforms.insert(0, sep_transforms.Zoom(*cfg.train_shape['general'], gray=False))
        train1 = SelfRender(cfg.root_glass, n_frames=cfg.train_n_frames,
                            split='train',
                            with_flow=True,
                            transform=input_transform,
                            co_transform=co_transform,
                            target_transform={"flow": sep_transforms.ArrayToTensor(),
                                              "flowsec": sep_transforms.ArrayToTensor(),
                                              "mask": sep_transforms.ArrayToTensor()}
                            )
        train2 = SelfRender(cfg.root_specular, n_frames=cfg.train_n_frames,
                            split='train',
                            with_flow=True,
                            transform=input_transform,
                            co_transform=co_transform,
                            target_transform={"flow": sep_transforms.ArrayToTensor(),
                                              "flowsec": sep_transforms.ArrayToTensor(),
                                              "mask": sep_transforms.ArrayToTensor()}
                            )
        train3 = SelfRender(cfg.root_frosted_glass, n_frames=cfg.train_n_frames,
                            split='train',
                            with_flow=True,
                            transform=input_transform,
                            co_transform=co_transform,
                            target_transform={"flow": sep_transforms.ArrayToTensor(),
                                              "flowsec": sep_transforms.ArrayToTensor(),
                                              "mask": sep_transforms.ArrayToTensor()}
                            )

        train_set = ConcatDataset([train1, train2, train3])
        valid_set = train1

    elif cfg.type == 'Diffuse':
        # co_transform_gray = copy.deepcopy(co_transform)
        # co_transform_sintel = copy.deepcopy(co_transform)
        co_transform.co_transforms.insert(0, sep_transforms.Zoom(*cfg.train_shape['general'], gray=False))
        train1 = SelfRender(cfg.root_diffuse, n_frames=cfg.train_n_frames,
                            split='train',
                            with_flow=True,
                            transform=input_transform,
                            co_transform=co_transform,
                            target_transform={"flow": sep_transforms.ArrayToTensor(),
                                              "flowsec": sep_transforms.ArrayToTensor(),
                                              "mask": sep_transforms.ArrayToTensor()}
                            )

        train_set = ConcatDataset([train1])
        valid_set = train1



    elif cfg.type == 'Specular+Diffuse':
        # co_transform_gray = copy.deepcopy(co_transform)
        # co_transform_sintel = copy.deepcopy(co_transform)
        co_transform.co_transforms.insert(0, sep_transforms.Zoom(*cfg.train_shape['general'], gray=False))
        train1 = SelfRender(cfg.root_diffuse, n_frames=cfg.train_n_frames,
                            split='train',
                            with_flow=True,
                            transform=input_transform,
                            co_transform=co_transform,
                            target_transform={"flow": sep_transforms.ArrayToTensor(),
                                              "flowsec": sep_transforms.ArrayToTensor(),
                                              "mask": sep_transforms.ArrayToTensor()}
                            )

        train2 = SelfRender(cfg.root_glass, n_frames=cfg.train_n_frames,
                            split='train',
                            with_flow=True,
                            transform=input_transform,
                            co_transform=co_transform,
                            target_transform={"flow": sep_transforms.ArrayToTensor(),
                                              "flowsec": sep_transforms.ArrayToTensor(),
                                              "mask": sep_transforms.ArrayToTensor()}
                            )
        train3 = SelfRender(cfg.root_specular, n_frames=cfg.train_n_frames,
                            split='train',
                            with_flow=True,
                            transform=input_transform,
                            co_transform=co_transform,
                            target_transform={"flow": sep_transforms.ArrayToTensor(),
                                              "flowsec": sep_transforms.ArrayToTensor(),
                                              "mask": sep_transforms.ArrayToTensor()}
                            )
        train4 = SelfRender(cfg.root_frosted_glass, n_frames=cfg.train_n_frames,
                            split='train',
                            with_flow=True,
                            transform=input_transform,
                            co_transform=co_transform,
                            target_transform={"flow": sep_transforms.ArrayToTensor(),
                                              "flowsec": sep_transforms.ArrayToTensor(),
                                              "mask": sep_transforms.ArrayToTensor()}
                            )
        train_set = ConcatDataset([train1, train2, train3, train4])
        valid_set = train2


    else:
        raise ValueError("Unknown dataset type: {}".format(cfg.type))

    return train_set, valid_set
