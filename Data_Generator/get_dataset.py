import copy
from torchvision import transforms
from torch.utils.data import ConcatDataset
from transforms.co_transforms import get_co_transforms, get_co_transforms_s, get_co_transforms_sup
from transforms.ar_transforms.ap_transforms import get_ap_transforms
from transforms import sep_transforms
from datasets.flow_datasets import SintelRaw, Sintel, SintelDualExtend, NonTex, DAVIS, SintelSlow, Sinewave
from datasets.flow_datasets import FlyingChairs, FlyingChairsRaw, ChairSDHom, ChairSDHomUn
from datasets.flow_datasets import KITTIFlow, KITTIFlowMV
from evaluation import dataset_eval


def get_dataset(all_cfg):
    cfg = all_cfg.data

    input_transform = transforms.Compose([
        sep_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0], std=[255]),  # normalize to [0，1]
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # normalize to [-1，1]
    ])
    zoom = sep_transforms.Zoom(*cfg.test_shape)
    zoom_single = sep_transforms.ZoomSingle(*cfg.test_shape)
    co_transform = get_co_transforms_sup(aug_args=all_cfg.data_aug)

    if cfg.type == 'Sintel_Flow':
        co_transform.co_transforms.insert(0, sep_transforms.Zoom(*cfg.train_shape))
        ap_transform = get_ap_transforms(cfg.at_cfg) if cfg.run_at else None

        train_set_1 = Sintel(cfg.root_sintel, n_frames=cfg.train_n_frames, type='clean',
                             split='training', subsplit=cfg.train_subsplit,
                             with_flow=True,
                             ap_transform=ap_transform,
                             transform=input_transform,
                             co_transform=co_transform,
                             target_transform={"flow": sep_transforms.ArrayToTensor()}
                             )
        train_set_2 = Sintel(cfg.root_sintel, n_frames=cfg.train_n_frames, type='final',
                             split='training', subsplit=cfg.train_subsplit,
                             with_flow=True,
                             ap_transform=ap_transform,
                             transform=input_transform,
                             co_transform=co_transform,
                             target_transform={"flow": sep_transforms.ArrayToTensor()}
                             )
        train_set_3 = Sintel(cfg.root_sintel, n_frames=cfg.train_n_frames, type='albedo',
                             split='training', subsplit=cfg.train_subsplit,
                             with_flow=True,
                             ap_transform=ap_transform,
                             transform=input_transform,
                             co_transform=co_transform,
                             target_transform={"flow": sep_transforms.ArrayToTensor()}
                             )

        train_set = ConcatDataset([train_set_1, train_set_2, train_set_3])

        # valid_input_transform = copy.deepcopy(input_transform)
        # # valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))
        #
        # valid_set_1 = Sintel(cfg.root_sintel, n_frames=cfg.val_n_frames, type='clean',
        #                      split='training', subsplit=cfg.val_subsplit,
        #                      transform=valid_input_transform,
        #                      target_transform={'flow': sep_transforms.ArrayToTensor()}
        #                      )
        # valid_set_2 = Sintel(cfg.root_sintel, n_frames=cfg.val_n_frames, type='final',
        #                      split='training', subsplit=cfg.val_subsplit,
        #                      transform=valid_input_transform,
        #                      target_transform={'flow': sep_transforms.ArrayToTensor()}
        #                      )
        valid_set_1 = Sintel(cfg.root_sintel, n_frames=cfg.train_n_frames, type='clean',
                             split='training', subsplit=cfg.train_subsplit,
                             with_flow=True,
                             ap_transform=None,
                             transform=input_transform,
                             co_transform=sep_transforms.Zoom(*cfg.test_shape),
                             target_transform={"flow": sep_transforms.ArrayToTensor()}
                             )
        valid_set_2 = Sintel(cfg.root_sintel, n_frames=cfg.train_n_frames, type='final',
                             split='training', subsplit=cfg.train_subsplit,
                             with_flow=True,
                             ap_transform=None,
                             transform=input_transform,
                             co_transform=sep_transforms.Zoom(*cfg.test_shape),
                             target_transform={"flow": sep_transforms.ArrayToTensor()}
                             )

        valid_set = ConcatDataset([valid_set_1, valid_set_2])


    elif cfg.type == 'Sintel_extend':

        co_transform_nonzoom = copy.deepcopy(co_transform)
        # co_transform_nonzoom.co_transforms.insert(0, sep_transforms.Zoom(-1, -1))
        co_transform.co_transforms.insert(0, sep_transforms.Zoom(*cfg.train_shape))
        co_transform_nonzoom.co_transforms.insert(0, sep_transforms.NonZoom())
        ap_transform = get_ap_transforms(cfg.at_cfg) if cfg.run_at else None

        train_set_1 = Sintel(cfg.root_sintel, n_frames=cfg.train_n_frames, type='clean',
                             split='training', subsplit=cfg.train_subsplit,
                             with_flow=True,
                             ap_transform=ap_transform,
                             transform=input_transform,
                             co_transform=co_transform,
                             target_transform={"flow": sep_transforms.ArrayToTensor()}
                             )
        train_set_2 = Sintel(cfg.root_sintel, n_frames=cfg.train_n_frames, type='final',
                             split='training', subsplit=cfg.train_subsplit,
                             with_flow=True,
                             ap_transform=ap_transform,
                             transform=input_transform,
                             co_transform=co_transform,
                             target_transform={"flow": sep_transforms.ArrayToTensor()}
                             )
        train_set_3 = Sintel(cfg.root_sintel, n_frames=cfg.train_n_frames, type='albedo',
                             split='training', subsplit=cfg.train_subsplit,
                             with_flow=True,
                             ap_transform=ap_transform,
                             transform=input_transform,
                             co_transform=co_transform,
                             target_transform={"flow": sep_transforms.ArrayToTensor()}
                             )

        train_set_4 = NonTex(cfg.root_nontex, n_frames=cfg.train_n_frames,
                             split='train', subsplit=cfg.train_subsplit,
                             with_flow=True,
                             ap_transform=ap_transform,
                             transform=input_transform,
                             co_transform=co_transform,
                             target_transform={"flow": sep_transforms.ArrayToTensor()}
                             )
        train_set_5 = DAVIS(cfg.root_davis, n_frames=cfg.train_n_frames,
                            split='train', subsplit=cfg.train_subsplit,
                            with_flow=True,
                            ap_transform=ap_transform,
                            transform=input_transform,
                            co_transform=co_transform,
                            target_transform={"flow": sep_transforms.ArrayToTensor()}
                            )

        train_set_6 = SintelSlow(cfg.root_sintel_slow, n_frames=cfg.train_n_frames,
                                 split='train', subsplit=cfg.train_subsplit,
                                 with_flow=True,
                                 ap_transform=ap_transform,
                                 transform=input_transform,
                                 co_transform=co_transform,
                                 target_transform={"flow": sep_transforms.ArrayToTensor()}
                                 )

        train_set_7 = Sinewave(cfg.root_sin, n_frames=cfg.train_n_frames,
                               split='train', subsplit=cfg.train_subsplit,
                               with_flow=True,
                               ap_transform=ap_transform,
                               transform=input_transform,
                               co_transform=co_transform,
                               target_transform={"flow": sep_transforms.ArrayToTensor()}
                               )

        print("non texture number %d" % len(train_set_4))
        print("davis number %d" % len(train_set_5))
        print("Sintel_slow number %d" % len(train_set_6))
        print("Sin wave number %d" % len(train_set_7))

        train_set = ConcatDataset([train_set_1, train_set_2, train_set_3, train_set_4, train_set_5,   train_set_6,
                                   train_set_7])

        # valid_input_transform = copy.deepcopy(input_transform)
        # # valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))
        #
        # valid_set_1 = Sintel(cfg.root_sintel, n_frames=cfg.val_n_frames, type='clean',
        #                      split='training', subsplit=cfg.val_subsplit,
        #                      transform=valid_input_transform,
        #                      target_transform={'flow': sep_transforms.ArrayToTensor()}
        #                      )
        # valid_set_2 = Sintel(cfg.root_sintel, n_frames=cfg.val_n_frames, type='final',
        #                      split='training', subsplit=cfg.val_subsplit,
        #                      transform=valid_input_transform,
        #                      target_transform={'flow': sep_transforms.ArrayToTensor()}
        #                      )
        valid_set_1 = Sintel(cfg.root_sintel, n_frames=cfg.train_n_frames, type='clean',
                             split='training', subsplit=cfg.train_subsplit,
                             with_flow=True,
                             ap_transform=None,
                             transform=input_transform,
                             co_transform=sep_transforms.Zoom(*cfg.test_shape),
                             target_transform={"flow": sep_transforms.ArrayToTensor()}
                             )
        valid_set_2 = Sintel(cfg.root_sintel, n_frames=cfg.train_n_frames, type='final',
                             split='training', subsplit=cfg.train_subsplit,
                             with_flow=True,
                             ap_transform=None,
                             transform=input_transform,
                             co_transform=sep_transforms.Zoom(*cfg.test_shape),
                             target_transform={"flow": sep_transforms.ArrayToTensor()}
                             )

        valid_set = ConcatDataset([valid_set_1, valid_set_2])

    elif cfg.type == 'Sintel_Raw':
        train_set = SintelRaw(cfg.root_sintel_raw, n_frames=cfg.train_n_frames,
                              transform=input_transform, co_transform=co_transform)
        # valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))
        valid_set_1 = dataset_eval.MpiSintel(aug_params=zoom, split='training', if_seq=True, frames=0,
                                             if_test_sequence=True,
                                             subsplit=cfg.val_subsplit,
                                             dstype='clean', root='../opticalflowdataset/'
                                                                  'MPI-Sintel-complete')
        valid_set_2 = dataset_eval.MpiSintel(aug_params=zoom, split='training', if_seq=True, frames=0,
                                             if_test_sequence=True,
                                             subsplit=cfg.val_subsplit,
                                             dstype='final', root='../opticalflowdataset/'
                                                                  'MPI-Sintel-complete')
        valid_set = ConcatDataset([valid_set_1, valid_set_2])
    elif cfg.type == 'Chair':
        ap_transform = get_ap_transforms(cfg.at_cfg) if cfg.run_at else None
        train_input_transform = copy.deepcopy(input_transform)
        co_transform.co_transforms.insert(0, sep_transforms.Zoom(*cfg.train_shape))
        train_set = FlyingChairsRaw(cfg.root_chair_raw, n_frames=cfg.train_n_frames, transform=train_input_transform,
                                    co_transform=co_transform, ap_transform=ap_transform)
        valid_input_transform = copy.deepcopy(input_transform)
        valid_input_transform.transforms.insert(0, sep_transforms.ZoomSingle(*cfg.test_shape))
        # valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))
        valid_set = FlyingChairs(cfg.root_chair, n_frames=2,
                                 split='training', subsplit="val",
                                 transform=valid_input_transform,
                                 target_transform=sep_transforms.ArrayToTensor()
                                 )
    elif cfg.type == 'ChairSD':
        train_input_transform = copy.deepcopy(input_transform)
        co_transform.co_transforms.insert(0, sep_transforms.Zoom(*cfg.train_shape))
        train_set = ChairSDHomUn(cfg.root_chairsd, n_frames=cfg.train_n_frames, split="train",
                                 transform=train_input_transform, co_transform=co_transform)
        valid_input_transform = copy.deepcopy(input_transform)
        valid_input_transform.transforms.insert(0, sep_transforms.ZoomSingle(*cfg.test_shape))
        # valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))
        valid_set = ChairSDHom(cfg.root_chairsd, n_frames=2,
                               split='test', subsplit="val",
                               transform=valid_input_transform,
                               target_transform=sep_transforms.ArrayToTensor()
                               )

    elif cfg.type == 'ChairMixed':
        train_input_transform = copy.deepcopy(input_transform)
        co_transform.co_transforms.insert(0, sep_transforms.Zoom(*cfg.train_shape))
        train_set_1 = FlyingChairsRaw(cfg.root_chair_raw, n_frames=cfg.train_n_frames,
                                      transform=train_input_transform, co_transform=co_transform)

        train_input_transform = copy.deepcopy(input_transform)
        co_transform.co_transforms.insert(0, sep_transforms.Zoom(*cfg.train_shape))
        train_set_2 = ChairSDHomUn(cfg.root_chairsd, n_frames=cfg.train_n_frames, split="train",
                                   transform=train_input_transform, co_transform=co_transform)
        valid_input_transform = copy.deepcopy(input_transform)
        valid_input_transform.transforms.insert(0, sep_transforms.ZoomSingle(*cfg.test_shape))

        valid_set = FlyingChairs(cfg.root_chair, n_frames=2,
                                 split='training', subsplit="val",
                                 transform=valid_input_transform,
                                 target_transform=sep_transforms.ArrayToTensor()
                                 )
        train_set = train_set_1 + train_set_2
    elif cfg.type == 'KITTI':
        train_input_transform = copy.deepcopy(input_transform)
        co_transform.co_transforms.insert(0, sep_transforms.Zoom(*cfg.train_shape))
        ap_transform = get_ap_transforms(cfg.at_cfg) if cfg.run_at else None
        if cfg.train_15 and cfg.train_12:
            train_set_1 = KITTIFlowMV(
                cfg.root_kitti15,
                cfg.train_n_frames,
                transform=train_input_transform,
                ap_transform=ap_transform,
                co_transform=co_transform  # no target here
            )
            train_set_2 = KITTIFlowMV(
                cfg.root_kitti12,
                cfg.train_n_frames,
                transform=train_input_transform,
                ap_transform=ap_transform,
                co_transform=co_transform  # no target here
            )
            train_set = ConcatDataset([train_set_1, train_set_2])
        elif cfg.train_12:
            train_set = KITTIFlowMV(
                cfg.root_kitti12,
                cfg.train_n_frames,
                transform=train_input_transform,
                ap_transform=ap_transform,
                co_transform=co_transform  # no target here
            )
        elif cfg.train_15:
            train_set = KITTIFlowMV(
                cfg.root_kitti15,
                cfg.train_n_frames,
                transform=train_input_transform,
                ap_transform=ap_transform,
                co_transform=co_transform  # no target here
            )

        valid_input_transform = copy.deepcopy(input_transform)
        valid_input_transform.transforms.insert(0, sep_transforms.ZoomSingle(*cfg.test_shape))

        valid_set_1 = KITTIFlow(cfg.root_kitti15,
                                transform=valid_input_transform
                                )
        valid_set_2 = KITTIFlow(cfg.root_kitti12,
                                transform=valid_input_transform
                                )
        valid_set = ConcatDataset([valid_set_1, valid_set_2])
    else:
        raise NotImplementedError(cfg.type)
    return train_set, valid_set
