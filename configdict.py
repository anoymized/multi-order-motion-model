from easydict import EasyDict as edict

# Note please correct the path to the dataset accordingly
config_dict = {"data": {"root_mini_diffuse": "path to your dataset",
                        "root_mini_nondiffuse": "path to your dataset",
                        # "root_sintel_slow": "/home/4TSSD/opticalflowdataset/sintel_slow",
                        # "root_davis": "/home/4TSSD/opticalflowdataset/davis2016flow/",
                        # "root_sinwave1st": "/home/4TSSD/opticalflowdataset/Sinwave1st/",
                        # "root_glass": '/home/4TSSD/opticalflowdataset/sec/glass',
                        # "root_specular": '/home/4TSSD/opticalflowdataset/sec/specular',
                        # "root_diffuse": '/home/4TSSD/opticalflowdataset/sec/dynamic',
                        # "root_diffuse_static": '/home/4TSSD/opticalflowdataset/sec/all_static',
                        # 'root_glass_dynamic': '/home/4TSSD/opticalflowdataset/sec/glass_dynamic',
                        # 'root_frosted_glass': '/home/4TSSD/opticalflowdataset/sec/frostedglass',
                        # 'root_kitti_2015': '../opticalflowdataset/kitti2015/data_scene_flow_multiview/',
                        # 'root_kitti_2012': '../opticalflowdataset/kitti2012/data_stereo_flow_multiview/',
                        "test_shape": [960, 448],  # resize the input image to this shape
                        "train_n_frames": 15,
                        "type": "NMI6_KITTI",  # '"SecondOrderHuman",
                        "val_n_frames": 15,
                        "train_shape": {"general": [768, 768],
                                        "sintel": [1024, 448],
                                        "sintel_2K": [2048, 872],
                                        "kitti": [960, 448]}
                        },
               "data_aug": {"crop": True,
                            "hflip": True,
                            'vflip': True,
                            "para_crop": [416, 768],
                            "swap": False},
               "train": {
                   "eval_first": False,
                   "print_freq": 20,
                   "record_freq": 30,
                   "save_iter": 5000,
               },
               "loss": {"weight1": 0.5,
                        "weight2": 0.5, }}


config = edict(config_dict)
