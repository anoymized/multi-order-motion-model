from easydict import EasyDict as edict

# Note please correct the path to the dataset accordingly
config_dict = {"data": {"root_sintel": "/home/4TSSD/opticalflowdataset/MPI-Sintel-complete/",
                        "root_nontex": "/home/4TSSD/opticalflowdataset/flow_nontexture/",
                        "root_sintel_slow": "/home/4TSSD/opticalflowdataset/sintel_slow",
                        "root_davis": "/home/4TSSD/opticalflowdataset/davis2016flow/",
                        "root_sinwave1st": "/home/4TSSD/opticalflowdataset/Sinwave1st/",
                        "root_sinwave2nd": "/home/4TSSD/opticalflowdataset/Sinwave2nd/",
                        "root_sec": "/home/4TSSD/opticalflowdataset/sec/",
                        "root_sintelsec": "/home/4TSSD/opticalflowdataset/sec/sintel_sec",
                        "root_sec_test": "/home/4TSSD/opticalflowdataset/sec/testbenchmark",
                        "root_glass": '/home/4TSSD/opticalflowdataset/sec/glass',
                        "root_specular": '/home/4TSSD/opticalflowdataset/sec/specular',
                        "root_denoise1": '/home/4TSSD/opticalflowdataset/sec/noisescene1',
                        "root_diffuse": '/home/4TSSD/opticalflowdataset/sec/dynamic',
                        "root_diffuse_static": '/home/4TSSD/opticalflowdataset/sec/all_static',
                        'root_glass_dynamic': '/home/4TSSD/opticalflowdataset/sec/glass_dynamic',
                        'root_frosted_glass': '/home/4TSSD/opticalflowdataset/sec/frostedglass',
                        "test_shape": [768, 768],  # resize the input image to this shape
                        "train_n_frames": 15,
                        "type": "Dual-final",  # '"SecondOrderHuman",
                        "val_n_frames": 15,
                        "train_shape": {"general": [768, 768],
                                        "sintel": [1024, 448],
                                        "sintel_2K": [2048, 872],}
                        },
               "data_aug": {"crop": True,
                            "hflip": True,
                            "vflip": True,  # "vflip": True,
                            "para_crop": [384, 512],
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
