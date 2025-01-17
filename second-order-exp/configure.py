import easydict


def get_config():
    config = easydict.EasyDict({
        "image_size": 1024,
        "number_of_frame": 16,
        "probe_onset_frame": 7,  # 14 to 15
        "frame_rate": 30,
        "upampling_factor": 1.5,  # we enlarge the MPI movie by 2 times
        'aperture': 300,  # radius of aperture
        # create window and stimuli,
        # On Macbook Pro 13, 1 degree of VA=50 pixels = 1/18 ratio of (900pixels high) when viewing distance is 57cm 0.6
        'an2px': 50,
        "ratM": 1,
        'ctlsize': 300,
        'NumTrials': 40,  # number of trials for each location
        "feedback": False,
        "modulation_num": 7,
        "img_root": ".\\human_static\\",
        "TrailXlsxFile": '.\\Static_AllTrails_Final_shuffled.xlsx',
        "repetition_per_trail": 1,
        "TrailBlock": 5,  # from 1 to 4
        "response_wait_time": 1,
    })
    return config
