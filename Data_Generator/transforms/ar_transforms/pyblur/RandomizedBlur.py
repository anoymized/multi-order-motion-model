import numpy as np
from .BoxBlur import BoxBlur_random
from .GaussianBlur import GaussianBlur_random
from .LinearMotionBlur import LinearMotionBlur_random
from .PsfBlur import PsfBlur_random
from .DefocusBlur import DefocusBlur_random


class RandomizedBlur:
    def __init__(self):
        self.blurFunctions = {"0": BoxBlur_random, "1": GaussianBlur_random, "2": LinearMotionBlur_random,
                              "3": PsfBlur_random, "4": DefocusBlur_random}

    def get_parm(self):
        blurToApply = self.blurFunctions[str(np.random.randint(0, len(self.blurFunctions)))]
        return blurToApply
