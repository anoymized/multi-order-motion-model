# -*- coding: utf-8 -*-
import numpy as np
import pickle
from PIL import Image
from scipy.signal import convolve2d
import os.path
import cv2

pickledPsfFilename =os.path.join(os.path.dirname( __file__),"psf.pkl")

with open(pickledPsfFilename, 'rb') as pklfile:
    psfDictionary = pickle.load(pklfile,encoding='bytes')


def PsfBlur(img, psfid):
    imgarray = np.array(img)
    imgarray = cv2.cvtColor(imgarray, cv2.COLOR_RGB2BGR)
    kernel = psfDictionary[psfid]
    convolved = cv2.filter2D(imgarray, -1, kernel)
    convolved = cv2.cvtColor(convolved, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(convolved)
    return img
    
def PsfBlur_random(img):
    psfid = np.random.randint(0, len(psfDictionary))
    return PsfBlur(img, psfid)
    
    
