
"""
Visualize the generated localization synthetic
data stored in h5 data-bases
"""
from __future__ import division
import os
import os.path as osp
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt 
import h5py 
from common import *
import itertools
from PIL import Image, ImageDraw
import re
import json

from scipy.io import savemat, loadmat

def concat(dest, array):
    if type(dest) == np.ndarray:
        dest = np.concatenate((dest, array), axis=0)

    else:
        dest = array
    return dest

def main():
    x = os.listdir('./')
    x = [t for t in x if os.path.splitext(t)[1] == '.mat']
    x = sorted(x)
    print(x)
    syndict = dict()
    charBB = ''
    imnames = []
    txt = []
    wordBB = []

    for i in x:
        path = os.path.join('./', i)
        gt = loadmat(path)
        charBB=concat(charBB, gt['charBB'][0])
        imnames=concat(imnames, gt['imnames'][0])
        txt=concat(txt, gt['txt'][0])
        wordBB=concat(wordBB, gt['wordBB'][0])
    print(charBB.shape)
    print(imnames.shape)
    print(txt.shape)
    print(wordBB.shape)
    syndict['charBB'] = charBB
    syndict['wordBB'] = wordBB
    syndict['imnames'] = imnames
    syndict['txt'] = txt    
    
    savemat('gt.mat', syndict)



if __name__=='__main__':
    main()
