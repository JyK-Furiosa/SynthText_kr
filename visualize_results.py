
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

# initialize index
num = 1
char_ind = 0
word_ind = 6473388

def viz_textbb(text_im, charBB_list, wordBB, index, txt, alpha=1.0):
    """
    text_im : image containing text
    charBB_list : list of 2x4xn_i bounding-box matrices
    wordBB : 2x4xm matrix of word coordinates
    """
    plt.close(1)
    plt.figure(1)
    plt.imshow(text_im)
    img = Image.fromarray(text_im, 'RGB')
    img.save('images/'+index+'.jpg')
    # plt.hold(True)
    H,W = text_im.shape[:2]
    print(wordBB.shape)
    char = ''
    text=[]
    words = ''
    imgs  = []
    # plot the word-BB:
    tot = 0
    # plot the word-BB:
    import re
    
    txt =[re.split(' \n|\n|\n ', t.strip()) for t in txt]
    txt = list(itertools.chain(*txt))
    txt2 =[t.rstrip().lstrip() for t in txt]
    print(txt)
    print(len(txt))
    imgs.append('images/'+index+'.jpg')
    # print(charBB_list.shape)
    # print(txt2)
    for i in range(wordBB.shape[-1]):
        bb = wordBB[:,:,i]
        # bb = np.c_[bb,bb[:,0]]
        if np.any(abs(bb[0,:]) > 2*W) or np.any(abs(bb[1,:]) > 2*H):
            tot += len(txt2[i])
            continue

        bbs = bb.reshape(bb.shape[0],bb.shape[1],1) 
        if type(words) != np.ndarray:
            words = bbs
        else:
            words = np.concatenate([words, bbs], axis=2)

        text.append(txt[i])
        if type(char) != np.ndarray:
            char = charBB_list[:,:,tot:tot+len(txt2[i])]
        else:
            char = np.concatenate([char, charBB_list[:,:,tot:tot+len(txt2[i])]], axis=2)
        
        tot += len(txt2[i])
        # print(tot)
        text_ = txt[i]
        plt.plot(bb[0,:], bb[1,:], 'g', alpha=alpha)
        coords(bb[0,:], bb[1,:], index, text_im, "word", text_)

        # visualize the indiv vertices:
        vcol = ['r','g','b','k']
        for j in range(4):
            plt.scatter(bb[0,j],bb[1,j],color=vcol[j]) 


    # plot the character-BB:
    # bbs = charBB_list
    # ni = bbs.shape[-1]

    # for j in range(ni):
    #     bb = bbs[:,:,j]
    #     bb = np.c_[bb,bb[:,0]]
    #     plt.plot(bb[0,:], bb[1,:], 'r', alpha=alpha/2)
    #     coords(bb[0,:], bb[1,:], index, text_im, "char")

    return np.array(text), words, char, np.array(imgs)
# generate image files and coordinate files
def coords(x_list, y_list, k, img, typ, txt=None):
    tmp = []
    global char_ind
    global word_ind
    
    # remove alpha
    x_list=x_list[:-1]
    y_list=y_list[:-1] 

    # change datatype
    x_list=[int(x) for x in x_list]
    y_list=[int(y) for y in y_list]

    # generate coords list
    tmp=list(zip(x_list, y_list))

    # coords to image file
    k=k.split('.')[0]
    if typ == "char":
        char_ind = char_ind + 1
        _tojpg(img, tmp, char_ind, k, typ)
    elif typ == "word":
        word_ind = word_ind + 1
        _tojpg(img, tmp, word_ind, k, typ, txt)
 

#  split words and create jpg files
def _tojpg(text_im, polygon, index, k, typ, txt=None):
    img = Image.fromarray(text_im, 'RGB')
    # img.save('images/'+k+'.jpg')
    

    # convert to numpy (for convenience)
    imArray = np.asarray(img)

    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)

    ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
    mask = np.array(maskIm)

    # assemble new image (uint8: 0-255)
    newImArray = np.empty(imArray.shape,dtype='uint8')

    # colors (three first columns, RGB)
    newImArray[:,:,:3] = imArray[:,:,:3]

    # back to Image from numpy
    newIm = Image.fromarray(newImArray, "RGB")
    x_min = min([x[0] for x in polygon])
    x_max = max([x[0] for x in polygon])
    y_min = min([x[1] for x in polygon])
    y_max = max([x[1] for x in polygon])

    
    newIm = newIm.crop((x_min, y_min, x_max, y_max))

    if typ == "char":
        newIm.save("characters/"+typ+"_"+str(index)+".jpg")
        f1 = open('ch_coords.txt', 'a')
        f1.write(typ+str(index)+".jpg"+"\t"+str(x_min)+","+str(y_min)+","+str(x_max)+","+str(y_max)+"\n")

    elif typ == "word":
        newIm.save("words/"+typ+"_"+str(index)+".jpg")
        f2 = open('wd_coords.txt', 'a')
        f2.write(typ+str(index)+".jpg"+"\t"+str(x_min)+","+str(y_min)+","+str(x_max)+","+str(y_max)+"\n")
        f3 = open('wd_gt.txt', 'a')
        f3.write(typ+str(index)+".jpg"+"\t"+str(txt)+"\n")


def main(db_fname):
    db = h5py.File(db_fname, 'r')
    dsets = sorted(db['data'].keys())
    print( "total number of images : ", colorize(Color.RED, len(dsets), highlight=True))
    imnames=[]
    txt_=[]
    charBB_=np.empty(len(dsets), dtype=object)
    wordBB_=np.empty(len(dsets), dtype=object)
    for i, k in enumerate(dsets):
        print(k)
        rgb = db['data'][k][...]
        charBB = db['data'][k].attrs['charBB']
        wordBB = db['data'][k].attrs['wordBB']
        txt = db['data'][k].attrs['txt']

        text, words, chars, imgs = viz_textbb(rgb, charBB, wordBB, index=k, txt=txt)
        print("image name        : ", colorize(Color.RED, imgs, bold=True))
        print("  ** no. of chars : ", colorize(Color.YELLOW, chars.shape[-1]))
        print("  ** no. of words : ", colorize(Color.YELLOW, words.shape[-1]))
        print("  ** text         : ", colorize(Color.GREEN, text))
        # print "  ** text         : ", colorize(Color.GREEN, txt.encode('utf-8'))
        imnames.append(imgs)
        txt_.append(text)
        charBB_[i]=chars
        wordBB_[i]=words
        # if 'q' in raw_input("next? ('q' to exit) : "):
            # break
    db.close()
    dict_  = {}
    dict_['charBB']=np.array([charBB_], dtype=object)
    dict_['wordBB']=np.array([wordBB_], dtype=object)
    dict_['imnames']=np.array([imnames],dtype=object)
    dict_['txt']=np.array([txt_], dtype=object)
    from scipy.io import savemat
    savemat('gt4.mat', dict_)
if __name__=='__main__':
    main('gen/dset_kr.h5')
