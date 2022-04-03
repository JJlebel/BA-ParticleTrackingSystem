from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import os

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
#from Cython.Utility.MemoryView import memoryview
from pandas import DataFrame, Series  # for convenience

import pims
import trackpy as tp
import inspect

# %matplotlib inline
from trackpy.utils import memo


@pims.pipeline
def gray(image):
    return image[:, :, 1]  # Take just the green channel


@pims.pipeline
def as_grey(frame):
    red = frame[:, :, 0]
    green = frame[:, :, 1]
    blue = frame[:, :, 2]
    return 0.2125 * red + 0.7154 * green + 0.0721 * blue


def convert_into_image_sequence(path):
    dirName = 'ImageSequence'
    try:
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")
    os.system("cd " + str(dirName))
    os.system("ffmpeg -i " + path + " -f image2 " + dirName + "/video-frame%05d.png")
    return pims.open('./' + dirName + '/*.png')


# TODO Do not forget to delete the content of the directory after using it.
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Hi PyCharm')
    frames = gray(convert_into_image_sequence('./BW/BW-Isil-video4.avi'))
    print('----------')
    frames
    print('Type of Frames ' + str(type(frames)))
    print('----------')
    print(frames[0])
    print('Type of Frames[0] ' + str(type(frames[0])))
    print('----------')
    plt.imshow(frames[0])
    plt.show()
    print('------##----')
    # Localise les taches de types Gaussiens d'une taille approxi. dans une image.
    #Ici dans l'image 0
    print('Data of 1st image')
    f = tp.locate(frames[0], 5, False)
    print('Type of f ' + str(type(f)))
    # print(f.head())
    # Localise les taches de types Gaussiens d'une taille approxi. dans une image.
    #Ici dans l'image 1
    print("first print")
    f = tp.locate(frames[0], 5, minmass=200.0, maxsize=None, separation=2.5, noise_size=1,
                  smoothing_size=None, threshold=None, invert=False, topn=None, preprocess=True,
                  max_iterations=10, filter_before=None, filter_after=True, characterize=True, engine='python')
    plt.figure(figsize=(14, 10))
    # tp.annotate(f, frames[0])
    print(f.head())
    ii = f.index
    print(len(ii))

    print("Second print")
    f = tp.locate(frames[0], 5, minmass=200.0, maxsize=None, separation=2, noise_size=1,
                  smoothing_size=None, threshold=None, invert=False, topn=400, preprocess=True,
                  max_iterations=10, filter_before=None, filter_after=True, characterize=True, engine='python')
    plt.figure(figsize=(14, 10))
    # tp.annotate(f, frames[0])
    print(f.head(400))

    f = tp.locate(frames[59], 5, minmass=200.0, maxsize=None, separation=2, noise_size=1,
                  smoothing_size=None, threshold=None, invert=False, topn=400, preprocess=True,
                  max_iterations=10, filter_before=None, filter_after=True, characterize=True, engine='python')
    plt.figure(figsize=(14, 10))
    tp.annotate(f, frames[20])
    print(f.head())
    print(len(f))
    # print(len(ii))
    # for i in ii:
    #     print(i)
    # f = tp.locate(frames[1], 5, minmass=200.0, maxsize=None, separation=2, noise_size=1,
    #               smoothing_size=None, threshold=None, invert=False, topn=None, preprocess=True,
    #               max_iterations=10, filter_before=None, filter_after=True, characterize=True, engine='python')
    # plt.figure(figsize=(14, 10))
    # tp.annotate(f, frames[0])
    #
    # f = tp.locate(frames[2], 5, minmass=200.0, maxsize=None, separation=2, noise_size=1,
    #               smoothing_size=None, threshold=None, invert=False, topn=None, preprocess=True,
    #               max_iterations=10, filter_before=None, filter_after=True, characterize=True, engine='python')
    # plt.figure(figsize=(14, 10))
    # tp.annotate(f, frames[0])
    #
    # f = tp.locate(frames[3], 5, minmass=200.0, maxsize=None, separation=2, noise_size=1,
    #               smoothing_size=None, threshold=None, invert=False, topn=None, preprocess=True,
    #               max_iterations=10, filter_before=None, filter_after=True, characterize=True, engine='python')
    # plt.figure(figsize=(14, 10))
    # tp.annotate(f, frames[0])
    print('Bye PyCharm')
