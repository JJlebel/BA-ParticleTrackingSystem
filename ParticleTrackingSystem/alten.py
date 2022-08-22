from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import os

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
# from Cython.Utility.MemoryView import memoryview
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
    tgfg = pims.PyAVReaderIndexed("./BW/BW-Isil-video4.avi")
    frames = gray(convert_into_image_sequence('./BW/BW-Isil-video4.avi'))
    # Localise les taches de types Gaussiens d'une taille approxi. dans une image.
    # Ici dans l'image 0
    print('Data of 1st image')
    f = tp.locate(frames[0], 5, False)
    print('Type of f ' + str(type(f)))

    print("Second print")
    # f = tp.locate(frames[1], 11, minmass=2400.0, maxsize=3.3, separation=3, noise_size=1,
    #               smoothing_size=None, threshold=None, invert=False, topn=400, preprocess=True,
    #               max_iterations=10, filter_before=None, filter_after=True, characterize=True, engine='python')
    f = tp.locate(frames[62], 5, minmass=210.0, separation=6.3, engine='python')
    plt.figure(figsize=(14, 10))
    tp.annotate(f, frames[62])
    t = tp.subpx_bias(f)
    print(len(f))
    print(t)
    plt.savefig('foo.png')
    plt.show()
    print(len(f))
    # # ii = f.index
    # # print(ii[0])
    # print(f.head(100))
    # new = f['mass'].copy()
    # new.tolist()
    # print(type(new))
    # hh = 0
    # for i in new:
    #     hh += i
    # print(hh)
    # hh /= len(new)
    # print(hh)
    #
    # from scipy.spatial import distance
    #
    # point_a = (131.852081, 12.565719)
    # point_b = (127.343480, 11.030440)
    # print("distance.euclidean: " )
    # print(distance.euclidean(point_a, point_b))
    print('Bye PyCharm')
