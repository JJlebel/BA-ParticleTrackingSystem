from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import os

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
#from Cython.Utility.MemoryView import memoryview
from pandas import DataFrame, Series  # for convenience

import pims
import trackpy as tp


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
    frames = gray(convert_into_image_sequence('../BW/BW-Isil-video4.avi'))
    print('----------')
    frames
    print('Type of Frames ' + str(type(frames)))
    print('----------')
    print(frames[0])
    print('Type of Frames[0] ' + str(type(frames[0])))
    print('----------')
    plt.imshow(frames[0])
    plt.show()
    print('----------')
    # Localise les taches de types Gaussiens d'une taille approxi. dans une image.
    f = tp.locate(frames[0], 11, True)
    # Returns information of the first 5 founded particles(y,x,mass,size,ecc,signal,raw_mass,ep,frame)
    print(f.head())
    # Mark identified features with white circles.
    tp.annotate(f, frames[0])

    fig, ax = plt.subplots()
    ax.hist(f['mass'], bins=20)

    # Optionally, label the axes.
    ax.set(xlabel='mass', ylabel='count')
    plt.show()

    f = tp.locate(frames[0], 11, minmass=30, invert=True)
    tp.annotate(f, frames[0])

    # Check for subpixel accuracy
    #tp.subpx_bias(f)
    #plt.show()

    #tp.subpx_bias(tp.locate(frames[0], 15, minmass=30, invert=True))
    #plt.show()

    # f = tp.batch(frames[:2], 11, minmass=30, invert=True)
    f = tp.batch(frames[:2], 11, minmass=30, invert=True)
    print("Batch function returns a: " + str(type(f)))
    tt = pd.DataFrame(f)
    #print(tt)
    #for line in f
    df = pd.DataFrame({'c1': [10, 11, 12], 'c2': [100, 110, 120]})
    print(df)
    increase = []
    decrease = []
    for index, row in tt.iterrows():
        if index == 0:
            continue
        print("Idx: ")
        print(index, row['mass'])


    # plt.figure(figsize=(14, 10))
    plt.figure(figsize=(9, 5))
    tp.annotate(f, frames[0])
    plt.show()



    t = tp.link_df(f, search_range=5, memory=20)
    tp.plot_traj(t, label=False, superimpose=None)
    print('Bye PyCharm')
{'i': 4, 'x': 112.11347, 'y': 4.26299}
