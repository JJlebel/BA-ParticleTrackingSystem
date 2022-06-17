from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import os

import matplotlib.pyplot as plt

from array import *
import numpy as np
import pandas as pd
# from Cython.Utility.MemoryView import memoryview
from pandas import DataFrame, Series  # for convenience

import pims
import trackpy as tp
import inspect

# %matplotlib inline
from trackpy.utils import memo

from ParticleTrackingSystem.tracker import is_a_dictionary


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


def calculate_start_percentage(m, siz, sig):
    result = {'mass_%': 0, 'size_%': 0, 'signal_%': 0}
    # for m, siz, sig in kwargs.items():
    print(m, siz, sig)
    result['mass_%'] = (int(m) * int(m)) / 100
    result['size_%'] = (int(siz) * int(siz)) / 100
    result['signal_%'] = (int(sig) * int(sig)) / 100
    return result


# def set_start_percentage():

def print_2d(array):
    for r in array:
        for c in r:
            print(c, end=" ")
        print()


def set_frames_number_in_array(array):
    i = 0
    for n in array:
        array[i][0] = i
        i += 1


# def set_particle_in_2d_array(frames, ):
#     arr1 = []
#     print("len(tp.locate(frames[0], 5, False) " + str(len(tp.locate(frames[0], 5, False))))
#     print("len(frames) " + str(len(frames)))
#     ite = 0
#     for i in range(len(frames)):
#         col = []
#         ppf = particle_pre_frame[ite]
#         for j in range(ppf):
#             col.append(0)
#         arr1.append(col)
#         if ite < len(particle_pre_frame) - 1:
#             ite += 1
#     print("Print arr1")
#     print(arr1)
#     return arr1

# TODO Do not forget to delete the content of the directory after using it.
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Hi PyCharm')
    frames = gray(convert_into_image_sequence('./BW/BW-Isil-video4.avi'))
    print('----------')
    # plt.imshow(frames[0])
    # plt.show()

    # print('------##------')
    #
    # # Localise les taches de types Gaussiens d'une taille approxi. dans une image.
    # # Ici dans l'image 0
    # print('Data of 1st image')
    # f = tp.locate(frames[0], 5, False)
    # print('Type of f ' + str(type(f)))
    # print(f.head())
    # print(f.at[99, 'mass'])
    # print(f.columns)
    # ii = f.index
    # for i in ii:
    #     print(i)
    # print(len(f))
    # # Localise les taches de types Gaussiens d'une taille approxi. dans une image.
    # # Ici dans l'image 1
    # print('Data of 2nd image')
    # f = tp.locate(frames[1], 5, False)
    # print(f.head())
    # print(len(f))

    # Stores the number of particles per image in array
    # print("Loop is starting...")
    # particle_pre_frame = []
    # print("type of particle_pre_frame: " + str(type(particle_pre_frame)))
    # cnt = 0
    # for i in frames:
    #     f = tp.locate(frames[cnt], 5, False)
    #     print("Number of particle frame[" + str(cnt) + "]" + str(len(f)))
    #     particle_pre_frame.append(len(f))
    #     cnt += 1
    #     # break
    # print("Loop is ending...")
    mi = 210
    sep = 5+1.3
    noi = 1.1
    m_ite = 100
    top = 250
    thre = 20
    f3 = tp.locate(frames[0], 5, minmass=mi, separation=sep, threshold=thre)
    plt.figure(figsize=(14, 10))
    tp.annotate(f3, frames[0])
    # print(f3.head(5))

    # tp.subpx_bias(f3)
    plt.show()
    print(f"By threshold:{thre}  ==>  len: {len(f3)}")

    # f3 = tp.batch(frames[:99], 5, minmass=mi, separation=sep)


    # def elt_decimal(number, decimal_number):
    #     return round(number % 1, decimal_number)
    #
    # print(f"elt_decimal of 2.654321 gives => {elt_decimal(2.654321, 3)}")


    # def gg(hd):
    #     xx, yy = [], []
    #     for e in hd:
    #         if is_a_dictionary(e):
    #             xx.append(e['x'])
    #             yy.append(e['y'])
    #         else:
    #             continue
    #
    #     plt.plot(xx, xy)
    #     plt.gca().set_aspect("equal")
    #     plt.show()

    # print(frames[0])
    # f4 = tp.locate(frames[0], 11, minmass=1000.0, maxsize=None, separation=2, noise_size=1,
    # smoothing_size=None, threshold=None, invert=False, topn=300, preprocess=True,
    # max_iterations=10, filter_before=None, filter_after=True, characterize=True, engine='python')
    # plt.figure(figsize=(14, 10))
    # tp.annotate(f4, frames[0])
    # print(f4.head(5))
    #
    # tp.subpx_bias(f4)
    # plt.show()


    # f3 = tp.locate(frames[0], 11, minmass=1000.0, maxsize=None, separation=2, noise_size=1,
    #           smoothing_size=None, threshold=None, invert=False, topn=400, preprocess=True,
    #           max_iterations=10, filter_before=None, filter_after=True, characterize=True, engine='python')
    # plt.figure(figsize=(14, 10))
    # tp.annotate(f3, frames[0])
    # print(f3.head(5))

    # f7 = tp.locate(frames[7], 11, minmass=1000.0, maxsize=None, separation=2, noise_size=1,
    #             smoothing_size=None, threshold=None, invert=False, topn=400, preprocess=True,
    #             max_iterations=10, filter_before=None, filter_after=True, characterize=True, engine='python')
    # plt.figure(figsize=(14, 10))
    # tp.annotate(f7, frames[7])
    # Returns information of the first 5 founded particles(y,x,mass,size,ecc,signal,raw_mass,ep,frame)
    # print(f.head())

    # # Mark identified features with white circles and show it up.
    # tp.annotate(f, frames[0])
    # print(f.at[0, 'mass'])
    # print(f.at[99, 'mass'])
    # # Gets all attributes of the given label/index
    # print(f.loc[[0]])

    # print("Trying to add some attributes of a particle in a specific arrays position")
    # rows, cols = (5, 5)
    # arr = [[0 for i in range(cols)] for j in range(rows)]
    #
    # arr[0][1] = {'mass': f.at[0, 'mass'], 'size': f.at[0, 'size'], 'signal': f.at[0, 'signal']}
    # print(arr)
    # i = 0
    # for n in arr:
    #     arr[i][0] = i
    #     i += 1
    #
    # print_2d(arr)
    # f = tp.batch(frames[:30], 5, minmass=30, invert=True)
    # print("Batch function returns a: " + str(type(f)))

    # rows, cols = (len(frames), len(tp.locate(frames[0], 5, False)))
    # arr2 = [[0 for i in range(cols)] for j in range(rows)]
    #
    # i = 0
    # for n in arr2:
    #     arr2[i][0] = i
    #     i += 1
    # print("type of arr2: " + str(type(arr2)))
    # rows, cols = (5, 5)
    # print("After looping arrange_array(frames, particle_pre_frame) :")
    # arr1 = []
    # print("len(tp.locate(frames[0], 5, False) " + str(len(tp.locate(frames[0], 5, False))))
    # print("len(frames) " + str(len(frames)))
    # ite = 0
    # for i in range(len(frames)):
    #     col = []
    #     ppf = particle_pre_frame[ite]
    #     for j in range(ppf):
    #         col.append(0)
    #     arr1.append(col)
    #     if ite < len(particle_pre_frame) - 1:
    #         ite += 1
    # print("Print arr1")
    # print(arr1)
    #
    # set_frames_number_in_array(arr1)
    # frame_index, particle_index = 0, 1
    # for r in arr1:
    #     re = int(len(r))
    #     if frame_index in range(0, len(frames)):
    #         f = tp.locate(frames[frame_index], 5, False)
    #         for c in r:
    #             arr1[frame_index][particle_index] = {'mass': f.at[particle_index, 'mass'],
    #                                                  'size': f.at[particle_index, 'size'],
    #                                                  'signal': f.at[particle_index, 'signal']}
    #             re = int(len(r))
    #             re = int(len(r)) - particle_index
    #             if int(len(r)) - particle_index != 1:
    #                 particle_index += 1
    #         particle_index = 1
    #         re = int(len(arr1))
    #         if frame_index <= int(len(arr1)) - 1:
    #             frame_index += 1
    #         else:
    #             break
    # print_2d(arr1)
    # print("After looping set_frames_number_in_array(p_array) :")
    # i = 0
    # for n in arr1:
    #     arr1[i][0] = i
    #     i += 1
    # print("type of arr1: " + str(type(arr1)))
    # print(arr1)
    # frame_index, particle_index = 0, 1
    # for r in arr1:
    #     re = int(len(r))
    #     if frame_index in range(0, len(frames)):
    #         f = tp.locate(frames[frame_index], 5, False)
    #         for c in r:
    #             arr1[frame_index][particle_index] = {'x': f.at[particle_index, 'x'],
    #                                                  'y': f.at[particle_index, 'y']}
    #             re = int(len(r))
    #             re = int(len(r)) - particle_index
    #             if int(len(r)) - particle_index != 1:
    #                 particle_index += 1
    #         particle_index = 1
    #         re = int(len(arr1))
    #         if frame_index <= int(len(arr1)) - 1:
    #             frame_index += 1
    #         else:
    #             break
    # # print_2d(arr1)
    # df = pd.DataFrame()
    # df.insert(0, "Part_index", pd.NA)
    # index = 0
    # for elt in arr1:
    #     df.insert(index+1, "F" + str(index), pd.NA)
    #     index += 1
    # print(df)
    # indexes = []
    # for elt in arr1:
    #     for elt_1 in elt:
    #         if not is_a_dictionary(elt_1):
    #             continue
    #         indexes.append(elt_1["i"])
    print('Bye PyCharm')
