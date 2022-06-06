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
    # Ici dans l'image 0
    print('Data of 1st image')
    f = tp.locate(frames[0], 5, False)
    print('Type of f ' + str(type(f)))
    # print(f.head())
    # Localise les taches de types Gaussiens d'une taille approxi. dans une image.
    # Ici dans l'image 1
    # print("first print")
    # f = tp.locate(frames[0], 5, minmass=200.0, maxsize=None, separation=2.5, noise_size=1,
    #               smoothing_size=None, threshold=None, invert=False, topn=None, preprocess=True,
    #               max_iterations=10, filter_before=None, filter_after=True, characterize=True, engine='python')
    # plt.figure(figsize=(14, 10))
    # # tp.annotate(f, frames[0])
    # print(f.head())
    # ii = f.index
    # print(ii[0])

    # print("Second print")
    # f = tp.locate(frames[1], 11, minmass=300.0, maxsize=2.0, separation=3, noise_size=1,
    #               smoothing_size=None, threshold=None, invert=False, topn=400, preprocess=True,
    #               max_iterations=10, filter_before=None, filter_after=True, characterize=True, engine='python')
    # plt.figure(figsize=(14, 10))
    # tp.annotate(f, frames[0])
    # t = tp.subpx_bias(f)
    # print(len(f))
    # print(t)
    # plt.savefig('foo.png')
    # plt.show()
    # print(len(f))
    # # ii = f.index
    # # print(ii[0])
    # print(f.head(100))
    # x = [0.99363,0.41047,0.65067,0.54331,0.23312,0.80303,0.02905,0.84578,0.64103,0.46270,0.75459,0.68338,0.24247,0.99618,0.76862,0.19102,0.84525,0.47395,0.57941,0.89530,0.59659,0.44935,0.43946,0.44527,0.64598,0.17049,0.64024,0.83191,0.98111,0.24522,0.85168,0.38578,0.49868,0.68000,0.39526,0.69292,0.60484,0.44439,0.00461,0.08581,0.42577,0.13393,0.98284,0.37304,0.92926,0.82763,0.94699,0.69952,0.14646,0.56987,0.61160,0.53558,0.09826,0.31014,0.38407,0.77849,0.85794,0.19359,0.17439,0.26331,0.44196,0.42843,0.39108,0.59850,0.21330,0.75980,0.49547,0.35178,0.03157,0.02901,0.53292,0.89606,0.12503,0.03883,0.65770,0.00365,0.39500,0.68707,0.15606,0.85479,0.21169,0.07439,0.98063,0.76332,0.77363,0.16435,0.99306,0.58536,0.09274,0.33778,0.46487,0.33810,0.06394,0.91166,0.47199,0.57233,0.32608,0.72178,0.18631,0.73730,0.49598,0.37634,0.72967,0.59536,0.90051,0.82526,0.81868,0.93010,0.87363,0.68859,0.75573,0.74362,0.41131,0.50486,0.61884,0.89635,0.57701,0.84879,0.72946,0.91770,0.31854,0.35712,0.43648,0.49091,0.45550,0.10998,0.18496,0.03819,0.76411,0.25216,0.73102,0.63453,0.12479,0.97855,0.95814,0.79340,0.99463,0.14677,0.30904,0.50000,0.62320,0.87398,0.84275,0.11115,0.15578,0.01905,0.27792,0.88091,0.23281,0.47424,0.79263,0.93532,0.00313,0.75544,0.99622,0.65932,0.76192,0.82716,0.59214,0.50751,0.40206,0.21265,0.65837,0.01024,0.44150,0.23275,0.92686,0.10342,0.23949,0.41123,0.65388,0.87285,0.94021,0.46427,0.85907,0.02993,0.15615,0.47468,0.84233,0.82262,0.90892,0.11885,0.86246,0.40712,0.89652,0.34197,0.11111,0.84904,0.13589,0.80368,0.08727,0.96457,0.77447,0.94978,0.52791,0.46995,0.37083,0.27593,0.17320,0.89832,0.08491,0.11968,0.56516,0.50798,0.18225,0.28848,0.88402,0.46349,0.64420,0.89318,0.57147,0.08504,0.63687,0.91757,0.06355,0.15040,0.25213,0.90674,0.47083,0.92900,0.70842,0.79394,0.09667,0.77553,0.83696,0.97905,0.44106,0.17357,0.29898,0.17669,0.64359,0.15712,0.93589,0.14498,0.03343,0.18313,0.42626,0.30644,0.36081,0.73647,0.87912,0.55766,0.29069,0.31777,0.46729,0.56141,0.76987,0.58058,0.44630,0.14514,0.34896,0.23638,0.03942,0.78049,0.22657,0.65872,0.72489,0.21253,0.20305,0.87577,0.37536,0.19388,0.42431,0.25392,0.22388,0.48593,0.69671,0.62173,0.82354,0.41278,0.51993,0.15496,0.95347,0.02151,0.87378,0.81640,0.17243,0.96386,0.71742,0.77138,0.81851,0.76803,0.87091,0.89647,0.81583,0.05027,0.63766,0.44303,0.75668,0.92169,0.52198,0.77663,0.13981,0.13618,0.58682,0.78891,0.91296,0.84968,0.83961,0.26883,0.90253,0.13222,0.48778,0.97809,0.70022,0.71293,0.91091,0.44884,0.63736,0.60103,0.05194,0.63377,0.39355,0.72470,0.09995,0.70667,0.69519,0.21061,0.40310,0.83493,0.89571,0.79668,0.14398,0.11947,0.36944,0.12866,0.26492,0.42923,0.36703,0.20848,0.26572,0.00673,0.60691,0.39795,0.47964,0.74463,0.29698,0.70182,0.60491,0.73438,0.90654,0.57517,0.76598,0.91208,0.01484,0.04211,0.06120,0.54987,0.02783,0.33749,0.43939,0.92068,0.48547,0.59332,0.54326,0.87954,0.68217,0.87978,0.52597,0.74824,0.44326,0.06329,0.92625,0.03748,0.89965,0.24239,0.55082,0.82181,0.06716,0.42396,0.94786,0.63904,0.53919,0.26017,0.15059,0.76122,0.54075,0.75823,0.77573,0.98111,0.78133,0.83593,0.65166,0.02214,0.93431,0.57552,0.84134,0.86506,0.28616,0.12804,0.89805,0.75097,0.17568,0.38061,0.72636,0.72401,0.51279,0.46681,0.68366,0.28325]
    # y = [0.43714,0.34844,0.21689,0.34459,0.99686,0.47767,0.37500,0.42309,0.12536,0.59675,0.05999,0.09679,0.64234,0.56775,0.35346,0.59995,0.61958,0.40174,0.74391,0.09004,0.44030,0.77480,0.50996,0.95414,0.21322,0.73268,0.91775,0.58482,0.57664,0.48662,0.75819,0.42052,0.58628,0.59886,0.43534,0.99936,0.18619,0.86004,0.83634,0.44413,0.09758,0.77009,0.46790,0.10668,0.63175,0.00047,0.15711,0.70673,0.31225,0.52430,0.79968,0.51433,0.27824,0.94527,0.81839,0.52111,0.68547,0.87657,0.53736,0.19876,0.22879,0.79340,0.89895,0.57941,0.45983,0.41759,0.51039,0.77322,0.99921,0.93659,0.28997,0.44014,0.69008,0.08239,0.51292,0.50243,0.52179,0.47062,0.70510,0.19205,0.75818,0.08584,0.45771,0.52541,0.18593,0.71663,0.76467,0.76360,0.87298,0.13481,0.37214,0.85238,0.10585,0.56351,0.76650,0.05758,0.07457,0.03838,0.05278,0.48694,0.43226,0.83602,0.54306,0.91175,0.35765,0.76407,0.19211,0.69490,0.94856,0.02754,0.62198,0.81522,0.95349,0.84529,0.30648,0.28335,0.61655,0.18775,0.39261,0.76955,0.17885,0.76283,0.20686,0.41727,0.54377,0.98500,0.54439,0.28784,0.50712,0.08193,0.30033,0.68317,0.97876,0.92219,0.14043,0.57173,0.73792,0.66395,0.79622,0.64132,0.09004,0.21735,0.51424,0.32189,0.68425,0.16809,0.49062,0.41743,0.88905,0.84269,0.15932,0.83810,0.76475,0.79311,0.02525,0.03555,0.90381,0.20939,0.12635,0.56361,0.05777,0.54248,0.42409,0.85847,0.61637,0.47722,0.40093,0.06924,0.44124,0.80915,0.53039,0.78298,0.57667,0.58933,0.65154,0.27372,0.47029,0.07495,0.08569,0.87047,0.77891,0.49044,0.27744,0.94772,0.73027,0.92024,0.63363,0.71310,0.51295,0.48482,0.96412,0.50207,0.74716,0.02930,0.75557,0.00227,0.37873,0.25486,0.88706,0.04551,0.98282,0.14932,0.58853,0.46353,0.76275,0.20623,0.03936,0.47915,0.99096,0.18398,0.42853,0.60928,0.27573,0.45510,0.50467,0.99208,0.94310,0.14754,0.59122,0.67469,0.86271,0.58020,0.39501,0.67495,0.92165,0.63973,0.46791,0.26244,0.46054,0.18662,0.86138,0.11734,0.21762,0.52228,0.44783,0.19946,0.63965,0.44873,0.50580,0.35946,0.33040,0.52646,0.41604,0.36625,0.45219,0.30093,0.77244,0.17817,0.60704,0.80395,0.59608,0.77887,0.01484,0.60011,0.74965,0.02096,0.53130,0.73937,0.01262,0.12947,0.03863,0.23081,0.38910,0.53112,0.40620,0.52709,0.78394,0.07709,0.48068,0.62239,0.48516,0.32238,0.34333,0.82733,0.17310,0.29792,0.97604,0.68675,0.50877,0.67496,0.93882,0.57680,0.98878,0.81486,0.50919,0.04093,0.06444,0.68571,0.52510,0.45748,0.19985,0.59699,0.79202,0.27886,0.11048,0.84648,0.06086,0.12471,0.33864,0.56021,0.88362,0.23278,0.79435,0.27364,0.03701,0.66160,0.42164,0.52326,0.26548,0.97527,0.55414,0.29132,0.89977,0.61086,0.90262,0.96437,0.90252,0.05701,0.59845,0.09280,0.88140,0.38947,0.01642,0.54408,0.38175,0.10047,0.01622,0.79350,0.52722,0.39061,0.42033,0.73545,0.38871,0.86793,0.66968,0.08565,0.46601,0.00655,0.24688,0.36890,0.20136,0.43265,0.08845,0.83925,0.10509,0.16080,0.05469,0.09247,0.01167,0.85354,0.07055,0.94824,0.95025,0.68867,0.74610,0.43861,0.63738,0.40165,0.06192,0.24648,0.30531,0.32595,0.16633,0.02699,0.36801,0.61923,0.60136,0.47192,0.67872,0.45797,0.68827,0.50823,0.26188,0.82586,0.09137,0.26107,0.68966,0.82249,0.74308,0.72459,0.36489,0.78771,0.29404,0.84978,0.24255,0.45212,0.39710,0.48780,0.70182,0.58434,0.47331,0.80267,0.25352,0.25801,0.07320,0.11700,0.80247,0.49544,0.14464,0.86802]
    # plt.hist(x)
    # plt.show()
    #
    # col_x_list = f['x'].tolist()
    # col_x_list[0] = round(col_x_list[0] % 1, 5)
    # print(col_x_list[0])

    def plot_hist():
        col_x_list, col_y_list = [], []


        def elt_decimal(number):
            return round(number % 1, 5)


        for i in f['x'].tolist():
            col_x_list.append(elt_decimal(i))

        for i in f['y'].tolist():
            col_y_list.append(elt_decimal(i))

        plt.hist(col_x_list)
        plt.show()
        plt.hist(col_y_list)
        plt.show()

    # f = tp.locate(frames[59], 5, minmass=200.0, maxsize=None, separation=2, noise_size=1,
    #               smoothing_size=None, threshold=None, invert=False, topn=400, preprocess=True,
    #               max_iterations=10, filter_before=None, filter_after=True, characterize=True, engine='python')
    # plt.figure(figsize=(14, 10))
    # tp.annotate(f, frames[20])
    # print(f.head())
    # print(len(f))

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

    # print("sixteenth print")
    # f = tp.locate(frames[45], 11, minmass=200.0, maxsize=None, separation=3, noise_size=1,
    #               smoothing_size=None, threshold=float(5), invert=False, topn=400, preprocess=True,
    #               max_iterations=10, filter_before=None, filter_after=True, characterize=True, engine='python')
    # plt.figure(figsize=(14, 10))
    # tp.annotate(f, frames[0])
    # t = tp.subpx_bias(f)
    # print(len(f))
    # print(t)
    # plt.show()
    # print(len(f))
    # print(f.head(3))
    # print(f)

    print("Second print")
    f = tp.locate(frames[1], 11, minmass=2400.0, maxsize=3.3, separation=3, noise_size=1,
                  smoothing_size=None, threshold=None, invert=False, topn=400, preprocess=True,
                  max_iterations=10, filter_before=None, filter_after=True, characterize=True, engine='python')
    plt.figure(figsize=(14, 10))
    tp.annotate(f, frames[0])
    t = tp.subpx_bias(f)
    print(len(f))
    print(t)
    plt.savefig('foo.png')
    plt.show()
    print(len(f))
    # ii = f.index
    # print(ii[0])
    print(f.head(100))
    new = f['mass'].copy()
    new.tolist()
    print(type(new))
    hh = 0
    for i in new:
        hh += i
    print(hh)
    hh /= len(new)
    print(hh)

    from scipy.spatial import distance

    point_a = (131.852081, 12.565719)
    point_b = (127.343480, 11.030440)
    print("distance.euclidean: " )
    print(distance.euclidean(point_a, point_b))
    print('Bye PyCharm')
