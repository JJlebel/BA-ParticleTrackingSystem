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

from ParticleTrackingSystem.video_utility import Video_Utility, gray
from ParticleTrackingSystem.tracker import Tracker, set_frames_number_in_array,\
    print_2d, set_empty_panda, is_a_dictionary, tp_locate#, get_particles_per_image_as_array

if __name__ == '__main__':
    video_utility = Video_Utility()
    video_utility.set_path('./BW/BW-Isil-video4.avi')
    video_utility.frames = gray(video_utility.convert_into_image_sequence())
    frames = video_utility.frames

    tracker = Tracker(5)
    tracker.set_frames(frames)
    tracker.set_minmass(210)
    tracker.set_separation(6.3)

    particle_per_frame = tracker.get_particles_per_image_as_array(frames)

    set_frames_number_in_array(frames)
    tracker.arrange_array(frames, particle_per_frame)

    set_frames_number_in_array(tracker.array)

    tracker.set_particle_value_in_array(frames)

    print_2d(tracker.array)

    tracker.arrange_panda(tracker.array)

    xx, yy, labels = [], [], []


    def plot_row(series):
        if len(xx) > 0 or len(yy) > 0 or len(labels) > 0:
            xx.clear()
            yy.clear()
            labels.clear()
        for e in series:
            if is_a_dictionary(e):
                xx.append(e['x'])
                yy.append(e['y'])
            else:
                continue

        plt.plot(xx, yy, 'bo')
        plt.gca().invert_yaxis()
        plt.figure(figsize=(14, 10))
        # plt.gca().set_aspect("equal")
        plt.show()
        return xx, yy


    def plot_column_points(series):
        if len(xx) > 0 or len(yy) > 0 or len(labels) > 0:
            xx.clear()
            yy.clear()
            labels.clear()
        for e in series:
            if is_a_dictionary(e):
                xx.append(e['x'])
                yy.append(e['y'])
                labels.append(e["i"])
            else:
                continue
        colors = random_color_generator(len(xx))

        plt.scatter(xx, yy, c=colors)
        plt.gca().invert_yaxis()
        plt.figure(figsize=(14, 10))
        plt.show()
        return xx, yy, labels, colors


    def random_color_generator(no_of_colors):
        import random
        colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for i in range(6)]) for j in range(no_of_colors)]
        return colors

    def show_rows_particle(number):
        plot_row(tracker.dataframe.iloc[number])

    def show_frames_particle(number):
        plot_column_points(tracker.dataframe["F"+str(number)])

    def show_tracked_particle(f_no):
        f4 = tp_locate(frames, f_no, 5)
        plt.figure(figsize=(14, 10))
        tp.annotate(f4, frames[f_no])
        plt.show()

    def non_nan_len(series):
        res = 0
        for e in series:
            if is_a_dictionary(e):
                res += 1
        return res

    plot_column_points(tracker.dataframe["F66"])
    plot_row(tracker.dataframe.iloc[0])
    show_tracked_particle(66)
