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
from ParticleTrackingSystem.tracker import Tracker, set_frames_number_in_array, tp_locate, \
    get_particles_per_image_as_array, print_2d, test, is_a_dictionary

if __name__ == '__main__':
    video_utility = Video_Utility()
    video_utility.set_path('./BW/BW-Isil-video4.avi')
    video_utility.frames = gray(video_utility.convert_into_image_sequence())
    frames = video_utility.frames

    tracker = Tracker()
    tracker.set_frames(frames)

    particle_per_frame = get_particles_per_image_as_array(frames)

    set_frames_number_in_array(frames)
    tracker.arrange_array(frames, particle_per_frame)

    set_frames_number_in_array(tracker.array)

    # print_2d(tracker.array)

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
        # for label, xi, yi in zip(labels, xx, yy):
        #     plt.annotate(label, xy=(xi, yi), textcoords='offset pixels', xytext=(xi, yi),
        #                  ha='center', va='center_baseline', arrowprops={'width': 0.01})
        # for label, xi, yi in zip(labels, xx, yy):
        #     plt.annotate(label, xy=(xi, yi), textcoords='offset pixels', xytext=(xi, yi),
        #                  ha='left', va='bottom', arrowprops={'width': 0.0001})
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
        f4 = tp_locate(frames, f_no)
        plt.figure(figsize=(14, 10))
        tp.annotate(f4, frames[f_no])
        plt.show()

    plot_column_points(tracker.dataframe["F0"])
    plot_row(tracker.dataframe.iloc[0])
    # tracker.event_finder(tracker.dataframe)
    # tracker.testt(tracker.array)
