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

try:
    from ParticleTrackingSystem.video_utility import Video_Utility, gray
except ImportError:
    from video_utility import Video_Utility, gray

try:
    from ParticleTrackingSystem.tracker import Tracker, set_frames_number_in_array, \
    print_2d, set_empty_panda, is_a_dictionary, tp_locate  # , get_particles_per_image_as_array
except ImportError:
    from tracker import Tracker, set_frames_number_in_array, \
        print_2d, set_empty_panda, is_a_dictionary, tp_locate  # , get_particles_per_image_as_array

if __name__ == '__main__':
    video_utility = Video_Utility()
    video_utility.set_path('./BW/BW-Isil-video4.avi')
    video_utility.frames = gray(video_utility.convert_into_image_sequence())
    frames = video_utility.frames

    tracker = Tracker(5)
    tracker.set_frames(frames)
    tracker.set_minmass(210)
    tracker.set_separation(6.3)

    particle_per_frame = tracker.get_particles_per_image_as_array(frames, max_particle_percentage=100)

    set_frames_number_in_array(frames)
    tracker.arrange_array(frames, particle_per_frame)

    set_frames_number_in_array(tracker.array)

    tracker.set_particle_value_in_array(frames)

    # print_2d(tracker.array)

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
        plot_column_points(tracker.dataframe["F" + str(number)])


    def show_tracked_particle(f_no):
        min = particle_per_frame[f_no]["minmass"]
        f4 = tp_locate(frames, f_no, tracker.get_diameter(), minmass=min)
        plt.figure(figsize=(14, 10))
        fig = tp.annotate(f4, frames[f_no])
        plt.show()
        return fig


    def non_nan_len(series):
        res = 0
        for e in series:
            if is_a_dictionary(e):
                res += 1
        return res


    def save_all_frame():
        i = 0
        for i in range(0, len(particle_per_frame)):
            if i < 10:
                name = "./locatedImages/frame_00" + str(i) + ".png"
            elif 10 >= i < 100:
                name = "./locatedImages/frame_0" + str(i) + ".png"
            else:
                name = "./locatedImages/frame_" + str(i) + ".png"
            r = show_tracked_particle(i)
            r.get_figure().savefig(name)
            i += 1


    plot_column_points(tracker.dataframe["F66"])
    plot_row(tracker.dataframe.iloc[0])
    show_tracked_particle(66)
    print(f"Array len of F66 before: {particle_per_frame[66]['len']}")
    print(f"Dataframe len of F66 before: {non_nan_len(tracker.dataframe['F66'])}")
    # tracker.updated_frame(frames, 66, minmass=170)
    # print(f"Array len of F66 after: {particle_per_frame[66]['len']}")
    # print(f"Dataframe len of F66 after: {non_nan_len(tracker.dataframe['F66'])}")
    # show_tracked_particle(66)


    from bokeh.io import curdoc
    from bokeh.plotting import figure, show, output_file
    from bokeh.layouts import layout
    from bokeh.models import (Button, ColumnDataSource)
    # from .data import process_data
    from os import listdir
    from os.path import isfile, join


    locatedImages = [f"./locatedImages/{f}" for f in listdir('./locatedImages/') if isfile(join('./locatedImages/', f))]
    sorted(locatedImages, key=lambda i: i[0][-7:-4])

    output_file('image.html', title='Tracked particle')


    # Add plot
    p = figure(
        x_range=(0, 0.7),
        y_range=(0, 0.7),
        x_axis_label='x-coordinate',
        y_axis_label='y-coordinate',
        title='Frame55'
    )


    # Render glyph
    p.image_url(url=[locatedImages[55]], x=-0.1, y=0.6, w=0.8, h=0.6)

    # Show results
    show(p)
