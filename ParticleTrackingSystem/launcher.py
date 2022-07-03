from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import os

import matplotlib.pyplot as plt

from array import *
import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience
from os import listdir, remove
from os.path import isfile, join

import pims
import trackpy as tp
import inspect

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

    tracker.arrange_panda(tracker.array)

    xx, yy, labels = [], [], []

    try:
        locatedImages = [f"./static/locatedImages/{f}" for f in
                         listdir('./static/locatedImages/')
                         if isfile(join('./static/locatedImages/', f))]
        sorted(locatedImages, key=lambda i: i[0][-7:-4])
    except FileNotFoundError:
        pass


    def plot_row(series):
        """
            Plots the position of a specific particle over time.
            From frame F0 ... Fn
            With the given series.

        Parameters
        ----------
        series:  pandas.core.series.Series
            the given series to with xy-coordinates to plot
        Returns
        -------
            A dict of array with all x- and y- position
        """
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
        plt.show()
        return xx, yy


    def plot_column_points(series):
        """
            Plots the position of all particles in one frame.
            With the given series.

        Parameters
        ----------
        series:  pandas.core.series.Series
            the given series to with xy-coordinates to plot
        Returns
        -------
            A dict of array with
            all x- and y- positions, labels and colors
        """
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
        """
            Generates random x colors.

        Parameters
        ----------
        no_of_colors:  int
            the given number of colors to generate
        Returns
        -------
            An array with x colors
        """
        import random
        colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for i in range(6)]) for j in range(no_of_colors)]
        return colors


    def show_rows_particle(number):
        """
            Plots the row number from the given number.

        Parameters
        ----------
        number:  int
            the row number to generate the plot from
        Returns
        -------
            Nothing
        """
        plot_row(tracker.dataframe.iloc[number])


    def show_frames_particle(number):
        """
            Plots the row number from the given number.

        Parameters
        ----------
        number:  int
            the frame number to generate the plot from
        Returns
        -------
        Nothing
        """
        plot_column_points(tracker.dataframe["F" + str(number)])

    def non_nan_len(series):
        """
            Gives number of element in the given series without
            counting the NAN values

        Parameters
        ----------
        series:  pandas.core.series.Series
            the given series to count the element from
        Returns
        -------
            number of element in the series
        """
        res = 0
        for e in series:
            if is_a_dictionary(e):
                res += 1
        return res

    tracker.save_all_frame()
    tracker.generate_output()

    # plot_column_points(tracker.dataframe["F66"])
    # plot_row(tracker.dataframe.iloc[0])
    # tracker.show_tracked_particle(66)
    # print(f"Array len of F66 before: {particle_per_frame[66]['Len']}")
    # print(f"Dataframe len of F66 before: {non_nan_len(tracker.dataframe['F66'])}")
    # tracker.updated_frame(frames, 66, minmass=170)
    # print(f"Array len of F66 after: {particle_per_frame[66]['Len']}")
    # print(f"Dataframe len of F66 after: {non_nan_len(tracker.dataframe['F66'])}")
    # tracker.show_tracked_particle(66)