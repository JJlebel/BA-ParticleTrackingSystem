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


    def gg(hd):
        xx, yy = [], []
        for e in hd:
            if is_a_dictionary():
                xx.append(e['x'])
                yy.append(e['y'])
            else:
                continue
        return xx, yy
        # tracker.event_finder(tracker.dataframe)
    # tracker.testt(tracker.array)
