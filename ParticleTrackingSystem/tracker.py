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

from ParticleTrackingSystem.event import Event, EventType
from ParticleTrackingSystem.video_utility import Video_Utility


def set_frames_number_in_array(p_array):
    i = 0
    for n in p_array:
        p_array[i][0] = i
        i += 1


# Stores the number of particles per image in array
def get_particles_per_image_as_array(frames):
    print("Loop is starting...")
    particle_pre_frame = []
    print("type of particle_pre_frame: " + str(type(particle_pre_frame)))
    cnt = 0
    for i in frames:
        f = tp_locate(frames, cnt)
        print("Number of particle frame[" + str(cnt) + "]: " + str(len(f)))
        particle_pre_frame.append(len(f))
        cnt += 1
    print("Loop is ending...")
    return particle_pre_frame


def tp_locate(frames, image):
    return tp.locate(frames[image], 11, minmass=1000.0, maxsize=None, separation=2, noise_size=1,
                     smoothing_size=None, threshold=None, invert=False, topn=400, preprocess=True,
                     max_iterations=10, filter_before=None, filter_after=True, characterize=True, engine='python')


def print_2d(array_to_print):
    for r in array_to_print:
        for c in r:
            print(c, end=" ")
        print()


def elt_decimal(number, decimal_number):
    return round(number % 1, decimal_number)


def is_a_dictionary(element):
    return True if isinstance(element, dict) else False


def range_with_floats(start, stop, step):
    while stop > start:
        yield start
        start += step


class Tracker:
    def __init__(self):
        self._frames = None
        self.array = None
        self.partikel = None
        self.video_utility = Video_Utility()

    def get_frames(self):
        return self._frames

    def set_frames(self, frames):
        self._frames = frames

    def get_array(self):
        return self.array

    def set_array(self, array):
        self.array = array

    def get_partikel(self):
        return self.partikel

    def set_partikel(self, partikel):
        self.partikel = partikel

    def get_video_utility(self):
        return self.video_utility

    def set_video_utility(self, video):
        self.video_utility = video

    def arrange_array(self, frames, particle_pre_frame):
        # particle_pre_frame = get_particles_per_image_as_array(frames)
        self.array = []
        ite = 0
        for i in range(len(frames)):
            col = []
            ppf = particle_pre_frame[ite]
            for j in range(ppf):
                col.append(0)
            self.array.append(col)
            if ite < len(particle_pre_frame) - 1:
                ite += 1

    def set_particle_value_in_array(self, frames):
        frame_index, particle_index = 0, 1
        for r in self.array:
            re = int(len(r))
            if frame_index in range(0, len(frames)):
                f = tp_locate(frames, frame_index)
                index = f.index
                print(index)
                for c in r:
                    jj = index[particle_index - 1]
                    self.array[frame_index][particle_index] = {'i': index[particle_index - 1],
                                                               'x': elt_decimal(f.at[index[particle_index - 1], 'x'],
                                                                                5),
                                                               'y': elt_decimal(f.at[index[particle_index - 1], 'y'],
                                                                                5)}
                    re = int(len(r))
                    re = int(len(r)) - particle_index
                    if int(len(r)) - particle_index != 1:
                        particle_index += 1
                particle_index = 1
                re = int(len(self.array))
                if frame_index <= int(len(self.array)) - 1:
                    frame_index += 1
                else:
                    break

    def testt(self, arr):
        from scipy.spatial import distance
        merge, split = [], []
        eucli_dist = None
        for elt in arr:
            print(elt)
            index, pointer = 0, 1
            for elt_1 in elt:
                event = Event()
                if not is_a_dictionary(elt_1):
                    continue
                print(elt_1)
                bb = elt[pointer]

                while elt_1 == elt[pointer] and pointer < (len(elt_1) - 2):
                    pointer += 1
                if elt_1 != elt[pointer]:
                    point_a = (elt_1["x"], elt_1["y"])
                    point_b = (elt[pointer]["x"], elt[pointer]["y"])
                    eucli_dist = distance.euclidean(point_a, point_b)

                    tmp = True if elt_decimal(eucli_dist, 1) in range_with_floats(0.0, 0.2, 0.1) else False

                    if tmp:
                        event.frame = elt[0]
                        event.first_particle = elt_1
                        event.second_particle = elt[pointer]
                        event.event_type = EventType.MERGE
                        merge.append(event)

                # else:
                #     pointer += 1
            break
        print("Merge list: (" + str(len(merge)) + ")")
        for i in merge:
            print("first_particle: " + str(i.first_particle) + " \n" + " second_particle: "
                  + str(i.second_particle) + " frame: " + str(i.frame))
        print("Merge list: (" + str(len(merge)) + ")")
