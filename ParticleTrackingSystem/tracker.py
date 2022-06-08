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
    return tp.locate(frames[image], 5, minmass=210, separation=6.3, engine='python')


def print_2d(array_to_print):
    for r in array_to_print:
        for c in r:
            print(c, end=" ")
        print()


def elt_decimal(number, decimal_number):
    # % 1 ensures that integer part disappears
    # return round(number % 1, decimal_number)
    return round(number, decimal_number)


def is_a_dictionary(element):
    return True if isinstance(element, dict) else False


def range_with_floats(start, stop, step):
    while stop > start:
        yield start
        start += step


def test(arr: list):
    df = pd.DataFrame()
    df.insert(0, "Part_index", pd.NA)
    index = 0
    for elt in arr:
        df.insert(index + 1, "F" + str(index), pd.NA)
        index += 1
    print(df)
    tmp = []
    for elt in arr:
        for elt_1 in elt:
            if not is_a_dictionary(elt_1):
                continue
            tmp.append(elt_1["i"])
    indexes = list(set(tmp))
    i = 0
    for lists in arr:
        for dic in lists:
            tmp = 0
            if not is_a_dictionary(dic):
                continue
            while dic["i"] != indexes[i]:
                if i < len(indexes):
                    i += 1
                else:
                    break

            while tmp < len(indexes):
                df.loc[tmp, 'Part_index'] = indexes[tmp]
                tmp += 1
        break
    return df, indexes


class Tracker:
    def __init__(self):
        self._frames = None
        self.array = None
        self.partikel = None
        self.dataframe = None
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

    def get_dataframe(self):
        return self.dataframe

    def set_dataframe(self, dataframe):
        self.dataframe = dataframe

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

    def arrange_panda(self, p_array: list):
        te = test(p_array)
        self.dataframe = te[0].copy()
        col_ind = 0
        for col_name, data in self.dataframe.items():
            if str(col_name) == "Part_index": continue
            f_ind = 0
            cell = 1
            for c in data:
                if not is_a_dictionary(c) and not isinstance(c, int):
                    try:
                        p = p_array[col_ind][cell]["i"]
                        print(
                            f"Pi (2D Array): {p} VS Part_index (Dataframe): {self.dataframe.loc[f_ind, 'Part_index']}")
                        if p == self.dataframe.at[f_ind, 'Part_index']:
                            data[f_ind] = p_array[col_ind][cell]
                        else:
                            f_ind = p
                        print(p_array[col_ind][cell])
                    except IndexError:
                        break
                if cell <= len(p_array[col_ind]) and f_ind <= len(data):
                    cell += 1
                    f_ind += 1
                    print(f"f_ind: {f_ind}  and  cell: {cell}")
                else:
                    break
            if col_ind < len(p_array):
                col_ind += 1
            else:
                break

    def event_finder(self, df):
        df = self.dataframe
        from scipy.spatial import distance
        merge, split = [], []
        eucli_dist = None
        col_ind = 0
        for col_name, data in df.items():
            if str(col_name) == "Part_index": continue
            f_ind = 0
            frame = 0
            for c in data:
                if c == pd.NA:
                    f_ind += 1
                    continue
                event = Event()
                tmp = None
                print(is_a_dictionary(c))
                print(c != data[f_ind])
                print(is_a_dictionary(data[f_ind]))
                if is_a_dictionary(c) and c != data[f_ind] and is_a_dictionary(data[f_ind]):
                    point_a = (c["x"], c["y"])
                    point_b = (data[f_ind]["x"], data[f_ind]["y"])
                    print(f"point_a: {point_a} and point_b: {point_b}")
                    eucli_dist = distance.euclidean(point_a, point_b)
                    tmp = True if elt_decimal(eucli_dist, 3) in range_with_floats(0.0, 0.055, 0.001) else False

                elif data[f_ind] == pd.NA:
                    while data[f_ind] == pd.NA and f_ind < len(data):
                        f_ind += 1
                    point_a = (c["x"], c["y"])
                    point_b = (data[f_ind]["x"], data[f_ind]["y"])
                    print(f"point_a: {point_a} and point_b: {point_b}")
                    eucli_dist = distance.euclidean(point_a, point_b)
                    f_ind += 1

                    tmp = True if elt_decimal(eucli_dist, 3) in range_with_floats(0.0, 0.055, 0.001) else False

                if tmp:
                    event.frame = frame
                    event.first_particle = c
                    event.second_particle = data[f_ind]
                    event.event_type = EventType.MERGE
                    merge.append(event)
            frame += 1
            print(merge)
            break

    def merge_event_finder(self, arr):
        from scipy.spatial import distance
        merge, split = [], []
        eucli_dist = None
        for elt in arr:
            print(elt)
            for elt_1 in elt:
                index, pointer = 0, 1
                event = Event()
                if not is_a_dictionary(elt_1):
                    continue
                print(elt_1)
                bb = elt[pointer]
                cc = len(elt)
                cb = len(elt) - 2
                while pointer < (len(elt) - 2):
                    if elt_1 == elt[pointer]:
                        pass
                    pointer += 1
                    if elt_1 != elt[pointer]:
                        point_a = (elt_1["x"], elt_1["y"])
                        point_b = (elt[pointer]["x"], elt[pointer]["y"])
                        eucli_dist = distance.euclidean(point_a, point_b)

                        tmp = True if elt_decimal(eucli_dist, 3) in range_with_floats(0.0, 0.055, 0.001) else False

                    if tmp:
                        event.frame = elt[0]
                        event.first_particle = elt_1
                        event.second_particle = elt[pointer]
                        event.event_type = EventType.MERGE
                        if not event in merge:
                            merge.append(event)
                # else:
                #     pointer += 1
            # break
        print("Merge list: (" + str(len(merge)) + ")")
        for i in merge:
            print("first_particle: " + str(i.first_particle) + " \n" + "second_particle: "
                  + str(i.second_particle) + " frame: " + str(i.frame))
        print("Merge list: (" + str(len(merge)) + ")")
