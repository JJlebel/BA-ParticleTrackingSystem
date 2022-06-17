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
# def get_particles_per_image_as_array(frames):
# particle_pre_frame = []
# cnt = 0
#
# ci = 0
# mean = 0
# for i in frames:
#     if ci == 10:
#         mean = np.mean(particle_pre_frame[cnt-ci:cnt])
#         print(f"Mean of {cnt-ci}-{cnt}: {mean}")
#         ci = 0
#
#     f = tp_locate(frames, cnt, 5, minmass=210, separation=6.3)
#     # if len(f) < mean:
#     print("Number of particle frame[" + str(cnt) + "]: " + str(len(f)))
#     particle_pre_frame.append(len(f))
#     cnt += 1
#     ci += 1

# particle_pre_frame = []
# cnt = 0
# max_size = 0
# # the less percentage of particle that the image should localise.
# min_particle_percentage = 85.0
# i_percent = 0
# for i in frames:
#     f = tp_locate(frames, cnt, 5, minmass=210, separation=6.3)
#     if cnt == 0:
#         max_size = len(f)
#
#     if cnt > 0 and ((len(f)/max_size)*100) <= min_particle_percentage:
#         while ((len(f) / max_size) * 100) <= min_particle_percentage:
#             f = tp_locate(frames, cnt, 5, minmass=210, separation=6.3)
#
#     print("Number of particle frame[" + str(cnt) + "]: " + str(len(f)))
#     particle_pre_frame.append(len(f))
#     cnt += 1
#
# return particle_pre_frame


def tp_locate(frames, image, diameter, minmass=None,
              separation=None, maxsize=None, noise_size=1,
              smoothing_size=None, threshold=None, topn=None, preprocess=True,
              max_iterations=10, characterize=True, engine='python'):
    if separation is None:
        separation = diameter+1
    if smoothing_size is None:
        smoothing_size = diameter
    # arguments["minmass"] = None,
    # arguments["separation"] = diameter + 1,
    # # arguments["percentile"] = self.percentile,
    # arguments["maxsize"] = None,
    # arguments["noise_size"] = 1,
    # arguments["smoothing_size"] = diameter,
    # arguments["threshold"] = None,
    # arguments["topn"] = None,
    # arguments["preprocess"] = True,
    # arguments["max_iterations"] = 10,
    # # arguments["filter_before"] = self.filter_before,
    # arguments["characterize"] = True
    # arguments["engine"] = self.engine
    return tp.locate(frames[image],
                     diameter,
                     minmass=minmass,
                     separation=separation,
                     # arguments["percentile"],
                     maxsize=maxsize,
                     noise_size=noise_size,
                     smoothing_size=smoothing_size,
                     threshold=threshold,
                     topn=topn,
                     preprocess=preprocess,
                     max_iterations=max_iterations,
                     characterize=characterize,
                     engine=engine)


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


def set_empty_panda(arr: list):
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
    def __init__(self, diameter):
        self._frames = None
        self.array = None
        self.partikel = None
        self.dataframe = None
        self.video_utility = Video_Utility()
        self.particle_pre_frame = []
        self.diameter = diameter
        self.minmass = None
        self.separation = self.diameter + 1
        self.percentile = False
        self.maxsize = None
        self.noise_size = 1
        self.smoothing_size = self.diameter
        self.threshold = None
        self.topn = None
        self.preprocess = True
        self.max_iterations = 10
        # self.filter_before = None
        self.characterize = True
        # {'auto', 'python', 'numba'}
        self.engine = 'python'

    def get_diameter(self):
        return self.diameter

    def set_diameter(self, new_diameter):
        self.diameter = new_diameter

    def get_minmass(self):
        return self.minmass

    def set_minmass(self, new_minmass):
        self.minmass = new_minmass

    def get_separation(self):
        return self.separation

    def set_separation(self, new_separation):
        self.separation = new_separation

    def get_percentile(self):
        return self.percentile

    def set_percentile(self, new_percentile):
        self.percentile = new_percentile

    def get_maxsize(self):
        return self.maxsize

    def set_maxsize(self, new_maxsize):
        self.maxsize = new_maxsize

    def get_noise_size(self):
        return self.noise_size

    def set_noise_size(self, new_noise_size):
        self.noise_size = new_noise_size

    def get_smoothing_size(self):
        return self.smoothing_size

    def set_smoothing_size(self, new_smoothing_size):
        self.smoothing_size = new_smoothing_size

    def get_threshold(self):
        return self.threshold

    def set_threshold(self, new_threshold):
        self.threshold = new_threshold

    def get_topn(self):
        return self.topn

    def set_topn(self, new_topn):
        self.topn = new_topn

    def get_preprocess(self):
        return self.preprocess

    def set_preprocess(self, new_preprocess):
        self.preprocess = new_preprocess

    def get_max_iterations(self):
        return self.max_iterations

    def set_max_iterations(self, new_max_iterations):
        self.max_iterations = new_max_iterations

    # def get_filter_before(self):
    #     return self.filter_before
    #
    # def set_filter_before(self, new_filter_before):
    #     self.filter_before = new_filter_before

    def get_characterize(self):
        return self.characterize

    def set_characterize(self, new_characterize):
        self.characterize = new_characterize

    def get_engine(self):
        return self.engine

    def set_engine(self, new_engine):
        self.filter_before = new_engine

    # -------------------------------------------------------------------
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

    def get_particle_pre_frame(self):
        return self.particle_pre_frame

    def set_particle_pre_frame(self, particle_pre_frame):
        self.particle_pre_frame = particle_pre_frame

    # def tp_locate(self, frames, image, diameter, **arguments):
    #     arguments["minmass"] = self.minmass,
    #     arguments["separation"] = self.separation,
    #     arguments["percentile"] = self.percentile,
    #     arguments["maxsize"] = self.maxsize,
    #     arguments["noise_size"] = self.noise_size,
    #     arguments["smoothing_size"] = self.smoothing_size,
    #     arguments["threshold"] = self.threshold,
    #     arguments["topn"] = self.topn,
    #     arguments["preprocess"] = self.preprocess,
    #     arguments["max_iterations"] = self.max_iterations,
    #     # arguments["filter_before"] = self.filter_before,
    #     arguments["characterize"] = self.characterize,
    #     # arguments["engine"] = self.engine
    #     return tp.locate(frames[image],
    #                      diameter,
    #                      arguments["minmass"],
    #                      arguments["separation"],
    #                      arguments["percentile"],
    #                      arguments["maxsize"],
    #                      arguments["noise_size"],
    #                      arguments["smoothing_size"],
    #                      arguments["threshold"],
    #                      arguments["topn"],
    #                      arguments["preprocess"],
    #                      arguments["max_iterations"],
    #                      # arguments["filter_before"],
    #                      arguments["characterize"])#,
    #                      # arguments["engine"])

    # Stores the number of particles per image in array
    def get_particles_per_image_as_array(self, frames):
        # particle_pre_frame = []
        self.particle_pre_frame.clear()
        cnt = 0
        max_size = 0
        # the less percentage of particle that the image should localise.
        min_particle_percentage = 85.0
        max_particle_percentage = 110.0
        i_percent = 0
        new_minmass = self.minmass
        for i in frames:
            # f = tp_locate(frames, cnt, 5, minmass=210, separation=6.3)
            f = tp_locate(frames, cnt, self.diameter, minmass=self.minmass, separation=self.separation)
            if cnt == 0:
                max_size = len(f)

            if cnt > 0 and ((len(f) / max_size) * 100) <= min_particle_percentage:
                i_percent = ((len(f) / max_size) * 100)
                while i_percent <= min_particle_percentage:
                    if new_minmass > 0:
                        f = tp_locate(frames, cnt, self.diameter, minmass=new_minmass, separation=self.separation)
                        i_percent = ((len(f) / max_size) * 100)
                        if i_percent > min_particle_percentage:
                            break
                        new_minmass -= 5
                        len(f)
                self.particle_pre_frame.append({"len": len(f), "minmass": new_minmass})

            elif cnt > 0 and i_percent >= max_particle_percentage:
                while i_percent >= max_particle_percentage:
                    if new_minmass > 0:
                        new_minmass += 5
                        f = tp_locate(frames, cnt, self.diameter, minmass=new_minmass, separation=self.separation)
                        len(f)
                self.particle_pre_frame.append({"len": len(f), "minmass": new_minmass})

            else:
                self.particle_pre_frame.append({"len": len(f), "minmass": self.minmass})

            print("Number of particle frame[" + str(cnt) + "]: " + str(len(f)))
            cnt += 1
        return self.particle_pre_frame

    def arrange_array(self, frames, particle_pre_frame):
        self.array = []
        ite = 0
        for i in range(len(frames)):
            col = []
            ppf = particle_pre_frame[ite]["minmass"]
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
                f = tp_locate(frames, frame_index, 5, minmass=self.particle_pre_frame[frame_index]["minmass"], separation=self.separation)
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
        te = set_empty_panda(p_array)
        self.dataframe = te[0].copy()
        col_ind = 0
        for col_name, data in self.dataframe.items():
            if str(col_name) == "Part_index": continue
            r_ind = 0
            cell = 1
            # print(f"Type of data in col_name {type(data)}")
            for c in data:
                # print(f"Type of c in data: {type(c)}")
                if not is_a_dictionary(c) and not isinstance(c, int):
                    try:
                        # takes the index of the particle within the dictionary in the initial 2D-array
                        pi_array = p_array[col_ind][cell]["i"]
                        # takes the index of the particle in the 'Part_index'-column of the dataframe
                        pi_dataframe = self.dataframe.loc[r_ind, 'Part_index']

                        if pi_array != pi_dataframe:
                            r_ind += 1
                            continue

                        # print(f"Pi (2D Array): {pi_array} VS Part_index (Dataframe): {pi_dataframe}")

                        if pi_array == pi_dataframe:
                            data[r_ind] = p_array[col_ind][cell]
                        # else:
                        #     r_ind = pi_array
                        #     # data[r_ind] = p_array[col_ind][cell]
                        print(p_array[col_ind][cell])
                    except IndexError:
                        break
                if cell <= len(p_array[col_ind]) and r_ind <= len(data):
                    cell += 1
                    r_ind += 1
                    # print(f"r_ind: {r_ind}  and  cell: {cell}")
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
