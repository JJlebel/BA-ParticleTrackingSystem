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

try:
    from ParticleTrackingSystem.event import Event, EventType
    from ParticleTrackingSystem.video_utility import Video_Utility
except:
    from event import Event, EventType
    from video_utility import Video_Utility


def set_frames_number_in_array(p_array):
    """
    Sets the length of the given array.
    Will be generally use set Tracker._frames

    :param p_array: The given array
    :return: Nothing
    """
    i = 0
    for n in p_array:
        p_array[i][0] = i
        i += 1


def tp_locate(frames, image, diameter, minmass=None,
              separation=None, maxsize=None, noise_size=1,
              smoothing_size=None, threshold=None, topn=None, preprocess=True,
              max_iterations=10, characterize=True, engine='python'):
    """
    Calls the trackpy's function locate.
    By using only specific parameters of the initial function, which are no longer deprecated.
    And also those who seems to procure any changes

    :param frames: array
        Processed image used for centroid-finding and most particle
        measurements.
    :param image:
    :param diameter: odd integer or tuple of odd integers
        This may be a single number or a tuple giving the feature's
        extent in each dimension, useful when the dimensions do not have
        equal resolution (e.g. confocal microscopy). The tuple order is the
        same as the image shape, conventionally (z, y, x) or (y, x). The
        number(s) must be odd integers. When in doubt, round up.
    :param minmass:float
        The minimum integrated brightness. This is a crucial parameter for
        eliminating spurious features.
        Recommended minimum values are 100 for integer images and 1 for float
        images. Defaults to 0 (no filtering).
        .. warning:: The mass value is changed since v0.3.0
        .. warning:: The default behaviour of minmass has changed since v0.4.0
    :param separation:float or tuple
        Minimum separation between features.
        Default is diameter + 1. May be a tuple, see diameter for details.
    :param maxsize:float
        maximum radius-of-gyration of brightness, default None
    :param noise_size:float or tuple
        Width of Gaussian blurring kernel, in pixels
        Default is 1. May be a tuple, see diameter for details.
    :param smoothing_size: float or tuple
        The size of the sides of the square kernel used in boxcar (rolling
        average) smoothing, in pixels
        Default is diameter. May be a tuple, making the kernel rectangular.
    :param threshold: float
        Clip bandpass result below this value. Thresholding is done on the
        already background-subtracted image.
        By default, 1 for integer images and 1/255 for float images.
    :param topn: integer
        Return only the N brightest features above minmass.
        If None (default), return all features above minmass.
    :param preprocess: boolean
        Set to False to turn off bandpass preprocessing.
    :param max_iterations: integer
        max number of loops to refine the center of mass, default 10
    :param characterize:boolean
        Compute "extras": eccentricity, signal, ep. True by default.
    :param engine:{'auto', 'python', 'numba'}. Default is 'python'
    :return:
    DataFrame([x, y, mass, size, ecc, signal, raw_mass])
        where "x, y" are appropriate to the dimensionality of the image,
        mass means total integrated brightness of the blob,
        size means the radius of gyration of its Gaussian-like profile,
        ecc is its eccentricity (0 is circular),
        and raw_mass is the total integrated brightness in raw_image.
    """
    if separation is None:
        separation = diameter + 1
    if smoothing_size is None:
        smoothing_size = diameter
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


def print_2d(array_to_print: list):
    """
    Prints the 2D-Array in a readable manner for human eye
    :param array_to_print:  Array
        The array to print
    :return:  Nothing
    """
    for r in array_to_print:
        for c in r:
            print(c, end=" ")
        print()


def elt_decimal(number, decimal_number):
    """Rounds up a given number to a specific (decimal_number) amount of number after the comma

    :param number: float
        Number to round up
    :param decimal_number: int
        Amount of number after the comma

    :return: float
        The rounded 'number'
    """
    return round(number, decimal_number)


def is_a_dictionary(element):
    """
        Checks if a given parameter 'element' is of type 'dict'.
    :param element: Any
        the given parameter
    :return:
        True if  :var elment is a 'dict' otherwise returns False
    """
    return True if isinstance(element, dict) else False


def range_with_floats(start, stop, step):
    """
        Return an object that produces a sequence of integers from start (inclusive) to stop (exclusive) by step.
        Whereby 'step' is adjustable

    :param start: float
        The beginning of the range(included)
    :param stop: float
        The end of the range (excluded)
    :param step: float
        The step of each skip
    :return:
        An object that produces a sequence of floats from start (inclusive) to stop (exclusive) by step.
    """
    while stop > start:
        yield start
        start += step


def set_empty_panda(arr: list):
    """
        Sets an empty Panda.Dataframe with the data from the given 'arr'
        Whereby the columns of the Panda.Dataframe will be set to the length of the 'arr'
            (e.g. length of 'arr'=100, Panda.Dataframe' columns => F0, F1,...,F99)
        And the is a special column named 'Part_index', which could be use to store index of particle.

    Parameters
    ----------
    arr: Array
        the given array

    :return: 'dict'
        A dictionary of a
        {An arranged  Panda.Dataframe(df),
        An array of indexes(indexes)}
    """
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
        self.particle_per_frame = []
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
        self.engine = new_engine

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

    def get_particle_per_frame(self):
        return self.particle_per_frame

    def set_particle_per_frame(self, particle_per_frame):
        self.particle_per_frame = particle_per_frame

    def updated_frame(self, frames, f_no, minmass=None, separation=None, maxsize=None, topn=None, engine='python'):
        """
            Updates the particle detection made on a specific image. This is done using the given parameters.
            Only parameters with values other than None will be considered. Otherwise, their values will be reset.


        Parameters
        ----------
        frames: Array
            the list of all frames
        f_no: int
            The number of the image to be modified.
        minmass: float
            The minimum integrated brightness. This is a crucial parameter for
            eliminating spurious features.
            Recommended minimum values are 100 for integer images and 1 for float
            images. Defaults to 0 (no filtering).
        separation: float or tuple
            Minimum separation between features.
            Default is diameter + 1. May be a tuple, see diameter for details.
        maxsize:float
            maximum radius-of-gyration of brightness, default None
        topn:   integer
            Return only the N brightest features above minmass.
            If None (default), return all features above minmass.
        engine: {'auto', 'python', 'numba'}

        :return: Nothting
        """
        if minmass is None:
            minmass = self.get_minmass()
        if separation is None:
            separation = self.get_separation()
        if maxsize is None:
            maxsize = self.get_maxsize()
        if topn is None:
            topn = self.get_topn()
        ppf = self.get_particle_per_frame()
        f = tp_locate(frames, f_no, self.get_diameter(), minmass=minmass, separation=separation, maxsize=maxsize,
                      topn=topn, engine=engine)
        ppf[f_no] = {"Len": len(f), "Minmass": minmass, "Mod": 0,
                     "Separation": separation, "Maxsize": maxsize,
                     "Topn": topn, "Engine": engine}
        self.set_particle_per_frame(ppf)

        self.set_particle_value_in_array(frames)
        self.arrange_panda(self.array)

    # Stores the number of particles per image in array
    def get_particles_per_image_as_array(self, frames, min_particle_percentage=85.0, max_particle_percentage=110.0):
        """
            Stores several attributes used to obtain the detection made on an image.
            These parameters are:
                len: Number of detected particles
                minmass:  The minimum integrated brightness
                mod: Number of times that parameter has been changed to stay within the desired range.
            Whereby the len will always stay in the range of
            (min_min_particle_percentage < len > max_particle_percentage).
            Due to an algorithm that modifies the values of the parameters
             until the len is in the previously given range.

        Parameters
        ----------
        frames: Array
            the sequence of image of the video
        min_particle_percentage: float
            the minimum percentage of particle that should be detected
        max_particle_percentage: float
            the maximum  percentage of particle that should be detected.
            100% is equal to the amount of particle found on the first frame
        Returns
        -------
        An array of dict.
        """
        # particle_pre_frame = []
        self.particle_per_frame.clear()
        cnt = 0
        max_size = 0
        # the lowest percentage of particles that the image should localise.
        min_particle_percentage = min_particle_percentage
        # the highest percentage of particles that the image should localise.
        max_particle_percentage = max_particle_percentage
        i_percent = 0
        new_minmass = self.minmass
        mod = 0
        for i in frames:
            f = tp_locate(frames, cnt, self.diameter, minmass=self.minmass, separation=self.separation,
                          maxsize=self.maxsize, topn=self.topn, engine=self.engine)
            if cnt == 0:
                max_size = len(f)
            if cnt > 0 and ((len(f) / max_size) * 100) <= min_particle_percentage:
                i_percent = ((len(f) / max_size) * 100)
                while i_percent <= min_particle_percentage:
                    mod += 1
                    if new_minmass > 0:
                        f = tp_locate(frames, cnt, self.diameter, minmass=new_minmass, separation=self.separation,
                                      maxsize=self.maxsize, topn=self.topn, engine=self.engine)
                        i_percent = ((len(f) / max_size) * 100)
                        if min_particle_percentage < i_percent < max_particle_percentage:
                            break
                        elif i_percent >= max_particle_percentage:
                            while i_percent >= max_particle_percentage:
                                mod += 1
                                new_minmass += 5
                                f = tp_locate(frames, cnt, self.diameter, minmass=new_minmass,
                                              separation=self.separation,
                                              maxsize=self.maxsize, topn=self.topn, engine=self.engine)
                                i_percent = ((len(f) / max_size) * 100)
                                if min_particle_percentage < i_percent < max_particle_percentage:
                                    break
                        new_minmass -= 5
                        # len(f)
                self.particle_per_frame.append({"Len": len(f), "Minmass": new_minmass, "Mod": mod,
                                                "Separation": self.separation, "Maxsize": self.maxsize,
                                                "Topn": self.topn, "Engine": self.engine})
                mod = 0

            elif cnt > 0 and ((len(f) / max_size) * 100) >= max_particle_percentage:
                i_percent = ((len(f) / max_size) * 100)
                while i_percent >= max_particle_percentage:
                    mod += 1
                    if new_minmass > 0:
                        f = tp_locate(frames, cnt, self.diameter, minmass=new_minmass, separation=self.separation,
                                      maxsize=self.maxsize, topn=self.topn, engine=self.engine)
                        i_percent = ((len(f) / max_size) * 100)
                        if min_particle_percentage < i_percent < max_particle_percentage:
                            break
                        elif i_percent <= min_particle_percentage:
                            while i_percent >= max_particle_percentage:
                                mod += 1
                                new_minmass -= 5
                                f = tp_locate(frames, cnt, self.diameter, minmass=new_minmass,
                                              separation=self.separation,
                                              maxsize=self.maxsize, topn=self.topn, engine=self.engine)
                                i_percent = ((len(f) / max_size) * 100)
                                if min_particle_percentage < i_percent < max_particle_percentage:
                                    break
                        new_minmass += 5
                        # len(f)
                self.particle_per_frame.append({"Len": len(f), "Minmass": new_minmass, "Mod": mod,
                                                "Separation": self.separation, "Maxsize": self.maxsize,
                                                "Topn": self.topn, "Engine": self.engine})
                mod = 0

            else:
                self.particle_per_frame.append({"Len": len(f), "Minmass": new_minmass, "Mod": mod,
                                                "Separation": self.separation, "Maxsize": self.maxsize,
                                                "Topn": self.topn, "Engine": self.engine})

            print("Number of particle frame[" + str(cnt) + "]: " + str(len(f)))
            cnt += 1
        return self.particle_per_frame

    def arrange_array(self, frames, particle_per_frame):
        """
            Sets up the self.array.
            For each element in particle_per_frame creates an index
            And fills it with as many zeros as particle_per_frame has len at that particular index.
        Parameters
        ----------
        frames: Array
            the sequence of image of the video
        particle_per_frame: Array
            several attributes used to obtain the detection made on an image
        Returns
        -------
        Nothing
        """
        self.array = []
        ite = 0
        for i in range(len(frames)):
            col = []
            ppf = particle_per_frame[ite]["Len"]
            for j in range(ppf):
                col.append(0)
            self.array.append(col)
            if ite < len(particle_per_frame) - 1:
                ite += 1

    def set_particle_value_in_array(self, frames):
        """
            Fills the self.array with a dict of
            {'i': index,
            'x': x-position of particle,
            'y': y-position of particle}

        Parameters
        ----------
        frames: Array
            the sequence of image of the video
        Returns
        -------
        Nothing
        """
        if self.get_array() is not None and is_a_dictionary(self.get_array()[0][1]):
            self.array.clear()
            set_frames_number_in_array(frames)
            self.arrange_array(frames, self.particle_per_frame)
            set_frames_number_in_array(self.array)
        #
        frame_index, particle_index = 0, 1
        for r in self.array:
            re = int(len(r))
            if frame_index in range(0, len(frames)):
                f = tp_locate(frames, frame_index, self.diameter, minmass=self.particle_per_frame[frame_index]["Minmass"],
                              separation=self.particle_per_frame[frame_index]["Separation"],
                              maxsize=self.particle_per_frame[frame_index]["Maxsize"],
                              topn=self.particle_per_frame[frame_index]["Topn"],
                              engine=self.particle_per_frame[frame_index]["Engine"])
                index = f.index
                # print(index)
                for c in r:
                    try:
                        jj = index[particle_index - 1]
                        self.array[frame_index][particle_index] = {'i': index[particle_index - 1],
                                                                   'x': elt_decimal(f.at[index[particle_index - 1], 'x'],
                                                                                    5),
                                                                   'y': elt_decimal(f.at[index[particle_index - 1], 'y'],
                                                                                    5)}
                    except IndexError:
                        continue
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
        """
        Fills the self.dataframe with the data of he given p_p_array.

        Parameters
        ----------
        p_array: Array
            The given array. Should be self.array
        Return
        ------
        Nothing
        """

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
                        try:
                            # takes the index of the particle within the dictionary in the initial 2D-array
                            pi_array = p_array[col_ind][cell]["i"]
                            # takes the index of the particle in the 'Part_index'-column of the dataframe
                            pi_dataframe = self.dataframe.loc[r_ind, 'Part_index']
                        except TypeError:
                            continue

                        if pi_array != pi_dataframe:
                            r_ind += 1
                            continue

                        if pi_array == pi_dataframe:
                            data[r_ind] = p_array[col_ind][cell]
                        # print(p_array[col_ind][cell])
                    except IndexError:
                        break
                if cell <= len(p_array[col_ind]) and r_ind <= len(data):
                    cell += 1
                    r_ind += 1
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
