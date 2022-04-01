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

from ParticleTrackingSystem.video_utility import Video_Utility


class Tracker:
    def __init__(self):
        self.__frames = None
        self.array = None
        self.partikel = None
        self.video_utility = Video_Utility()

    def get__frames(self):
        return self.__frames

    def set__frames(self, frames):
        self.__frames = frames

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
