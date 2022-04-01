from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import os
import pims

from ParticleTrackingSystem.main import convert_into_image_sequence


class Video_Utility:
    def __init__(self):
        self.path = None
        self.image = None
        self.frames = None

    @pims.pipeline
    def gray(self):
        return self.image[:, :, 1]  # Take just the green channel

    def convert_into_image_sequence(self):
        dir_name = 'ImageSequence'
        try:
            os.mkdir(dir_name)
            print("Directory ", dir_name, " Created ")
        except FileExistsError:
            print("Directory ", dir_name, " already exists")
        os.system("cd " + str(dir_name))
        os.system("ffmpeg -i " + self.path + " -f image2 " + dir_name + "/video-frame%05d.png")
        return pims.open('./' + dir_name + '/*.png')

