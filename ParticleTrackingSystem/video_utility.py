from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import os
import pims

# from ParticleTrackingSystem.main import convert_into_image_sequence


@pims.pipeline
def gray(image):
    return image[:, :, 1]  # Take just the green channel


class Video_Utility:
    def __init__(self):
        self._path = None
        self.image = None
        self.frames = None

    def get_path(self):
        return self._path

    def set_path(self, path):
        self._path = path

    # def get_frames(self):
    #     return self._frames
    #
    # def set_frames(self, frames):
    #     self._frames = frames

    def convert_into_image_sequence(self):
        dir_name = 'ImageSequence'
        try:
            os.mkdir(dir_name)
            print("Directory ", dir_name, " Created ")
        except FileExistsError:
            print("Directory ", dir_name, " already exists")
        os.system("cd " + str(dir_name))
        os.system("ffmpeg -i " + self._path + " -f image2 " + dir_name + "/video-frame%05d.png")
        return pims.open('./' + dir_name + '/*.png')
