from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import os
import pims


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
        """
            Converts a video in a sequence of image.
            These images are store in a folder.
        Returns
        -------
        The folder with all the data(images) it contents.
        """
        dir_name = 'ImageSequence'
        try:
            os.mkdir(dir_name)
            print("Directory ", dir_name, " Created ")
        except FileExistsError:
            print("Directory ", dir_name, " already exists")

        if len(os.listdir("./ImageSequence")) > 0:
            for e in os.listdir('./ImageSequence/'):
                print("video_utilitie")
                os.remove(f"./ImageSequence/{e}")
        os.system("cd " + str(dir_name))
        os.system("ffmpeg -i " + self._path + " -f image2 " + dir_name + "/video-frame%05d.png")
        return pims.open('./' + dir_name + '/*.png')
