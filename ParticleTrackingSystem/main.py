from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import os

import matplotlib.pyplot as plt
import pims


# %matplotlib inline


@pims.pipeline
def gray(image):
    return image[:, :, 1]  # Take just the green channel


@pims.pipeline
def test(image):
    return image[:, :, 1]  # Take just the green channel


def convert_into_image_sequence(path):
    dirName = 'ImageSequence'
    try:
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")
    os.system("cd " + str(dirName))
    os.system("ffmpeg -i " + path + " -f image2 " + dirName + "/video-frame%05d.png")
    return pims.open('./' + dirName + '/*.png')


# TODO Do not forget to delete the content of the directory after using it.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Hi PyCharm')
    frames = test(convert_into_image_sequence('./BW/BW-Isil-video4.avi'))
    print('----------')
    frames
    print('Type of Frames ' + str(type(frames)))
    print('----------')
    print(frames[0])
    print('Type of Frames[0] ' + str(type(frames[0])))
    print('----------')
    frames[0]
    print('----------')
    plt.imshow(frames[0])
    plt.show()
    print('----------')
    print('Bye PyCharm')
