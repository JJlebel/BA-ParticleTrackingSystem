from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import os

import matplotlib.pyplot as plt

from array import *
import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience
from os import listdir, remove
from os.path import isfile, join

import pims
import trackpy as tp
import inspect

from trackpy.utils import memo

try:
    from ParticleTrackingSystem.video_utility import Video_Utility, gray
except ImportError:
    from video_utility import Video_Utility, gray

try:
    from ParticleTrackingSystem.tracker import Tracker, set_frames_number_in_array, \
        print_2d, set_empty_panda, is_a_dictionary, tp_locate  # , get_particles_per_image_as_array
except ImportError:
    from tracker import Tracker, set_frames_number_in_array, \
        print_2d, set_empty_panda, is_a_dictionary, tp_locate  # , get_particles_per_image_as_array

if __name__ == '__main__':
    video_utility = Video_Utility()
    video_utility.set_path('./BW/BW-Isil-video4.avi')
    video_utility.frames = gray(video_utility.convert_into_image_sequence())
    frames = video_utility.frames

    tracker = Tracker(5)
    tracker.set_frames(frames)
    tracker.set_minmass(210)
    tracker.set_separation(6.3)

    particle_per_frame = tracker.get_particles_per_image_as_array(frames, max_particle_percentage=100)

    set_frames_number_in_array(frames)
    tracker.arrange_array(frames, particle_per_frame)

    set_frames_number_in_array(tracker.array)

    tracker.set_particle_value_in_array(frames)

    tracker.arrange_panda(tracker.array)

    xx, yy, labels = [], [], []

    try:
        locatedImages = [f"./static/locatedImages/{f}" for f in
                         listdir('./static/locatedImages/')
                         if isfile(join('./static/locatedImages/', f))]
        sorted(locatedImages, key=lambda i: i[0][-7:-4])
    except FileNotFoundError:
        pass


    def plot_row(series):
        """
            Plots the position of a specific particle over time.
            From frame F0 ... Fn
            With the given series.

        Parameters
        ----------
        series:  pandas.core.series.Series
            the given series to with xy-coordinates to plot
        Returns
        -------
            A dict of array with all x- and y- position
        """
        if len(xx) > 0 or len(yy) > 0 or len(labels) > 0:
            xx.clear()
            yy.clear()
            labels.clear()
        for e in series:
            if is_a_dictionary(e):
                xx.append(e['x'])
                yy.append(e['y'])
            else:
                continue

        plt.plot(xx, yy, 'bo')
        plt.gca().invert_yaxis()
        plt.figure(figsize=(14, 10))
        plt.show()
        return xx, yy


    def plot_column_points(series):
        """
            Plots the position of all particles in one frame.
            With the given series.

        Parameters
        ----------
        series:  pandas.core.series.Series
            the given series to with xy-coordinates to plot
        Returns
        -------
            A dict of array with
            all x- and y- positions, labels and colors
        """
        if len(xx) > 0 or len(yy) > 0 or len(labels) > 0:
            xx.clear()
            yy.clear()
            labels.clear()
        for e in series:
            if is_a_dictionary(e):
                xx.append(e['x'])
                yy.append(e['y'])
                labels.append(e["i"])
            else:
                continue
        colors = random_color_generator(len(xx))

        plt.scatter(xx, yy, c=colors)
        plt.gca().invert_yaxis()
        plt.figure(figsize=(14, 10))
        plt.show()
        return xx, yy, labels, colors


    def random_color_generator(no_of_colors):
        """
            Generates random x colors.

        Parameters
        ----------
        no_of_colors:  int
            the given number of colors to generate
        Returns
        -------
            An array with x colors
        """
        import random
        colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for i in range(6)]) for j in range(no_of_colors)]
        return colors


    def show_rows_particle(number):
        """
            Plots the row number from the given number.

        Parameters
        ----------
        number:  int
            the row number to generate the plot from
        Returns
        -------
            Nothing
        """
        plot_row(tracker.dataframe.iloc[number])


    def show_frames_particle(number):
        """
            Plots the row number from the given number.

        Parameters
        ----------
        number:  int
            the frame number to generate the plot from
        Returns
        -------
        Nothing
        """
        plot_column_points(tracker.dataframe["F" + str(number)])


    def show_tracked_particle(f_no):
        """
            Shows the figure of all the tracked particle from the given frame number.

        Parameters
        ----------
        f_no:  int
            the frame number to show the figure from
        Returns
        -------
            figure
        """
        min = particle_per_frame[f_no]["minmass"]
        f4 = tp_locate(frames, f_no, tracker.get_diameter(), minmass=min)
        plt.figure(figsize=(14, 10))
        fig = tp.annotate(f4, frames[f_no])
        plt.show()
        return fig


    def non_nan_len(series):
        """
            Gives number of element in the given series without
            counting the NAN values

        Parameters
        ----------
        series:  pandas.core.series.Series
            the given series to count the element from
        Returns
        -------
            number of element in the series
        """
        res = 0
        for e in series:
            if is_a_dictionary(e):
                res += 1
        return res


    def save_all_frame():
        """
           Save all tracked images in a folder.
           It firstly, clean up the folder.

        Returns
        -------
        Nothing
        """
        if len(listdir("./static/locatedImages/")) > 0:
            for e in listdir('./static/locatedImages/'):
                remove(f"./static/locatedImages/{e}")
        i = 0
        for i in range(0, len(particle_per_frame)):
            if i < 10:
                name = "./static/locatedImages/frame_00" + str(i) + ".png"
            elif 10 >= i < 100:
                name = "./static/locatedImages/frame_0" + str(i) + ".png"
            else:
                name = "./static/locatedImages/frame_" + str(i) + ".png"
            r = show_tracked_particle(i)
            r.get_figure().savefig(name)
            i += 1


    def generate_output():
        """
            Generates a csv-file with several data.
            Such as path to get tracked image of each frame,
            length of each frame, as well as Minmass  and Mod
        Returns
        -------
        """
        if 'output.csv' in listdir('./static/'):
            print("remove(output.csv)")
            remove("./static/output.csv")
        for_csv = pd.DataFrame()
        locatedImages = [f"./static/locatedImages/{f}" for f in
                         listdir('./static/locatedImages/')
                         if isfile(join('./static/locatedImages/', f))]
        sorted(locatedImages, key=lambda i: i[0][-7:-4])
        length = [x["len"] for x in particle_per_frame]
        minmass = [x["minmass"] for x in particle_per_frame]
        mod = [x["Mod"] for x in particle_per_frame]
        i = 0
        h = ["Images", "Length", "Minmass", "Mod"]
        hh = [locatedImages, length, minmass, mod]
        for i in range(0, 4):
            for_csv.insert(i, h.pop(0), hh.pop(0))
            i += 1
        for_csv.to_csv('./static/output.csv', columns=["Images", "Length", "Minmass", "Mod"])


    save_all_frame()
    generate_output()

    # plot_column_points(tracker.dataframe["F66"])
    # plot_row(tracker.dataframe.iloc[0])
    # show_tracked_particle(66)
    # print(f"Array len of F66 before: {particle_per_frame[66]['len']}")
    # print(f"Dataframe len of F66 before: {non_nan_len(tracker.dataframe['F66'])}")
    # tracker.updated_frame(frames, 66, minmass=170)
    # print(f"Array len of F66 after: {particle_per_frame[66]['len']}")
    # print(f"Dataframe len of F66 after: {non_nan_len(tracker.dataframe['F66'])}")
    # show_tracked_particle(66)

from bokeh.io import curdoc
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import layout
from bokeh.models import (Button, SingleIntervalTicker, ColumnDataSource, Slider, Label, CustomJS)
from os import listdir
from os.path import isfile, join
import pandas as pd

try:
    locatedImages = [f"ParticleTrackingSystem/static/locatedImages/{f}" for f in
                     listdir('ParticleTrackingSystem/static/locatedImages/')
                     if isfile(join('ParticleTrackingSystem/static/locatedImages/', f))]
    sorted(locatedImages, key=lambda i: i[0][-7:-4])

    df = pd.read_csv('ParticleTrackingSystem/static/output.csv')
    df['Images'] = locatedImages

    # Create ColumnDataSource from data frame
    source = ColumnDataSource(df)

    # lists of differents values
    images = source.data['Images'].tolist()
    minmass = source.data['Minmass'].tolist()
    length = source.data['Length'].tolist()
    mod = source.data['Mod'].tolist()

    # Add plot
    p = figure(
        x_range=(0, 4),
        y_range=(0, 4),
        x_axis_label='x-coordinate',
        y_axis_label='y-coordinate',
        plot_width=950,
        plot_height=820,
        title='Evolution of tracked particles over time'
    )

    # Render glyph
    p.image_url(url=[images[0]], x=-0.76, y=4.11, w=5, h=4.6)

    # Show results
    label = Label(x=0.2, y=3.6, text=f"Minmass: {str(minmass[0])}, Length: {str(length[0])},\nMod: {str(mod[0])}",
                  text_font_size='17px', text_color='#0521f7')
    p.add_layout(label)

    def animate_update():
        """
            Plays what should be do when the state of the button is ► Play

        Returns
        -------
        Nothing
        """
        frame = slider.value + slider_2.value
        if frame > images.index(images[-1]):
            frame = images.index(images[0])
        slider.value = frame
        p.image_url(url=[images[frame]], x=-0.76, y=4.11, w=5, h=4.6)

    def slider_update(attr, old, new):
        """
            Updates the value of the slider when it is moved manually.

        Returns
        -------
        Nothing
        """
        frame = slider.value
        label.text = f"Minmass: {str(minmass[frame])}, Length: {str(length[frame])},\nMod: {str(mod[frame])}"
        p.image_url(url=[images[frame]], x=-0.76, y=4.11, w=5, h=4.6)
        pass

    slider = Slider(start=0, end=100, value=0, step=1, title="Frames")
    slider.on_change('value', slider_update)

    slider_2 = Slider(start=1, end=5, value=1, step=1, title="Speed (frames/second)", width=60)

    callback_id = None

    def animate():
        """
            Animates the button when it is clicked.
            And calls the appropriate function linked to the state of the button
        Returns
        -------
        Nothing
        """
        global callback_id
        if button.label == '► Play':
            button.label = '❚❚ Pause'
            callback_id = curdoc().add_periodic_callback(animate_update, 200)
        else:
            button.label = '► Play'
            curdoc().remove_periodic_callback(callback_id)


    button = Button(label='► Play', width=60)
    button.on_event('button_click', animate)

    layout = layout([
        [p],
        [button],
        [slider, slider_2],
    ])

    curdoc().add_root(layout)
    curdoc().title = "Particle visualisation"
except FileNotFoundError:
    pass
