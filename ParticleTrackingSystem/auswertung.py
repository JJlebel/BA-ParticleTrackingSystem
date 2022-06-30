# from ParticleTrackingSystem.launcher import *
from bokeh.io import curdoc
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import layout
from bokeh.models import (Button, ColumnDataSource, Slider, Label, CustomJS)
# from .data import process_data
from os import listdir
from os.path import isfile, join
from PIL import Image
import pandas as pd

# im = Image.open('./locatedImages/frame_8.png')
# im.show()

locatedImages = [f"./locatedImages/{f}" for f in listdir('./locatedImages/') if isfile(join('./locatedImages/', f))]
sorted(locatedImages, key=lambda i: i[0][-7:-4])

# output_file('image.html', title='Tracked particle')

# Add plot
p = figure(
    x_range=(0, 0.7),
    y_range=(0, 0.7),
    x_axis_label='x-coordinate',
    y_axis_label='y-coordinate',
    title='Frame55'
)

# Render glyph
p.image_url(url=[locatedImages[55]], x=-0.1, y=0.6, w=0.8, h=0.6)



# Show results
# show(p)


def animate_update():
    pass


def slider_update(attrname, old, new):
    pass


slider = Slider(start=0.5, end=1.5, value=0, step=0.2, title="Speed")
slider.on_change('value', slider_update)

callback_id = None

def animate():
    global callback_id
    if button.label == '► Play':
        button.label = '❚❚ Pause'
        callback_id = curdoc().add_periodic_callback(animate_update, 200)
    else:
        button.label = '► Play'
        curdoc().remove_periodic_callback(callback_id)


button = Button(label='► Play', width=60)
button.on_event('button_click', animate)
# button.js_on_click(animate)


layout = layout([
    [p],
    [slider, button],
])

curdoc().add_root(layout)
curdoc().title = "Gapminder"
# show(layout)