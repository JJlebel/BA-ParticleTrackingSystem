from bokeh.io import curdoc
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import layout
from bokeh.models import (Button, SingleIntervalTicker, ColumnDataSource, Slider, Label, CustomJS)
from os import listdir
from os.path import isfile, join
import pandas as pd

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
sep = source.data['Separation'].tolist()
maxsize = source.data['Maxsize'].tolist()
topn = source.data['Topn'].tolist()
engine = source.data['Engine'].tolist()

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
label = Label(x=0.2, y=3.6,
              text=f"Minmass: {str(minmass[0])}, Length: {str(length[0])}, Separation: {str(sep[0])},\n"
                   f"Maxsize: {str(maxsize[0])}, Topn: {str(topn[0])}, Engine: {str(engine[0])}"
                   f"\nMod: {str(mod[0])}", text_font_size='17px', text_color='#0521f7')
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
    label.text = f"Minmass: {str(minmass[frame])}, Length: {str(length[frame])}, Separation: {str(sep[frame])},\n" \
                 f"Maxsize: {str(maxsize[frame])}, Topn: {str(topn[frame])}, Engine: {str(engine[frame])}," \
                 f"\nMod: {str(mod[frame])}"
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
