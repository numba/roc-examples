from __future__ import print_function

from bokeh.browserlib import view
from bokeh.document import Document
from bokeh.embed import file_html
from bokeh.models.glyphs import Circle, ImageRGBA
from bokeh.models import (
    GMapPlot, Range1d, ColumnDataSource,
    PanTool, WheelZoomTool, BoxSelectTool,
    BoxZoomTool, GMapOptions)
from bokeh.resources import INLINE
from bokeh.io import curdoc

import numpy as np
import logging

logging.basicConfig(level=logging.ERROR)


def create_map_plot():
    x_range = Range1d()
    y_range = Range1d()

    # JSON style string taken from: https://snazzymaps.com/style/1/pale-dawn
    map_options = GMapOptions(lat=30.2861, lng=-97.7394, map_type="roadmap", zoom=10, styles="""
    [{"featureType":"administrative","elementType":"all","stylers":[{"visibility":"on"},{"lightness":33}]},{"featureType":"landscape","elementType":"all","stylers":[{"color":"#f2e5d4"}]},{"featureType":"poi.park","elementType":"geometry","stylers":[{"color":"#c5dac6"}]},{"featureType":"poi.park","elementType":"labels","stylers":[{"visibility":"on"},{"lightness":20}]},{"featureType":"road","elementType":"all","stylers":[{"lightness":20}]},{"featureType":"road.highway","elementType":"geometry","stylers":[{"color":"#c5c6c6"}]},{"featureType":"road.arterial","elementType":"geometry","stylers":[{"color":"#e4d7c6"}]},{"featureType":"road.local","elementType":"geometry","stylers":[{"color":"#fbfaf7"}]},{"featureType":"water","elementType":"all","stylers":[{"visibility":"on"},{"color":"#acbcc9"}]}]
    """)

    plot = GMapPlot(
        x_range=x_range, y_range=y_range,
        map_options=map_options,
        title="Austin"
    )

    source = ColumnDataSource(
        data=dict(
            lat=[30.2861, 30.2855, 30.2869],
            lon=[-97.7394, -97.7390, -97.7405],
            fill=['orange', 'blue', 'green']
        )
    )

    data_points = draw_data(plot, source)


    pan = PanTool()
    wheel_zoom = WheelZoomTool()
    box_select = BoxSelectTool()

    ## XXX: BoxZoomTool doesn't work?
    # box_zoom = BoxZoomTool()
    # tools = [pan, wheel_zoom, box_select, box_zoom]
    tools = [pan, wheel_zoom, box_select]
    plot.add_tools(*tools)

    source.on_change('selected', SourceChange(plot, data_points).on_change)
    ## XXX: Can I listen to change of view port (zoom, x-range, y-range)?
    return plot

def draw_data(plot, source):
    circle = Circle(x="lon", y="lat", size=15, fill_color="fill", line_color="black")
    plot.add_glyph(source, circle)
    return circle

def draw_density(plot):
    x_range = plot.x_range
    y_range = plot.y_range
    print(x_range.start, x_range.end)
    print(y_range.start, y_range.end)
    #
    # # create an array of RGBA data
    # N = 20
    # img = np.empty((N, N), dtype=np.uint32)
    # view = img.view(dtype=np.uint8).reshape((N, N, 4))
    # for i in range(N):
    #     for j in range(N):
    #         view[i, j, 0] = int(255 * i / N)
    #         view[i, j, 1] = 158
    #         view[i, j, 2] = int(255 * j / N)
    #         view[i, j, 3] = 255
    #
    # x = x_range.start
    # y = y_range.start
    # dw = x_range.end - x
    # dh = y_range.end - y
    # image_rgba = ImageRGBA(image=[1, 2, 3], x=x, y=y, dw=dw, dh=dh)
    # plot.add_glyph(image_rgba)


def on_change_range(attr, old, new):
    print("on_change_range")
    print("Old".center(80, '-'))
    print(old)
    print("New".center(80, '-'))
    print(new)


class SourceChange(object):
    def __init__(self, plot, data_points):
        self.plot = plot
        self.data_points = data_points

    def on_change(self, attr, old, new):
        print("on_change_source", attr)
        # Sample `old` and `new`
        # {'2d': {'indices': []}, '0d': {'indices': [], 'flag': False}, '1d': {'indices': [0, 1, 2]}}
        # {'2d': {'indices': []}, '0d': {'indices': [], 'flag': False}, '1d': {'indices': [0, 1]}}
        print("Old".center(80, '-'))
        print(old)
        print("New".center(80, '-'))
        print(new)

        ## XXX: how do I force plot to update?
        #       I have to wheel-zoom to make the plot redraw.
        draw_density(self.plot)

def main():
    doc = curdoc()
    doc.add(create_map_plot())


main()
