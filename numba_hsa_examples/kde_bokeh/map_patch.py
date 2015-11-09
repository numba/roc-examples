from bokeh.sampledata import us_states
from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.models import Range1d, ColumnDataSource
import numpy as np
import logging


# logging.basicConfig(level=logging.WARNING)


def get_us_state_outline():
    states = us_states.data.copy()

    # Remove HAWAII and Alaska
    del states["HI"]
    del states["AK"]

    bad = """
    KY
    RI
    FL
    CA
    NY
    """

    # Remove "bad" states
    for c in bad.split():
        del states[c.strip()]

    state_xs = [states[code]["lons"] for code in states]
    state_ys = [states[code]["lats"] for code in states]

    return state_xs, state_ys


def plot_state_outline(plot, state_xs, state_ys):
    plot.patches(state_xs, state_ys, fill_alpha=0.2, line_color="#884444",
                 line_width=2)


class ViewListener(object):
    def __init__(self, plot, name):
        self.plot = plot
        self.name = name

    def __call__(self, attrname, old, new):
        print(self.name)
        print("-" * (80))
        print(attrname)

        left = self.plot.x_range.start
        right = self.plot.x_range.end
        bottom = self.plot.y_range.start
        top = self.plot.y_range.start

        print(left, right, bottom, top)
        print("=" * (80))
        print()


def minmax(arr):
    expanded = [v for vl in arr for v in vl]
    return min(expanded), max(expanded)


def main():
    state_xs, state_ys = get_us_state_outline()

    plot = figure(title="US Map Lightning", plot_width=1000, plot_height=700,
                  tools="pan, box_zoom, wheel_zoom, reset",
                  x_range=Range1d(*minmax(state_xs)),
                  y_range=Range1d(*minmax(state_ys)))

    plot_state_outline(plot, state_xs, state_ys)

    listener = ViewListener(plot, name="viewport")

    plot.x_range.on_change("start", listener)
    plot.x_range.on_change("end", listener)
    plot.y_range.on_change("start", listener)
    plot.y_range.on_change("end", listener)

    doc = curdoc()
    doc.add(plot)


main()
