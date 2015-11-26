"""
Launch with
PYTHONPATH=`pwd` bokeh serve numba_hsa_examples/kde_bokeh/lightning_app.py
"""
from bokeh.sampledata import us_states
from bokeh.plotting import Figure
from bokeh.io import curdoc
from bokeh.models import Range1d, ColumnDataSource
from bokeh.models.widgets import HBox, VBox, Select, Slider
from bokeh import palettes
import numpy as np
import itertools
import pandas as pd
from numba_hsa_examples.kde_bokeh import kde
from numba_hsa_examples.kde_bokeh import dataloader
from numba_hsa_examples.kde_bokeh import plotting

TITLE = "US Lightning Density"


def get_us_state_outline():
    states = us_states.data.copy()

    # Remove HAWAII and Alaska
    del states["HI"]
    del states["AK"]

    state_xs = [states[code]["lons"] for code in states]
    state_ys = [states[code]["lats"] for code in states]

    return state_xs, state_ys


def plot_state_outline(plot, state_xs, state_ys):
    plot.patches(state_xs, state_ys, fill_alpha=0.2, line_color="black",
                 line_width=1)


class ViewListener(object):
    def __init__(self, plot, density_overlay, name):
        self.plot = plot
        self.density_overlay = density_overlay
        self.name = name

    def __call__(self, attrname, old, new):
        left = self.plot.x_range.start
        right = self.plot.x_range.end
        bottom = self.plot.y_range.start
        top = self.plot.y_range.end

        self.density_overlay.queue.append((left, right, bottom, top))


def minmax(arr):
    expanded = [v for vl in arr for v in vl]
    return min(expanded), max(expanded)


class DensityOverlay(object):
    INITIAL_GRID = 20

    def __init__(self, plot, left, right, bottom, top):
        self.count = 0
        self.gridcount = self.INITIAL_GRID
        self.radiance = 50000
        assert self.radiance > dataloader.MIN_RAD
        self.plot = plot
        self.queue = []
        self.use_hsa = bool(kde.USE_HSA)
        self.last_view_port = left, right, bottom, top

        print("Loading data")
        if True:
            df = dataloader.load_all_data(left, right, bottom, top)
            self.lon = df.lon.values
            self.lat = df.lat.values
            self.rad = df.rad.values
        else:
            self.lon = np.random.random(100) * (right - left) + left
            self.lat = np.random.random(100) * (top - bottom) + bottom
            self.rad = np.random.random(100) * 10 * self.radiance

        self.source = ColumnDataSource(data={
            'lon': [],
            'lat': [],
            'width': [],
            'height': [],
            'colors': [],
        })
        self.update(left, right, bottom, top)

    def _make_dict(self, left, right, bottom, top):

        ny = nx = self.gridcount
        x = np.linspace(left, right, nx)
        y = np.linspace(bottom, top, ny)

        xx, yy = zip(*itertools.product(x, y))

        dw = (right - left) / (nx - 1)
        dh = (top - bottom) / (ny - 1)

        print("Filter")
        df = pd.DataFrame({'lon': self.lon,
                           'lat': self.lat,
                           'rad': self.rad,})
        df = dataloader.filter_dataframes(df,
                                          lat_min=bottom,
                                          lat_max=top,
                                          lon_min=left,
                                          lon_max=right,
                                          rad_min=self.radiance)
        lon = df.lon.values
        lat = df.lat.values

        print("Compute density")
        pdf, count = kde.compute_density(lon, lat, xx, yy, use_hsa=self.use_hsa)
        self.count = count

        print("Done")
        cm = plotting.RGBColorMapper(0.0, 1.0, palettes.Reds9)
        cols = cm.color(1 - pdf / np.ptp(pdf))

        # cols = [random.choice(palettes.Spectral9) for _ in range(len(xx))]
        return {
            'lon': xx,
            'lat': yy,
            'width': [dw] * len(xx),
            'height': [dh] * len(yy),
            'colors': cols,
        }

    def update(self, left, right, bottom, top):
        dct = self._make_dict(left, right, bottom, top)
        self.source.data = dct
        self.plot.title = TITLE + " ({0} lightning strikes)".format(self.count)
        self.last_view_port = left, right, bottom, top

    def draw(self):
        self.plot.rect(x="lon", y="lat", width="width", height="height",
                       fill_color="colors", fill_alpha=0.50, line_alpha=0,
                       source=self.source)

    def backend_change_listener(self, attr, old, new):
        self.use_hsa = new == 'HSA'
        print("select", new)

    def periodic_callback(self):
        if not self.queue:
            return

        left, right, bottom, top = self.queue.pop()
        self.update(left, right, bottom, top)

        self.queue.clear()

    def grid_change_listener(self, attr, old, new):
        self.gridcount = new
        self.update(*self.last_view_port)

    def radiance_change_listener(self, attr, old, new):
        self.radiance = new
        self.update(*self.last_view_port)


def main():
    state_xs, state_ys = get_us_state_outline()
    left, right = minmax(state_xs)
    bottom, top = minmax(state_ys)
    plot = Figure(title=TITLE, plot_width=1000,
                  plot_height=700,
                  tools="pan, wheel_zoom, box_zoom, reset",
                  x_range=Range1d(left, right),
                  y_range=Range1d(bottom, top),
                  x_axis_label='Longitude',
                  y_axis_label='Latitude')

    plot_state_outline(plot, state_xs, state_ys)

    density_overlay = DensityOverlay(plot, left, right, bottom, top)
    density_overlay.draw()

    grid_slider = Slider(title="Details", value=density_overlay.gridcount,
                         start=10, end=100, step=10)
    grid_slider.on_change("value", density_overlay.grid_change_listener)

    radiance_slider = Slider(title="Min. Radiance",
                             value=density_overlay.radiance,
                             start=np.min(density_overlay.rad),
                             end=np.max(density_overlay.rad), step=10)
    radiance_slider.on_change("value", density_overlay.radiance_change_listener)

    listener = ViewListener(plot, density_overlay, name="viewport")

    plot.x_range.on_change("start", listener)
    plot.x_range.on_change("end", listener)
    plot.y_range.on_change("start", listener)
    plot.y_range.on_change("end", listener)

    backends = ["CPU", "HSA"]
    default_value = backends[kde.USE_HSA]
    backend_select = Select(name="backend", value=default_value,
                            options=backends)
    backend_select.on_change('value', density_overlay.backend_change_listener)

    doc = curdoc()
    doc.add(VBox(children=[plot, grid_slider, radiance_slider, backend_select]))
    doc.add_periodic_callback(density_overlay.periodic_callback, 0.5)

if __name__.startswith('bk_script_'):
    main()
