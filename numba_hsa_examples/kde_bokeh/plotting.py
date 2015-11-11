import numpy as np


# Color map
def hex_to_rgb(value):
    """Given a color in hex format, return it in RGB."""

    values = value.lstrip('#')
    lv = len(values)
    rgb = list(int(values[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    return rgb


def rgb_to_hex(red, green, blue):
    """Give three color arrays, return a list of hex RGB strings"""
    pat = "#{0:02X}{1:02X}{2:02X}"
    return [pat.format(r & 0xff, g & 0xff, b & 0xff)
            for r, g, b in zip(red, green, blue)]


class RGBColorMapper(object):
    """Maps floating point values to rgb values over a palette"""

    def __init__(self, low, high, palette):
        self.range = np.linspace(low, high, len(palette))
        self.r, self.g, self.b = np.array(
            list(zip(*[hex_to_rgb(i) for i in palette])))

    def color(self, data):
        """Maps your data values to the pallette with linear interpolation"""

        red = np.interp(data, self.range, self.r)
        blue = np.interp(data, self.range, self.b)
        green = np.interp(data, self.range, self.g)
        # Style plot to return a grey color when value is 'nan'
        red[np.isnan(red)] = 240
        blue[np.isnan(blue)] = 240
        green[np.isnan(green)] = 240

        return rgb_to_hex(red.astype(np.uint8),
                          green.astype(np.uint8),
                          blue.astype(np.uint8))

    def color_rgba(self, data):
        red = np.interp(data, self.range, self.r)
        blue = np.interp(data, self.range, self.b)
        green = np.interp(data, self.range, self.g)
        # Style plot to return a grey color when value is 'nan'
        red[np.isnan(red)] = 240
        blue[np.isnan(blue)] = 240
        green[np.isnan(green)] = 240
        alpha = np.zeros_like(red)
        alpha[:] = 0xff
        return np.dstack([red.astype(np.uint8),
                          green.astype(np.uint8),
                          blue.astype(np.uint8),
                          alpha.astype(np.uint8)])

