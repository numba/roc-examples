import math
import itertools
import numpy as np
from scipy import stats

import statsmodels.api as sm
from statsmodels.distributions.mixture_rvs import mixture_rvs

from bokeh.plotting import figure, show, output_file, vplot
from bokeh import palettes

from numba import njit

np.random.seed(12345)


@njit
def gaussian(x, mu, sigma):
    xmu = (x - mu)
    xmu2 = xmu * xmu
    div = 2 * sigma * sigma
    exp = math.exp(-(xmu2 / div))
    return exp / (sigma * math.sqrt(2 * math.pi))


@njit
def gaussian_kernel(x):
    return gaussian(x, mu=0, sigma=1)


def approx_bandwidth(xs):
    """
    Scott's rule of thumb as in SciPy
    """
    n = xs.size
    d = xs.ndim
    return n ** (-1 / (d + 4))


def uni_kde_seq_factory(kernel):
    @njit
    def uni_kde(support, samples, bandwidth, pdf):
        for i in range(support.size):
            supp = support[i]
            total = 0
            for j in range(samples.size):
                total += kernel((samples[j] - supp) / bandwidth) / bandwidth
            pdf[i] = total / samples.size

    return uni_kde


uni_kde_seq = uni_kde_seq_factory(gaussian_kernel)


def multi_kde_seq_factory(kernel):
    @njit
    def multi_kde(support, samples, bandwidths, pdf):
        """
        Expects 2d arrays for samples and support: (num_observations,
        num_variables)
        """
        assert support.shape[1] == samples.shape[1]
        nvar = support.shape[1]
        for i in range(support.shape[0]):
            sum = 0
            for j in range(samples.shape[0]):
                prod = 1
                for k in range(nvar):
                    bw = bandwidths[k]
                    diff = samples[j, k] - support[i, k]
                    prod *= kernel(diff / bw) / bw
                sum += prod
            pdf[i] = sum / samples.shape[0]

    return multi_kde


multi_kde_seq = multi_kde_seq_factory(gaussian_kernel)


def test_gaussian():
    output_file("kde.html")
    samples = np.linspace(-2, 2, 1000)
    p1 = figure()

    x = samples
    pdf = np.array([gaussian_kernel(i) for i in samples])

    p1.line(x, pdf)

    show(p1)


def build_support(samples, bandwidth, cut=3):
    minimum = np.min(samples) - cut * bandwidth
    maximum = np.max(samples) + cut * bandwidth
    nobs = samples.size
    support = np.linspace(minimum, maximum, nobs)
    return support


def calc_rms(expect, got, norm=False):
    """
    Caclualte the RMS error between two arrays of the same size.

    If `norm` is True, the result is normalized to the range of the expected
    distribution.
    """
    assert expect.size == got.size
    rms = np.sqrt(np.sum((expect - got) ** 2) / expect.size)
    if norm:
        return rms / np.ptp(expect)
    else:
        return rms


def test_uni():
    samples = mixture_rvs([.25, .75], size=10000,
                          dist=[stats.norm, stats.norm],
                          kwargs=(
                              dict(loc=-1, scale=.5), dict(loc=1, scale=.5)))

    # samples = np.random.normal(size=10000)
    bandwidth = approx_bandwidth(samples)

    print('bandwidth', bandwidth)
    print('size', samples.size)

    support = build_support(samples, bandwidth)
    pdf = np.zeros_like(support)

    uni_kde_seq(support, samples, bandwidth, pdf)

    # Plotting
    output_file("kde.html")

    p1 = figure(title="Hist")
    hist, edges = np.histogram(samples, bins=50, density=True)
    p1.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:])

    p2 = figure(title="KDE")
    p2.line(support, pdf)
    p2.circle(x=support, y=pdf, size=5)

    p3 = figure(title="KDE-SM")
    kde = sm.nonparametric.KDEUnivariate(samples)
    kde.fit(kernel="gau")

    p3.line(kde.support, kde.density)

    print(samples.size)
    print(len(kde.support), len(kde.density))

    print(kde.density.sum())
    show(vplot(p1, p2, p3))


def test_multi_kde():
    nelem = 100

    samples = np.ascontiguousarray(np.vstack([np.random.normal(size=nelem),
                                              np.random.normal(size=nelem)]).T)

    bwlist = [approx_bandwidth(samples[:, k])
              for k in range(samples.shape[1])]
    bandwidths = np.array(bwlist)

    support_dims = [build_support(samples[:, i], bandwidths[i])
                    for i in range(bandwidths.size)]

    support = np.array([(x, y) for x, y in itertools.product(*support_dims)])

    pdf = np.zeros(support.shape[0], dtype=np.float64)

    print('bandwidths', bandwidths)
    print('support', support.shape)
    print('samples', samples.shape)

    multi_kde_seq(support, samples, bandwidths, pdf)

    kde = sm.nonparametric.KDEMultivariate(samples, var_type='cc',
                                           bw=bwlist)
    expect_pdf = kde.pdf(support)

    # Plotting

    output_file("kde_multi.html")

    cm = RGBAColorMapper(0, 1, palettes.Spectral11)

    p1 = figure(title="KDE")
    p1.square(support[:, 0], support[:, 1], size=5,
              color=cm.color(pdf / np.ptp(pdf)))

    p2 = figure(title="KDE-SM")
    p2.square(support[:, 0], support[:, 1], size=5,
              color=cm.color(expect_pdf / np.ptp(expect_pdf)))

    show(vplot(p1, p2))


def test_uni_kde():
    samples = mixture_rvs([.25, .75], size=10000,
                          dist=[stats.norm, stats.norm],
                          kwargs=(
                              dict(loc=-1, scale=.5), dict(loc=1, scale=.5)))

    # samples = np.random.normal(size=10000)
    bandwidth = approx_bandwidth(samples)

    # Run statsmodel for reference
    kde = sm.nonparametric.KDEUnivariate(samples)
    kde.fit(kernel="gau")

    # Reuse statsmodel support for our test
    support = kde.support

    # Run custom KDE
    pdf = np.zeros_like(support)
    uni_kde_seq(support, samples, bandwidth, pdf)

    # Check value
    expect = kde.density
    got = pdf

    rms = calc_rms(expect, got, norm=True)
    assert rms < 1e-4, "RMS error too high"


# Color map
def hex_to_rgb(value):
    """Given a color in hex format, return it in RGB."""

    values = value.lstrip('#')
    lv = len(values)
    rgb = list(int(values[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    return rgb


class RGBAColorMapper(object):
    """Maps floating point values to rgb values over a palette"""

    def __init__(self, low, high, palette):
        self.range = np.linspace(low, high, len(palette))
        # self.r, self.g, self.b = np.array(zip(*[hex_to_rgb(i) for i in palette])) #python 2.7
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
        colors = np.dstack([red.astype(np.uint8),
                            green.astype(np.uint8),
                            blue.astype(np.uint8),
                            np.full_like(data, 255, dtype=np.uint8)])
        colors = colors.view(dtype=np.uint32).reshape(data.shape)

        def make_hex(x):
            h = hex(x)[4:]
            diff = 6 - len(h)
            prefix = '0' * diff
            return prefix + h

        return ["#{0}".format(make_hex(x)) for x in colors]


def test_multi_kde():
    nelem = 100

    samples = np.ascontiguousarray(np.vstack([np.random.normal(size=nelem),
                                              np.random.normal(size=nelem)]).T)

    bwlist = [approx_bandwidth(samples[:, k])
              for k in range(samples.shape[1])]
    bandwidths = np.array(bwlist)

    support_dims = [build_support(samples[:, i], bandwidths[i])
                    for i in range(bandwidths.size)]

    support = np.array([(x, y) for x, y in itertools.product(*support_dims)])

    pdf = np.zeros(support.shape[0], dtype=np.float64)

    print('bandwidths', bandwidths)
    print('support', support.shape)
    print('samples', samples.shape)

    multi_kde_seq(support, samples, bandwidths, pdf)

    kde = sm.nonparametric.KDEMultivariate(samples, var_type='cc',
                                           bw=bwlist)
    expect_pdf = kde.pdf(support)

    rms = calc_rms(expect_pdf, pdf, norm=True)
    assert rms < 1e-4, "RMS error too high"


if __name__ == '__main__':
    test_multi_kde()
    # test_uni_kde()
    # test_uni()
    # test_gaussian()
    # test_multi()
