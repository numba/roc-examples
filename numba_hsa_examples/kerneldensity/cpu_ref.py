import math
import numpy as np
from scipy import stats

import statsmodels.api as sm
from statsmodels.distributions.mixture_rvs import mixture_rvs

from bokeh.plotting import figure, show, output_file, vplot

from numba import njit


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


def approx_gaussian_bandwidth(xs):
    stddev = np.std(xs)
    return ((4 * stddev ** 5) / (3 * xs.size)) ** (1 / 5)


def uni_kde_seq_factory(kernel):
    @njit
    def kde(support, samples, bandwidth, pdf):
        for i in range(support.size):
            xi = support[i]
            total = 0
            for j in range(samples.size):
                total += kernel((xi - samples[j]) / bandwidth) / bandwidth
            pdf[i] = total / samples.size

    return kde


uni_kde_seq = uni_kde_seq_factory(gaussian_kernel)


def test_gaussian():
    output_file("kde.html")
    samples = np.linspace(-2, 2, 1000)
    p1 = figure()

    x = samples
    pdf = np.array([gaussian_kernel(i) for i in samples])

    p1.line(x, pdf)

    show(p1)


def test():
    np.random.seed(12345)

    samples = mixture_rvs([.25, .75], size=10000,
                          dist=[stats.norm, stats.norm],
                          kwargs=(
                              dict(loc=-1, scale=.5), dict(loc=1, scale=.5)))

    # samples = np.random.normal(size=10000)



    bandwidth = approx_gaussian_bandwidth(samples)
    bandwidth /= 2
    print('bandwidth', bandwidth)
    print('size', samples.size)

    cut = 3
    minimum = np.min(samples) - cut * bandwidth
    maximum = np.max(samples) + cut * bandwidth
    nobs = samples.size
    support = np.linspace(minimum, maximum, nobs)
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


#

if __name__ == '__main__':
    test()
    # test_gaussian()
