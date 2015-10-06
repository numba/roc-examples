import math
import numpy as np
from scipy import stats

import statsmodels.api as sm
from statsmodels.distributions.mixture_rvs import mixture_rvs

from bokeh.plotting import figure, show, output_file, vplot

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


def approx_gaussian_bandwidth(xs):
    stddev = np.std(xs)
    return ((4 * stddev ** 5) / (3 * xs.size)) ** (1 / 5)


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


def test_uni():
    samples = mixture_rvs([.25, .75], size=10000,
                          dist=[stats.norm, stats.norm],
                          kwargs=(
                              dict(loc=-1, scale=.5), dict(loc=1, scale=.5)))

    # samples = np.random.normal(size=10000)
    bandwidth = approx_gaussian_bandwidth(samples)

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


def test_multi():
    samples = np.ascontiguousarray(np.vstack([np.random.normal(size=1000),
                                              np.random.normal(size=1000)]).T)

    bwlist = [approx_gaussian_bandwidth(samples[:, k])
              for k in range(samples.shape[1])]
    bandwidths = np.array(bwlist)

    support = np.ascontiguousarray(
        np.vstack([build_support(samples[:, i], bandwidths[i])
                   for i in range(bandwidths.size)]).T)

    pdf = np.zeros(support.shape[0], dtype=np.float64)

    print('bandwidths', bandwidths)
    print('support', support.shape)
    print('samples', samples.shape)

    multi_kde_seq(support, samples, bandwidths, pdf)

    # print(pdf)

    # Plotting
    output_file("kde_multi.html")
    #
    # p1 = figure(title="Hist")
    # hist, edges = np.histogram(samples, bins=50, density=True)
    # p1.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:])

    p2 = figure(title="KDE")
    p2.line(support[:, 0], pdf, color='blue')
    p2.line(support[:, 1], pdf, color='red')
    # p2.circle(x=support, y=pdf, size=5)

    p3 = figure(title="KDE-SM")
    kde = sm.nonparametric.KDEMultivariate(samples, var_type='cc')
    kde_pdf = kde.pdf()

    idx = support[:, 0].argsort(kind='mergesort')
    p3.line(support[idx, 0], kde_pdf[idx], color='blue')

    show(vplot(p2, p3))


if __name__ == '__main__':
    # test_uni()
    # test_gaussian()
    test_multi()
