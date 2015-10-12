import math
import itertools
import numpy as np
from scipy import stats

import statsmodels.api as sm
from statsmodels.distributions.mixture_rvs import mixture_rvs

from bokeh.plotting import figure, show, output_file, vplot
from bokeh import palettes

from numba import njit

from .plotting import RGBColorMapper

from .cpu_ref import approx_bandwidth, calc_rms

from numba import hsa


@hsa.jit(device=True)
def hsa_gaussian(x, mu, sigma):
    xmu = (x - mu)
    xmu2 = xmu * xmu
    div = 2 * sigma * sigma
    exp = math.exp(-(xmu2 / div))
    return exp / (sigma * math.sqrt(2 * math.pi))


@hsa.jit(device=True)
def hsa_gaussian_kernel(x):
    return hsa_gaussian(x, mu=0, sigma=1)


def approx_bandwidth(xs):
    """
    Scott's rule of thumb as in SciPy
    """
    n = xs.size
    d = xs.ndim
    return n ** (-1 / (d + 4))


def hsa_uni_kde_seq_factory(kernel):
    @hsa.jit
    def hsa_uni_kde(support, samples, bandwidth, pdf):
        i = hsa.get_global_id(0)

        supp = support[i]
        total = 0
        for j in range(samples.size):
            total += kernel((samples[j] - supp) / bandwidth) / bandwidth
        pdf[i] = total / samples.size

    return hsa_uni_kde


hsa_uni_kde_seq = hsa_uni_kde_seq_factory(hsa_gaussian_kernel)


def test_uni_kde():
    np.random.seed(12345)

    samples = mixture_rvs([.25, .75], size=10000,
                          dist=[stats.norm, stats.norm],
                          kwargs=(
                              dict(loc=-1, scale=.5), dict(loc=1, scale=.5)))

    bandwidth = approx_bandwidth(samples)

    # Run statsmodel for reference
    kde = sm.nonparametric.KDEUnivariate(samples)
    kde.fit(kernel="gau")

    # Reuse statsmodel support for our test
    support = kde.support

    # Run custom KDE
    pdf = np.zeros_like(support)
    hsa_uni_kde_seq(support, samples, bandwidth, pdf)

    # Check value
    expect = kde.density
    got = pdf

    rms = calc_rms(expect, got, norm=True)
    assert rms < 1e-2, "RMS error too high: {0}".format(rms)


if __name__ == "__main__":
    test_uni_kde()
