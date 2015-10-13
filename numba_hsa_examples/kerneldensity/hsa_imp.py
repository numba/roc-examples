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

from numba import hsa, njit
from numba_hsa_examples.reduction.reduction import group_reduce_sum_float64


@hsa.jit(device=True)
def hsa_gaussian(x, mu, sigma):
    xmu = (x - mu)
    xmu2 = xmu * xmu
    div = 2 * sigma * sigma
    exp = math.exp(-(xmu2 / div))
    return exp / (sigma * math.sqrt(2 * math.pi))


@hsa.jit(device=True)
def hsa_gaussian_kernel(x):
    return hsa_gaussian(x, 0, 1)


def approx_bandwidth(xs):
    """
    Scott's rule of thumb as in SciPy
    """
    n = xs.size
    d = xs.ndim
    return n ** (-1 / (d + 4))


def hsa_uni_kde_factory(kernel):
    @hsa.jit
    def hsa_uni_kde(support, samples, bandwidth, pdf):
        i = hsa.get_global_id(0)

        supp = support[i]
        total = 0
        for j in range(samples.size):
            total += kernel((samples[j] - supp) / bandwidth) / bandwidth
        pdf[i] = total / samples.size

    def launcher(support, samples, bandwidth, pdf):
        assert pdf.ndim == 1
        assert support.ndim == 1
        assert samples.ndim == 1
        assert support.size == pdf.size
        with hsa.register(support, samples, pdf):
            threads = 64 * 4  # 4x wavesize
            blocks = (pdf.size + threads - 1) // threads
            hsa_uni_kde[blocks, threads](support, samples, bandwidth, pdf)

    return launcher


hsa_uni_kde = hsa_uni_kde_factory(hsa_gaussian_kernel)


def hsa_uni_kde_ver2_factory(kernel):
    @hsa.jit
    def hsa_uni_kde(support, samples, bandwidth, pdf):
        gid = hsa.get_group_id(0)
        tid = hsa.get_local_id(0)
        tsz = hsa.get_local_size(0)

        supp = support[gid]

        # all local threads cooperately computes the energy for a support
        energy = 0
        for base in range(0, samples.size, tsz):
            idx = tid + base
            if idx < samples.size:
                energy += kernel((samples[idx] - supp) / bandwidth) / bandwidth

        # reduce energy
        total = group_reduce_sum_float64(energy)
        pdf[gid] = total / samples.size

    def launcher(support, samples, bandwidth, pdf):
        assert pdf.ndim == 1
        assert support.ndim == 1
        assert samples.ndim == 1
        assert support.size == pdf.size
        with hsa.register(support, samples, pdf):
            threads = 64 * 4  # 4x wavesize
            blocks = support.size
            hsa_uni_kde[blocks, threads](support, samples, bandwidth, pdf)

    return launcher


hsa_uni_kde_ver2 = hsa_uni_kde_ver2_factory(hsa_gaussian_kernel)


def test_hsa_uni_kde():
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
    hsa_uni_kde(support, samples, bandwidth, pdf)

    # Check value
    expect = kde.density
    got = pdf

    rms = calc_rms(expect, got, norm=True)
    print("RMS", rms)
    assert rms < 1e-2, "RMS error too high: {0}".format(rms)


def test_hsa_uni_kde_ver2():
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
    hsa_uni_kde_ver2(support, samples, bandwidth, pdf)

    # Check value
    expect = kde.density
    got = pdf

    rms = calc_rms(expect, got, norm=True)
    print("RMS", rms)
    assert rms < 1e-2, "RMS error too high: {0}".format(rms)


if __name__ == "__main__":
    test_hsa_uni_kde()
    test_hsa_uni_kde_ver2()
