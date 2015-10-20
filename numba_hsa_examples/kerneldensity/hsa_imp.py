import math
import numpy as np
from scipy import stats
from pprint import pprint
from collections import OrderedDict

import statsmodels.api as sm
from statsmodels.distributions.mixture_rvs import mixture_rvs
from timeit import default_timer as timer

from .cpu_ref import (approx_bandwidth, calc_rms, uni_kde_seq, build_support_nd,
                      multi_kde_seq)

from numba import hsa
from numba_hsa_examples.reduction.reduction import group_reduce_sum_float64

WAVESIZE = 64


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

        if i < support.size:
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
            threads = WAVESIZE * 4
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

        # all local threads cooperatively computes the energy for a support
        energy = 0
        for base in range(0, samples.size, tsz):
            idx = tid + base
            if idx < samples.size:
                energy += kernel((samples[idx] - supp) / bandwidth) / bandwidth

        # reduce energy
        total = group_reduce_sum_float64(energy)
        if tid == 0:
            pdf[gid] = total / samples.size

    def launcher(support, samples, bandwidth, pdf):
        assert pdf.ndim == 1
        assert support.ndim == 1
        assert samples.ndim == 1
        assert support.size == pdf.size
        with hsa.register(support, samples, pdf):
            threads = WAVESIZE * 8
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
    kde.fit(kernel="gau", fft=False)

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
    kde.fit(kernel="gau", fft=False)

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


def benchmark_hsa_uni_kde():
    def driver(imp_dict, retry=3, size=10000):
        print("Running univariate kde benchmark on size = {size}".format(
            size=size))
        samples = mixture_rvs([.25, .75], size=size,
                              dist=[stats.norm, stats.norm],
                              kwargs=(
                                  dict(loc=-1, scale=.5),
                                  dict(loc=1, scale=.5)))

        bandwidth = approx_bandwidth(samples)

        # Run statsmodel for reference
        kde = sm.nonparametric.KDEUnivariate(samples)
        kde.fit(kernel="gau", fft=False)

        # Reuse statsmodel support for our test
        support = kde.support
        expect = kde.density

        timing = OrderedDict()

        # Run timing loop
        for name, imp in imp_dict.items():
            print("Running {name}".format(name=name))
            times = []
            for t in range(retry):
                print(" trial = {t}".format(t=t), end=' ... ')
                got, elapsed = imp(support, samples, bandwidth, timer)
                print("elapsed =", elapsed, end=' | ')
                times.append(elapsed)
                rms = calc_rms(expect, got, norm=True)
                print("RMS =", rms)
                if rms > 0.01:
                    print("*** warning, RMS is too high")
            timing[name] = times

        return timing

    imp_dict = OrderedDict()

    def statsmodel_imp(support, samples, bandwidth, timer):
        kde = sm.nonparametric.KDEUnivariate(samples)
        ts = timer()
        kde.fit(kernel='gau', fft=False)
        te = timer()
        return kde.density, te - ts

    imp_dict['statsmodel'] = statsmodel_imp

    def numba_cpu_imp(support, samples, bandwidth, timer):
        pdf = np.zeros_like(support)
        ts = timer()
        uni_kde_seq(support, samples, bandwidth, pdf)
        te = timer()
        return pdf, te - ts

    imp_dict['numba-cpu'] = numba_cpu_imp

    def numba_hsa_ver1_imp(support, samples, bandwidth, timer):
        pdf = np.zeros_like(support)
        ts = timer()
        hsa_uni_kde(support, samples, bandwidth, pdf)
        te = timer()
        return pdf, te - ts

    imp_dict['numba-hsa-ver1'] = numba_hsa_ver1_imp

    def numba_hsa_ver2_imp(support, samples, bandwidth, timer):
        pdf = np.zeros_like(support)
        ts = timer()
        hsa_uni_kde_ver2(support, samples, bandwidth, pdf)
        te = timer()
        return pdf, te - ts

    imp_dict['numba-hsa-ver2'] = numba_hsa_ver2_imp

    # Run benchmark
    for size in [5000, 10000, 15000]:
        timings = driver(imp_dict, size=size)
        pprint(timings)


# -----------------------------------------------------------------------------
# Multi KDE

def hsa_multi_kde_factory(kernel):
    @hsa.jit
    def hsa_multi_kde(support, samples, bandwidths, pdf):
        """
        Expects 2d arrays for samples and support: (num_observations,
        num_variables)
        """
        nvar = support.shape[1]

        i = hsa.get_global_id(0)
        if i < support.shape[0]:
            sum = 0
            for j in range(samples.shape[0]):
                prod = 1
                for k in range(nvar):
                    bw = bandwidths[k]
                    diff = samples[j, k] - support[i, k]
                    prod *= kernel(diff / bw) / bw
                sum += prod
            pdf[i] = sum / samples.shape[0]

    def launcher(support, samples, bandwidths, pdf):
        assert support.shape[0] == pdf.size
        assert support.shape[1] == samples.shape[1]
        assert bandwidths.size == support.shape[1]

        threads = WAVESIZE * 4
        blocks = (support.shape[0] + threads - 1) // threads

        with hsa.register(support, samples, bandwidths, pdf):
            hsa_multi_kde[blocks, threads](support, samples, bandwidths, pdf)

    return launcher


hsa_multi_kde = hsa_multi_kde_factory(hsa_gaussian_kernel)


def hsa_multi_kde_ver2_factory(kernel):
    from numba import float64

    BLOCKSIZE = WAVESIZE * 4
    MAX_NDIM = 4
    SAMPLES_SIZE = MAX_NDIM, BLOCKSIZE

    @hsa.jit
    def hsa_multi_kde(support, samples, bandwidths, pdf):
        """
        Expects 2d arrays for samples and support: (num_observations,
        num_variables)
        """
        nvar = support.shape[1]
        i = hsa.get_global_id(0)
        tid = hsa.get_local_id(0)
        valid = i < support.shape[0]

        sum = 0

        sm_samples = hsa.shared.array(SAMPLES_SIZE, dtype=float64)
        sm_bandwidths = hsa.shared.array(MAX_NDIM, dtype=float64)
        sm_support = hsa.shared.array(SAMPLES_SIZE, dtype=float64)

        if valid:
            for k in range(nvar):
                sm_support[k, tid] = support[i, k]

        if tid < nvar:
            sm_bandwidths[tid] = bandwidths[tid]

        for base in range(0, samples.shape[0], BLOCKSIZE):
            loadcount = min(samples.shape[0] - base, BLOCKSIZE)

            hsa.barrier()

            # Preload samples tile
            if tid < loadcount:
                for k in range(nvar):
                    sm_samples[k, tid] = samples[base + tid, k]

            hsa.barrier()

            # Compute on the tile
            if valid:
                for j in range(loadcount):
                    prod = 1
                    for k in range(nvar):
                        bw = sm_bandwidths[k]
                        diff = sm_samples[k, j] - sm_support[k, tid]
                        prod *= kernel(diff / bw) / bw
                    sum += prod

        if valid:
            pdf[i] = sum / samples.shape[0]

    def launcher(support, samples, bandwidths, pdf):
        assert support.shape[0] == pdf.size
        assert support.shape[1] == samples.shape[1]
        assert bandwidths.size == support.shape[1]
        assert bandwidths.size <= MAX_NDIM

        threads = BLOCKSIZE
        blocks = (support.shape[0] + threads - 1) // threads

        with hsa.register(support, samples, bandwidths, pdf):
            hsa_multi_kde[blocks, threads](support, samples, bandwidths, pdf)

    return launcher


hsa_multi_kde_ver2 = hsa_multi_kde_ver2_factory(hsa_gaussian_kernel)


def test_hsa_multi_kde():
    np.random.seed(12345)

    nelem = 100

    samples = np.squeeze(np.dstack([np.random.normal(size=nelem),
                                    np.random.normal(size=nelem)]))

    bwlist = [approx_bandwidth(samples[:, k])
              for k in range(samples.shape[1])]
    bandwidths = np.array(bwlist)

    support = build_support_nd(samples, bandwidths)

    pdf = np.zeros(support.shape[0], dtype=np.float64)

    hsa_multi_kde(support, samples, bandwidths, pdf)

    kde = sm.nonparametric.KDEMultivariate(samples, var_type='cc',
                                           bw=bwlist)
    expect_pdf = kde.pdf(support)

    rms = calc_rms(expect_pdf, pdf, norm=True)
    print('RMS', rms)
    assert rms < 1e-4, "RMS error too high: {0}".format(rms)


def test_hsa_multi_kde_ver2():
    np.random.seed(12345)

    nelem = 100

    samples = np.squeeze(np.dstack([np.random.normal(size=nelem),
                                    np.random.normal(size=nelem)]))

    bwlist = [approx_bandwidth(samples[:, k])
              for k in range(samples.shape[1])]
    bandwidths = np.array(bwlist)

    support = build_support_nd(samples, bandwidths)

    pdf = np.zeros(support.shape[0], dtype=np.float64)

    hsa_multi_kde_ver2(support, samples, bandwidths, pdf)

    kde = sm.nonparametric.KDEMultivariate(samples, var_type='cc',
                                           bw=bwlist)
    expect_pdf = kde.pdf(support)

    rms = calc_rms(expect_pdf, pdf, norm=True)
    print('RMS', rms)
    assert rms < 1e-4, "RMS error too high: {0}".format(rms)


def benchmark_hsa_multi_kde():
    def driver(imp_dict, retry=3, size=100):
        print("Running multivariate kde benchmark on size = {size}".format(
            size=size))

        samples = np.squeeze(np.dstack([np.random.normal(size=size),
                                        np.random.normal(size=size)]))

        bwlist = [approx_bandwidth(samples[:, k])
                  for k in range(samples.shape[1])]
        bandwidths = np.array(bwlist)

        support = build_support_nd(samples, bandwidths)

        # Run statsmodel for reference
        kde = sm.nonparametric.KDEMultivariate(samples, var_type='cc',
                                               bw=bwlist)
        expect = kde.pdf(support)

        timing = OrderedDict()

        # Run timing loop
        for name, imp in imp_dict.items():
            print("Running {name}".format(name=name))
            times = []
            for t in range(retry):
                print(" trial = {t}".format(t=t), end=' ... ')
                got, elapsed = imp(support, samples, bandwidths, timer)
                print("elapsed =", elapsed, end=' | ')
                times.append(elapsed)
                rms = calc_rms(expect, got, norm=True)
                print("RMS =", rms)
                if rms > 0.01:
                    print("*** warning, RMS is too high")
            timing[name] = times

        return timing

    imp_dict = OrderedDict()

    def statsmodel_imp(support, samples, bandwidths, timer):
        kde = sm.nonparametric.KDEMultivariate(samples, var_type='cc',
                                               bw=bandwidths)
        ts = timer()
        pdf = kde.pdf(support)
        te = timer()
        return pdf, te - ts

    imp_dict['statsmodel'] = statsmodel_imp

    def numba_cpu_imp(support, samples, bandwidths, timer):
        pdf = np.zeros(support.shape[0], dtype=np.float64)
        ts = timer()
        multi_kde_seq(support, samples, bandwidths, pdf)
        te = timer()
        return pdf, te - ts

    imp_dict['numba-cpu'] = numba_cpu_imp

    def numba_hsa_ver1_imp(support, samples, bandwidths, timer):
        pdf = np.zeros(support.shape[0], dtype=np.float64)
        ts = timer()
        hsa_multi_kde(support, samples, bandwidths, pdf)
        te = timer()
        return pdf, te - ts

    imp_dict['numba-hsa-ver1'] = numba_hsa_ver1_imp

    def numba_hsa_ver2_imp(support, samples, bandwidths, timer):
        pdf = np.zeros(support.shape[0], dtype=np.float64)
        ts = timer()
        hsa_multi_kde_ver2(support, samples, bandwidths, pdf)
        te = timer()
        return pdf, te - ts

    imp_dict['numba-hsa-ver2'] = numba_hsa_ver2_imp

    # Run benchmark
    for size in [100, 200]:  # , 300]:
        timings = driver(imp_dict, size=size)
        pprint(timings)


if __name__ == "__main__":
    # test_hsa_uni_kde()
    # test_hsa_uni_kde_ver2()
    # test_hsa_multi_kde()
    # test_hsa_multi_kde_ver2()
    benchmark_hsa_uni_kde()
    benchmark_hsa_multi_kde()
