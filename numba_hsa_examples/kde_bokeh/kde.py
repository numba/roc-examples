import numpy as np
from timeit import default_timer as timer
from numba_hsa_examples.kerneldensity import cpu_ref, hsa_imp
from numba import vectorize
import os

USE_HSA = int(os.environ.get('HSA', 0))


@vectorize(nopython=True)
def filter_array(lon, lat, min_lon, min_lat, max_lon, max_lat):
    return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


def compute_density(lon, lat, xx, yy, use_hsa):
    min_lon = np.min(xx)
    max_lon = np.max(xx)

    min_lat = np.min(yy)
    max_lat = np.max(yy)

    selected = filter_array(lon, lat, min_lon, min_lat, max_lon, max_lat)

    lon = lon[selected]
    lat = lat[selected]

    samples = np.dstack([lon, lat])
    assert samples.shape[0] == 1
    samples = samples.reshape(samples.shape[1:])
    support = np.squeeze(np.dstack([xx, yy]))

    bwlist = np.array([cpu_ref.approx_bandwidth(support[:, k])
                       for k in range(support.shape[1])])

    pdf = np.zeros(support.shape[0], dtype=np.float64)

    if samples.size:
        print(samples.shape, samples.dtype)

        start_time = timer()
        if use_hsa:
            print("HSA".center(80, '-'))
            hsa_imp.hsa_multi_kde(support, samples, bwlist, pdf)
        else:
            print("CPU".center(80, '-'))
            cpu_ref.multi_kde_seq(support, samples, bwlist, pdf)
        end_time = timer()
        print("duration", "{0:0.2f} seconds".format(end_time - start_time))
    return pdf, samples.size
