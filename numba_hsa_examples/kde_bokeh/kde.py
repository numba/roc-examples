import numpy as np
import cpu_ref
from numba import vectorize


@vectorize(nopython=True)
def filter_array(lon, lat, min_lon, min_lat, max_lon, max_lat):
    return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


def compute_density(lon, lat, xx, yy):
    min_lon = np.min(xx)
    max_lon = np.max(xx)

    min_lat = np.min(yy)
    max_lat = np.max(yy)

    selected = filter_array(lon, lat, min_lon, min_lat, max_lon, max_lat)
    lon = lon[selected]
    lat = lat[selected]

    samples = np.squeeze(np.dstack([lon, lat]))
    support = np.squeeze(np.dstack([xx, yy]))

    print("sample size", samples.size)

    # bwlist = [cpu_ref.approx_bandwidth(samples[:, k])
    #           for k in range(samples.shape[1])]
    bwlist = [cpu_ref.approx_bandwidth(support[:, k])
              for k in range(support.shape[1])]

    pdf = np.zeros(support.shape[0], dtype=np.float64)

    cpu_ref.multi_kde_seq(support, samples, bwlist, pdf)

    return pdf
