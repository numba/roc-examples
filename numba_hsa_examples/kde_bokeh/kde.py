import numpy as np
import cpu_ref


def compute_density(lon, lat, xx, yy):
    samples = np.squeeze(np.dstack([lon, lat]))
    support = np.squeeze(np.dstack([xx, yy]))

    # bwlist = [cpu_ref.approx_bandwidth(samples[:, k])
    #           for k in range(samples.shape[1])]
    bwlist = [cpu_ref.approx_bandwidth(support[:, k])
              for k in range(support.shape[1])]

    pdf = np.zeros(support.shape[0], dtype=np.float64)

    cpu_ref.multi_kde_seq(support, samples, bwlist, pdf)

    return pdf
