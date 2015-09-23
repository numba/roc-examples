from __future__ import print_function, absolute_import

from collections import OrderedDict
from timeit import default_timer as timer
import logging

import pandas as pd
import numpy as np

from .groupby import HSAGrouper

logging.basicConfig(level=logging.DEBUG)


def benchmark(nelem, numgroup):
    print('------ benchmark', nelem, numgroup)

    out = OrderedDict()

    a = np.random.randint(0, numgroup, nelem).astype(np.intp)
    b = np.random.random(nelem)

    df = pd.DataFrame({'a': a, 'b': b})

    ts = timer()
    expected_grouper = df.groupby(pd.Grouper('a', sort=True))
    out['cpu_groupby'] = timer() - ts

    ts = timer()
    got_grouper = df.groupby(HSAGrouper('a', sort=True))
    out['hsa_groupby'] = timer() - ts

    expeceted_groups = expected_grouper.groups
    got_groups = got_grouper.groups

    # Pandas does not use a stable sort,
    # Only the keys are checked. The values are in different order.
    assert expeceted_groups.keys() == got_groups.keys()

    ts = timer()
    expected_mean = expected_grouper.mean()
    out['cpu_mean'] = timer() - ts

    ts = timer()
    got_mean = got_grouper.mean()
    out['hsa_mean'] = timer() - ts

    assert np.allclose(expected_mean.values, got_mean.values)

    ts = timer()
    expected_var = expected_grouper.var()
    out['cpu_var'] = timer() - ts

    ts = timer()
    got_var = got_grouper.var()
    out['hsa_var'] = timer() - ts

    assert np.allclose(expected_var.values, got_var.values)

    return out


def main():
    # numgroup = 8000  # roughly the number of symbols at NYSE
    numgroup = 8
    # Warm up JIT
    print('warm up'.center(80, '='))
    benchmark(10 ** 6 * 2, 2)
    # Start real benchmark
    print('benchmark'.center(80, '='))
    for i in range(1):
        print("### round", i + 1)
        # With 10^6 element, CPU and GPU speed is similar
        print(benchmark(10 ** 6, numgroup))
        # With 10^7 element, GPU speed is 2-3 seconds faster
        print(benchmark(10 ** 7, numgroup))

        print(benchmark(2 * 10 ** 7, numgroup))


if __name__ == '__main__':
    main()
