from __future__ import print_function, division, absolute_import
import numpy as np
from .sort_driver import HsaRadixSortDriver

from timeit import default_timer as timer


def speed(compare_driver, nelem, dtype=np.intp):
    data = np.random.randint(0, 0xffffffff, nelem).astype(dtype)
    sorter = compare_driver()

    # CPU Reference: Do argsort and then scatter.
    ts = timer()
    expected_indices = np.argsort(data, kind='mergesort')
    expected = data[expected_indices]
    print('numpy mergesort', timer() - ts)

    # HSA: sort_with_indices will produce the sorted data and the indices
    #      for scattering.
    ts = timer()
    sorted, indices = sorter.sort_with_indices(data)
    print('hsa radixsort  ', timer() - ts)
    print("internal timings:", sorter.last_timings)

    np.testing.assert_equal(expected, sorted)
    np.testing.assert_equal(expected_indices, indices)


def main():
    nelemlist = [100,
                 10000,
                 100000,
                 1000000,
                 10000000,
                 ]
    for n in nelemlist:
        print('=' * 80)
        print('n = {0}'.format(n))
        speed(HsaRadixSortDriver, n)


if __name__ == '__main__':
    main()
