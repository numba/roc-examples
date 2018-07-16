from __future__ import print_function, absolute_import, division

from timeit import default_timer as timer
import numpy as np

from .reduction import device_reduce_sum


def benchmark_intp(nelem):
    data = np.random.randint(0, 100, nelem).astype(np.intp)

    ts = timer()
    expected_res = data.sum()
    cpu_time = timer() - ts

    ts = timer()
    got_res = device_reduce_sum(data)
    gpu_time = timer() - ts

    assert got_res == expected_res
    return cpu_time, gpu_time


def benchmark_float64(nelem):
    data = np.random.random(nelem).astype(np.float64)

    ts = timer()
    expected_res = data.sum()
    cpu_time = timer() - ts

    ts = timer()
    got_res = device_reduce_sum(data)
    gpu_time = timer() - ts

    np.allclose(got_res, expected_res)
    return cpu_time, gpu_time


def main():
    print('benchmark intp'.center(80, '='))
    for n in [100, 1000, 10000, 100000, 1000000, 10000000]:
        print('n = {0}'.format(n))
        for t in range(3):
            print(benchmark_intp(n))

    print('benchmark float64'.center(80, '='))
    for n in [100, 1000, 10000, 100000, 1000000, 10000000]:
        print('n = {0}'.format(n))
        for t in range(3):
            print(benchmark_float64(n))

    # Note: On Carrizo, speedup is attained at n=1,000,000


if __name__ == '__main__':
    main()
