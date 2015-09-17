from __future__ import print_function, division, absolute_import
import sys
from collections import defaultdict
from timeit import default_timer as timer
import numpy as np
from .sort_ref import RadixSortReference


class SortPass(object):
    RADIX = 4

    def __init__(self, sorter, chunksize, data_size, data_dtype):
        self._sorter = sorter
        self._chunksize = chunksize
        self._size = data_size
        self._dtype = data_dtype

        # Initialize states
        numchunk, chunksize = self.calculate_buffer_shape(data_size)

        self._bufsize = numchunk * chunksize

        self._shuffled = np.zeros((numchunk, chunksize), dtype=data_dtype)
        self._localscan = np.zeros((numchunk, chunksize), dtype=np.intp)

        self._blocksum = np.zeros((self.RADIX, numchunk), dtype=np.intp)
        self._scanblocksum = np.zeros_like(self._blocksum)
        self._shuffled_sorted = np.zeros(data_size, dtype=data_dtype)

        self._dummy_indices = np.empty(0, dtype=np.intp)

        self._timings = defaultdict(float)

    @property
    def timings(self):
        return self._timings

    def allocate_data_buffer(self):
        return np.zeros(self._bufsize, dtype=self._dtype)

    def run(self, data, shift, sorted):
        self._run(data, shift, sorted, self._dummy_indices,
                  self._dummy_indices, store_indices=False)

    def run_with_indices(self, data, shift, sorted, indices, sorted_indices):
        self._run(data, shift, sorted, indices[:self._size],
                  sorted_indices[:self._size], store_indices=True)

    def _run(self, data, shift, sorted, indices, sorted_indices, store_indices):
        sorter = self._sorter

        ts = timer()
        sorter.local_shuffle(data[:self._size],
                             self._size,
                             shift,
                             self._blocksum,
                             self._localscan,
                             self._shuffled,
                             indices[:self._size],
                             store_indices=store_indices)
        self._timings['local_shuffle'] += timer() - ts

        ts = timer()
        sorter.scan_block_sum(self._blocksum, self._scanblocksum)
        self._timings['scan_block_sum'] += timer() - ts

        ts = timer()
        sorter.scatter(self._size, shift, self._shuffled, self._scanblocksum,
                       self._localscan, sorted, indices[:self._size],
                       sorted_indices[:self._size], store_indices=store_indices)
        self._timings['scatter'] += timer() - ts

    def calculate_buffer_shape(self, nelem):
        chunksize = self._chunksize
        numchunk = (nelem + chunksize - 1) // chunksize
        return numchunk, chunksize


class DoubleBuffer(object):
    def __init__(self, front, back):
        self.front = front
        self.back = back

    def copy_to_front(self, data):
        self.front[:data.size] = data

    def swap(self):
        self.front, self.back = self.back, self.front


class RadixSortDriver(object):
    def __init__(self):
        self._sorter, self._chunksize = self._init_sorter()
        self._last_timings = None

    @property
    def last_timings(self):
        return self._last_timings

    @classmethod
    def _init_sorter(cls):
        """
        Override this to use a different sorter and chunksize.
        """
        return RadixSortReference(), 32

    def sort(self, data):
        data_bits = data.dtype.itemsize * 8

        sortpass = SortPass(self._sorter, self._chunksize, data.size,
                            data.dtype)

        # Allocate buffer
        buffers = DoubleBuffer(sortpass.allocate_data_buffer(),
                               sortpass.allocate_data_buffer())
        buffers.copy_to_front(data)

        # Run sort passes
        for shift in range(0, data_bits, 2):
            sortpass.run(buffers.front, shift, buffers.back)
            buffers.swap()

        self._last_timings = sortpass.timings
        return buffers.front[:data.size]

    def sort_with_indices(self, data):
        data_bits = data.dtype.itemsize * 8

        sortpass = SortPass(self._sorter, self._chunksize, data.size,
                            data.dtype)

        # Allocate buffer
        data_buffers = DoubleBuffer(sortpass.allocate_data_buffer(),
                                    sortpass.allocate_data_buffer())
        data_buffers.copy_to_front(data)

        indices_buffers = DoubleBuffer(np.arange(data.size, dtype=np.intp),
                                       np.zeros(data.size, dtype=np.intp))

        # Run sort passes
        for shift in range(0, data_bits, 2):
            sortpass.run_with_indices(data_buffers.front,
                                      shift,
                                      data_buffers.back,
                                      indices_buffers.front,
                                      indices_buffers.back)
            data_buffers.swap()
            indices_buffers.swap()

        self._last_timings = sortpass.timings
        return data_buffers.front[:data.size], indices_buffers.front


class HsaRadixSortDriver(RadixSortDriver):
    @classmethod
    def _init_sorter(cls):
        from .hsa_sort import HsaRadixSort, BLOCKSIZE

        return HsaRadixSort(), BLOCKSIZE


def full_sort_test_helper(driver, nelem, dtype=np.intp):
    print("testing full sort driver={0} nelem={1}".format(driver, nelem))
    data = np.random.randint(0, 0xffffffff, nelem).astype(dtype)
    sorter = driver()

    expected = np.sort(data, kind='mergesort')
    sorted = sorter.sort(data)
    np.testing.assert_equal(expected, sorted)


def full_radix_sort_test_template(driver):
    for nelem in [1, 3, 7, 11, 111, 128, 1024, 10000]:
        full_sort_test_helper(driver, nelem)


def test_cpu_full_radix_sort():
    full_radix_sort_test_template(RadixSortDriver)


def test_hsa_full_radix_sort():
    full_radix_sort_test_template(HsaRadixSortDriver)


def sort_tester(nelem):
    hi = np.random.randint(0, 0xffffffff, nelem).astype(np.uintp)
    lo = np.random.randint(0, 0xffffffff, nelem).astype(np.uintp)
    data = (hi << 64) | lo
    sorter = HsaRadixSortDriver()

    expected = np.sort(data)

    sorted = sorter.sort(data)
    print(sorted)

    np.testing.assert_equal(expected, sorted)


def argsort_tester(nelem):
    hi = np.random.randint(0, 0xffffffff, nelem).astype(np.uintp)
    lo = np.random.randint(0, 0xffffffff, nelem).astype(np.uintp)
    data = (hi << 64) | lo
    print(data.dtype)
    sorter = HsaRadixSortDriver()

    expected = np.sort(data, kind='mergesort')
    print(expected)

    sorted, indices = sorter.sort_with_indices(data)
    print(sorted)
    print(indices)

    np.testing.assert_equal(expected, sorted)
    np.testing.assert_equal(expected, data[indices])


def test_sort():
    for nelem in [10, 100, 1000]:
        sort_tester(nelem)


def test_argsort():
    for nelem in [10, 100, 1000]:
        argsort_tester(nelem)


def main_sort():
    nelem = int(sys.argv[1])
    print('nelem {0}'.format(nelem))
    sort_tester(nelem)


def main_argsort():
    nelem = int(sys.argv[1])
    print('nelem {0}'.format(nelem))
    argsort_tester(nelem)


if __name__ == '__main__':
    main_argsort()
