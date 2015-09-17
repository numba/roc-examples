from __future__ import print_function, division, absolute_import
import numpy as np

RADIX = 4
RADIX_MINUS_1 = RADIX - 1


class RadixSortReference(object):
    """
    Reference implementation of radixsort
    """

    def local_shuffle(self, data, size, shift, blocksum, localscan, shuffled,
                      indices, store_indices):
        numchunk, chunksize = localscan.shape
        chunk_offset = np.zeros(numchunk, dtype=np.intp)

        templocalscan = np.zeros_like(localscan)

        old_indices = indices.copy()

        for digit in range(RADIX):
            mask = np.zeros_like(shuffled)
            mask.ravel()[:size] = (((data >> shift) & RADIX_MINUS_1) == digit)

            chk_scan_inc = mask.cumsum(axis=1)
            chk_scan_exc = np.zeros_like(chk_scan_inc)
            chk_scan_exc[:, 1:] = chk_scan_inc[:, :-1]
            blocksum[digit] = chk_scan_inc[:, -1]

            templocalscan[mask == 1] = chk_scan_exc[mask == 1]

            for chunkid in range(numchunk):
                for id in range(chunksize):
                    if mask[chunkid, id]:
                        where = (chk_scan_exc[chunkid, id] +
                                 chunk_offset[chunkid])
                        shuffled[chunkid, where] = \
                            data[chunkid * chunksize + id]

                        localscan[chunkid, where] = templocalscan[chunkid, id]
                        if store_indices:
                            indices[chunkid * chunksize + where] = \
                                old_indices[chunkid * chunksize + id]

            chunk_offset += blocksum[digit]

        shuffled.ravel()[size:] = -1
        localscan.ravel()[size:] = -1

    def scan_block_sum(self, blocksum, scanblocksum):
        np.cumsum(blocksum.ravel()[:-1], out=scanblocksum.ravel()[1:])

    def scatter(self, size, shift, shuffled, scanblocksum, localscan,
                shuffled_sorted, indices, indices_sorted, store_indices):
        numchunk, chunksize = localscan.shape
        data_radix = np.zeros(shuffled.size, dtype=np.int8)
        data_radix[:size] = ((shuffled.ravel()[:size] >> shift) & RADIX_MINUS_1)
        data_chunk = np.arange(data_radix.size) // chunksize
        globalpos = scanblocksum[data_radix, data_chunk] + localscan.ravel()
        shuffled_sorted.ravel()[globalpos.ravel()[:size]] = shuffled.ravel()[
                                                            :size]
        if store_indices:
            indices_sorted.ravel()[globalpos.ravel()[:size]] = indices.ravel()[
                                                               :size]


class RadixSorterTester(object):
    """
    Test a single pass of the radixsort
    """

    def __init__(self, sorter_class):
        self.sorter_class = sorter_class

    def init_reference_data(self):
        chunksize = 4
        data = np.array([1, 2, 0, 3,
                         0, 1, 1, 0,
                         3, 3, 3, 2,
                         1, 2, 2, 0,
                         2, 0, 0, 2])
        self._init_data(data=data, chunksize=chunksize, size=data.size)

    def init_random_data(self, numchunk, chunksize, size=None):
        size = chunksize * numchunk if size is None else size
        data = np.random.randint(0, RADIX, size).astype(np.int32)
        assert not np.any(data >= RADIX)
        assert not np.any(data < 0)
        self._init_data(data=data, chunksize=chunksize, size=size)

    def init_copy_from(self, othertest):
        self._init_data(data=othertest.data.copy(),
                        chunksize=othertest.localscan.shape[1],
                        size=othertest.size)

    def _init_data(self, data, chunksize, size):
        expected_sorted = np.sort(data[:size], kind='mergesort')
        expected_argsorted = np.argsort(data[:size], kind='mergesort')
        numchunk = (data.size + chunksize - 1) // chunksize

        shuffled = np.zeros((numchunk, chunksize), dtype=data.dtype)
        localscan = np.zeros((numchunk, chunksize), dtype=np.intp)

        blocksum = np.zeros((RADIX, numchunk), dtype=np.intp)
        scanblocksum = np.zeros_like(blocksum)
        shuffled_sorted = np.zeros_like(data.ravel())
        indices = np.arange(size, dtype=np.intp)
        indices_sorted = np.zeros_like(indices)

        self.data = data
        self.size = size
        self.expected_sorted = expected_sorted
        self.expected_argsorted = expected_argsorted
        self.shuffled = shuffled
        self.localscan = localscan
        self.scanblocksum = scanblocksum
        self.shuffled_sorted = shuffled_sorted
        self.indices = indices
        self.indices_sorted = indices_sorted
        self.blocksum = blocksum

    def test_sort(self):
        # Construct an instance
        sorter = self.sorter_class()

        shift = 0

        # Normal sort sequence
        # local_shuffle -> scan_block_sum -> scatter

        sorter.local_shuffle(self.data, self.size, shift, self.blocksum,
                             self.localscan, self.shuffled, self.indices,
                             store_indices=True)

        print(self.data)
        print('shuffled')
        print(self.shuffled)

        print('indices')
        print(self.indices)

        print('blocksum')
        print(self.blocksum)

        print('local scan')
        print(self.localscan)

        sorter.scan_block_sum(self.blocksum, self.scanblocksum)

        print('scanblocksum')
        print(self.scanblocksum)

        sorter.scatter(self.size, shift, self.shuffled, self.scanblocksum,
                       self.localscan, self.shuffled_sorted, self.indices,
                       self.indices_sorted, store_indices=True)

        print('shuffled_sorted')
        print(self.shuffled_sorted)

        np.testing.assert_equal(self.shuffled_sorted[:self.size],
                                self.expected_sorted)

        idx = self.indices_sorted
        np.testing.assert_equal(self.data[idx], self.expected_sorted)


class RadixSortCrossTester(object):
    """
    Cross testing two implementation
    """

    def __init__(self, refimpl, testimpl):
        self.refimpl = refimpl
        self.testimpl = testimpl

        self.ref_test = RadixSorterTester(self.refimpl)
        self.test_test = RadixSorterTester(self.testimpl)

    def test_sort_reference_data(self):
        self.ref_test.init_reference_data()
        self.test_test.init_reference_data()
        self._run_sort()

    def test_sort_random(self, numchunk, chunksize, size=None):
        print('test_sort_random numchunk={0} chunksize={1} size={2}'.format(
            numchunk, chunksize, size))
        self.ref_test.init_random_data(numchunk=numchunk, chunksize=chunksize,
                                       size=size)
        self.test_test.init_copy_from(self.ref_test)
        self._run_sort()

    def test_sort_random_sized(self):
        self.ref_test.init_reference_data()
        self.test_test.init_reference_data()
        self._run_sort()

    def _run_sort(self):
        ref_obj = self.ref_test.sorter_class()
        test_obj = self.test_test.sorter_class()

        shift = 0

        ref_obj.local_shuffle(self.ref_test.data,
                              self.ref_test.size,
                              shift,
                              self.ref_test.blocksum,
                              self.ref_test.localscan,
                              self.ref_test.shuffled,
                              self.ref_test.indices,
                              store_indices=True)

        test_obj.local_shuffle(self.test_test.data,
                               self.test_test.size,
                               shift,
                               self.test_test.blocksum,
                               self.test_test.localscan,
                               self.test_test.shuffled,
                               self.test_test.indices,
                               store_indices=True)

        np.testing.assert_equal(self.ref_test.localscan,
                                self.test_test.localscan)

        np.testing.assert_equal(self.ref_test.indices,
                                self.test_test.indices)

        np.testing.assert_equal(self.ref_test.blocksum,
                                self.test_test.blocksum)

        np.testing.assert_equal(
            self.ref_test.shuffled.ravel()[:self.ref_test.size],
            self.test_test.shuffled.ravel()[:self.ref_test.size])

        ref_obj.scan_block_sum(self.ref_test.blocksum,
                               self.ref_test.scanblocksum)

        test_obj.scan_block_sum(self.test_test.blocksum,
                                self.test_test.scanblocksum)

        np.testing.assert_equal(self.ref_test.scanblocksum,
                                self.test_test.scanblocksum)

        ref_obj.scatter(self.ref_test.size,
                        shift,
                        self.ref_test.shuffled,
                        self.ref_test.scanblocksum,
                        self.ref_test.localscan,
                        self.ref_test.shuffled_sorted,
                        self.ref_test.indices,
                        self.ref_test.indices_sorted,
                        store_indices=True)

        test_obj.scatter(self.test_test.size,
                         shift,
                         self.test_test.shuffled,
                         self.test_test.scanblocksum,
                         self.test_test.localscan,
                         self.test_test.shuffled_sorted,
                         self.test_test.indices,
                         self.test_test.indices_sorted,
                         store_indices=True)

        np.testing.assert_equal(
            self.ref_test.shuffled_sorted.ravel()[:self.ref_test.size],
            self.test_test.shuffled_sorted.ravel()[:self.test_test.size])

        np.testing.assert_equal(self.ref_test.indices_sorted,
                                self.test_test.indices_sorted)


def check_implementation(impcls):
    tester = RadixSorterTester(impcls)

    # Test reference data
    tester.init_reference_data()
    tester.test_sort()

    # # Test small random data
    tester.init_random_data(numchunk=4, chunksize=4)
    tester.test_sort()
    tester.init_random_data(numchunk=4, chunksize=32)
    tester.test_sort()

    # Test bigger random data
    tester.init_random_data(numchunk=4, chunksize=64)
    tester.test_sort()

    # Test extra random data
    tester.init_random_data(numchunk=4, chunksize=4, size=14)
    tester.test_sort()


def test_reference_implementation():
    check_implementation(RadixSortReference)
