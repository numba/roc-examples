from __future__ import absolute_import, print_function, division

from numba import roc
from numba import intp, int32, uintp

from .sort_ref import RADIX, RADIX_MINUS_1
from .sort_ref import RadixSortReference, RadixSortCrossTester
from .roc_scan import local_inclusive_scan, shuf_wave_exclusive_scan

_WARPSIZE = 64


@roc.jit(device=True)
def blockwise_prefixsum_naive(data, nelem):
    last = data[nelem - 1]
    roc.barrier()

    tid = roc.get_local_id(0)

    if tid == 0:
        psum = 0
        for i in range(nelem):
            cur = data[i]
            data[i] = psum
            psum += cur

    roc.barrier()

    return last + data[nelem - 1]


@roc.jit(device=True)
def blockwise_prefixsum(value, temp, nelem):
    tid = roc.get_local_id(0)

    # inc_val = local_inclusive_scan_shuf(tid, value, nelem, data)
    inc_val = local_inclusive_scan(tid, value, nelem, temp)

    roc.barrier()
    if tid + 1 == nelem:
        # the last value stores the sum at index 0
        temp[0] = inc_val
    else:
        # the other value stores at the next slot for exclusive scan value
        temp[tid + 1] = inc_val

    roc.barrier()

    # Read the sum
    the_sum = temp[0]

    roc.barrier()

    # Reset first slot to zero
    if tid == 0:
        temp[0] = 0

    roc.barrier()
    return the_sum


BLOCKSIZE = _WARPSIZE * 4


class RocRadixSortKernels(object):
    def __init__(self, block_size=BLOCKSIZE):
        @roc.jit(device=True)
        def four_way_scan(data, sm_masks, sm_blocksum, blksz, valid):
            sm_chunkoffset = roc.shared.array(4, dtype=int32)

            tid = roc.get_local_id(0)

            laneid = tid & (_WARPSIZE - 1)
            warpid = tid >> 6

            my_digit = -1

            for digit in range(RADIX):
                sm_masks[digit, tid] = 0
                if valid and data == digit:
                    sm_masks[digit, tid] = 1
                    my_digit = digit

            roc.barrier()

            offset = 0
            base = 0
            while offset < blksz:
                # Exclusive scan
                if warpid < RADIX:
                    val = intp(sm_masks[warpid, offset + laneid])
                    cur, psum = shuf_wave_exclusive_scan(val)
                    sm_masks[warpid, offset + laneid] = cur + base
                    base += psum

                roc.barrier()
                offset += _WARPSIZE

            roc.barrier()

            # Store blocksum from the exclusive scan
            if warpid < RADIX and laneid == 0:
                sm_blocksum[warpid] = base

            roc.barrier()
            # Calc chunk offset (a short exclusive scan)
            if tid == 0:
                sm_chunkoffset[0] = 0
                sm_chunkoffset[1] = sm_blocksum[0]
                sm_chunkoffset[2] = sm_chunkoffset[1] + sm_blocksum[1]
                sm_chunkoffset[3] = sm_chunkoffset[2] + sm_blocksum[2]

            roc.barrier()
            # Prepare output
            chunk_offset = -1
            scanval = -1

            if my_digit != -1:
                chunk_offset = sm_chunkoffset[my_digit]
                scanval = sm_masks[my_digit, tid]

            roc.wavebarrier()
            roc.barrier()

            return chunk_offset, scanval

        mask_shape = 4, block_size

        assert block_size >= _WARPSIZE * 4

        @roc.jit
        def kernel_local_shuffle(data, size, shift, blocksum, localscan,
                                 shuffled, indices, store_indices):
            tid = roc.get_local_id(0)
            blkid = roc.get_group_id(0)
            blksz = localscan.shape[1]

            sm_mask = roc.shared.array(shape=mask_shape, dtype=int32)
            sm_blocksum = roc.shared.array(shape=4, dtype=int32)
            sm_shuffled = roc.shared.array(shape=block_size, dtype=uintp)
            sm_indices = roc.shared.array(shape=block_size, dtype=uintp)
            sm_localscan = roc.shared.array(shape=block_size, dtype=int32)
            sm_localscan[tid] = -1

            dataid = blkid * blksz + tid
            valid = dataid < size and tid < blksz
            curdata = uintp(data[dataid] if valid else uintp(0))
            processed_data = uintp((curdata >> uintp(shift)) &
                                   uintp(RADIX_MINUS_1))

            chunk_offset, scanval = four_way_scan(processed_data, sm_mask,
                                                  sm_blocksum, blksz, valid)

            if tid < RADIX:
                blocksum[tid, blkid] = sm_blocksum[tid]

            if tid < blksz:
                # Store local scan value
                where = chunk_offset + scanval
                # Store shuffled value and indices
                shuffled[blkid, where] = curdata
                if store_indices and valid:
                    sm_indices[where] = indices[dataid]
                sm_localscan[where] = scanval

            # Cleanup
            roc.barrier()
            if tid < blksz:
                # shuffled[blkid, tid] = sm_shuffled[tid]
                if store_indices and valid:
                    indices[dataid] = sm_indices[tid]
                localscan[blkid, tid] = sm_localscan[tid]

        @roc.jit
        def kernel_scatter(size, shift, shuffled, scanblocksum, localscan,
                           shuffled_sorted, indices, indices_sorted,
                           store_indices):
            tid = roc.get_local_id(0)
            blkid = roc.get_group_id(0)
            gid = roc.get_global_id(0)

            if gid < size:
                curdata = uintp(shuffled[blkid, tid])
                data_radix = uintp((curdata >> uintp(shift)) &
                                   uintp(RADIX_MINUS_1))
                pos = scanblocksum[data_radix, blkid] + localscan[blkid, tid]
                shuffled_sorted[pos] = curdata

                if store_indices:
                    indices_sorted[pos] = indices[gid]

        self.block_size = block_size
        self.kernel_local_shuffle = kernel_local_shuffle
        self.kernel_scatter = kernel_scatter


class RocRadixSort(RadixSortReference):
    _kernel = RocRadixSortKernels(block_size=BLOCKSIZE)

    def __init__(self):
        self.block_size = self._kernel.block_size
        self.kernel_local_shuffle = self._kernel.kernel_local_shuffle
        self.kernel_scatter = self._kernel.kernel_scatter

    def local_shuffle(self, data, size, shift, blocksum, localscan, shuffled,
                      indices, store_indices):
        numblock, blocksize = localscan.shape
        assert data.dtype == shuffled.dtype
        blocksize = max(blocksize, 4 * _WARPSIZE)
        assert blocksize <= self.block_size
        assert blocksize >= 4 * _WARPSIZE
        self.kernel_local_shuffle[numblock, blocksize](
            data, size, shift, blocksum, localscan, shuffled,
            indices, store_indices,
        )

    def scatter(self, size, shift, shuffled, scanblocksum, localscan,
                shuffled_sorted, indices, indices_sorted, store_indices):
        numblock, blocksize = shuffled.shape
        assert blocksize <= self.block_size
        assert shuffled.dtype == shuffled_sorted.dtype
        self.kernel_scatter[numblock, blocksize](
            size, shift, shuffled, scanblocksum, localscan, shuffled_sorted,
            indices, indices_sorted, store_indices
        )


def test_single_pass_against_reference():
    tester = RadixSortCrossTester(RadixSortReference, RocRadixSort)
    # Test sort
    tester.test_sort_reference_data()
    tester.test_sort_random(numchunk=4, chunksize=4)
    tester.test_sort_random(numchunk=4, chunksize=32)
    tester.test_sort_random(numchunk=4, chunksize=64)
    tester.test_sort_random(numchunk=4, chunksize=128)
    tester.test_sort_random(numchunk=1, chunksize=BLOCKSIZE)
    # Sized
    tester.test_sort_random(numchunk=4, chunksize=4, size=14)
    tester.test_sort_random(numchunk=4, chunksize=64, size=4 * 64 - 3)
    tester.test_sort_random(numchunk=4, chunksize=64, size=4 * 64 - 20)
    tester.test_sort_random(numchunk=2, chunksize=128, size=2 * 128 - 27)

