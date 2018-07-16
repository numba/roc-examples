from __future__ import print_function, division, absolute_import
import numpy as np
from numba import roc, intp

_WARPSIZE = 64


@roc.jit(device=True)
def local_shuffle(tid, value, mask, temp):
    """
    * temp: shared array
        Size of the array must be at least the number of threads

    Note: This function must be called by all threads in the block
    """
    roc.barrier(0)
    temp[tid] = value
    roc.barrier(0)
    output = temp[mask]
    roc.barrier(0)
    return output


@roc.jit
def kernel_shuffle(arr, masks):
    tid = roc.get_local_id(0)
    temp = roc.shared.array(256, dtype=intp)
    val = arr[tid]
    mask = masks[tid]
    out = local_shuffle(tid, val, mask, temp)
    arr[tid] = out


@roc.jit(device=True)
def local_inclusive_scan(tid, value, nelem, temp):
    """
    * temp: shared array
        Size of the array must be at least the number of threads

    Note: This function must be called by all threads in the block
    """
    offset = 1
    while offset < nelem:
        mask = tid - offset if tid > offset else 0
        out = local_shuffle(tid, value, mask, temp)
        if offset <= tid:
            value += out
        offset *= 2
    return value


@roc.jit
def kernel_scan(values):
    tid = roc.get_local_id(0)
    nelem = values.size
    temp = roc.shared.array(256, dtype=intp)
    value = values[tid]
    out = local_inclusive_scan(tid, value, nelem, temp)
    values[tid] = out


###############################################################################
# Alternative implementation

@roc.jit(device=True)
def shuffle_up(val, width):
    tid = roc.get_local_id(0)
    roc.wavebarrier()
    res = roc.activelanepermute_wavewidth(val, tid - width, 0, False)
    return res


@roc.jit(device=True)
def broadcast(val, src):
    roc.wavebarrier()
    return roc.activelanepermute_wavewidth(val, src, 0, False)


@roc.jit(device=True)
def shuf_wave_inclusive_scan(val):
    tid = roc.get_local_id(0)
    lane = tid & (_WARPSIZE - 1)

    roc.wavebarrier()
    shuf = shuffle_up(val, 1)
    if lane >= 1:
        val += shuf

    roc.wavebarrier()
    shuf = shuffle_up(val, 2)
    if lane >= 2:
        val += shuf

    roc.wavebarrier()
    shuf = shuffle_up(val, 4)
    if lane >= 4:
        val += shuf

    roc.wavebarrier()
    shuf = shuffle_up(val, 8)
    if lane >= 8:
        val += shuf

    roc.wavebarrier()
    shuf = shuffle_up(val, 16)
    if lane >= 16:
        val += shuf

    roc.wavebarrier()
    shuf = shuffle_up(val, 32)
    if lane >= 32:
        val += shuf

    roc.wavebarrier()
    return val


@roc.jit(device=True)
def shuf_wave_exclusive_scan(val):
    tid = roc.get_local_id(0)
    lane = tid & (_WARPSIZE - 1)

    incl = shuf_wave_inclusive_scan(val)
    excl = shuffle_up(incl, 1)
    the_sum = broadcast(excl, 0)
    if lane == 0:
        excl = 0
    return excl, the_sum


@roc.jit(device=True)
def shuf_device_inclusive_scan(data, temp):
    """
    Args
    ----
    data: scalar
        input for tid
    temp: shared memory for temporary work, requires at least
    threadcount/wavesize storage
    """
    tid = roc.get_local_id(0)
    lane = tid & (_WARPSIZE - 1)
    warpid = tid >> 6

    roc.barrier()

    # Scan warps in parallel
    warp_scan_res = shuf_wave_inclusive_scan(data)

    roc.barrier()

    # Store partial sum into shared memory
    if lane == (_WARPSIZE - 1):
        temp[warpid] = warp_scan_res

    roc.barrier()

    # Scan the partial sum by first wave
    if warpid == 0:
        temp[lane] = shuf_wave_inclusive_scan(temp[lane])

    roc.barrier()

    # Get block sum for each wave
    blocksum = 0  # first wave is 0
    if warpid > 0:
        blocksum = temp[warpid - 1]

    return warp_scan_res + blocksum


@roc.jit(device=True)
def local_inclusive_scan_shuf(tid, value, nelem, temp):
    """
    * temp: shared array
        Size of the array must be at least the number of active wave

    Note: This function must be called by all threads in the block
    """
    roc.barrier()
    roc.wavebarrier()
    res = shuf_device_inclusive_scan(value, temp)
    roc.barrier()
    return res


@roc.jit
def kernel_scan_shuf(values):
    tid = roc.get_local_id(0)
    nelem = values.size
    temp = roc.shared.array(256, dtype=intp)
    value = values[tid]
    out = local_inclusive_scan(tid, value, nelem, temp)
    values[tid] = out


@roc.jit
def kernel_wave_excl_scan_shuf(values, out, psum):
    tid = roc.get_local_id(0)
    val = values[tid]
    out[tid], psum[tid] = shuf_wave_exclusive_scan(val)


###############################################################################
# Tests


def test_scan():
    arr = np.arange(256) + 1
    print('inp', arr)
    expect = np.cumsum(arr)
    kernel_scan[1, arr.size](arr)
    print('got', arr)
    print('expect', expect)
    np.testing.assert_equal(arr, expect)


def test_scan_shuf():
    arr = np.arange(256) + 1
    print('inp', arr)
    expect = np.cumsum(arr)
    kernel_scan_shuf[1, arr.size](arr)
    print('got', arr)
    print('expect', expect)
    np.testing.assert_equal(arr, expect)


def test_wave_excl_scan_shuf():
    arr = np.arange(64)
    out = np.arange(64)
    psum = np.arange(64)
    print('inp', arr)
    expect = np.cumsum(arr)
    kernel_wave_excl_scan_shuf[1, arr.size](arr, out, psum)
    print('out', out)
    print('psum', psum)
    print('expect', expect)
    np.testing.assert_equal(out[1:], expect[:-1])
    np.testing.assert_equal(psum, expect[-1])
    np.testing.assert_equal(out[0], 0)


def main():
    test_scan()
    test_scan_shuf()
    test_wave_excl_scan_shuf()


if __name__ == '__main__':
    main()
