from __future__ import print_function, absolute_import, division

import numpy as np
from numba import hsa
from numba import intp, float64

WAVESIZE = 64
WAVEBITS = 6

NEG_INF = float('-inf')
POS_INF = float('+inf')


@hsa.jit(device=True)
def shuffle_down(val, width):
    tid = hsa.get_local_id(0)
    hsa.wavebarrier()
    res = hsa.activelanepermute_wavewidth(val, tid + width, 0, False)
    return res


@hsa.jit(device=True)
def broadcast(val, src):
    hsa.wavebarrier()
    return hsa.activelanepermute_wavewidth(val, src, 0, False)


@hsa.jit(device=True)
def wave_reduce_sum(val):
    """
    First thread in wave gets the result
    """
    width = WAVESIZE // 2
    while width > 0:
        val += shuffle_down(val, width)
        width //= 2

    return val


@hsa.jit(device=True)
def wave_reduce_max(val):
    """
    First thread in wave gets the result
    """
    width = WAVESIZE // 2
    while width > 0:
        val = max(val, shuffle_down(val, width))
        width //= 2

    return val


@hsa.jit(device=True)
def wave_reduce_min(val):
    """
    First thread in wave gets the result
    """
    width = WAVESIZE // 2
    while width > 0:
        val = min(val, shuffle_down(val, width))
        width //= 2

    return val


@hsa.jit(device=True)
def wave_reduce_sum_all(val):
    """
    All threads get the result
    """
    res = wave_reduce_sum(val)
    return broadcast(res, 0)


@hsa.jit(device=True)
def wave_reduce_max_all(val):
    """
    All threads get the result
    """
    res = wave_reduce_max(val)
    return broadcast(res, 0)


@hsa.jit(device=True)
def wave_reduce_min_all(val):
    """
    All threads get the result
    """
    res = wave_reduce_min(val)
    return broadcast(res, 0)


def test_wave_reduce_sum():
    @hsa.jit
    def test_wave_reduce(inp, out):
        tid = hsa.get_local_id(0)
        val = inp[tid]
        out[tid] = wave_reduce_sum_all(val)

    # Test one wave
    inp = np.arange(WAVESIZE, dtype=np.intp)
    out = np.zeros_like(inp)

    test_wave_reduce[1, WAVESIZE](inp, out)
    np.testing.assert_equal(inp.sum(), out)

    # Test two waves
    inp = np.arange(2 * WAVESIZE, dtype=np.intp)
    out = np.zeros_like(inp)

    test_wave_reduce[1, 2 * WAVESIZE](inp, out)
    np.testing.assert_equal(inp[:WAVESIZE].sum(), out[:WAVESIZE])
    np.testing.assert_equal(inp[WAVESIZE:].sum(), out[WAVESIZE:])


def test_wave_reduce_max():
    @hsa.jit
    def test_wave_reduce(inp, out):
        tid = hsa.get_local_id(0)
        val = inp[tid]
        out[tid] = wave_reduce_max_all(val)

    # Test one wave
    inp = np.arange(WAVESIZE, dtype=np.intp)
    out = np.zeros_like(inp)

    test_wave_reduce[1, WAVESIZE](inp, out)
    np.testing.assert_equal(inp.max(), out)

    # Test two waves
    inp = np.arange(2 * WAVESIZE, dtype=np.intp)
    out = np.zeros_like(inp)

    test_wave_reduce[1, 2 * WAVESIZE](inp, out)
    np.testing.assert_equal(inp[:WAVESIZE].max(), out[:WAVESIZE])
    np.testing.assert_equal(inp[WAVESIZE:].max(), out[WAVESIZE:])


def test_wave_reduce_min():
    @hsa.jit
    def test_wave_reduce(inp, out):
        tid = hsa.get_local_id(0)
        val = inp[tid]
        out[tid] = wave_reduce_min_all(val)

    # Test one wave
    inp = np.arange(WAVESIZE, dtype=np.intp)
    out = np.zeros_like(inp)

    test_wave_reduce[1, WAVESIZE](inp, out)
    np.testing.assert_equal(inp.min(), out)

    # Test two waves
    inp = np.arange(2 * WAVESIZE, dtype=np.intp)
    out = np.zeros_like(inp)

    test_wave_reduce[1, 2 * WAVESIZE](inp, out)
    np.testing.assert_equal(inp[:WAVESIZE].min(), out[:WAVESIZE])
    np.testing.assert_equal(inp[WAVESIZE:].min(), out[WAVESIZE:])


def test_wave_reduce_min_real():
    @hsa.jit
    def test_wave_reduce(inp, out):
        tid = hsa.get_local_id(0)
        val = inp[tid]
        out[tid] = wave_reduce_min_all(val)

    # Test one wave
    inp = np.random.random(WAVESIZE).astype(np.float64)
    out = np.zeros_like(inp)

    test_wave_reduce[1, WAVESIZE](inp, out)
    np.testing.assert_equal(inp.min(), out)

    # Test two waves
    inp = np.random.random(2 * WAVESIZE).astype(np.float64)
    out = np.zeros_like(inp)

    test_wave_reduce[1, 2 * WAVESIZE](inp, out)
    np.testing.assert_equal(inp[:WAVESIZE].min(), out[:WAVESIZE])
    np.testing.assert_equal(inp[WAVESIZE:].min(), out[WAVESIZE:])


def group_reduce_sum_builder(dtype):
    """
    Build reducer of max 64*64 threads per group.
    """

    @hsa.jit(device=True)
    def group_reduce_sum(val):
        """
        First thread of first wave get the result
        """
        tid = hsa.get_local_id(0)
        blksz = hsa.get_local_size(0)
        wid = tid >> WAVEBITS
        lane = tid & (WAVESIZE - 1)

        sm_partials = hsa.shared.array(WAVESIZE, dtype=dtype)

        val = wave_reduce_sum(val)

        if lane == 0:
            sm_partials[wid] = val

        hsa.barrier()

        val = sm_partials[lane] if tid < (blksz // WAVESIZE) else 0

        if wid == 0:
            val = wave_reduce_sum(val)

        return val

    return group_reduce_sum


def group_reduce_max_builder(dtype):
    """
    Build reducer of max 64*64 threads per group.
    """

    @hsa.jit(device=True)
    def group_reduce_max(val):
        """
        First thread of first wave get the result
        """
        tid = hsa.get_local_id(0)
        blksz = hsa.get_local_size(0)
        wid = tid >> WAVEBITS
        lane = tid & (WAVESIZE - 1)

        sm_partials = hsa.shared.array(WAVESIZE, dtype=dtype)

        val = wave_reduce_max(val)

        if lane == 0:
            sm_partials[wid] = val

        hsa.barrier()

        val = sm_partials[lane] if tid < (blksz // WAVESIZE) else dtype(NEG_INF)

        if wid == 0:
            val = wave_reduce_max(val)

        return val

    return group_reduce_max


def group_reduce_min_builder(dtype):
    """
    Build reducer of min 64*64 threads per group.
    """

    @hsa.jit(device=True)
    def group_reduce_min(val):
        """
        First thread of first wave get the result
        """
        tid = hsa.get_local_id(0)
        blksz = hsa.get_local_size(0)
        wid = tid >> WAVEBITS
        lane = tid & (WAVESIZE - 1)

        sm_partials = hsa.shared.array(WAVESIZE, dtype=dtype)

        val = wave_reduce_min(val)

        if lane == 0:
            sm_partials[wid] = val

        hsa.barrier()

        val = sm_partials[lane] if tid < (blksz // WAVESIZE) else dtype(POS_INF)

        if wid == 0:
            val = wave_reduce_min(val)

        return val

    return group_reduce_min


group_reduce_sum_intp = group_reduce_sum_builder(intp)
group_reduce_sum_float64 = group_reduce_sum_builder(float64)

group_reduce_max_intp = group_reduce_max_builder(intp)
group_reduce_max_float64 = group_reduce_max_builder(float64)

group_reduce_min_intp = group_reduce_min_builder(intp)
group_reduce_min_float64 = group_reduce_min_builder(float64)


def test_group_reduce_sum_intp():
    @hsa.jit
    def test_group_reduce(inp, out):
        gid = hsa.get_global_id(0)
        val = inp[gid]
        val = group_reduce_sum_intp(val)
        out[gid] = val

    inp = np.arange(4 * WAVESIZE, dtype=np.intp)
    out = np.zeros_like(inp)
    test_group_reduce[1, 4 * WAVESIZE](inp, out)
    np.testing.assert_equal(inp.sum(), out[0])


def test_group_reduce_sum_float64():
    @hsa.jit
    def test_group_reduce(inp, out):
        gid = hsa.get_global_id(0)
        val = inp[gid]
        val = group_reduce_sum_float64(val)
        out[gid] = val

    inp = np.linspace(0, 1, 4 * WAVESIZE).astype(np.float64)
    out = np.zeros_like(inp)
    test_group_reduce[1, 4 * WAVESIZE](inp, out)
    np.testing.assert_equal(inp.sum(), out[0])


def test_group_reduce_max_intp():
    @hsa.jit
    def test_group_reduce(inp, out):
        gid = hsa.get_global_id(0)
        val = inp[gid]
        val = group_reduce_max_intp(val)
        out[gid] = val

    inp = np.arange(4 * WAVESIZE, dtype=np.intp)
    out = np.zeros_like(inp)
    test_group_reduce[1, 4 * WAVESIZE](inp, out)
    np.testing.assert_equal(inp.max(), out[0])


def test_group_reduce_max_float64():
    @hsa.jit
    def test_group_reduce(inp, out):
        gid = hsa.get_global_id(0)
        val = inp[gid]
        val = group_reduce_max_float64(val)
        out[gid] = val

    inp = np.linspace(0, 1, 4 * WAVESIZE).astype(np.float64)
    out = np.zeros_like(inp)
    test_group_reduce[1, 4 * WAVESIZE](inp, out)
    np.testing.assert_equal(inp.max(), out[0])


def test_group_reduce_min_intp():
    @hsa.jit
    def test_group_reduce(inp, out):
        gid = hsa.get_global_id(0)
        val = inp[gid]
        val = group_reduce_min_intp(val)
        out[gid] = val

    inp = np.arange(4 * WAVESIZE, dtype=np.intp)
    out = np.zeros_like(inp)
    test_group_reduce[1, 4 * WAVESIZE](inp, out)
    np.testing.assert_equal(inp.min(), out[0])


def test_group_reduce_min_float64():
    @hsa.jit
    def test_group_reduce(inp, out):
        gid = hsa.get_global_id(0)
        val = inp[gid]
        val = group_reduce_min_float64(val)
        out[gid] = val

    inp = np.random.random(4 * WAVESIZE).astype(np.float64)
    out = np.zeros_like(inp)
    test_group_reduce[1, 4 * WAVESIZE](inp, out)
    np.testing.assert_equal(inp.min(), out[0])


def kernel_reduce_sum_builder(dtype):
    group_fn_map = {
        intp: group_reduce_sum_intp,
        float64: group_reduce_sum_float64,
    }

    group_reducer = group_fn_map[dtype]

    @hsa.jit
    def kernel_reduce_sum(inp, out, nelem):
        tid = hsa.get_local_id(0)
        blkid = hsa.get_group_id(0)
        blksz = hsa.get_local_size(0)
        numgroup = hsa.get_num_groups(0)

        i = blkid * blksz + tid

        accum = dtype(0)
        while i < nelem:
            accum += inp[i]
            i += blksz * numgroup

        accum = group_reducer(accum)
        if tid == 0:
            out[blkid] = accum

    return kernel_reduce_sum


def kernel_reduce_max_builder(dtype):
    group_fn_map = {
        intp: group_reduce_max_intp,
        float64: group_reduce_max_float64,
    }

    group_reducer = group_fn_map[dtype]

    @hsa.jit
    def kernel_reduce_max(inp, out, nelem):
        tid = hsa.get_local_id(0)
        blkid = hsa.get_group_id(0)
        blksz = hsa.get_local_size(0)
        numgroup = hsa.get_num_groups(0)

        i = blkid * blksz + tid

        accum = dtype(NEG_INF)
        while i < nelem:
            accum = max(inp[i], accum)
            i += blksz * numgroup

        accum = group_reducer(accum)
        if tid == 0:
            out[blkid] = accum

    return kernel_reduce_max


def kernel_reduce_min_builder(dtype):
    group_fn_map = {
        intp: group_reduce_min_intp,
        float64: group_reduce_min_float64,
    }

    group_reducer = group_fn_map[dtype]

    @hsa.jit
    def kernel_reduce_min(inp, out, nelem):
        tid = hsa.get_local_id(0)
        blkid = hsa.get_group_id(0)
        blksz = hsa.get_local_size(0)
        numgroup = hsa.get_num_groups(0)

        i = blkid * blksz + tid

        accum = dtype(POS_INF)
        while i < nelem:
            accum = min(accum, inp[i])
            i += blksz * numgroup

        accum = group_reducer(accum)
        if tid == 0:
            out[blkid] = accum

    return kernel_reduce_min


kernel_reduce_sum_intp = kernel_reduce_sum_builder(intp)
kernel_reduce_sum_float64 = kernel_reduce_sum_builder(float64)

kernel_reduce_max_intp = kernel_reduce_max_builder(intp)
kernel_reduce_max_float64 = kernel_reduce_max_builder(float64)

kernel_reduce_min_intp = kernel_reduce_min_builder(intp)
kernel_reduce_min_float64 = kernel_reduce_min_builder(float64)

_device_reduce_sum_map = {
    np.dtype(np.intp): kernel_reduce_sum_intp,
    np.dtype(np.float64): kernel_reduce_sum_float64,
}


def device_reduce_sum(inp):
    """
    Run device-wide reduction on the given array
    """
    # Select kernel
    reduce_kernel = _device_reduce_sum_map[inp.dtype]

    # Run reduction
    nelem = inp.size

    threads = WAVESIZE * 8
    partial_size = threads
    groups = min((nelem + threads - 1) // threads, partial_size)

    partials = np.zeros(groups, dtype=inp.dtype)
    reduce_kernel[groups, threads](inp, partials, nelem)
    if groups < partial_size // 2:
        # Less than half is filled
        return partials.sum()
    else:
        reduce_kernel[1, partial_size](partials, partials, groups)
    return partials[0]


def test_device_reduce_sum_intp():
    inp = np.arange(10000, dtype=np.intp)
    out = device_reduce_sum(inp)

    assert out == inp.sum()


def test_device_reduce_sum_float64():
    inp = np.linspace(0, 1, 10000, dtype=np.float64)
    out = device_reduce_sum(inp)

    assert np.allclose(out, inp.sum())


_device_reduce_max_map = {
    np.dtype(np.intp): kernel_reduce_max_intp,
    np.dtype(np.float64): kernel_reduce_max_float64,
}


def device_reduce_max(inp):
    """
    Run device-wide reduction on the given array
    """
    # Select kernel
    reduce_kernel = _device_reduce_max_map[inp.dtype]
    # Run reduction
    nelem = inp.size

    threads = WAVESIZE * 8
    partial_size = threads
    groups = min((nelem + threads - 1) // threads, partial_size)

    partials = np.zeros(groups, dtype=inp.dtype)
    reduce_kernel[groups, threads](inp, partials, nelem)
    if groups < partial_size // 2:
        # Less than half is filled
        return partials.max()
    else:
        reduce_kernel[1, partial_size](partials, partials, groups)
    return partials[0]


def test_device_reduce_max_intp():
    inp = np.arange(10000, dtype=np.intp)
    out = device_reduce_max(inp)

    assert out == inp.max()


def test_device_reduce_max_float64():
    inp = np.random.random(10000).astype(np.float64)
    out = device_reduce_max(inp)

    assert out == inp.max()


_device_reduce_min_map = {
    np.dtype(np.intp): kernel_reduce_min_intp,
    np.dtype(np.float64): kernel_reduce_min_float64,
}


def device_reduce_min(inp):
    """
    Run device-wide reduction on the given array
    """
    # Select kernel
    reduce_kernel = _device_reduce_min_map[inp.dtype]

    # Run reduction
    nelem = inp.size

    threads = WAVESIZE * 8
    partial_size = threads
    groups = min((nelem + threads - 1) // threads, partial_size)

    partials = np.zeros(groups, dtype=inp.dtype)

    reduce_kernel[groups, threads](inp, partials, nelem)
    
    if groups < partial_size // 2:
        # Less than half is filled
        return partials.min()
    else:
        reduce_kernel[1, partial_size](partials, partials, groups)
    return partials[0]


def test_device_reduce_min_intp():
    inp = np.arange(10000, dtype=np.intp)
    out = device_reduce_min(inp)

    assert out == inp.min()


def test_device_reduce_min_float64():
    inp = np.random.random(10000).astype(np.float64)
    out = device_reduce_min(inp)
    assert out == inp.min()
