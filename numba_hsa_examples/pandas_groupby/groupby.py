from __future__ import print_function, division, absolute_import

import numpy as np
from pandas.core.groupby import Grouper, BaseGrouper, Grouping, _is_label_like
from pandas.core.index import Index, MultiIndex

from pandas import compat
from pandas.core.series import Series
from pandas.core.frame import DataFrame
import pandas.core.common as com

from numba_hsa_examples.radixsort.sort_driver import HsaRadixSortDriver
from numba import hsa, jit

from numba_hsa_examples.reduction.reduction import (device_reduce_sum,
                                                    device_reduce_max,
                                                    device_reduce_min)

import logging

_logger = logging.getLogger(__name__)


class HSAGrouper(Grouper):
    def __init__(self, *args, **kwargs):
        kwargs['sort'] = True
        super(HSAGrouper, self).__init__(*args, **kwargs)

    def _get_grouper(self, obj):

        """
        Parameters
        ----------
        obj : the subject object

        Returns
        -------
        a tuple of binner, grouper, obj (possibly sorted)
        """

        self._set_grouper(obj)
        self.grouper, exclusions, self.obj = _get_grouper(self.obj, [self.key],
                                                          axis=self.axis,
                                                          level=self.level,
                                                          sort=self.sort)
        return self.binner, self.grouper, self.obj

    def _set_grouper(self, obj, sort=False):
        """
        given an object and the specifcations, setup the internal grouper for this particular specification

        Parameters
        ----------
        obj : the subject object

        """

        # NOTE: the following code is based on the base Grouper class with
        #       additional hook to specify custom sorter

        if self.key is not None and self.level is not None:
            raise ValueError(
                "The Grouper cannot specify both a key and a level!")

        # the key must be a valid info item
        if self.key is not None:
            key = self.key
            if key not in obj._info_axis:
                raise KeyError("The grouper name {0} is not found".format(key))
            ax = Index(obj[key], name=key)

        else:
            ax = obj._get_axis(self.axis)
            if self.level is not None:
                level = self.level

                # if a level is given it must be a mi level or
                # equivalent to the axis name
                if isinstance(ax, MultiIndex):
                    level = ax._get_level_number(level)
                    ax = Index(ax.get_level_values(level), name=ax.names[level])

                else:
                    if level not in (0, ax.name):
                        raise ValueError(
                            "The level {0} is not valid".format(level))

        # possibly sort
        if (self.sort or sort) and not ax.is_monotonic:
            # The following line is different from the base class for
            # possible extension.
            ax, indexer = self._make_sorter(ax)
            self.indexer = indexer
            obj = obj.take(indexer, axis=self.axis, convert=False,
                           is_copy=False)

        self.obj = obj
        self.grouper = ax
        return self.grouper

    def _make_sorter(self, ax):
        """
        Returns the index that would sort the given axis `ax`.
        """
        np_array = ax.get_values()
        # return np_array.argsort()
        # ax = ax.take(indexer)
        sorter = HsaRadixSortDriver()
        sorted_array, indices = sorter.sort_with_indices(np_array)
        return sorted_array, indices


def _get_grouper(obj, key=None, axis=0, level=None, sort=True):
    """
    create and return a BaseGrouper, which is an internal
    mapping of how to create the grouper indexers.
    This may be composed of multiple Grouping objects, indicating
    multiple groupers

    Groupers are ultimately index mappings. They can originate as:
    index mappings, keys to columns, functions, or Groupers

    Groupers enable local references to axis,level,sort, while
    the passed in axis, level, and sort are 'global'.

    This routine tries to figure of what the passing in references
    are and then creates a Grouping for each one, combined into
    a BaseGrouper.

    """

    # The implementation is essentially the same as pandas.core.groupby

    group_axis = obj._get_axis(axis)

    # validate thatthe passed level is compatible with the passed
    # axis of the object
    if level is not None:
        if not isinstance(group_axis, MultiIndex):
            if isinstance(level, compat.string_types):
                if obj.index.name != level:
                    raise ValueError('level name %s is not the name of the '
                                     'index' % level)
            elif level > 0:
                raise ValueError('level > 0 only valid with MultiIndex')

            level = None
            key = group_axis

    # a passed in Grouper, directly convert
    if isinstance(key, Grouper):
        binner, grouper, obj = key._get_grouper(obj)
        if key.key is None:
            return grouper, [], obj
        else:
            return grouper, set([key.key]), obj

    # already have a BaseGrouper, just return it
    elif isinstance(key, BaseGrouper):
        return key, [], obj

    if not isinstance(key, (tuple, list)):
        keys = [key]
    else:
        keys = key

    # what are we after, exactly?
    match_axis_length = len(keys) == len(group_axis)
    any_callable = any(callable(g) or isinstance(g, dict) for g in keys)
    any_arraylike = any(isinstance(g, (list, tuple, Series, Index, np.ndarray))
                        for g in keys)

    try:
        if isinstance(obj, DataFrame):
            all_in_columns = all(g in obj.columns for g in keys)
        else:
            all_in_columns = False
    except Exception:
        all_in_columns = False

    if (not any_callable and not all_in_columns
        and not any_arraylike and match_axis_length
        and level is None):
        keys = [com._asarray_tuplesafe(keys)]

    if isinstance(level, (tuple, list)):
        if key is None:
            keys = [None] * len(level)
        levels = level
    else:
        levels = [level] * len(keys)

    groupings = []
    exclusions = []

    # if the actual grouper should be obj[key]
    def is_in_axis(key):
        if not _is_label_like(key):
            try:
                obj._data.items.get_loc(key)
            except Exception:
                return False

        return True

    # if the the grouper is obj[name]
    def is_in_obj(gpr):
        try:
            return id(gpr) == id(obj[gpr.name])
        except Exception:
            return False

    for i, (gpr, level) in enumerate(zip(keys, levels)):

        if is_in_obj(gpr):  # df.groupby(df['name'])
            in_axis, name = True, gpr.name
            exclusions.append(name)

        elif is_in_axis(gpr):  # df.groupby('name')
            in_axis, name, gpr = True, gpr, obj[gpr]
            exclusions.append(name)

        else:
            in_axis, name = False, None

        if com.is_categorical_dtype(gpr) and len(gpr) != len(obj):
            raise ValueError(
                "Categorical dtype grouper must have len(grouper) == len(data)")

        ping = Grouping(group_axis, gpr, obj=obj, name=name,
                        level=level, sort=sort, in_axis=in_axis)

        groupings.append(ping)

    if len(groupings) == 0:
        raise ValueError('No group keys passed!')

    # create the internals grouper
    # Modified to insert CustomGrouper
    grouper = CustomGrouper(group_axis, groupings, sort=sort)
    return grouper, exclusions, obj


class CustomGrouper(BaseGrouper):
    def _aggregate(self, result, counts, values, labels, agg_func, is_numeric):
        _logger.info("_aggregate %s", agg_func)
        # NOTE: Intercept aggregate that has a HSA equivalent
        # The rest are the same as the base class. XXX call base class, perhaps?
        comp_ids, _, ngroups = self.group_info
        if values.ndim > 3:
            # punting for now
            raise NotImplementedError("number of dimensions is currently "
                                      "limited to 3")
        elif values.ndim > 2:
            for i, chunk in enumerate(values.transpose(2, 0, 1)):
                chunk = chunk.squeeze()
                agg_func(result[:, :, i], counts, chunk, comp_ids)
        else:
            try:
                fn = _optimized_aggregate_functions.get(agg_func.__name__,
                                                        agg_func)
                fn(result, counts, values, comp_ids)
            except:
                _logger.exception("HSA cusotm grouper failed with exception")
                raise

        return result


SPEED_BARRIER = 10 ** 5 * 5


def _hsa_group_agg(cpu_agg, gpu_agg, result, counts, values, comp_ids):
    assert comp_ids.size == values.shape[0]
    assert values.shape[1] == result.shape[1]

    from timeit import default_timer as timer

    group_count(counts, comp_ids)

    start = 0
    for i, stop in enumerate(counts):
        for j in range(values.shape[1]):
            inputs = values[start:stop, j]
            if inputs.size < SPEED_BARRIER:
                res = cpu_agg(inputs)
            else:
                tss = timer()
                res = gpu_agg(inputs)
                _logger.debug("%s runtime %s", gpu_agg.__name__, timer() - tss)
            result[i, j] = res

        # next
        start = stop


def hsa_group_mean(result, counts, values, comp_ids):
    def device_mean(inputs):
        return device_reduce_sum(inputs) / inputs.size

    def host_mean(inputs):
        return inputs.mean()

    _hsa_group_agg(host_mean, device_mean, result, counts, values, comp_ids)


def hsa_group_max(result, counts, values, comp_ids):
    def device_max(inputs):
        return device_reduce_max(inputs)

    def host_max(inputs):
        return inputs.max()

    _hsa_group_agg(host_max, device_max, result, counts, values, comp_ids)


def hsa_group_min(result, counts, values, comp_ids):
    def device_min(inputs):
        return device_reduce_min(inputs)

    def host_min(inputs):
        return inputs.min()

    _hsa_group_agg(host_min, device_min, result, counts, values, comp_ids)


def hsa_group_var(result, counts, values, comp_ids):
    def device_var(inputs):
        div = inputs.size - 1
        if div == 0:
            return NAN
        mean = device_reduce_sum(inputs) / inputs.size
        diff = np.empty(inputs.size, dtype=result.dtype)
        nelem = inputs.size
        threads = 256
        groups = (nelem + threads - 1) // threads
        hsa_var_diff_kernel[groups, threads](diff, inputs, mean)
        psum = device_reduce_sum(diff)
        return psum / div

    def host_var(inputs):
        return comp_var(inputs, inputs.mean(), 1)

    _hsa_group_agg(host_var, device_var, result, counts, values, comp_ids)


NAN = float('nan')


@hsa.jit
def hsa_var_diff_kernel(diff, inputs, mean):
    gid = hsa.get_global_id(0)
    if gid < inputs.size:
        val = inputs[gid]
        x = val - mean
        diff[gid] = x * x


@jit(nopython=True)
def comp_var(inp, mean, ddof):
    psum = 0
    for i in range(inp.size):
        x = (inp[i] - mean)
        xx = x * x
        psum += xx
    div = (inp.size - ddof)
    if div == 0:
        return NAN
    return psum / div


"""SAMPLE

    _cython_functions = {
        'add': 'group_add',
        'prod': 'group_prod',
        'min': 'group_min',
        'max': 'group_max',
        'mean': 'group_mean',
        'median': {
            'name': 'group_median'
        },
        'var': 'group_var',
        'first': {
            'name': 'group_nth',
            'f': lambda func, a, b, c, d: func(a, b, c, d, 1)
        },
        'last': 'group_last',
        'count': 'group_count',
    }
"""

_optimized_aggregate_functions = {
    'group_mean_float64': hsa_group_mean,
    'group_max_float64': hsa_group_max,
    'group_min_float64': hsa_group_min,
    'group_var_float64': hsa_group_var,

}


@jit(nopython=True)
def group_count(counts, comp_ids):
    """
    Store inclusive prefixsum of number of element per label into ``counts``.
    The last element will be the total number of element.
    """
    # binning
    for i in range(comp_ids.size):
        val = comp_ids[i]
        counts[val] += 1
    # inclusive scan
    total = 0
    for i in range(counts.size):
        ct = counts[i]
        counts[i] = ct + total
        total += ct
