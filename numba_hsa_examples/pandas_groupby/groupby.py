from __future__ import print_function, division, absolute_import

import numpy as np
from pandas.core.groupby import Grouper, BaseGrouper, Grouping, _is_label_like
from pandas.core.index import Index, MultiIndex

from pandas import compat
from pandas.core.series import Series
from pandas.core.frame import DataFrame
import pandas.core.common as com


from numba_hsa_examples.radixsort.sort_driver import HsaRadixSortDriver


class NotYet(NotImplementedError):
    pass


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
            indexer = self.indexer = self._make_sorter(ax)
            ax = ax.take(indexer)
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
        sorter = HsaRadixSortDriver()
        _, indices = sorter.sort_with_indices(np_array)
        return indices


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
    def _aggregate(self, result, counts, values, agg_func, is_numeric):
        print("_aggregate", agg_func)
        # TODO: Intercept aggregate that has a HSA equivalent
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
            agg_func(result, counts, values, comp_ids)

        return result
