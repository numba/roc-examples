from __future__ import print_function

import logging

import pandas as pd
import numpy as np
from numba import unittest_support as unittest

from .groupby import HSAGrouper, SPEED_BARRIER

logging.basicConfig(level=logging.DEBUG)


class TestCustomGrouper(unittest.TestCase):
    def make_groupers(self, nelem=10, numgroup=4, ncols=1):
        column_names = 'bcdefg'
        dct = {'a': np.random.randint(0, numgroup, nelem).astype(np.int32)}
        cols = column_names[:ncols]
        self.assertEqual(len(cols), ncols)
        for col in cols:
            dct[col] = np.random.random(nelem).astype(np.float64)
        df = pd.DataFrame(dct)
        expected_grouper = df.groupby('a')
        got_grouper = df.groupby(HSAGrouper('a'))
        return expected_grouper, got_grouper

    def test_iterator(self):
        expected_grouper, got_grouper = self.make_groupers()

        for expect, got in zip(expected_grouper, got_grouper):
            expect_i, expect_group = expect
            got_i, got_group = got
            self.assertEqual(got_i, expect_i)
            pd.util.testing.assert_frame_equal(got_group, expect_group)

    def test_len(self):
        expected_grouper, got_grouper = self.make_groupers()
        self.assertEqual(len(expected_grouper), len(got_grouper))

    def test_groups(self):
        expected_grouper, got_grouper = self.make_groupers()
        self.assertEqual(expected_grouper.groups, got_grouper.groups)

    def test_indices(self):
        expected_grouper, got_grouper = self.make_groupers()
        self.assertEqual(expected_grouper.groups, got_grouper.groups)

    def test_name(self):
        expected_grouper, got_grouper = self.make_groupers()
        self.assertEqual(expected_grouper.name, got_grouper.name)

    def test_ngroups(self):
        expected_grouper, got_grouper = self.make_groupers()
        self.assertEqual(expected_grouper.ngroups, got_grouper.ngroups)

    def test_first(self):
        expected_grouper, got_grouper = self.make_groupers()
        expect = expected_grouper.first()
        got = got_grouper.first()
        pd.util.testing.assert_frame_equal(expect, got)

    def test_mean(self):
        expected_grouper, got_grouper = self.make_groupers()
        expect = expected_grouper.mean()
        got = got_grouper.mean()
        pd.util.testing.assert_frame_equal(expect, got)

    def test_mean_larger(self):
        nelem = int(2.5 * SPEED_BARRIER)
        expected_grouper, got_grouper = self.make_groupers(nelem=nelem,
                                                           numgroup=2)
        expect = expected_grouper.mean()
        got = got_grouper.mean()
        pd.util.testing.assert_frame_equal(expect, got)

    def test_max(self):
        expected_grouper, got_grouper = self.make_groupers()
        expect = expected_grouper.max()
        got = got_grouper.max()
        pd.util.testing.assert_frame_equal(expect, got)

    def test_max_larger(self):
        nelem = int(2.5 * SPEED_BARRIER)
        expected_grouper, got_grouper = self.make_groupers(nelem=nelem,
                                                           numgroup=2)
        expect = expected_grouper.max()
        got = got_grouper.max()
        pd.util.testing.assert_frame_equal(expect, got)

    def test_min(self):
        expected_grouper, got_grouper = self.make_groupers()
        expect = expected_grouper.min()
        got = got_grouper.min()
        pd.util.testing.assert_frame_equal(expect, got)

    def test_min_larger(self):
        nelem = int(2.5 * SPEED_BARRIER)
        expected_grouper, got_grouper = self.make_groupers(nelem=nelem,
                                                           numgroup=2)
        expect = expected_grouper.min()
        got = got_grouper.min()
        pd.util.testing.assert_frame_equal(expect, got)

    def test_var(self):
        expected_grouper, got_grouper = self.make_groupers()
        expect = expected_grouper.var()
        got = got_grouper.var()
        for x, y in zip(expect.values, got.values):
            np.testing.assert_allclose(x, y)

    def test_var_larger(self):
        nelem = int(2.5 * SPEED_BARRIER)
        expected_grouper, got_grouper = self.make_groupers(nelem=nelem,
                                                           numgroup=2)
        expect = expected_grouper.var()
        got = got_grouper.var()
        for x, y in zip(expect.values, got.values):
            np.testing.assert_allclose(x, y)

    def test_var_two_value_columns(self):
        nelem = int(2.5 * SPEED_BARRIER)
        expected_grouper, got_grouper = self.make_groupers(nelem=nelem,
                                                           numgroup=2,
                                                           ncols=2)
        expect = expected_grouper.var()
        got = got_grouper.var()
        for x, y in zip(expect.values, got.values):
            np.testing.assert_allclose(x, y)

    def test_var_multi_value_columns(self):
        for ncol in range(3, 5):
            nelem = int(2.5 * SPEED_BARRIER)
            expected_grouper, got_grouper = self.make_groupers(nelem=nelem,
                                                               numgroup=2,
                                                               ncols=ncol)
            expect = expected_grouper.var()
            got = got_grouper.var()
            for x, y in zip(expect.values, got.values):
                np.testing.assert_allclose(x, y)


if __name__ == '__main__':
    unittest.main()
