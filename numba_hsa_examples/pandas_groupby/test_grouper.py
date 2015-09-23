from __future__ import print_function

import pandas as pd
import numpy as np
from numba import unittest_support as unittest

from .groupby import HSAGrouper


class TestCustomGrouper(unittest.TestCase):
    def make_groupers(self):
        nelem = 10
        numgroup = 4
        df = pd.DataFrame(
            {'a': np.random.randint(0, numgroup, nelem).astype(np.int32),
             'b': np.random.randint(0, nelem, nelem).astype(np.int32)})
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


if __name__ == '__main__':
    unittest.main()
