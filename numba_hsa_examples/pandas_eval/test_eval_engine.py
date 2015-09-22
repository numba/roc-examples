from __future__ import print_function, division, absolute_import

from numba import unittest_support as unittest

import pandas as pd
import numpy as np
from timeit import default_timer as timer
from numba_hsa_examples.pandas_eval import eval_engine

eval_engine.register()


def _best_time(fn):
    ts = timer()
    fn()
    te = timer()
    return te - ts


def best_time(fn, repeat):
    return min(_best_time(fn) for _ in range(repeat))


def eval_template(expr, engine, nelem=4, repeat=3):
    print("Eval:", expr, "Nelem:", nelem)
    a = pd.DataFrame(dict(x=np.arange(nelem, dtype=np.float64),
                          y=np.arange(1, 1 + nelem, dtype=np.float64)))

    # print('Input:', type(a), '\n', a)

    b = a.eval(expr)
    # print('Output:', type(b), '\n', b)

    c = a.eval(expr, engine=engine)
    # print('Output:', type(c), '\n', c)

    np.testing.assert_allclose(b, c)  # , rtol=1e-5)

    runtime = best_time(lambda: a.eval(expr), repeat=repeat)
    # print('Output:', type(b), '\n', b)
    print('numexpr time', runtime)

    runtime = best_time(lambda: a.eval(expr, engine=engine), repeat=repeat)
    # print('Output:', type(c), '\n', c)
    print('{0} time'.format(engine), runtime)


def query_template(expr, engine, nelem=4, repeat=3):
    print("Query:", expr, "Nelem:", nelem)
    a = pd.DataFrame(dict(x=np.arange(nelem, dtype=np.float64),
                          y=np.arange(1, 1 + nelem, dtype=np.float64)))

    # print('Input:', type(a), '\n', a)

    b = a.query(expr)
    # print('Output:', type(b), '\n', b)

    c = a.query(expr, engine=engine)
    # print('Output:', type(c), '\n', c)

    pd.util.testing.assert_frame_equal(b, c)

    runtime = best_time(lambda: a.query(expr), repeat=repeat)
    # print('Output:', type(b), '\n', b)
    print('numexpr time', runtime)

    runtime = best_time(lambda: a.query(expr, engine=engine), repeat=repeat)
    # print('Output:', type(c), '\n', c)
    print('{0} time'.format(engine), runtime)


def driver_test_template(driver, expr, nelem=4, repeat=10):
    print("test cpu")
    driver(expr, engine='numba.cpu', nelem=nelem, repeat=repeat)

    print("test hsa")
    driver(expr, engine='numba.hsa', nelem=nelem, repeat=repeat)


class TestEvalEngine(unittest.TestCase):
    def test_simple_query(self):
        driver_test_template(query_template, "x > 2 or y > 1")

    def test_simple_eval(self):
        driver_test_template(eval_template, "x + y")

    def test_special_case_eval_sqrt(self):
        driver_test_template(eval_template, "x + y ** 0.5")

    def test_special_case_eval_square(self):
        driver_test_template(eval_template, "x + y ** 2")

    def test_special_case_boundaries(self):
        driver_test_template(eval_template, "x + y ** 1.9")
        driver_test_template(eval_template, "x + y ** 0.49")

    def test_math_calls(self):
        driver_test_template(eval_template, "sin(x) + cos(y)")

    def test_all_unary_math_calls(self):
        from pandas.computation.ops import _unary_math_ops

        for op in _unary_math_ops:
            expr = "{0}(x)".format(op)
            driver_test_template(eval_template, expr)

    def test_all_binary_math_calls(self):
        from pandas.computation.ops import _binary_math_ops

        for op in _binary_math_ops:
            expr = "{0}(x, y)".format(op)
            driver_test_template(eval_template, expr)


if __name__ == '__main__':
    unittest.main()
