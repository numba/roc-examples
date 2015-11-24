from __future__ import print_function, absolute_import
import math
import numpy as np
from pandas.computation import engines, ops
from numba import vectorize
from numba.numpy_support import from_dtype

_operator_mapping = {'&': 'and', '|': 'or', '~': 'not'}
_implicit_math_module = '__auto_import_module_math__'

_math_call_mapping = {
    'arcsin': 'asin',
    'arccos': 'acos',
    'arctan': 'atan',
    'arcsinh': 'asinh',
    'arccosh': 'acosh',
    'arctanh': 'atanh',
    'arctan2': 'atan2',
}


def _fix_operator_string(op):
    if op in _operator_mapping:
        return _operator_mapping[op]
    else:
        return op


def _stringify_eval_op_tree(op, nameset):
    """
    Return a string presentation of the eval tree that is suitable for
    use in numba.  The argument `nameset` must be a set that will be populated
    with term names.
    """
    if isinstance(op, ops.BinOp):

        if op.op == '**' and op.rhs.isscalar and op.rhs.value in [2, 0.5]:
            # Special case power operator with certain RHS constants
            lhs = _stringify_eval_op_tree(op.lhs, nameset)
            if op.rhs.value == 0.5:
                return "__auto_import_module_math__.sqrt({lhs})".format(lhs=lhs)
            elif op.rhs.value == 2:
                return "({lhs}) * ({lhs})".format(lhs=lhs)
            else:
                raise NotImplementedError
        else:
            fmt = "({lhs}) {op} ({rhs})"
            data = {'op': _fix_operator_string(op.op),
                    'lhs': _stringify_eval_op_tree(op.lhs, nameset),
                    'rhs': _stringify_eval_op_tree(op.rhs, nameset)}
        return fmt.format(**data)
    elif isinstance(op, ops.Term):
        if ((not op.isscalar and not op.is_datetime) or
                (isinstance(op.name, str) and
                     op.name.startswith('__pd_eval_local_'))):
            name = str(op)
            nameset.add(name)
            return name
        else:
            return op.value
    elif isinstance(op, ops.MathCall):
        fname = op.func.name
        fname = _math_call_mapping.get(fname, fname)
        if fname == 'abs':
            fn = 'abs'
        else:
            fn = '.'.join([_implicit_math_module, fname])
        args = [_stringify_eval_op_tree(op, nameset) for op in op.operands]
        out = "{0}({1})".format(fn, ','.join(args))
        return out
    else:
        raise NotImplementedError(op, type(op))


class NumbaEngine(engines.AbstractEngine):
    """Evaluate an expression using the Numba target.
    """
    has_neg_frac = False
    target = ''
    _func_cache = {}

    def __init__(self, expr):
        super(NumbaEngine, self).__init__(expr)
        self._args = [x for x in expr.names if isinstance(x, str)]

    def _compile(self, expr, args, argtypes, function_name):
        function_str = '''def %s(%s):
            return %s
        ''' % (function_name, ','.join(args), expr)
        scope = {_implicit_math_module: math}
        exec(function_str, scope)
        vectorizer = vectorize([argtypes], target=self.target)
        return vectorizer(scope[function_name])

    def _evaluate(self):
        # Get argument values
        env = self.expr.env
        is_local = lambda x: x.startswith("__pd_eval_local_")
        call_args = [np.asarray(env.resolve(name, is_local(name)))
                     for name in self._args]
        # Get argument types
        call_types = tuple(from_dtype(a.dtype) for a in call_args)
        # Check if the expression has already been compiled
        cache_key = (self.target, str(self.expr), call_types)
        fn = self._func_cache.get(cache_key)
        if fn is None:
            # Not cached.  Compile new one

            # Stringify the eval tree and get arg names
            nameset = set()
            exprstr = _stringify_eval_op_tree(self.expr.terms, nameset)
            assert set(self._args) == nameset

            function_name = '__numba_pandas_eval_ufunc'
            fn = self._compile(exprstr, self._args, call_types, function_name)
            self._func_cache[cache_key] = fn

        # Execute
        return fn(*call_args)


class NumbaCpuEngine(NumbaEngine):
    target = 'cpu'


class NumbaHsaEngine(NumbaEngine):
    target = 'hsa'


def register():
    from numba.hsa import is_available

    engines._engines['numba.cpu'] = NumbaCpuEngine
    if is_available():
        engines._engines['numba.hsa'] = NumbaHsaEngine
