Pandas ``.eval()`` engine extension
===================================

The "eval_engine.py" file implements an extension to the ``pandas.eval``.
Its adds two numba evaluation engines--a CPU backend and a HSA GPU backend.


Software Requirement
--------------------

* numba: development version of 0.22.0 that is not yet released
* pandas: development version of 0.17.0 that is not yet released
