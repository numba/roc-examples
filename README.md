Numba ROC Examples
==================

This repository contains several example of ROC GPU programming using Numba.
It is a fork and update of examples written for HSA
https://github.com/ContinuumIO/numba-hsa-examples.

WARNING: The examples here and the underlying compilation chain are both under
development!

For the purposes of this README, the `$` symbol indicates the command prompt.


Set up machine
==============

 * First, install the ROC stack from AMD as per the installation instructions
   provided [here](
https://github.com/RadeonOpenCompute/ROCm#installing-from-amd-rocm-repositories)
   .

 * Create and activate a conda environment (named e.g. `amd_roc`) as follows:

```bash
$ conda create -n amd_roc -c numba numba roctools jupyter bokeh statsmodels \
python=3 h5py -y
$ source activate amd_roc
```

  * Note that users should be a member of the `video` group to have access to the GPU.

 <!-- Note: NOT MERGED INTO NUMBA YET
* Check that your ROC installation and AMDGCN hardware is recognised by Numba
   with:

```bash
$ numba -s
```
-->

Run examples
------------

There are two examples working at present, both revolving around kernel density
estimation. The first is a Jupyter notebook, `multi_variate_kde_example.ipynb`
which can be launched with the following:

    $ jupyter notebook numba_roc_examples/kerneldensity/

You can see the rendered notebook [here](https://nbviewer.jupyter.org/github/numba/roc-examples/blob/master/numba_roc_examples/kerneldensity/multi_variate_kde_example.ipynb).
The AMD GPU implementation used in notebook can be found [here](https://github.com/numba/roc-examples/blob/master/numba_roc_examples/kerneldensity/roc_imp.py).

The second is a ``bokeh`` application that can be launched following the
instructions in the ``README.md`` of the ``numba_roc_examples/kde_bokeh``
directory.


Run Tests
---------

Run the full test suite, it is expected that this will fail (still under
development):

```bash
$ ./runtests.sh
```

Selectively run tests from subdirectories:

```bash
$ ./runtests.sh <subdirectory>
```

Example:

```bash
$ ./runtests.sh numba_roc_examples/kerneldensity
```
