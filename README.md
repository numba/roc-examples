Numba HSA Examples
==================

This repository contains several example of HSA GPU programming using numba.

System Requirements
-------------------

HSA requires 64-bit linux machines with specific platform
requirements.
See https://github.com/HSAFoundation/HSA-Docs-AMD/wiki/HSA-Platforms-&-Installation
for details.

Installation
------------

The installation requires the use of conda.
Please follow the instruction at http://conda.pydata.org/docs/installation.html
to install conda.

Once conda is installed, user can create a conda environment with numba and
all the dependencies necessary to run the examples using the following command:

```bash
conda create --name hsa_examples_env --file env_spec.txt
```

To activate the environment, run

```bash
source activate hsa_examples_env
```

Run Tests
---------

Run full test suite:

```bash
./runtests.sh
```

Selectively run tests from subdirectories:

```bash
./runtests.sh <subdirectory>
```

Example:

```
./runtests.sh numba_hsa_examples/kerneldensity
```
