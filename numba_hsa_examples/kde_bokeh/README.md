HSA Accelerated Kernel Density Estimation
=========================================

This example uses bokeh to setup an interactive plot of the density of US
lightning data.  This example uses the HSA-accelerated kernel density
estimation (KDE) code in "numba_hsa_examples/kerneldensity" and the HSA pandas
eval/query backend in "numba_hsa_examples/pandas_eval".

To start the bokeh app server, run the following from the root of the source
tree:

```bash
PYTHONPATH=`pwd` bokeh serve  numba_hsa_examples/kde_bokeh/lightning_app.py
```

