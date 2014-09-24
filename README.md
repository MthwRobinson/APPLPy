A Probability Progamming Language (APPL) -- Python Edition

ABOUT:

APPLPy is a computational probability package written in Python. The project
runs using the open source SymPy computer algebra system (CAS). Because all of the dependencies that APPLPy requires to run are open source, APPLPy is available free of charge. APPLPy is currently capable of performing a wide range of random variable algebra operations on univariate probaility distributions. The package supports both continuous and discrete distributions, as well as piece-wise distributions. The current capabilities of APPLPy include:

1. Conversion between PDF,CDF,SF,HF,CHF and IDF representations of random variables
2. Computation of expected values, with both numeric and symbolic output
3. Plotting distributions, including piece-wise distributions
4. One-to-one and many-to-one transformations of piecewise distributions
5. Random Variable Algebra (Sums/Differences/Products/Division)
6. Random sampling from distributions
7. Bootstrapping data sets

In addition, the Bayesian statistics module provides the following Bayesian
capabilities:

1. Computation of posterior and posterior predictive distributions 
for unknown parameters
2. Derivation of Jeffrey's Priors
3. Computation of credible sets

APPLPy is derived from A Probability Programming Language (APPL), which runs
in Maple. The idea behind APPLPy is to make the capabilites of APPL available
on a free of charge, open source platform. This has the potential to make APPL much more effective, both as an educational resource and a research tool.

INSTALLATION:

ApplPy requires the following dependencies in order to run properly:

1. SymPy
2. Matplotlib

The latests stable release of both of these packages can be downloading
from the python package index at https://pypi.python.org/pypi

The latest working edition of APPLPy is available on GitHub and the latest
stable release is available from the python package index. To install the
software, open the directory where APPLPy has been downloaded and type
the following command

$ python setup.py install

If you have any comments or suggestions for APPLPy, feel free to contact the author
at mthw.wm.robinson@gmail.com. Users with Python experience are encouraged to
get in touch and contribute.
