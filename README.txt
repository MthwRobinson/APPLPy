ApplPy

ABOUT:

ApplPy is a computational probability package written in Python. The project
runs using the open source SymPy computer algebra system (CAS). Because all of
the dependencies that ApplPy requires to run are open source, ApplPy is available
free of charge. While still incomplete, ApplPy will eventually be capable of
performing a wide range of random variable algebra operations on univariate
probaility distributions. The package supports both continuous and discrete 
distributions, as well as piece-wise distributions. The current capabilities of
ApplPy include:

1. Conversion between PDF,CDF,SF,HF,CHF and IDF representations of random variables
2. Computation of expected values, with both numeric and symbolic output
3. Plotting distributions, including piece-wise distributions
4. One-to-one and many-to-one transformations of piecewise distributions
5. Convolutions of lifetime distributions
6. Random sampling from distributions
7. Bootstrapping data sets

ApplPy is derived from A Probability Programming Language (APPL), which runs
in Maple. The idea behind ApplPy is to make the capabilites of APPL available
on a free of charge, open source platform. This has the potential to make APPL
much more effective, both as an educational resource and a research tool.

INSTALLATION:

ApplPy requires the following dependencies in order to run properly:
1. SymPy, available at https://github.com/sympy/sympy
2. Pyglet or Matplotlib, for plotting functions

Functions plot more smoothly using Matplotlib, however, it is not yet fully
debugged. As such, it may not yet work for all function. Pyglet is the main
plotting library for SymPy. If both are installed, then ApplPy will default to a 
Pyglet plot if Matplotlib raises an exception. Full installation instructions
will be added for Microsoft Windows, Mac OS X and various Linux distributions
once the project is further along.

If you have any comments or suggestions for ApplPy, feel free to contact the author
at mthw.wm.robinson@gmail.com. Users with Python experience are encouraged to
get in touch and contribute.