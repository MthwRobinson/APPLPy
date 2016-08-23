# APPLPy
### A Probability Programming Language -- Python Edition

## What is APPLPy?

APPLPy stands for A Probability Programming Language -- Python Edition. The primary goal of APPLPy is to provide an open-source conceptual probability package capable of manipulating random variables symbolically. Although the Python implementation is a recent development, a version based on the Maple computer algebra system has been used for over a decade. The Maple implementation, called APPL, has been successfully integrated into mathematical statistics and computing courses at the College of William and Mary, the United States Military Academy and Colorado College, while also facilitating research in areas ranging from order statistics to queuing theory. The hope of APPLPy is to make the computational capabilities of APPL available to researchers and educators on an open-source platform.

The current capabilities of APPLPy include:

1. Conversion between PDF,CDF,SF,HF,CHF and IDF representations of random variables
2. Computation of expected values, with both numeric and symbolic output
3. Plotting distributions, including piece-wise distributions
4. One-to-one and many-to-one transformations of piecewise distributions
5. Random Variable Algebra (Sums/Differences/Products/Division)
6. Random sampling from distributions
7. Bootstrapping data sets
8. Bayesian inference with ad-hoc prior distributions
9. Computation of distributions for M/M/s queues
10. Analysis of Discrete Time Markov Chains

## How is APPLPy Used?

Although APPLPy can be used for a variety of purposes, it is best suited to fill three special roles. First, it enables students to gain an intuitive understanding of mathematical statistics by automating tedious, calculus-intensive algorithms. As such, students can experiment with different models without having to perform difficult derivations or produce ad-hoc code. Second, it allows students to check hand derived results. This aids the learning process by providing a quick and reliable answer key. Finally, it allows researchers to explore systems whose properties would be intractable to derive by hand. As mentioned above, the Maple-based APPL software has already spawned a variety of insightful research. APPLPy has the potential to continue along this pathway. The simplicity of APPLPy's syntax allow users to explore stochastic models with ease.

# Installing APPLPy

APPLPy is coded in Python 2 and requires Python 2.6 or later. The software has not yet been tested for forward compatibility with Python 3. Support for Python 3 will be added in future versions. In addition to Python, APPLPy requires the following software packages:

1. [sympy](https://pypi.python.org/pypi/sympy/)
2. [numpy](https://pypi.python.org/pypi/numpy/)
3. [scipy](https://pypi.python.org/pypi/scipy/)
4. [mpmath](https://pypi.python.org/pypi/mpmath/)
5. [matplotlib](https://pypi.python.org/pypi/matplotlib/)
6. [seaborn](https://pypi.python.org/pypi/seaborn/)
7. [pandas](https://pypi.python.org/pypi/pandas/)

## Anaconda

[Anaconda](https://store.continuum.io/cshop/anaconda/) is a free Python distribution that includes the most commonly used scientific computing packages, including all of the dependencies for APPLPy. While not required, installing Anaconda is highly recommended. Useful packages that ship with the Anaconda distribution include [pandas](http://pandas.pydata.org/) (a data analysis package), [iPython](http://ipython.org/) (a convenient interactive shell) and iPython Notebook (a Mathematica-style notebook that allows for easy collaboration). Anaconda helps ensure that all of these packages are well integrated and up-to-date.

## Installing APPLPy

The latest version of APPLPy is 0.4.3, which was released on 20 July 2016. It is available for download from the [Python Package Index](https://pypi.python.org/pypi/APPLPy/0.4.3). APPLPy and its dependencies can be installed by issuing the following terminal commands. Note that an internet connection is required to install APPLPy through pip.
```shell
$ pip install applpy
```
Once it is installed, APPLPy can be updated to the latest release with
```shell
$ pip install applpy --upgrade
```
pip can also be used to install APPLPy from inside of an Python session with the follow syntax:
```python
import pip
pip.main(['install','applpy'])
```
or upgraded with
```python
import pip
pip.main(['install','applpy','--upgrade'])
```

Users who want the latest development version of APPLPy can also download source code from the [APPLPy Github repo](https://github.com/MthwRobinson/APPLPy) with the following command
```shell
$ git clone https://github.com/MthwRobinson/APPLPy.git
```
 To install APPLPy, navigate to the location of the downloaded files and type 
```shell
$python setup.py install
``` 
into the terminal or the command prompt. After the installation is complete, users can run APPLPy from any Python interactive session by typing 
```python
from applpy import *
```

## Options for Running APPLPy

APPLPy can be run from any python interactive session. For fast performance, users can run APPLPy from the Python command line interface. Another option is the iPython Notebook, which executes procedures more slowly, but provides a notebook interface that is similar to Maple or Mathematica. iPython Notebook is the most convenient option for collaborating and sharing code.

SymPy includes a convenient command for initializing variables and setting up optimal plotting and printing environments. To leverage this initialization procedure, the recommended series of commands to begin an APPLPy session are as follows:
```python
from sympy import *; init_session()

from applpy import *
```

## Operating System Compatibility

Since APPLPy has been developed entirely in Python, it is compatible with almost any operating system. APPLPy will run happily on Window, Linux or OSX.

If you have any comments or suggestions for APPLPy, feel free to contact the author at mthw.wm.robinson@gmail.com. Users with Python experience are encouraged to get in touch and contribute.
