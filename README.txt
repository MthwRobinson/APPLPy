APPLPy
A Probability Programming Language -- Python Edition

What is APPLPy?

APPLPy stands for A Probability Programming Language -- Python Edition. The primary goal of APPLPy is to provide an open-source conceptual probability package capable of manipulating random variables symbolically. Although the Python implementation is a recent development, a version based on the Maple computer algebra system has been used for over a decade. The Maple implementation, called APPL, has been successfully integrated into mathematical statistics and computing courses at the College of William and Mary and the United States Military Academy, while also facilitating research in areas ranging from order statistics to queuing theory. The hope of APPLPy is to make the computational capabilities of APPL available to researchers and educators on an open-source platform.

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

How is APPLPy Used?

Although APPLPy can be used for a variety of purposes, it is best suited to fill three special roles. First, it enables students to gain an intuitive understanding of mathematical statistics by automating tedious, calculus-intensive algorithms. As such, students can experiment with different models without having to perform difficult derivations or produce ad-hoc code. Second, it allows students to check hand derived results. This aids the learning process by providing a quick and reliable answer key. Finally, it allows researchers to explore systems whose properties would be intractable to derive by hand. As mentioned above, the Maple-based APPL software has already spawned a variety of insightful research. APPLPy has the potential to continue along this pathway. The simplicity of APPLPy's syntax allow users to explore stochastic models with ease.

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

If you have any comments or suggestions for APPLPy, feel free to contact the author at mthw.wm.robinson@gmail.com. Users with Python experience are encouraged to get in touch and contribute.
