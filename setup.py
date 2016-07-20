from distutils.core import setup

setup(
    name='APPLPy',
    version='0.4.3',
    author='Matthew Robinson',
    author_email='mthw.wm.robinson@gmail.com',
    packages=['applpy','applpy.test'],
    scripts=[],
    url='https://pypi.python.org/pypi/APPLPy/',
    license='LICENSE.txt',
    description='open source computational probability software',
    long_description=open('README.txt').read(),
    install_requires=[
        "sympy",
        "scipy",
        "matplotlib",
        "seaborn",
        "numpy",
        "pylab",
        "mpmath",
        "pandas",
        "pickle"

        ]
    )
