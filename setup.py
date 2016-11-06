from distutils.core import setup
# To upload to PyPi run python setup.py sdist upload -r pypi
setup(
    name='APPLPy',
    version='0.4.7',
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
        "mpmath",
        #"pandas"
        ]
    )
