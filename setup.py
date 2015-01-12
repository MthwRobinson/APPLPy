from distutils.core import setup

setup(
    name='APPLPy',
    version='0.3.0',
    author='Matthew Robinson',
    author_email='mthw.wm.robinson@gmail.com',
    packages=['applpy','applpy.test'],
    scripts=[],
    url='http://pypi.python.org/pypi/APPLPy/',
    license='LICENSE.txt',
    description='open source computational probability software',
    long_description=open('README.txt').read(),
    install_requires=[
        "sympy >= 0.7.0",
        "scipy >= 0.14.0",
        "matplotlib >= 1.3.1"
        ]
    )
