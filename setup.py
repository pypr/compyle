import sys
from setuptools import setup, find_packages


def get_version():
    import os
    data = {}
    fname = os.path.join('compyle', '__init__.py')
    exec(compile(open(fname).read(), fname, 'exec'), data)
    return data.get('__version__')


install_requires = ['mako', 'pytools', 'cython', 'numpy', 'pytest']
if sys.version_info[0] < 3:
    install_requires += ['mock>=1.0']
tests_require = ['pytest']

classes = '''
Development Status :: 4 - Beta
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Natural Language :: English
Operating System :: MacOS :: MacOS X
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Programming Language :: Python
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3
Topic :: Scientific/Engineering
Topic :: Software Development :: Code Generators
Topic :: Software Development :: Compilers
Topic :: Software Development :: Libraries
Topic :: Utilities
'''
classifiers = [x.strip() for x in classes.splitlines() if x]

setup(
    name='compyle',
    version=get_version(),
    author='Prabhu Ramachandran',
    author_email='prabhu@aero.iitb.ac.in',
    description='Execute a subset of Python on HPC platforms',
    long_description=open('README.rst').read(),
    license="BSD",
    url='https://github.com/pypr/compyle',
    classifiers=classifiers,
    packages=find_packages(),
    install_requires=install_requires,
    tests_require=tests_require,
)
