# PyThermo

[![DOI](osf_io_BNUVZ.svg)](https://doi.org/10.17605/OSF.IO/BNUVZ)
[![CI][ci-img]][ci-url]
[![PyPI - Version][pypi-img]][pypi-url]
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/OpenThermochronology/PyThermo/HEAD?urlpath=https%3A%2F%2Fgithub.com%2FOpenThermochronology%2FPyThermo%2Fblob%2Fmain%2Fexamples%2Ftemplate.ipynb)


A set of classes and methods for performing various modeling and computational tasks in the field of low-temperature thermochronology. The current focus is on forward modeling of apatite and zircon (U-Th)/He data using various diffusion and damage annealing kinetic models. Future releases will expand upon the available kinetic models and mineral systems, and introduce additional methods, such as forward modeling of Arrhenius relationships.

The primary objective of this software is to provide an open-source, python-based toolkit for user adaptability and experimentation. The software includes routines for forward modeling and data plotting at higher levels that can be run in a simple fashion, but lower level algorithms are accessible as well. To that end, a secondary objective of this software is as a learning tool to remove some of the black box nature of thermal history modeling routines. Several methods are included (for example, a tridiagonal matrix solver) for instructional purposes, although the main program calls nominally faster scipy routines.

## Organization

The source code consists of three separate classes and accompanying methods and/or subclasses. `crystal.py` contains the class `crystal` and currently two sub-classes `apatite` and `zircon`. Methods are devoted to calculating and parameterizing damage-diffusivity relationships and numerically solving the diffusion equation using a Crank-Nicolson approach. `tT_path.py` methods interpolate and discretize time-temperature (tT) paths from a handful of tT points, and calculate fission track annealing for apatite and zircon using the equivalent time concept. `tT_model.py` methods currently allow for one particular approach to forward modeling and plotting (U-Th)/He date-effective Uranium (eU) trends.

## Requirements

If you've installed Python through the open data science platform [Anaconda](https://www.anaconda.com/download), you should be all set. In detail, the specific libraries you'll need to have installed are:

* Numpy
* Scipy
* Matplotlib
* Pandas

## Installation

PyThermo can be installed as a package by using pip (see this [helpful](https://packaging.python.org/en/latest/tutorials/installing-packages/) guide for using pip if you are unfamiliar):

```python-repl
pip install pythermo
```

 You can alternatively download the `/src` folder and place it in your working directory. See the `template.ipynb` file in the `examples` folder for usage under either circumstance.

## Usage

It is recommended that you use this package in [Jupyter Notebooks](https://jupyter.org), which are included in the [Anaconda](https://www.anaconda.com/download) platform. Once you've downloaded or installed the package, and if you're just interested in running some forward models, the quickest way to get started is to modify the Jupyter Notebook file `template.ipynb` that is included in the `examples` folder. The notebook contains markdown and code that explains and demonstrates forward model date-eU comparisons for the apatite and zircon (U-Th)/He system. The forward modeling method is one particular approach and you can (and should!) call lower level methods to suite your needs. The `tT_path` and `crystal` classes contain several methods that you may want to call. A basic example of one way to use lower-level methods is included in `template.ipynb`. Please read the descriptions for each method in the source code for more details.

## Citation

You can find various citation styles for this package [here](https://doi.org/10.17605/OSF.IO/BNUVZ).

[ci-img]: https://github.com/OpenThermochronology/PyThermo/actions/workflows/CI.yml/badge.svg?branch=main
[ci-url]: https://github.com/OpenThermochronology/PyThermo/actions/workflows/CI.yml
[pypi-img]: https://img.shields.io/pypi/v/pythermo
[pypi-url]: https://pypi.org/project/pythermo/
