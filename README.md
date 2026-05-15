apoCHARMM: High-performance molecular dynamics simulations on GPUs for advanced
simulation methods
================================================================================

## About

apoCHARMM is a GPU-only molecular dynamics package. Features that are used in
the working unit tests (c.f. run_all_tests.sh) can be used for performing
molecular dynamics simulations. The Python API is currently being overhauled and
is deprecated.

Reference:

[S. Prasad, F. Aviat, J. E. Gonzales II, and B. R. Brooks, "apoCHARMM:
High-performance molecular dynamics simulations on GPUs for advanced simulation
methods," J. Chem. Phys. 162, 182501
(2025).](https://pubs.aip.org/aip/jcp/article/162/18/182501/3346618)

## License

apoCHARMM is distributed under the
[BSD-3-clause](https://opensource.org/licenses/BSD-3-Clause) open source
license, as described in the LICENSE file in the top level of the repository.
Some external dependencies are used that are licensed under different terms, as
enumerated below.

## Dependencies

* [Catch2](https://github.com/catchorg/Catch2) for unit testing
  [(BSL license)](https://opensource.org/licenses/BSL-1.0).

## Authors

Samarjeet Prasad (Nvidia)

James E. Gonzales II (NIH)

Bernard R. Brooks (NIH)

## Installation

Currently, only building from source is supported. Once the Python API has been
overhauled, the installation procedure may change.

While the source code was developed using the following tool and compiler
versions, other versions may work.
* GCC [12.2.0]
* CUDA [12.2.140]
* CMake [3.25.1]

### 0. Clone this repository
```
git clone git@github.com:jeg7/apocharmm --recursive
cd apocharmm/
```

(If you already cloned this repository without the `--recursive` flag, simply
run `git submodule update --init --force --remote` from within the `apocharmm/`
directory).

### 1. Compile the source code
```
mkdir build
cd build/
cmake ..
make -j
```

## Testing

To ensure that your installation of apoCHARMM is working correctly, you can run
the [test shell script](run_all_tests.sh).

## Getting Help, Feature Requests, and Contributing

Please contact <james.gonzales@nih.gov> for any of the following:

* Trouble installing
* Trouble using
* Bugs
* Feature requests
* Code/feature contributions
