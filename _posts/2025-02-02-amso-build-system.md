---
layout: blog
author: Ludwig Schneider
tags: [soma, amso, software engineering]
---

# Build Systems

[AMSO](https://github.com/InnocentBug/AMSO) is software project that combines C++, python, CUDA and other technologies.
To bring it all together it needs a build system, that is convenient and can handle the difficulties to bring these technologies together now, and stably in the future.
This post roughly outlines how and why the build system is chosen. I will show some examples, but it is not a beginners introduction nor am I covering all the details.
If you want to see the whole system in action, check it out directly on GitHub.

Build systems are tools that we used to reliably build our software as painless as possible.
The first build system, I came in touch with is the [`Make`](<https://en.wikipedia.org/wiki/Make_(software)>).
`Make` is great and very versatile since you can define arbitrary targets to build. And `Make` requirements of if something needs to be build based on the file timestamps.

However, AMSO is going to be a little bit bigger then Make can comfortably handle. Plus AMSO is written in 2 programming languages (C++ & Python) and we need to combine and handle both.
Let's go through it one by one:

## CMake

[CMake](https://cmake.org/) is build system that shines for C++.
It is great to define software targets in a meta-language. You (and CMake) can usually find everything you need to know about a project in a `CMakeLists.txt` file.
CMake doesn't use my favorite syntax, but it nice to define how a C++ project should be compile.

This is how a `CMakeLists.txt` may look like in the root of the directory.

```CMake
cmake_minimum_required(VERSION 3.25)
project(
 ${SKBUILD_PROJECT_NAME}
 VERSION ${SKBUILD_PROJECT_VERSION}
 LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_CUDA "Enable compilation with CUDA for GPU acceleration"
       $ENV{ENABLE_CUDA})
# Add source files
add_subdirectory(src)
```

`project` defines the general project usually directly by name version and languages used.
Don't worry that we are using variables that start with `SKBUILD` here for now, we will explain in [SKBuild section](#python/c++-build-via-scikit-build-core).
We can also define the language this project is in, here we choose `CXX` for C++. Followed by defining and enforcing the C++ standard.

We can also define `option`s that the user can set, here I am showing an option that allows builds with or without CUDA support.
But other are normal, like which precision strategy to take or if certain modules of the project should be compiled or not.

And lastly, I define the sub-directory `src` should be included in the project. You can add as many sub-directories as you like and it good to structure your project.
For CMake it is just important that each included sub-directory has a `CMakeLists.txt` file to process.

For `src`, which bundles all C++ source files (`.cpp`) of AMSO it may look like this:

```CMake
# Add your source files
set(_amso_cpu_sources platform_status.cpp logger.cpp)

# Define the target first
add_library(amso_core ${_amso_cpu_sources})

# Update target sources
target_sources(amso_core PRIVATE ${_amso_sources})
# Include directories
target_include_directories(amso_core
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

```

First we define a list of all the source (`.cpp`) files, here as `_amso_cpu_source` files.
Then we make it a library that we can link to our application: `amso_core`
The idea here is to compile a separate C++ library first to reduce build time, later we build a python library and link this one to it.
And lastly, we define where to find the header files (`.h`) for these source files with `target_include_directories`.
I keep the file in `AMSO/include` so from `AMSO/src` which is in the variable `CMAKE_CURRENT_SOURCE_DIR` we can link to them.

To build with CMake, we create a new build directory at the top of AMSO.

```shell
mkdir build
cd build
cmake ..
```

The `cmake ..` initializes the build. It is always good to start with a clean directory with CMake if problem occur.
CMake cashes some variables that may not expect, especially choices of compilers and libraries.

Next is to build the C++ library.

```shell
make
make install
```

CMake uses `Make` as a build engine here. You can also uses others, like `ninja`.
This is useful, but we will soon relinquish this building to a python building tool and we don't have to do this manually anymore.
Although, sometimes it is useful during debugging to reduce build times while chasing typos.

## Connect C++ to Python

SOMA was C-only, but AMSO we want to be more user friendly make it available through Python, so we need to connect the two.
The most fundamental one is probably [`ctypes`](https://docs.python.org/3/library/ctypes.html), but it is quite low-level.
We would want framework that is more C++ like so that we can directly map our Object-Oriented design directly between Python and C++.
[`pybind11`](https://pybind11.readthedocs.io/en/stable/) is a nice framework to do just that.
I have had good experience with it in the past, that is what AMSO is starting out with.
Recently, I came across [`nanobind`](https://github.com/wjakob/nanobind) which seems like an interesting project, maybe we change AMSO later to that one.

## C++ Dependency Management: CPM

Now, that we want to use `pybind11` we need to make sure that it is available to us as a dependency.
CMake has a `find_package` functionality that checks your system for a given dependency.

Let's take a look at `CMakeLists.txt` in the `AMSO/python` directory where I keep all python related files.

```CMake
find_package(
  Python
  COMPONENTS Interpreter Development Development.Module
  REQUIRED)
```

Here we are finding an installation of `python` and ensure that all the necessary components for development are present.
This is fine, because we assume that a new user has python already installed.

But most people will not have `pybind11` installed on their machine and CMake itself has no mechanism to install missing dependency.
This inconvenient and can be very confusing for new users. We want something like [`PyPI`](https://pypi.org/) but for C++ instead of python.
This is where [`CPM`](https://github.com/cpm-cmake/CPM.cmake) comes, their tag line is even: "Cmake's missing package manager".
You can use `CPM` by simply downloading a file from their GitHub, you can find AMSO's at [`AMSO/cmake/CPM.cmake`](https://github.com/cpm-cmake/CPM.cmake) and activating in your CMakeLists.txt with `include(cmake/CPM.cmake)`.
After that we can easily include many different C++ projects directly. Many even directly from GitHub as long as they have `CMakeLists.txt` in their root directory.
This is great to interface with big established projects like `pybind11`, but it is also great if you want to include another project from your own GitHub.

Let's take a look again at `AMSO/python/CMakeLists.txt`:

```CMake
# Pybind11
cpmaddpackage(NAME pybind11 GITHUB_REPOSITORY pybind/pybind11 VERSION 2.10.0)
# Define the Python module
pybind11_add_module(_amso module_amso.cpp)

# Link against your main library
target_link_libraries(_amso PUBLIC amso_core pybind11::module Python::Python)

# Install the compiled Python module
install(TARGETS _amso LIBRARY DESTINATION amso)
install(TARGETS amso_core LIBRARY DESTINATION amso)

# Install the pure Python files
install(DIRECTORY amso/ DESTINATION amso)
```

Now we could use `CPM` to add `pybind11` with an exact version directly from GitHub. When you build with `CMake` you will notice that `CPM` downloads the dependencies and makes them available in `build/_deps/pybind11*`.

Next is to introduce AMSO as a python module `_amso` via `pybind11`.
I deliberately choose here the leading underscore, because the C++ library acts like a hiding module that the `amso` module uses, so we can use `from _amso import MemArray1DInt32` for example.
This way we have control of what the user can see as a public interface.
We link everything we need to our module and install the libraries as well.

And finally, you can see how we `install` the pure python library `AMSO/python/amso`.
That directory is structured like any python library you may build which includes the main `__init__.py` file to make it a package, but we can also have multiple different module files like `memarray.py`.
Now we setup our CMake to handle the compilation and installation AMSO as a python package with a compiled C++ extension.

## Python build via `pyproject.toml`

## Python/C++ build via SciKit-Build-Core

##

## Super Linting with trunk.io
