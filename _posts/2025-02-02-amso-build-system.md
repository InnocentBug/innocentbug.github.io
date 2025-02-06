---
layout: blog
author: Ludwig Schneider
tags: [soma, amso, software engineering]
---

# Build System for AMSO

[AMSO](https://github.com/InnocentBug/AMSO) is a software project that combines C++, Python, CUDA, and other technologies. To bring it all together, it needs a build system that is convenient and can handle the complexities of integrating these technologies now and stably in the future. This post roughly outlines how and why the build system is chosen. I will show some examples, but it is not a beginner's introduction, nor am I covering all the details. If you want to see the whole system in action, check it out directly on GitHub.

Build systems are tools that we use to reliably build our software as painlessly as possible. The first build system I came in touch with is [`Make`](<https://en.wikipedia.org/wiki/Make_(software)>). Make is great and very versatile since you can define arbitrary targets to build. Make determines if something needs to be built based on file timestamps.

However, AMSO is going to be a little bit bigger than Make can comfortably handle. Plus, AMSO is written in two programming languages (C++ & Python), and we need to combine and handle both. Let's go through it one by one:

## CMake

[CMake](https://cmake.org/) is a build system that shines for C++. It is great for defining software targets in a meta-language. You (and CMake) can usually find everything you need to know about a project in a `CMakeLists.txt` file. CMake doesn't use my favorite syntax, but it's nice to define how a C++ project should be compiled.

This is how a `CMakeLists.txt` may look like in the root of the directory:

```text
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

`project` defines the general project usually directly by name, version, and languages used.
Don't worry that we are using variables that start with `SKBUILD` here for now; we will explain in the [SKBuild section](#scikit-build-core).
We can also define the language this project is in; here we choose `CXX` for C++, followed by defining and enforcing the C++ standard.

We can also define `option`s that the user can set. Here, I am showing an option that allows builds with or without CUDA support. But others are normal, like which precision strategy to take or if certain modules of the project should be compiled or not.

Lastly, I define that the sub-directory `src` should be included in the project. You can add as many sub-directories as you like, and it's good to structure your project. For CMake, it's just important that each included sub-directory has a `CMakeLists.txt` file to process.

For `src`, which bundles all C++ source files (`.cpp`) of AMSO, it may look like this:

```text
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

First, we define a list of all the source (`.cpp`) files, here as `_amso_cpu_source` files. Then we make it a library that we can link to our application: `amso_core`. The idea here is to compile a separate C++ library first to reduce build time; later, we build a Python library and link this one to it. Lastly, we define where to find the header files (`.h`) for these source files with `target_include_directories`. I keep the files in `AMSO/include`, so from `AMSO/src`, which is in the variable `CMAKE_CURRENT_SOURCE_DIR`, we can link to them.

To build with CMake, we create a new build directory at the top of AMSO:

```shell
mkdir build
cd build
cmake ..
```

The `cmake ..` initializes the build. It's always good to start with a clean directory with CMake if problems occur. CMake caches some variables that you may not expect, especially choices of compilers and libraries.

Next is to build the C++ library:

```shell
make
make install
```

CMake uses `Make` as a build engine here. You can also use others, like `ninja`. This is useful, but we will soon relinquish this building to a Python building tool, and we won't have to do this manually anymore. Although, sometimes it's useful during debugging to reduce build times while chasing typos.

## Connect C++ to Python

SOMA was C-only, but for AMSO, we want to be more user-friendly and make it available through Python, so we need to connect the two. The most fundamental one is probably [`ctypes`](https://docs.python.org/3/library/ctypes.html), but it is quite low-level. We want a framework that is more C++-like so that we can directly map our Object-Oriented design between Python and C++. [`pybind11`](https://pybind11.readthedocs.io/en/stable/) is a nice framework to do just that. I have had good experience with it in the past, so that's what AMSO is starting out with. Recently, I came across [`nanobind`](https://github.com/wjakob/nanobind), which seems like an interesting project; maybe we'll change AMSO to that one later.

## C++ Dependency Management: CPM

Now that we want to use `pybind11`, we need to make sure that it's available to us as a dependency. CMake has a `find_package` functionality that checks your system for a given dependency.

Let's take a look at `CMakeLists.txt` in the `AMSO/python` directory where I keep all Python-related files:

```text
find_package(
  Python
  COMPONENTS Interpreter Development Development.Module
  REQUIRED)
```

Here we are finding an installation of `python` and ensuring that all the necessary components for development are present. This is fine because we assume that a new user has Python already installed.

But most people will not have `pybind11` installed on their machine, and CMake itself has no mechanism to install missing dependencies. This is inconvenient and can be very confusing for new users. We want something like [`PyPI`](https://pypi.org/) but for C++ instead of Python. This is where [`CPM`](https://github.com/cpm-cmake/CPM.cmake) comes in; their tagline is even: "CMake's missing package manager". You can use `CPM` by simply downloading a file from their GitHub; you can find AMSO's at [`AMSO/cmake/CPM.cmake`](https://github.com/cpm-cmake/CPM.cmake) and activating it in your CMakeLists.txt with `include(cmake/CPM.cmake)`. After that, we can easily include many different C++ projects directly. Many even directly from GitHub as long as they have `CMakeLists.txt` in their root directory. This is great for interfacing with big established projects like `pybind11`, but it's also great if you want to include another project from your own GitHub.

Let's take a look again at `AMSO/python/CMakeLists.txt`:

```text
# Pybind11
cpmaddpackage(NAME pybind11 GITHUB_REPOSITORY pybind/pybind11 VERSION 2.10.0)
# Define the Python module
pybind11_add_module(_amso module_amso.cpp)

# Link against your main library
target_link_libraries(_amso PUBLIC amso_core pybind11::module Python::Python)

# Install the compiled Python module
install(TARGETS _amso LIBRARY DESTINATION amso)
```

Now we could use `CPM` to add `pybind11` with an exact version directly from GitHub. When you build with `CMake`, you will notice that `CPM` downloads the dependencies and makes them available in `build/_deps/pybind11*`.

Next is to introduce AMSO as a Python module `_amso` via `pybind11`. I deliberately chose the leading underscore here because the C++ library acts like a hidden module that the `amso` module uses, so we can use `from _amso import MemArray1DInt32`, for example. This way, we have control over what the user can see as a public interface. We link everything we need to our module and install the libraries as well.

## Python build via `pyproject.toml`

Up to this point, we can compile our C++ library and make it ready to be used as a Python module. But it isn't a Python package yet; for that, we need to include some pure Python code as well and describe the Python package. To describe packages, Python uses [`pyproject.toml`](https://peps.python.org/pep-0621/), and AMSO's looks like this:

```toml
[build-system]
requires = ["scikit-build-core"]
build-backend = "scikit_build_core.build"

[project]
name = "amso"
version = "0.0.0"
authors = [
  { name = "Ludwig Schneider", email = "ludwigschneider@uchicago.edu" },
]
dependencies = ["numpy"]
description = "C++/GPU accelerated code for polymer multi-block systems. This is a refactoring of SOMA https://doi.org/10.1016/j.cpc.2018.08.011."
readme = "README.md"
requires-python = ">=3.10"
license = { file = "LICENSE" }
urls = { homepage = "https://github.com/InnocentBug/AMSO" }
```

The project section describes the name, author, license, etc. It notably also includes detailed information about other Python packages that are needed. In AMSO's case, we define a dependency on `numpy` for now.

The interesting part in terms of the blog post is the `build-system` section. Most Python packages use `setuptools` to describe how the pure Python packages can be packaged. This information can then be used to automatically install that Python package, for example, with `pip` like:

```shell
pip install .
```

Where `.` refers to the location of the `pyproject.toml` file, which is in AMSO's root directory.

However, since we have a C++ extension to compile and not just pure Python, we use a different Python build backend: `scikit_build_core`.

## SciKit-Build-Core

[SciKit](https://scikit-learn.org/stable/index.html) is a scientific machine learning package for Python. The developers of that package had a very similar need: a Python build backend that supports C++ extensions described via `CMake`. So they developed [`SciKit-Build-Core`](https://github.com/scikit-build/scikit-build-core) for this purpose, which is now independent of the actual SciKit project. And it serves exactly our needs for AMSO. It identifies the `CMakeLists.txt` file, builds the C++ extension for us, and installs the Python as a Python package!

You can see what's going on in the background when installing with verbose options:

```shell
pip install . -vvv
```

Output:

```shell
Using pip 23.0.1 from /home/XXX/amso/lib/python3.11/site-packages/pip (python 3.11)
Non-user install because user site-packages disabled
Created temporary directory: /tmp/pip-build-tracker-pxezdgr4
....
Processing /home/XXX/git/AMSO
...
  Collecting scikit-build-core
    Using cached scikit_build_core-0.10.7-py3-none-any.whl (165 kB)
...
  Successfully installed packaging-24.2 pathspec-0.12.1 scikit-build-core-0.10.7
...
```

I shorten the output a little bit, but we can see how pip creates a temporary directory to build for us and installs scikit-build-core to do the building.

```shell
  *** scikit-build-core 0.10.7 using CMake 3.25.1 (metadata_wheel)
  Preparing metadata (pyproject.toml) ... done
  Source in /home/XXX/git/AMSO has version 0.0.0, which satisfies requirement amso==0.0.0 from file:///home/XXX/git/AMSO
...
  *** scikit-build-core 0.10.7 using CMake 3.25.1 (wheel)
  *** Configuring CMake...
  loading initial cache file /tmp/tmpnbo58tzh/build/CMakeInit.txt
  -- The CXX compiler identification is GNU 12.2.0
...
  -- CPM: Adding package pybind11@2.10.0 (v2.10.0)
  -- pybind11 v2.10.0
...
  -- Build files have been written to: /tmp/tmpnbo58tzh/build
```

Next, we see the usual `CMake` configuration output including CPM adding `pybind11` for us.
Here is important to note, that `SKBuild` inserts the variables for the project name and project version from the `pyproject.toml` file.
So everything is in just one place.

After configuring, building is next and this looks a bit different then before and the reason is that it uses [`ninja`](https://ninja-build.org/) instead of `Make` to build the project.
That can be a bit faster, but we don't have to focus on that too much.

```shell
  *** Building project with Ninja...
...
  [4/22] Building CXX object src/CMakeFiles/amso_core.dir/logger.cpp.o
  [5/22] Building CXX object src/CMakeFiles/amso_core.dir/platform_status.cpp.o
...
  [12/22] Linking CXX static library src/libamso_core.a
...
  [21/22] Building CXX object python/CMakeFiles/_amso.dir/module_amso.cpp.o
  [22/22] Linking CXX shared module python/_amso.cpython-311-x86_64-linux-gnu.so
```

We see how our different targets are build and linked together.

The next step is to package everything into a python wheel:

```shell
  *** Installing project into wheel...
  -- Installing: /tmp/tmpnbo58tzh/wheel/platlib/include
...
  -- Installing: /tmp/tmpnbo58tzh/wheel/platlib/amso/_amso.cpython-311-x86_64-linux-gnu.so
  -- Set runtime path of "/tmp/tmpnbo58tzh/wheel/platlib/amso/_amso.cpython-311-x86_64-linux-gnu.so" to ""
  -- Up-to-date: /tmp/tmpnbo58tzh/wheel/platlib/include
...
  *** Making wheel...
  *** Created amso-0.0.0-cp311-cp311-linux_x86_64.whl
  Building wheel for amso (pyproject.toml) ... done
  Created wheel for amso: filename=amso-0.0.0-cp311-cp311-linux_x86_64.whl size=2063167 sha256=f51f5b72d1e4ac922536b6d066197125ab10fb732ea59480dc984a3c2699b9dd
  Stored in directory: /tmp/pip-ephem-wheel-cache-iq85ho3q/wheels/09/67/7c/81092adbc8c0d1458fa653c542c7ae08a4f37e6809b8f82e3d
  Successfully built amso
```

Now the wheel is completely build, and we is being installed by pip into a location where you python interpreter will find it, ready to import.

```shell
Installing collected packages: amso
  Attempting uninstall: amso
    Found existing installation: amso 0.0.0
    Uninstalling amso-0.0.0:
      Created temporary directory: /home/XXX/py-env/amso/lib/python3.11/site-packages/~mso-0.0.0.dist-info
      Removing file or directory /home/XXX/py-env/amso/lib/python3.11/site-packages/amso-0.0.0.dist-info/
...
      Successfully uninstalled amso-0.0.0

Successfully installed amso-0.0.0
Removed build tracker: '/tmp/pip-build-tracker-pxezdgr4'
```

Now it's installed, and we can import the module with Python:

```shell
cd ~
python -c "import amso"
```

If you want to modify some of the options for CMake during the build process, you can either export those as environment variables or place the settings as meta-data into `pyproject.toml`:

```toml
[tool.scikit-build]
cmake.define = { "ENABLE_CUDA" = "ON" }
```

## Conclusion

This whole setup allows us to exactly describe how the C++ extension should be built via `CMake`, but it offers the convenience of installing a Python project. CPM offers package management for C++ libraries, `pyproject.toml` defines the Python dependencies, and specifies via `scikit-build-core` to compile, package, and install everything with just a single command.

### Next steps

Next will be building a class that can hold our C++ memory on both the CPU (host) and device (GPU) and export this conveniently to Python. After that, we can write and tune some CUDA kernels for part of the AMSO task, calculating density fields.
