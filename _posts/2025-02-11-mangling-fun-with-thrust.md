---
layout: blog
author: Ludwig Schneider
tags: [software engineering, c++-linking, name mangling, amso]
---

# Translation and Mangling Fun

[Name Mangling](https://en.wikipedia.org/wiki/Name_mangling) happens when compilers translate source code with overloaded functions into object code. Since overloaded functions have the same function name but different template or function arguments, the compiler translates these function names into unique names for each overloaded function. Understanding mangling is important to follow the translation your code undergoes before linking.

For example, the function declaration `void foo(std::vector<int>&a)` will be translated by GCC to `_Z3fooRSt6vectorIiSaIiEE` in a library for linking. This unique string is then used during the linking stage if an external library wants to link to our `foo` function. We can use the tool [`c++filt`](https://www.man7.org/linux/man-pages/man1/c++filt.1.html) to translate mangled names back into human-readable declarations[^1].

```shell
$ c++filt _Z3fooRSt6vectorIiSaIiEE
foo(std::vector<int, std::allocator<int> >&)
```

Usually, we do not have to worry about this at all, and it happens behind the scenes.

To follow along with this repository, you can find all files to reproduce this on [GitHub](https://github.com/InnocentBug/thrust_mangling) as well.

## Multi-Language Translation

When we use multiple languages, things get a bit more interesting, because with different languages we may have different compilers translating different units. After translation, the units get linked together. Now, the name mangling has to be identical from the different compilers to allow linking the differently translated units.

While building code for the [AMSO](http://ludwigschneider.net/motivation-for-amso) project, I came across an instance where this step failed. In my particular instance, I was using `gcc` to translate regular C++ code and `nvcc` to translate CUDA/Thrust code. And in this instance, `gcc` and `nvcc` translate the names of functions containing `thrust` objects in the signature differently. This is particularly annoying when building the C++ project as a Python extension, because the Python extension is a library that is dynamically loaded by Python. As a result, at the compilation stage, no linking errors are reported, but when loading the module with Python, the missing linked objects result in errors.

### Simple Example Problem[^2]

Let's look at a minimal example. A library handling CUDA-specific operations may have a source file `nvcc.cu`:

```cpp
#include <iostream>
#include "nvcc.cuh"

void foo(std::vector<int>&a){
  //Potentially some CUDA code
  std::cout<<"std::vector "<<a.size()<<std::endl;
}

void bar(thrust::host_vector<int>&a){
  std::cout<<"thrust::vector "<<a.size()<<std::endl;
}
```

With the corresponding header file `nvcc.cuh`:

```cpp
#pragma once
#include <vector>
#include <thrust/host_vector.h>

void foo(std::vector<int>&a);
void bar(thrust::host_vector<int>&a);
```

We can translate this into an object file using `nvcc` like this[^3]:

```shell
$ nvcc -std=c++17 -x cu -c nvcc.cu -o nvcc.cu.o
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
```

Here, we select `C++17` as our standard and instruct `nvcc` with `-x cu` to build this as a CUDA source file. With [nm](https://www.man7.org/linux/man-pages/man1/nm.1.html) we can inspect the resulting object file and use [grep](https://www.man7.org/linux/man-pages/man1/grep.1.html) to only see our declared functions.

```shell
$ nm nvcc.cu.o | grep foo
0000000000000cbf t _GLOBAL__sub_I__Z3fooRSt6vectorIiSaIiEE
000000000000002e T _Z3fooRSt6vectorIiSaIiEE
$ c++filt _Z3fooRSt6vectorIiSaIiEE
foo(std::vector<int, std::allocator<int> >&)
```

The first column of the output is the memory address of our objects, and the second, `t`/`T`, specify that the functions are dynamic/static symbols in the object file. The first row, we can ignore, but the second row gives us our expected signature of `foo` when demangled. We do not expect issues for `foo` since there is no `thrust` object in the signature.

The situation is different for the `bar` function:

```shell
$ nm nvcc.cu.o | grep bar
0000000000000087 T _Z3barRN6thrust20THRUST_200700_520_NS11host_vectorIiSaIiEEE
$ c++filt _Z3barRN6thrust20THRUST_200700_520_NS11host_vectorIiSaIiEEE
bar(thrust::THRUST_200700_520_NS::host_vector<int, std::allocator<int> >&)
```

And here is where we notice the first unexpected thing. The translation now includes a namespace addition, `thrust::THRUST_200700_520_NS`, to the `thrust` namespace. This is fine, and we usually ignore these details. It probably includes specific optimizations for my specific GPU architecture. However, it has to be identical if we translate this code with `g++`.

So, let's take a look at some C++ code that uses this code: `gcc.cpp`

```cpp
#include "nvcc.cuh"

void do_work(){
  auto std_vector = std::vector<int>(11);
  foo(std_vector);

  thrust::host_vector<int> thrust_vector(std_vector);
  bar(thrust_vector);
}
```

This code we want to translate with `g++` since it does not contain any CUDA-specific code.

```shell
g++ -isystem /usr/local/cuda-12.8/include -std=c++17 -c gcc.cpp -o gcc.cpp.o
```

Same as before, we can inspect the translated object file.

```shell
$ nm gcc.cpp.o | grep foo
                 U _Z3fooRSt6vectorIiSaIiEE
$ c++filt _Z3fooRSt6vectorIiSaIiEE
foo(std::vector<int, std::allocator<int> >&)
```

For the foo function, we find the signature mangled in the object file, but there is no first column with a memory address, and the second column shows `U`. The `U` stands for "undefined," and this is OK, since the object file uses the `foo` function, but it does not have an implementation for it. For the implementation, we want to link to the `nvcc.cu.o` object file from before that provides the translation.

The situation is different for the `bar` function:

```shell
$ nm gcc.cpp.o | grep bar
                 U _Z3barRN6thrust35THRUST_200700___CUDA_ARCH_LIST___NS11host_vectorIiSaIiEEE
$ c++filt _Z3barRN6thrust35THRUST_200700___CUDA_ARCH_LIST___NS11host_vectorIiSaIiEEE
bar(thrust::THRUST_200700___CUDA_ARCH_LIST___NS::host_vector<int, std::allocator<int> >&)
```

At first, it looks similar; the function is also undefined in the source file, and we expect it to link against the function from `nvcc.cu.o`. However, upon closer inspection, we notice that

```text
bar(thrust::THRUST_200700___CUDA_ARCH_LIST___NS::host_vector<int, std::allocator<int> >&)  // gcc
bar(thrust::THRUST_200700_520_NS::host_vector<int, std::allocator<int> >&)                 // nvcc
```

are not identical. The namespace additions after `thrust::` are different.
During the translation with `nvcc` we probably have a macro definition that replaces `_CUDA_ARCH_LIST_` with the appropriate device specification, here `520`. And part is missing when translating with `gcc`.
Unfortunately, this just passes without error or warning, So trying to link those together will result in a linking error.

Let's examine a short main file to show this[^4].

```cpp
void do_work();

int main(){
  do_work();
}
```

And compile and link it like so:

```shell
$ g++ -std=c++17 main.cpp nvcc.cu.o gcc.cpp.o -L /usr/local/cuda-12/lib64/ -lcudart -o main
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
/usr/bin/ld: gcc.cpp.o: in function `do_work()':
gcc.cpp:(.text+0x5d): undefined reference to `bar(thrust::THRUST_200700___CUDA_ARCH_LIST___NS::host_vector<int, std::allocator<int> >&)'
collect2: error: ld returned 1 exit status
```

As expected, the linker `ld` cannot find a reference to the function `bar(thrust::THRUST_200700___CUDA_ARCH_LIST___NS::host_vector<int, std::allocator<int> >&)` anywhere, because in the object file `nvcc.cu.o` defines the function as `bar(thrust::THRUST_200700_520_NS::host_vector<int, std::allocator<int> >&)` instead. This is a tedious problem that we can only identify by diving into the details of translation, name mangling, and linking.

## Solution: NVCC all the way

The problem arises because `g++` and `nvcc` translate the `thrust` object signatures differently. This may be a bug within the `thrust` library, but we need to find a workaround to get our project working. The easiest solution is to compile everything as CUDA code, since C++ is a subset of CUDA and it should compile just fine. We lose the flexibility of choosing the compiler with its specific features, like robustness and optimization, but we can get the project to work. This applies to all source files that use function signatures from `thrust`. Here, that is `gcc.cpp` but not `main.cpp`.

So, in our example system, we use the nvcc to compile `gcc.cpp`:

```shell
$ nvcc -std=c++17 -x cu -c gcc.cpp -o gcc.cpp.o
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
```

and inspect it as before:

```shell
$ nm gcc.cpp.o | grep bar
                 U _Z3barRN6thrust20THRUST_200700_520_NS11host_vectorIiSaIiEEE
$ c++filt _Z3barRN6thrust20THRUST_200700_520_NS11host_vectorIiSaIiEEE
bar(thrust::THRUST_200700_520_NS::host_vector<int, std::allocator<int> >&)
```

Now the function signature matches between the two object files, and we can link and execute the code.

```shell
$ g++ -std=c++17 main.cpp nvcc.cu.o gcc.cpp.o -L /usr/local/cuda-12/lib64/ -lcudart -o main
$ ./main
std::vector 11
thrust::vector 11
```

In a project using `CMake`, we can achieve this by explicitly setting the required source files as CUDA as follows:

```shell
set_source_files_properties(gcc.cpp PROPERTIES LANGUAGE CUDA)
```

This will be the workaround in AMSO for now until this bug gets fixed within `thrust`.

## Bug Report

I reported this bug to nvidia and you can find it [here](https://developer.nvidia.com/bugs/5154007).
You may need to have an NVIDIA Developer account to see the bug report unfortunately.
I am trying to update this post if this gets resolved.

---

[^1]: Notice how we declared `void foo(std::vector<int>&a)` without the default template argument for the allocator, but the translated name includes that template argument.

[^2]: This problem may be very version specific, so for reference I encounter this problem with `g++` version `g++ (Debian 12.2.0-14) 12.2.0` with a CUDAToolKit 12.8 and `nvcc` version `Cuda compilation tools, release 12.8, V12.8.61 Build cuda_12.8.r12.8/compiler.35404655_0` and my GPU is a NVIDIA RTX2080 super with the deprecating compute capability `7.5`.

[^3]: My desktop GPU is an NVIDIA 2080 super which compute capability (7.5) is deprecating. I don't think the warnings are related to the linking issue.

[^4]:
    I forward declare `void do_work()` in order to avoid another header file for `gcc.cpp`. The results are identical though.
    {: data-content="footnotes"}
