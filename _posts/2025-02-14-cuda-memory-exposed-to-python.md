---
layout: blog
author: Ludwig Schneider
tags: [software engineering, amso, CUDA, thrust, dlpack]
---

# A (CUDA) Memory Handling Class for AMSO

For [AMSO](http://ludwigschneider.net/motivation-for-amso), efficient memory management is at the core of handling the complex computations required for particle positions, density fields, and interaction calculations. Designing a memory handling class that seamlessly integrates with both C++ and Python while leveraging CUDA for peak performance is no small feat—but it’s exactly what I set out to do.

In this blog post, I’ll walk you through the design and implementation of a custom memory management class tailored for AMSO. Along the way, we’ll explore some key highlights, including:

- **Templated C++ Design**: How I used templates to create a flexible and type-safe memory class that supports various data types and dimensions.
- **Host-Device Memory Transfers**: Leveraging `thrust` for asynchronous and efficient memory movement between host and device.
- **Python Interoperability**: Binding the C++ class to Python using `pybind11` and enabling zero-copy data sharing with popular libraries like NumPy, PyTorch, and JAX via DLPack.
- **CUDA Optimization**: Implementing lightweight indexing with NVIDIA's CUDA Core Compute Libraries (CCCL) for efficient array access on both host and device.
- **Testing and Debugging**: Setting up robust testing frameworks in both C++ (with Google Test) and Python (with pytest) to ensure correctness.

Whether you're working on a similar project or just curious about how to build high-performance memory classes from scratch, this post will provide insights into balancing low-level control with high-level usability. For a full working code study, refer to this [pull-request](https://github.com/InnocentBug/AMSO/pull/1) on GitHub. Keep in mind that this blog is for illustration purposes—I may leave out some implementation details here, so check GitHub for the complete, maintained version.
If you are new to C++ and CUDA programming this blog post may be challenging, but I encourage you to stick with me as we will cover a lot interesting ground.

Let’s dive into the details of how this memory handling class was designed and built!

## AMSO's Requirements

For AMSO, we will have to handle memory for a variety of things. Most importantly, probably the positions of all the particles, density & interaction fields, etc.
In this blog post, I discuss how I went about designing this memory holding class. This covers some general talking points, but also some technical considerations and details.
For a full working code study, refer to this [pull-request](https://github.com/InnocentBug/AMSO/pull/1) on GitHub. Please do not copy code directly from here, as I may leave important implementation details out. This blog is for illustration purposes; check the `GitHub` for a maintained, including all details, and working version.

Since AMSO is written in C++ for peak performance with a Python binding for convenience, we need to be able to read and write the memory from both C++ and Python.
We could use external libraries like [CUPY](https://docs.cupy.dev/en/stable/reference/generated/cupy.array.html), [NUMBA](https://numba.readthedocs.io/en/stable/reference/types.html#arrays), or even Deep Learning Frameworks like [JAX](https://docs.jax.dev/en/latest/_autosummary/jax.Array.html) or [PyTorch](https://pytorch.org/docs/stable/tensors.html) and use their memory handling implementations.
However, that limits us in some aspects like how the memory is allocated and moved, a level of control that I do not want to give up for AMSO.
So instead, I am building a C++ class that handles all of this for us, exactly the way I want it to, but we do not have to give up the convenience of using such frameworks either. More on that and [DLPack](https://github.com/dmlc/dlpack) later.

The positions have a floating point type, and we can represent them as 2D arrays, with one dimension counting the number of particles and the other X, Y, Z component[^1].
Density fields are usually integral types in 3D, but for interactions, we multiply constants which transforms them into floating point types.

## A Templated C++ Class

From this requirement, we know that we will have to make a template class that supports two template parameters: the data type of the array and a dimension parameter.
The data type template parameter `T` allows type safety, and the dimension parameter `ndim` allows maximum performance.
Since template parameters are resolved at compile time, a lot of optimizations with a known dimension can be made.

However, C++ is a statically compiled language and Python is an interpreted language.
We could dynamically `jit` compile our C++ templates as they are requested from Python at runtime, but for AMSO, we usually know dimensions and types ahead of time.
So, we take the simpler approach here and explicitly instantiate the types and dimensions we need for AMSO at compile time.
And in case we request something unknown at runtime from Python, we give a meaningful error message prompting a recompile.

## The C++ Interface

Let's take a look at the actual interface.
I am going to omit parts of the source code, like includes and namespace, here for clarity. For a full working code piece, look at the GitHub [pull-request](https://github.com/InnocentBug/AMSO/pull/1).
The templated class definition can be found in `AMSO/include/memarray.h`.

```cpp
template <typename T, int ndim>
class MemArray {
private:
  thrust::host_vector _host_vec;
  thrust::device_vector _device_vec;

  std::array<int, ndim> _shape;
  bool _on_device;
  int _device_id;
  int64_t _lock_id; // Context manager lock, ID. Negative means unlocked.
                    // Positive, locked to that context manager
```

Having done quite a bit of Python development, I like to declare my member variables of classes starting with an underscore.
This doesn't have the same `private` meaning as in Python; C++ has the much more robust `private`, `public` & `protected` qualifiers for that.
But it helps me as a reminder to be careful whenever I directly access class member variables, and it's easier to ensure they don't get overshadowed by local variables.

This declares all the most important member variables of our class.
`_host_vec` and `_device_vec` are the containers that actually store the data.
We can use the `thrust` vectors here, instead of having to deal with pointers and cudaMalloc and cudaFree directly.
These containers not only allow us to write the code much more robustly, but they also open the door for later optimization and usage of the thrust library.

Since we know the number of dimensions as a template argument, we can use `std::array` to store the shape of our array.
`std::array` is a thin wrapper, so performance and storage are minimal. We don't need to dynamically allocate heap memory like we would have to with an `std::vector`, but we have all the conveniences that we love from standard containers.

`_on_device` keeps track of whether we consider the device or host copy of the data as current, `_device_id` remembers on which GPU ID we allocated the memory, and `_lock_id` is a bit more interesting.
`_lock_id` will handle locking the memory for the situation where an outside observer (i.e., Python) reserved access to the memory and we need to be careful.
Throughout the code, a positive `_lock_id` means that someone locked access, but a negative `ID` refers to the state where this class is the sole owner.

```cpp
public:
  static constexpr int NDIM = ndim;

  MemArray(const std::array<int, ndim> &shape, int device_id); // Regular constructor for new memory
```

We outwardly declare the number of dimensions as `constexpr`. Because it's a `constexpr`, it's very cheap to access, but it offers the convenience of accessing it when the template parameter is not directly reachable.

We follow with our main constructor. It requires only the shape and device ID as input. Except for error checking, it only initializes both thrust vectors. It automatically initializes the memory with `T()` everywhere. For more useful initialization, we'll later rely on NumPy and Python arrays.

```cpp
  ~MemArray(){};                                                 // Destructor
  MemArray(const MemArray<T, ndim> &other);                      // Copy constructor (expensive)
  MemArray<T, ndim> & operator=(const MemArray<T, ndim> &other); // Copy Assignment operator
```

Since this is a resource handling class, we also implement a destructor, copy constructor, and copy assignment operator. Next is a swap function that utilizes the thrust vector swap implementation for zero-copy swapping.

```cpp
  void swap(MemArray<T, ndim> &other) noexcept;
  template<typename U, int N>
  friend void swap(MemArray<U, N> &lhs, MemArray<U, N> &rhs) noexcept;
```

With swap, we can also implement the move constructor and move assignment operator without any resource allocation.

```cpp
  MemArray(MemArray<T, ndim> &&other) noexcept; // Move constructor (cheap)
  MemArray<T, ndim> &operator=(MemArray<T, ndim> &&other) noexcept; // Move Assignment operator
```

We also have public getter functions to allow reading the state of our variables. Note that `get_lock_id()` makes it possible to maliciously get access to locked memory, even when you weren't supposed to. However, this is C++, and it's only intended for users in the know. This mirrors the analogy of C++ from Herb Sutter: "In C++ we have some very sharp knives; they are very useful if you know what you're doing, but you can also cut yourself easily." If you circumvent the locking mechanism, you're taking the sharp knives out of the drawer. Be careful not to cut yourself.

```cpp
  int get_device_id() const { return _device_id; }
  const std::array<int, ndim> &get_shape() const { return _shape; }
  int64_t get_size() const { return _size; }
  bool get_on_device() const { return _on_device; }
  int64_t get_lock_id() const { return _lock_id; }
```

However, if you want to make sure to protect your access to the memory, here's a function that will throw an exception if you accidentally access memory you're not supposed to:

```cpp
  void throw_invalid_lock_access(const int64_t requested_lock_id) const;
```

Now we get to the juicy bits of this class: moving memory to and from device and host. We use the `DLPack` definitions of host or device as the first parameter to indicate the destination. This implementation also lets you do the transfer operations in a specific CUDA stream and asynchronously if desired.

```cpp
  void move_memory(const DLDeviceType requested_device_type,
                   const int64_t requested_cuda_stream_id,
                   const int requested_device_id,
                   const int64_t requested_lock_id = -1,
                   const bool async = true);
```

This comprehensive interface provides a robust and flexible way to manage memory for AMSO, giving us the control we need while still allowing for efficient operations and interoperability with other frameworks.

Next are the two methods to lock and unlock memory access. We'll later see that these are primarily intended to be used in Python context managers.

```cpp
  // Context manager methods
  void enter(const int64_t lock_id);
  void exit() { _lock_id = -1; }
  void read_numpy_array(pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast> np_array);
```

The `read_numpy_array` function I'll explain [later](#bind-it-to-python-with-pybind11); it's a bit unfortunate that we have to implement it explicitly.

Now, let's look at some more interesting aspects: how we're going to access the arrays. There are basically two ways. One is the safe way that checks memory access, lock status, device location, and out-of-bounds for you, but that comes with a bit of overhead. I overload the `()` operator for read access, and we have a `write` function for write access. Indices are lightweight `std::array` again. We could implement it as variable arguments and check statically against the template parameters, but this way it's very transparent, and you can just do `mem_array({3, 5, 7})` if you want to access the location `x=3, y=5, z=7`. That's straightforward. Notice how the functions also take `cuda::std::array`; that has to do with the NVIDIA CCCL library, and I'll explain it in more detail when we discuss the indexer.

```cpp
  // encapsulated access, not zero-overhead
  const T operator()(const std::array<int, ndim> &indices, const int64_t requested_lock_id = -1) const;
  const T operator()(const cuda::std::array<int, ndim> &indices, const int64_t requested_lock_id = -1) const;

  void write(const std::array<int, ndim> &indices, const T &value, const int64_t requested_lock_id = -1);
  void write(const cuda::std::array<int, ndim> &indices, const T &value, const int64_t requested_lock_id = -1);
```

But I also implement a no-overhead, no-checking (i.e., sharp knives out of the drawer) access. You can directly access a pointer to the underlying memory. It's your responsibility to keep the memory valid while you have the pointer and also your responsibility to only do valid things with it. It's intended to be used to obtain pointers to write CUDA kernels, etc.

```cpp
  // Zero over-head pointer access
  // Taking the sharp knives out of the drawer: don't cut yourself
  T *ptr(const int64_t requested_lock_id = -1);
  const T *ptr(const int64_t requested_lock_id = -1) const;
```

Lastly, we have two functions that are intended to work with DLPack. These are interesting, and we'll examine them in detail [later](#bind-it-to-python-with-pybind11).

```cpp
  pybind11::capsule get_dlpack_tensor(const bool versioned,
                                      const int64_t requested_lock_id);

  std::pair get_dlpack_device() const;
};
```

### Bind it to Python with Pybind11

To use this class meaningfully from Python, we first bind the C++ to Python via `pybind11`, and then we write a convenience wrapper class around it in Python. The wrapping isn't necessarily needed, but I like that approach because it cleanly separates the C++ and Python sides, and gives us some of the conveniences of Python to write input checking, etc., with more meaningful error messages.

Let's look at binding it with `pybind11` first. We create the module `_amso` with `pybind11`. In `AMSO/python/module_amso.cpp`, we have:

```cpp
#include

#include "dlpack_pybind.h"
#include "memarray_cuda.cuh"
#include "memarray_pybind.h"

namespace py = pybind11;

PYBIND11_MODULE(_amso, m) {
  amso::pybind_dlpack(m);
  amso::bind_mem_array(m);
  amso::bind_add_index_tuple(m);
}
```

The `PYBIND11_MODULE(_amso, m)` is a macro that creates the module `m` in C++. Then, we add components to it with functions. I intentionally chose `_amso` with an underscore to separate it from the `amso` package, which contains the Python part of the project, but we frequently import members of `_amso` into `amso`.

For the `MemArray` class, we have a template function that can bind it individually to Python. For this, we look at `memarray.cpp`:

```cpp
template <typename MemArrayType>
void template_bind_mem_array(pybind11::module &m, std::string python_name) {
  pybind11::class_<MemArrayType>(m, python_name.c_str())
      .def(pybind11::init<const std::array<int, MemArrayType::NDIM> &, int>())

      .def("_enter", &MemArrayType::enter)
      .def("_exit", &MemArrayType::exit)
      .def("_move_memory", &MemArrayType::move_memory)
      .def("_dlpack", &MemArrayType::get_dlpack_tensor)
      .def("_read_numpy_array", &MemArrayType::read_numpy_array)
      .def("_dlpack_device", &MemArrayType::get_dlpack_device);
}
```

The line `pybind11::class_(m, python_name.c_str())` binds our `MemArray` to Python with a name given as a string. We follow with the individual methods that we want to export to Python. I declare them all as private (with leading underscores) since I don't envision a user directly calling C++ functions; they all will be wrapped in Python.

This approach provides a clean interface between C++ and Python, allowing us to leverage the strengths of both languages in our AMSO project.

The challenge here is that `MemArray` is a template class, so we have to instantiate the exact type and size combinations we later want to make usable in Python.

```cpp
template class MemArray<int32_t, 1>;
using MemArray1DInt32 = MemArray<int32_t, 1>;
```

This explicitly declares the class for 32-bit int types and one dimension.
I repeat similar lines for `int64_t`, `float`, `double` and `1`, `2`, and `3`.
This instructs the compiler to have compiled versions of these classes ready in the dynamic library that we later link to Python.
To link these individual instances to Python, we call the previously defined template function with the specified types.

```cpp
void bind_mem_array(pybind11::module &m) {
  template_bind_mem_array<MemArray1DInt32>(m, "MemArray1DInt32");
  template_bind_mem_array<MemArray2DInt32>(m, "MemArray2DInt32");
  template_bind_mem_array<MemArray3DInt32>(m, "MemArray3DInt32");
  ...
}
```

So the `bind_mem_array` function that we call with the `_amso` module binds the class to Python with a specific name and type combination.

Now, let's look at the Python side in `AMSO/python/amso/memarray.py`.
`AMSO/python/amso/` is a Python module and has a `__init__.py`, that will be installed with [`SKBUILD-CORE`](http://ludwigschneider.net/amso-build-system) (more on that later).

```python
from ._amso import (
    MemArray1DFloat,
    MemArray2DDouble,
    ...
)
import numpy as np
```

You can see that we individually import all the C++ classes from the `_amso` package here.
For the user's convenience, we don't want all of them to be separate Python classes. Instead, we want one unified class that maps to the correct C++ class effortlessly.

```python
class MemArray:
    _SUPPORTED_DEVICE_TYPES = {amso_dlpack.kDLCUDA, amso_dlpack.kDLCPU}
    # On the C++ side we instantiate only certain types from the template.
    # This map maps the user arguments for types and ndim to the correct instance
    _TYPE_INSTANCE_MAP = {
        (np.dtype(np.float32), 1): MemArray1DFloat,
        (np.dtype(np.float64), 2): MemArray2DDouble,
    }

    def __init__(self, shape: list[int] | np.ndarray, dtype: np.dtype | None = None, device_id: int = 0):
        if dtype is None:
            dtype = np.dtype(float)

        self._dtype = np.dtype(dtype)
        self._shape = [int(element) for element in shape]
        self._ndim = len(self._shape)

        try:
            self._cpp_type = self._TYPE_INSTANCE_MAP[(self._dtype, self._ndim)]
        except KeyError as exc:
            raise RuntimeError(
                f"The compiled C++ AMSO only supports a number of predefined types and ndim. You requested {self._dtype} and {self._ndim} ({self._shape}), but AMSO only knows {self._TYPE_INSTANCE_MAP.keys()}. If you need your types, consider adding more instances of the C++ template, and add this type combination to the map."
            ) from exc

        self._cpp_obj = self._cpp_type(self._shape, device_id)
```

For this purpose, we have a dictionary that maps types and sizes to C++ class types.
In the `__init__` function of the class, we call the appropriate constructor as requested.
Note how we keep the C++ instance just as a private member variable.
This is very similar to the "Pointer To Implementation" (Pimpl) idea in C++, except here the interface is in Python, and the pointer is to the C++ implementation.

The class also wraps some of the functions from C++ to Python:

```python
    def move_memory(self, device_type: amso_dlpack.DLDeviceType, device_id: int | None = None, stream: int = 0, async_copy: bool = False ):
        if device_id is None:
            current_device_type, current_device_id = self._cpp_obj._dlpack_device()
            device_id = current_device_id

        if device_type in self._SUPPORTED_DEVICE_TYPES:
            self._cpp_obj._move_memory(device_type, stream, device_id, -1, async_copy)
        else:
            raise RuntimeError(f"Requested device {device_type} not supported. Supported types: {self._SUPPORTED_DEVICE_TYPES}")
    def read_numpy_array(self, array):
        self._cpp_obj._read_numpy_array(array)
```

These wrappers can be very thin, but some add convenient default variables and/or error messages.

The most interesting function concerns the wrapping as a DLPack tensor.
[DLPack](https://github.com/dmlc/dlpack) is a specification of how tensors are represented to Python, so that different packages like `numpy`, `pytorch`, `jax`, `cupy` can process them with zero copying for performance.
We are implementing that here. This is the user-facing function:

```python
  def get_dlpack(self, device_type: amso_dlpack.DLDeviceType | None = None, device_id: int | None = None, stream: int = 0 ):
    ...
    self._cpp_obj._move_memory(device_type, stream, device_id, -1, False)
    return self.DLPackWrapper(int(uuid.uuid4()), self._cpp_obj)
```

This creates another thin wrapper and a unique `uuid` that we use to lock the memory.

That thin wrapper looks as follows:

```python
    class DLPackWrapper:
        def __init__(self, lock_id: int, cpp_memarray):
            self._lock_id: int = lock_id
            self._cpp_memarray = cpp_memarray

        def __enter__(self):
            self._cpp_memarray._enter(self._lock_id)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self._cpp_memarray._exit()
```

We keep a pointer to the C++ implementation and its specific lock id.
Since this is used with a context manager, we implement the corresponding `__enter__` and `__exit__` functions that wrap the C++ equivalents.
But to make it visible to other frameworks, like `numpy`, there is an exact [definition](https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html) of which and how these functions need to be implemented.
See the definition for details; here I'm cutting it short and only showing you how they wrap the C++ functions.

```python
        def __dlpack__(self, stream: int | None = None, max_version: tuple[int, int] | None = None, dl_device: tuple | None = None, copy: bool | None = None):
            dlpack = self._cpp_memarray._dlpack(False, self._lock_id)
            return dlpack

        def __dlpack_device__(self):
            return self._cpp_memarray._dlpack_device()
```

Unfortunately, `numpy` doesn't support the versioned DLPack Tensor at the time of this writing, so we need to specifically use the unversioned variant here.
Numpy is actively developing the DLPack support right now, so hopefully in the future we will be able remove these caveats.

This all ties together in user code as follows:

```python
import amso
import numpy as np
mem_array = amso.MemArray([2, 4, 7], float)
with mem_array.get_dlpack(amso.dlpack.kDLCPU) as dlpack:
   local_array = np.from_dlpack(dlpack)
```

With this, you get zero-copy read access to our MemArray class with all the convenience that `numpy` arrays offer.
Note how we request the DLPack here on the `kDLCPU` device; this is the CPU. Numpy can only handle CPU arrays, but for other frameworks like `JAX` or `PyTorch`, we can request `CUDA` as well.

Unfortunately, at the time of this writing, `numpy` only supports DLPack as read-only, so we cannot actually change the memory of our `MemArray` class.
This is a restriction on `numpy`'s part; our implementation is ready, and other packages may support writing directly into our `MemArray` class.
For this reason, I implemented the `read_numpy_array` function that we saw before. The details are not as interesting as it is directly supported by [pybind11](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html#numpy), and it is similar to DLPack.

## DLPack Magic

So how can we use [`DLPack`](https://github.com/dmlc/dlpack) to make our memory accessible to other libraries?
All of this is specified by DLPack, and we a) use the associated `GitHub` repo directly as a source for data structures and b) implement it for our `MemArray` class.
Here is the C++ implementation of `get_dlpack_tensor` from `AMSO/src/memarray.cpp`:

```cpp
template<typename T, int ndim>
pybind11::capsule MemArray::get_dlpack_tensor(const bool versioned, const int64_t requested_lock_id) {
```

This member function of `MemArray` returns a [Python Capsule](https://docs.python.org/3/c-api/capsule.html) from the C-API of Python.
It is basically a type of smart pointer for Python that associates a pointer with a name that Python can handle.
What will become important later is that we hand over the ownership of resources and associate this object directly with Python.
This feels uncomfortable as a C++ programmer because we cannot use "Resource Allocation Is Initialization" (RAII) or otherwise ensure that the resources are released correctly.
We have to trust Python to do that for us.
To break it down: we have to handle resource management for this part a more like in C, not C++, but we enable python later to ensure that resources are released properly.

But first, let's populate and describe our memory for `DLPack` to use as a tensor.
DLPack defines for this purpose a struct `DLTensor` in `dlpack/dlpack.h`, and we populate it.

```cpp
  // Access to internal data
  T *data_ptr = this->ptr(requested_lock_id);

  DLTensor dl_tensor;
  dl_tensor.data = data_ptr;
```

First, we acquire the raw pointer on device or host from the `thrust` vectors.
At this point, the memory should already be locked, since we entered this through a context manager as seen above, and the lock validity is checked by `this->ptr()`.

Then, we use the `DLTensor` structure to populate it.
For now, we can keep the `dl_tensor` instance on stack memory; we will later transfer it to Python-handled memory.

The `data` field is the pointer to our data, and everything that follows explains to a different package via DLPack how to use it.
It's important to understand that we retain ownership of the memory resource of the actual data itself, so we're going to ensure that a) Python does not release the memory and b) this C++ instance needs to be kept alive by Python as long as any DLPack view exists.
More on that later, but this is important because it allows zero-copy access between the packages on the same memory allocation, which is crucial for performance!

Next, we have to describe the device where the memory lives to DLPack:

```cpp
  auto device_pair = get_dlpack_device();
  dl_tensor.device.device_type = device_pair.first;
  dl_tensor.device.device_id = device_pair.second;
```

We reuse a different function `get_dlpack_device`:

```cpp
template <typename T, int ndim>
std::pair<DLDeviceType, int32_t>
MemArray<T, ndim>::get_dlpack_device() const noexcept {
  if (get_on_device())
    return std::make_pair(DLDeviceType::kDLCUDA, get_device_id());
  return std::make_pair(DLDeviceType::kDLCPU, 0);
}
```

It's pretty straightforward: if the memory is on a CUDA device, we use the `kDLCUDA` enum from DLPack and the requested `device_id`, and `kDLCPU` and `0` for host memory.

Next, we focus again on `get_dlpack_tensor` and populate the tensor further:

```cpp
  dl_tensor.dtype = DLDataType{detail::TypeToDLPackCode::code,
                               detail::TypeToDLPackCode::bits,
                               detail::TypeToDLPackCode::lanes};
```

This describes the data type of the memory. I decided to custom implement type traits for our data type to resolve this.
The type traits look as follows, for example:

```cpp
namespace detail {
template<typename T> struct TypeToDLPackCode;

template <> struct TypeToDLPackCode {
  static constexpr DLDataTypeCode code = kDLFloat;
  static constexpr uint8_t bits = 8 * sizeof(double);
  static constexpr uint8_t lanes = 1;
};
}
```

DLPack requires us to use `kDLFloat` for a floating-point type like `double`, followed by the number of bits.
Since `sizeof` reports memory size in units of `char`, which is 8 bits, we can reuse that.
The `lanes` field is for vector types like `float4` from CUDA; for now, we use non-vector types and set it to `1`.

Instead of using custom type traits here, I could use the `std::` defined type traits, but I decided to use custom ones instead. This will alert me to a new type that I haven't handled yet.
This would give me a compile-time error for a new unknown type, and I can think about it and introduce the new type if necessary.

Next, we have to describe the shape of the tensor for DLPack in `get_dlpack_tensor`.

```cpp
  dl_tensor.ndim = ndim;
  auto shape_ptr = std::make_unique(ndim);
  dl_tensor.shape = shape_ptr.get();
  std::copy(get_shape().begin(), get_shape().end(), dl_tensor.shape);
```

`ndim` is self-explanatory: the number of dimensions of the tensor, and we can use the template argument.

The `shape` is also almost self-explanatory, as it describes the tensor's shape.
We need to allocate memory of which we transfer ownership to Python later.
For now, we use a `std::unique_ptr` for that, but already store the raw pointer in the tensor.
If an exception is thrown for any reason before we hand over ownership to Python, the `unique_ptr` will handle the release.

After that, we fill the memory with our `MemArray` shape information with `std::copy`.

Next are `strides` and a potential `byte_offset`.

```cpp
  dl_tensor.strides = nullptr;
  dl_tensor.byte_offset = 0;
```

I could handle strides explicitly for a strided tensor, but for a dense, C-style tensor, DLPack allows us to use `nullptr`, and we make use of that for now.
We may decide later to change this to properly support strided tensors, but for now, we are implementing a memory class, not a tensor library.

Next, we consider the part that manages the lifetime of all the objects with Python.
There are two variants of this in DLPack: one that includes the supported DLPack version and the other one without it.
I show the versioned variant here, but we need to be careful. Numpy does not understand the versioned variant, and it results in segfaults if you use it!

```cpp
  if (versioned) {
    auto versioned_tensor = std::make_unique();

    versioned_tensor->version = DLPackVersion{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
```

Same as before, we create this memory object with a `unique_ptr` so that we can hand over the ownership to Python.

```cpp
    versioned_tensor->manager_ctx = this;
```

Setting our `MemArray` instance as the context manager signals to Python that it needs to keep the `MemArray` alive as long as we have this DLPack capsule.
This is important because the `MemArray` maintains the ownership of the tensor memory.

```cpp
    versioned_tensor->deleter = &detail::dl_tensor_deleter<DLManagedTensorVersioned *, T, ndim>;
```

Here is the next interesting bit, because we need Python to deallocate the memory we used for `shape`. We need to write this deallocation in a deleter function.
I wrote a templated version for it so that it works for both the `versioned` and `unversioned` variants without repeating the code.

Here is the implementation:

```cpp
namespace detail {
template<typename tensor_ptr_type, typename T, int ndim>
void dl_tensor_deleter(tensor_ptr_type self) {
  auto *wrapper = static_cast<MemArray<T, ndim> *>(self->manager_ctx);

  delete[] self->dl_tensor.shape;
  delete[] self->dl_tensor.strides;
  // Invalidate tensor
  self->dl_tensor.data = nullptr;
  self->dl_tensor.ndim = 0;
  self->dl_tensor.shape = nullptr;
  self->dl_tensor.strides = nullptr;
  self->dl_tensor.byte_offset = 0;
}
}
```

The most important part is to release the memory of `shape`; we use `delete []`. Maybe I will find a more elegant version to use the `std::unique_ptr::get_deleter` in the future.
After the release, we invalidate the tensor since it should not be used by Python anymore.
It is very important to set the pointers to `nullptr` because Python might call this function more than once.
We also delete `strides` here. In the current implementation, it's a `nullptr`, but `delete []` of a `nullptr` is well-defined and does nothing.
In future implementations of strides, we will not forget to release it.
Next, we can actually copy the tensor we prepared into this `DLManagedTensorVersioned` object.

```cpp
    versioned_tensor->flags = 0;
    versioned_tensor->dl_tensor = dl_tensor;
```

There is also a flag that could signal read-only, etc., but we do not need it here.

Now, we are turning ownership for all of this over to the Python capsule.

```cpp
    shape_ptr.release();
    return pybind11::capsule(versioned_tensor.release(), "dltensor", &detail::dl_capsule_deleter<DLManagedTensorVersioned *>);
```

We release the `unique_ptr`s of `shape` and `versioned_tensor` so that they do not get deallocated by our `unique_ptr`s.
The latter, we use to directly construct the capsule.
We give it the name `"dltensor"`, and for a capsule, we can also define a function that is called upon memory release.

My implementation looks like this:

```cpp
namespace detail{
template<typename tensor_ptr_type>  void dl_capsule_deleter(PyObject *capsule) {
  void *raw_ptr = nullptr;
  // Can be original name if unused "dltensor"
  if (strcmp("dltensor", PyCapsule_GetName(capsule)) == 0)
    raw_ptr = PyCapsule_GetPointer(capsule, "dltensor");
  else // "used_dltensor if capsule is consumed
    raw_ptr = PyCapsule_GetPointer(capsule, "used_dltensor");

  if (raw_ptr) // Unknown capsule or already freed capsule
  {
    tensor_ptr_type tensor_ptr = static_cast<tensor_ptr_type>(raw_ptr);
    if (tensor_ptr->deleter) // Execute custom deleter, here delete[] shape.
      tensor_ptr->deleter(tensor_ptr);
  }
}
};
```

We get a `PyObject` pointer straight from the Python C-API as an argument.
Because we know it is a Python capsule, we can use `PyCapsule_GetPointer` to obtain the pointer to our `DLManagedTensorVersioned` object.
When the capsule is being used by another package, it changes the name from `dltensor` to `used_dltensor`. In either case, we do some checks and finally call the previously defined function to delete the `DLManagedTensorVersioned` to release the shape memory.
After this call, Python will release the memory for the `DLManagedTensorVersioned` object itself.

This all requires a bit of thinking about the ownership of resources and how they are being transferred and released.
It's also getting a bit more complicated because Python and DLPack are pure C, so we have to let go of some of the safety nets of C++ like type safety.
But it is worth it because this way, we can achieve zero-copy access to device and host memory from our C++ class to Python and other tensor libraries.

## Array Indexing with CUDA Core Compute Libraries (CCCL)

All that we have discussed so far handled just the memory access and basically no tensor-like functionality, except that we package it with DLPack for Python as tensors.
That is intentional; there are bigger and better-implemented tensor libraries out there.
But one convenience functionality that I would like to implement is indexing the array efficiently on the CPU and GPU.
For that, I implement a lightweight `ArrayIndexer` as a member class of `MemArray`:

```cpp
  class ArrayIndexer {
  private:
    T*const _ptr = nullptr;
    const cuda::std::array<int, ndim> _shape;
```

`ArrayIndexer` has only two data members: a raw pointer to the memory of `MemArray` and a `cuda::std::array` to store the shape of the memory.
The total memory footprint of an instance of `ArrayIndexer` is the size of a pointer plus the number of dimensions times `int`.
So, on a 64-bit machine and a 3-dimensional array, this will be 4 x 64 bits.
This small footprint is important because we can copy this without overhead to, for example, GPU kernels.
Additionally, all member variables and member functions are `const`, so they can be easily shared between parallel threads without race conditions.
Notice how we use `cuda::std::array` instead of `std::array`; this is an implementation of the NVIDIA CUDA Core Compute Libraries [(CCCL)](https://github.com/NVIDIA/cccl).
It mirrors the `std::` implementation, with the difference that we can use it in host and device code.
I also declare the constructor `private` with `MemArray` as a friend class. So the only way you can instantiate a new `ArrayIndexer` is by calling `MemArray::get_indexer`.

```cpp
  private:
    friend class MemArray;
    ArrayIndexer() = delete;
    inline ArrayIndexer(const std::array<int, ndim> &shape)
        : _shape(detail::make_cuda_std_array(shape)) { }
```

This tight coupling is important: the `ArrayIndexer` acts like a very lightweight view for the `MemArray` class.
This is a bit like `std::span`, but instead of having just a size, we have the entire shape of the array at our disposal.

We also have a few other convenience functions to access the memory in multiple dimensions:

```cpp
  public:
    __host__ __device__ inline int operator()(const cuda::std::array<int, ndim> &indices) const {
      int index = 0;
      int stride = 1;
      for (int i = ndim - 1; i >= 0; --i) {
        assert(indices[i]  get_indices(int idx) const {
      cuda::std::array<int, ndim> indices;
      for (int i = ndim - 1; i >= 0; --i) {
        indices[i] = idx % _shape[i];
        idx /= _shape[i];
      }
      return indices;
    }
```

This almost concludes the implementation of `MemArray` and the `ArrayIndexer`; however, one last look at the `size` function:

```cpp
    __host__ __device__ inline int size() const {
      return cuda::std::accumulate(_shape.begin(), _shape.end(), 1, cuda::std::multiplies());
    }
  };
```

It would be trivial to precompute the `size` in the `ArrayIndexer` constructor since everything is `const`.
However, in practice, it would mean that the memory footprint of `ArrayIndexer` is one `int` larger, and since fast register memory is scarce in GPU threads, I decided not to store the value.
Instead, we use `CCCL` to compute the size from the `shape` directly, which should be again fully optimizable by the compiler.
But we need some real-world data to see which implementation is actually faster.

### Host-Device Memory Transfers with Thrust

A key capability of `MemArray` is handling the transfer between host and device memory.
Since we are holding the memory in `thrust::*_vector`s, we will use `thrust` for all these transfers and don't have to rely on the C-API like `cudaMemCopyAsync` directly.

The easiest way to transfer memory from a `thrust::host_vector` to a `thrust::device_vector` is to use the copy assignment operator `thrust` offers like so:

```cpp
thrust::host_vector h_vec(11); // Host memory init
thrust::device_vector d_vec = h_vec; // Device memory init with host memory vector.
```

However, this potentially uses memory allocation and we cannot do it asynchronously.

So instead, we're relying on `thrust::async::copy` for the transfer from `#include <thrust/async/copy.h>`.
And since our transfer is always between `host` and `device`, we use 2 execution policies.
For the host memory, we use the simple `thrust::host`, but `thrust::cuda::par.on(stream)` for the device with a specified stream.

The full implementation in `AMSO/src/memarray.cpp` looks like this:

```cpp
template<typename T, int ndim>
void MemArray::move_memory(const DLDeviceType requested_device_type,
                                    const int64_t requested_cuda_stream_id,
                                    const int requested_device_id,
                                    const int64_t requested_lock_id,
                                    const bool async) {
  // Sanity checks
  throw_invalid_lock_access(requested_lock_id);

  if (requested_device_id > 0 and get_device_id() != requested_device_id)
    throw std::runtime_error("Requested to move memory to different device (" + std::to_string(requested_device_id) +", not supported. Current device " + std::to_string(get_device_id()));
  const cudaStream_t stream = convert_python_int_to_stream(requested_cuda_stream_id);

  if (not async) {
    cudaStreamSynchronize(stream);
  }
```

Since we're going to implement the transfer asynchronously, we offer the user the option to automatically perform a stream sync before and after the call in case `async` is not safe.

```cpp
  thrust::device_event event;
  // Memory movement
  switch (requested_device_type) {
  case kDLCUDA:
    if (get_on_device()) // EARLY exit
      return;

    event = thrust::async::copy(thrust::host, thrust::cuda::par.on(stream), _host_vec.begin(), _host_vec.end(), _device_vec.begin());
    _on_device = true;
    break;
```

Notice the early exit, to avoid overhead and incorrectness of moving memory to the device if it's already present there.
This also enables us to just call `move_memory` to device before we do a GPU operation without checking first if it is on the device or not.
The bulk of the async copy is then handled by `thrust` as discussed.

The reverse looks similar:

```cpp
  case kDLCPU:
    if (not get_on_device()) // EARLY EXIT
      return;
    event = thrust::async::copy(thrust::cuda::par.on(stream), thrust::host, _device_vec.begin(), _device_vec.end(), _host_vec.begin());
    _on_device = false;
    break;
  default:
    throw std::runtime_error("Unsupported memory move requested " + std::to_string(requested_device_type) + ".");
  }
```

And since we seemingly allow all DLPack Device types, we catch errors for unsupported device types.

Finally, we do synchronization if desired by the user:

```cpp
  if (not async) {
    event.wait();
    cudaStreamSynchronize(stream);
  }
}
```

Using this without the synchronization can lead to overlapping transfers, and the user needs to know what they're doing. It's another "sharp knives out of the drawer" situation.

## A simple test application

In the next blog post, I want to dive deep into a useful kernel with this memory.
But until then, we need to make sure everything works as expected.
So, I've coded up a quick kernel that demonstrates how I envision the use of the class.
Since this will include kernel and direct CUDA code, we keep it in `memarray_cuda.cu`.
The kernel we're implementing just adds the sum of the indices to the array elements as a quick way to test the functionality of indexing.

```cpp
template<typename T, int ndim>
__global__ void
add_index_kernel(const typename MemArray<T, ndim>::ArrayIndexer indexer) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx ());
    const int new_idx = indexer(indices);
    *indexer.get(new_idx) += static_cast<T>(index_sum);
  }
}
```

You can see how we just pass the lightweight `ArrayIndexer` as a `const` argument.
So all threads can share it with minimal footprint, just like `std::span`.
And we easily check if the active thread is within bounds of the array using `size`.

From there, we convert the linear index to indices, sum them up via `CCCL` and add it to the underlying memory via `indexer.get`.
This is a very fast implementation and it doesn't include any explicit error checking, however the design allows us to easily prevent accidental undefined behavior.
I actually profiled this kernel with Nsight Compute and as long as the array is large enough that we can occupy all SMs of my GPU it reaches optimal throughput.
It measured a 97% throughput of the memory compared with theoretically optimal speed of light, but this kernel is also very simple with just one coalesced memory read, and one write.
All the other computation is done locally and can be done in the registers directly.

Launching this kernel is also pretty straightforward. I'll shorten it and cut out all the boilerplate:

```cpp
template<typename T, int ndim>
void add_index_tuple(MemArray<T, ndim> &arr, const int64_t stream) {
  arr.move_memory(kDLCUDA, stream, arr.get_device_id());
```

This ensures that our memory is on the device memory. If it's already on device memory, this is just a quick `if` and a return.

```cpp
//Finding optimal grid size for occupancy and device capability.

  auto indexer = arr.get_indexer();
  // Launch the kernel with calculated configuration
  add_index_kernel<T, ndim><<<gridSize, blockSize, 0, reinterpret_cast<cudaStream_t>(stream)>>>(indexer);
```

We create a copy of the indexer on the host, and then distribute it to all threads as an argument.
Since it contains shape and a data pointer, this is all the kernel needs.
Note the use of `reinterpret_cast` to convert a `int64_t` to a `cudaStream_t`.
This makes me very uncomfortable. It is not type safe at all, and we do not know if this will be a valid stream.
I peaked at `pytorch`'s code and see that they do it that way.
The reason is that DLPack requires an `int` type for a stream, so we need to convert `int` to a `cudaStream_t` and apparently they use the same number of bits.
Maybe in the future, projects like [CUDA python](https://developer.nvidia.com/cuda-python) will be able to this more smoothly and type safe.

## Testing

The implementation of the little test kernel brings us directly to the matter of testing.
At the moment, I've implemented everything such that it requires CUDA and a GPU present; there is no pure CPU fallback.
I'll probably add that later, but that's not particularly interesting.

However, that limits us to testing with GPU-enabled systems only.
And since free GitHub runners do not have GPUs equipped, I cannot enable automated tests at the moment.
There exist third party options for this like AWS or Azure, but this project does not have a budget for that.
See my other [blog post](http://ludwigschneider.net/public-private-github-repo) if you do have private GPU-enabled runners on GitHub available.

So for now, testing is local.

### C++ Testing with Google-Test

For the C++ functionality, I added the Google test suite via CPM to the repo.
So, I can directly run unit tests on the indexer, and reading and writing memory.
`AMSO/test/test_ArrayIndexer.cpp` contains unit tests for the ArrayIndexer and I use templated `TYPED_TEST` to ensure index calculation works as expected.

```cpp
    EXPECT_EQ(this->indexer->operator()(random_index), expected_index);
```

For `AMSO/test/test_MemArray.cpp` we use a similar strategy, but also can test the `add_index_tuple` kernel call.

```cpp
  auto indexer = this->mem_array->get_indexer();
  for (int64_t i = 0; i < this->mem_array->size(); ++i){
    auto indices = indexer.get_indices(i);
    mem_array->write(indices, 2);
    }
  }
```

First we write `2` in every space on the CPU.

```cpp
  add_index_tuple(*(this->mem_array));
  EXPECT_TRUE(this->mem_array->get_on_device());
```

Call the kernel, which also transfers the memory to the GPU.

```cpp
  // Access without copy to CPU
  for (int64_t i = 0; i < this->mem_array->size(); ++i){
    auto indices = indexer.get_indices(i);
    int sum = std::accumulate(indices.begin(), indices.end(), 0, std::plus<int>());
    EXPECT_EQ((*this->mem_array)(indices), static_cast<T>(2 + sum));
  }
  EXPECT_TRUE(this->mem_array->get_on_device());
```

And finally check the results via the `thrust` vector access method.

### Python Testing

For testing the Python interface and DLPack conversions, we can use `pytest` to run the tests.
In principle, we want to test the same high-level aspects as on C++, but now we can directly compare with `numpy` arrays.
That also automatically checks the DLPack implementation.
`AMSO/tests/pytest/test_memarray.py` contains a lot of boilerplate to test many different combinations and data types, but it boils down to this:

```python
def test_memarray(shape, dtype):
    mem_array = amso.MemArray(shape, dtype)

    with mem_array.get_dlpack() as dlpack:
        local_array = np.from_dlpack(dlpack)
        assert local_array.shape == shape
        assert local_array.dtype == dtype
```

This tests that we can create and read memory and read it via DLPack with the correct shape and type.

```python
    mem_array.move_memory(amso_dlpack.kDLCUDA)

    with mem_array.get_dlpack(amso_dlpack.kDLCPU) as dlpack:
        dlpack_array = np.from_dlpack(dlpack)
        assert dlpack_array.shape == shape
```

This moves it to the device and back to view it again.

```python
    mem_array.move_memory(amso_dlpack.kDLCUDA)
    init_array = get_init_array(shape, dtype)

    mem_array.read_numpy_array(init_array)
    mem_array.move_memory(amso_dlpack.kDLCUDA)
```

This initializes it with a prepared numpy array.

```python
    # Run the CUDA kernel
    mem_array.add_index_tuple()
    mem_array.move_memory(amso_dlpack.kDLCPU)
```

This runs the kernel and moves it back.

```python
    # Access the result and transfer it to CPU
    with mem_array.get_dlpack(amso_dlpack.kDLCPU) as dlpack:
        dlpack_array = np.from_dlpack(dlpack)

        # Do the same, what you expect the CUDA kernel to in numpy
        ref_array = get_numpy_ref_array(shape, dtype)
        assert np.allclose(dlpack_array, ref_array)
```

And finally, this compares the resulting DLPack/Numpy result with the exact same operations, but implemented with numpy only.

## Possible Optimizations in the Future

With this implementation, there's a lot of room for specific optimization.
For example, we could use pinned memory for the host by using the appropriate templated allocator for the `thrust::host_vector`.
This could help with Host Device memory transfers, but pinned memory comes with its own drawback like reduced availability of system memory.
Or we could use pooled allocators if we expect a lot of small memory bits to be frequently allocated and deallocated.
However, all of these optimizations will have to be tested against real-world applications and they may not be necessary for AMSO.

Additionally, the class is well equipped for extension such as dynamic resizing, since we rely on high level abstractions like the `thrust` vectors that enable that easily.

## Conclusion

In this blog post, we've explored the development of a custom CUDA memory handling class for AMSO. We've covered several key aspects:

1. **Design Philosophy**: We created a C++ class that offers fine-grained control over memory allocation and movement, while still allowing interoperability with other frameworks.

2. **C++ Implementation**: We implemented a templated `MemArray` class that handles both host and device memory, with features like safe access methods and a lightweight `ArrayIndexer` for efficient indexing.

3. **Python Binding**: Using pybind11, we created a Python interface that provides a unified class for different C++ template instantiations, making it easy to use from Python.

4. **DLPack Integration**: We implemented DLPack support, enabling zero-copy data sharing between our custom class and other popular Python libraries like NumPy, PyTorch, and JAX.

5. **CUDA Optimization**: We utilized NVIDIA's CUDA Core Compute Libraries (CCCL) for cross-platform compatibility and potential performance improvements.

6. **Testing**: We set up a testing framework using Google Test for C++ and pytest for Python, ensuring the correctness of our implementation across different data types and dimensions.

7. **Performance Considerations**: We discussed potential future optimizations, such as using pinned memory or pooled allocators.

This memory handling class serves as a foundation for AMSO, providing efficient and flexible memory management for particle simulations. By balancing low-level control with high-level ease of use, we've created a tool that can be easily integrated into both C++ and Python workflows.

As we continue to develop AMSO, this memory class will play a crucial role in managing the positions of particles, density fields, and interaction data. In future blog posts, we'll explore how we can build upon this foundation to implement more complex simulation algorithms and optimize performance for large-scale particle systems.

Remember, while this implementation is tailored for AMSO, the concepts and techniques discussed here can be applied to other projects requiring custom CUDA memory management with Python interoperability. As always in software development, the specific needs of your project should guide the implementation details and optimization strategies.

[^1]:
    Let's keep it simple for now and not use compound types like `float4`. These types can be good on the GPU as they nicely align and DLPack supports vector types like this, but we can keep that for the future.
    {: data-content="footnotes"}
