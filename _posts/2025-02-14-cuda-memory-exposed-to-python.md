---
layout: blog
author: Ludwig Schneider
tags: [software engineering, amso, CUDA, thrust, dlpack]
---

# A (CUDA) Memory Handling Class for AMSO

For AMSO we will have to handle memory for a variety of things. Most importantly probably the positions of all the particles, density & interactions fields etc..
In this blog post, I discuss how I went about designing this memory holding class. This cover some general talking points, but also some technical considerations and details.
For a full working code study refer to this [pull-request](https://github.com/InnocentBug/AMSO/pull/1) on GitHub.

And since AMSO is written in C++ for peak performance with a python binding for convenience, we need to be able to read and write the memory from C++ and python.
We could use external libraries like [CUPY](https://docs.cupy.dev/en/stable/reference/generated/cupy.array.html), [NUMBA](https://numba.readthedocs.io/en/stable/reference/types.html#arrays), or even Deep Learning Frameworks like [JAX](https://docs.jax.dev/en/latest/_autosummary/jax.Array.html) or [PyTorch](https://pytorch.org/docs/stable/tensors.html) and use their memory handling implementations.
However, that limits us in some aspects like how the memory is allocated and moved, a level of control, that I do not want to give up for AMSO.
So instead, I am building a C++-class that handles all of this for us, exactly the way I want it to, but we do not have to give up the convenience of using such frameworks either, more on that and DLPack later.

The positions have a floating point type, and we can represent them as 2D arrays, with one dimension counting the number of particles and the other X, Y, Z component[^1].
Density fields are usually integral types in 3D, but for interactions we multiply constants on which transforms them into floating point types.

## A Templated Class C++ class

From this requirement we know, that we will have to make a template class that supports 2 template parameters, the data type of the array and a dimension parameter.
The data type template paramater `T` allows type safety, and the dimension parameter `ndim` allows maximum performance.
Since template parameters are resolved at compile time, a lot of optimiztions with a known dimension can be made.

However, C++ is a statically compiled language and python is an interpreted language.
We could dynamically `jit` compile our C++ templates as they are requested from python at compile time, but for AMSO we usually know dimensions and types ahead of time.
So, we take the simpler approach here, and explictly instantiate the types and dimensions we need for AMSO at compile time.
And in case we request something unknown at run time from python, we give a meaningful error message prompting a recompile.

## The C++-Interface

Let's take a look into the in actual interface.
I am going to omit parts of the source code, like includes and namespace, here for clarity. For a full working code piece look at the GitHub [pull-request](https://github.com/InnocentBug/AMSO/pull/1).
The templated class definition can be found in `AMSO/include/memarray.h`.

```cpp
template <typename T, int ndim> class MemArray {
private:
  thrust::host_vector<T> _host_vec;
  thrust::device_vector<T> _device_vec;

  std::array<int, ndim> _shape;
  bool _on_device;
  int _device_id;
  int64_t _lock_id; // Context manager lock, ID. Negative means unlocked.
                    // Positive, locked to that context manager
```

Doing quite a bit of python development, I like to declare my member variables of classes starting with an underscore.
This doesn't have the same `private` meaning as in python, C++ has the much more robust `private`, `public` & `protected` qualifiers for that.
But it helps me as a reminder to be careful, whenever I directly access class member variables, and it is easier to ensure they don't get overshadowed by local variables.

This declares all the most important member variables of our class.
`_host_vec` and `_device_vec` are the containers that actually store the data.
We can use the `thrust` vectors here, instead of having to deal with pointers and cudaMalloc and cudaFree direct.
These containers not only allow us to write the code much more robust, it also opens the door for later optimization and usage of the thrust library.

Since we know the number of dimension as a template argument, we can use `std::array` to store the shape of our array.
`std::array` is a thin wrapper, so performance and storage is minimal, we don't need to dynamically allocate heap memory like we would have to with an `std::vector` but have all the conveniences that we love from standard containers.

`_on_device` keeps track if we consider the device or host copy of the data as current, `_device_id` rembers on with GPU ID we allocated the memory, `_lock_id` is a bit more interesting.
`_lock_id` will handle locking the memory for the situation that an outside observer (i.e. python) reserved access to the memory and we need to be careful.
Throughout the code, a positive `_lock_id` means that someone locked access, but a negative `ID` refers to the state that this class is the sole owner.

```cpp
public:
  static constexpr int NDIM = ndim;

  MemArray(const std::array<int, ndim> &shape,
           int device_id);                 // Regular constructor for new memory
```

We outwardly declare the number of dimensions as constexpr. Because it is a constexpr it very cheap but it offers the convenience to access it when the template parameter is not reachable.
And we follow with our main constructor. It requires only the shape of and device ID as input.
Except for error checking, it only initializes both thrust vectors. It automatically initializes the memory with `T()` everywhere.
For more useful initialization, we will later rely on numpy and python arrays.

```cpp
  ~MemArray(){};                           // Destructor
  MemArray(const MemArray<T, ndim> &other); // Copy constructor (expensive)
  MemArray<T, ndim> & operator=(const MemArray<T, ndim> &other); // Copy Assignment operator
```

Since this is a resource handling class, we also implement Destructor, copy Constructor and copy assignment operator.
Next is a swap, that utilizes the thrust vector swap implementation for a zero-copy swapping.

```cpp
  void swap(MemArray<T, ndim> &other) noexcept;
  template <typename U, int N>
  friend void swap(MemArray<U, N> &lhs, MemArray<U, N> &rhs) noexcept;
```

With swap we can also implement the move constructor and move assignement operator without any resource allocation.

```cpp
  MemArray(MemArray<T, ndim> &&other) noexcept; // Move constructor (cheap)
  MemArray<T, ndim> &operator=(MemArray<T, ndim> &&other) noexcept; // Move Assignment operator

```

We also have public getter function, to allow reading the state of our variables.
Note, that `get_lock_id()` makes it possible to maliciously get access to locked memory, even when you weren't supposed to.
However, this is C++ and only indented for users the in the know.
This mirrors the analogy of C++ from Herb Sutter: "In C++ we have some very sharp knives, they are very useful if you know what you do, but you can also cut yourself easily."
If you circumvent the locking mechanism, you are taking the sharp knives out of the drawer. Be careful not to cut yourself.

```cpp
  int get_device_id() const { return _device_id; }
  const std::array<int, ndim> &get_shape() const { return _shape; }
  int64_t get_size() const { return _size; }
  bool get_on_device() const { return _on_device; }
  int64_t get_lock_id() const { return _lock_id; }
```

However, if you want to make sure to protect your access to the memory, here is function that will throw an exception if you accidentally access memory you are not supposed to.

```cpp
  void throw_invalid_lock_access(const int64_t requested_lock_id) const;
```

Now we get to the juicy bits of this class: Moving memory from and to device and host.
We use the `DLPack` defintions of host or device as first parameter to indicate the destination.
But this implementation also lets you do the transfer operations in specifc CUDA stream and asynchronously if desired.

```cpp
  void move_memory(const DLDeviceType requested_device_type,
                   const int64_t requested_cuda_stream_id,
                   const int requested_device_id,
                   const int64_t requested_lock_id = -1,
                   const bool async = true);
```

Next are the two method to lock and unlock the memory access.
We will later see, that those are primarily intended to be used in python context managers.

```cpp
  // Context manager methods
  void enter(const int64_t lock_id);
  void exit() { _lock_id = -1; }

  void read_numpy_array(pybind11::array_t<T, pybind11::array::c_style |
                                                 pybind11::array::forcecast>
                            np_array);
```

The `read_numpy_array` I will explain later, it is a bit unfortunate that we have to implement it explictly.

But now, let's look at some more interesting aspects. How are we going to access the arrays. There are basically two ways, one is the safe way that checks memory access, lock status, device location and out of bounds for you, but that comes with a bit of overhead.
I overload the `()` operator for read access and we have a `write` function for write access.
Indices are light-weight `std::array` again. We could implement it as variable arguments and check statically aginst the template parameters, but this way it is very transparent and you can just do `mem_array({3, 5, 7})` if you want to access the location `x=3, y=5, z=7`. That is straight forward.
Notice, how the functions also take `cuda::std::array` that has to do with the NVIDIA CCCL library, and I will explain it in more detail, when we discuss the indexer.

```cpp
  // encaspulated access access, not zero-overhead
  const T operator()(const std::array<int, ndim> &indeces, const int64_t requested_lock_id = -1) const;
  const T operator()(const cuda::std::array<int, ndim> &indeces, const int64_t requested_lock_id = -1) const;

  void write(const std::array<int, ndim> &indeces, const T &value, const int64_t requested_lock_id = -1);
  void write(const cuda::std::array<int, ndim> &indeces, const T &value, const int64_t requested_lock_id = -1);
```

But I also implement a no overhead, no checking, i.e. sharp knives out of the drawer access.
You can directly access the a pointer to the underlying memory.
It is your responsibility to keep the memory valid, while you have the pointer and also your responsibility to only to valid things with it.
It is intended to be used to obtain pointer to write cuda kernels etc.

```cpp
  // Zero over-head pointer access
  // Taking the sharp knives out of the drawer: don't cut yourself
  T *ptr(const int64_t requested_lock_id = -1);
  const T *ptr(const int64_t requested_lock_id = -1) const;
```

And lastly, we have two functions that are intented to work with DLPack.
These are interesting and we wil examine them in the next section in detail.

```cpp
  pybind11::capsule get_dlpack_tensor(const bool versioned,
                                      const int64_t requested_lock_id);

  std::pair<DLDeviceType, int32_t> get_dlpack_device() const;
};
```

## Possible Optimizations in the Future

[^1]:
    Let's keep it simple for now and not use compound types like `float4`. These types can be good on the GPU as they nicely align and DLPack supports vector types like this, but we can keep that for the future.
    {: data-content="footnotes"}
