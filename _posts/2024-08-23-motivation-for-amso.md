---
layout: post
author: Ludwig Schneider
tags: [soma, amso, software engineering]
---

# AMSO: A New SOMA for better Maintenance and User Experience

AMSO is a project where I dive into refactoring SOMA.
Focus will be to bring the performance and features of SOMA to modern languages/parallelization and a user friendly python interface.

It will be accompanied by a series of blog posts explaining the process and the engineering decisions involved:

{% include amso_list.html %}

## SOMA: Original Design to Solve a Physics Challenge

SOMA is a scientific software designed to simulate large polymeric systems that exhibit phase separation. Polymeric materials are important for numerous applications, including membranes for filtration, directed self-assembly for integrated circuit manufacturing, and more. For a full introduction, please check out the complete SOMA publication[^1] or the ArXiv version[^2].

From a software engineering perspective, SOMA solves an approximate Monte Carlo problem. A detailed description of the SCMF algorithm that SOMA uses and its derivation for phase-separating polymer systems can be found in this [scientific article](https://doi.org/10.1063/1.2364506).

### SCMF Algorithm - Parallel Monte Carlo

Particles are connected to molecules, which we can view as undirected graphs. These molecules fill the simulation box, representing a nano-material with a coarse-grained model. There are two types of interactions present:

1. A strong bonded interaction between particles directly adjacent in the molecular graph.
2. A non-bonded interaction between particles that are spatially close.

The second interaction is more interesting. The SCMF algorithm describes how to approximate this interaction with a quasi-instantaneous field approximation. As a result, the particles don't interact directly with each other but with an intermediate field. This approach decouples most particles, meaning the behavior of other particles doesn’t change unless they are directly bonded.

For the Monte Carlo simulations, we select a set of particles where no two particles are directly bonded. This makes each Monte Carlo move independent. We can maintain global balance (though difficult to prove) as long as each move fulfills detailed balance (which is typically provable). After the particles move, the interaction field is updated, and the process repeats.

This high level of independence between particles allows efficient parallelization of the simulation.

## Software Engineering Decisions

Let’s review some of the major software engineering decisions originally made for SOMA.

### C as the Main Language

SOMA is primarily written in C, which enables low-level programming for optimal performance and integrates well with the software stacks of scientific high-performance computing (HPC) centers. C is also familiar to many scientists, particularly those in Prof. Müller’s lab at the University of Göttingen, where the science behind SOMA was developed.

At the time, support for C++ in [CUDA](https://en.wikipedia.org/wiki/CUDA) and/or OpenACC wasn’t mature enough for us to adopt it.

While C has limitations regarding security and maintainability in large codebases, these concerns were less important for SOMA, which is used in research settings, often on HPC systems with no sensitive data processing.

Now, with improved C++ and GPU computing support, we are developing AMSO with a tighter Python interface, which aligns better with an object-oriented design in C++ compared to C. This allows users to program AMSO through Python without needing to modify backend C code.

### Why C++ and Not Rust

Rust is an excellent language, designed to prevent many pitfalls and security issues found in C or C++. It is also on par in terms of performance. Rust was a strong contender to replace C++ in AMSO, but GPU parallelization support is not yet sufficient. While Rust provides [some](https://rust-gpu.github.io/Rust-CUDA/features.html) GPU computing support, crucial atomic operations (important in SOMA's design) are not yet available.

Moreover, Rust’s main advantage—security—was not a priority for research software like AMSO, where security is less critical.

### Initialization and Program Flow in SOMA

The original SOMA is a monolithic executable, with researchers controlling the simulation flow. This is done through:

1. A static input file containing the initial conditions and simulation parameters.
2. Command-line arguments controlling the execution algorithm, affecting performance but not the physical system.

This approach is limiting. Adding new features requires changes to the codebase, often resulting in diverging code branches that are difficult to maintain and merge.

#### Python Frontend for Control Flow

In AMSO, we plan to hand control flow over to the scientific user. Users will describe a simulation state, either through an input file or using common Python libraries like NumPy. This state will be serializable, allowing it to be shared and stored. From this state, users can call different algorithms implemented in C++/CUDA, updating the state as needed.

This approach offers greater flexibility and modularity, making it easier to add new algorithms and features, and improving maintainability.

### Parallel Programming

A key concept in SOMA is parallelism. In the original SOMA, parallelism was achieved at multiple levels:

1. On a particle basis, implemented via [OpenACC](https://www.openacc.org/)/[OpenMP](https://www.openmp.org/).
2. On a molecular basis, also controlled via OpenACC/OpenMP.
3. Across groups of molecules, distributed via MPI in a distributed memory model.
4. In domain decomposition, where spatially located groups of molecules are split into different compartments, also using MPI [^3].

#### OpenACC/OpenMP vs CUDA/HIP

The first two levels rely on the pragma-based approaches of OpenACC and OpenMP, chosen for their readability, which was important for collaboration with GPU-unaware researchers. However, OpenACC has not proven as stable as hoped, and its goal of device independence has not fully materialized.

For AMSO, we are switching to CUDA, which is now well-established in parallel programming. We are also considering a [HIP](https://github.com/ROCm/HIP) implementation for CPU and AMD fallback, which will be discussed in a future post.

#### Message Passing Interface (MPI)

[MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) is an established protocol for sending messages between compute nodes running in parallel. While MPI works well in scientific contexts, it struggles with GPU memory handling. We had limited success with CUDA-aware MPI libraries in SOMA.

Nvidia's NCCL library, designed for GPU memory, has proven more effective. AMSO will aim for a holistic integration of MPI and [NCCL](https://developer.nvidia.com/nccl) using C++.

### Hierarchical Data Format 5 (HDF5)

[HDF5](https://www.hdfgroup.org/) is a binary format for storing matrix-like data in different types. It has worked well in SOMA, and we plan to continue using it in AMSO. HDF5 is well-supported in Python with [h5py](https://docs.h5py.org/en/stable/quick.html), and it integrates with ParaView for visualization. HDF5 also supports MPI-aware I/O, allowing parallel read/write operations, crucial for the large systems handled by SOMA.

We are considering moving the HDF5 implementation from the backend to Python to reduce boilerplate code and potential bugs.

## Summary

AMSO provides an opportunity to improve the original SOMA’s design, focusing on usability, maintainability, and potential performance gains. This will be a step-by-step process, documented through blog posts that explore different aspects. These posts will include both instructional content and explanations of key decisions.

In the next post, we will introduce the build system chosen for AMSO.

---

[^1]: [Multi-architecture Monte Carlo (MC) simulation of soft coarse-grained polymeric materials: SOft coarse-grained Monte Carlo Acceleration (SOMA), Computer Physics Communications, Volume 235, February 2019, Pages 463-476](https://doi.org/10.1016/j.cpc.2018.08.011)

[^2]: [Multi-architecture Monte Carlo (MC) simulation of soft coarse-grained polymeric materials: SOft coarse-grained Monte Carlo Acceleration (SOMA), arXiv 1711.03828](https://arxiv.org/abs/1711.03828)

[^3]: [Unendorsed Explanation of Domain Decompositions](https://bbanerjee.github.io/ParSim/mpi/c++/parallel-domain-decomposition-part-1/)
