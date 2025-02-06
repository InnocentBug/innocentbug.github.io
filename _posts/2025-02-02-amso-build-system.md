---
layout: blog
author: Ludwig Schneider
tags: [soma, amso, software engineering]
---

# Build Systems

Build systems are tools that we used to reliably build our software as painless as possible.
The first build system, I came in touch with is the [`Make`](<https://en.wikipedia.org/wiki/Make_(software)>).
`Make` is great and very versatile since you can define arbritrary targets to build. And `Make` requirements of if something needs to be build based on the file timestamps.

However, AMSO is going to be a little bit bigger then Make can comfortably handle. Plus AMSO is written in 2 programming languages (C++ & Python) and we need to combine and handle both.
Let's go through it one by one:

## CMake

[CMake](https://cmake.org/) is build system that shines for C++.
It is great to define software targets in a meta-language. You (and CMake) can usually find everything you need to know about a project in a `CMakeLists.txt` file.

##

## Super Linting with trunk.io
