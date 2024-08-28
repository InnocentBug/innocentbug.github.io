---
layout: post
author: Ludwig Schneider
tags: [soma, soma2, software engineering]
---

# SOMA 2

SOMA 2 is a project, where I dive into the refactoring of SOMA.
It will be accompanied by a series of blog posts explaining the process and the engineering decisions.

## SOMA

SOMA is a scientific software product that is designed to simulate large polymeric systems that exhibit phase separation.
Polymeric materials are important for a number of applications, including membranes for filtrations, directed self-assembly for integrated circuit manufacting and many more.
If you are interested in a full introduction, please check out the full publication for SOMA[^1] or the ArXiv version[^2].

From a software engineering perspective SOMA solves an approximated Monte-Carlo problem.
A detailed description of the SCMF algorithm that SOMA uses and its derivation for phase-separating polymer systems can be found in this [scientific article](https://doi.org/10.1063/1.2364506).

Particles are connected to molecules, which we can view as undirected graphs.
Many of these molecules fill up the simulation box that describes a nano-material with the coarse-grained model.
There are 2 types of interactions present:

1. A strong bonded interaction between particles directly adjacent in the molecular graph.
2. Non-bonded interaction between particles that spatially close.

The second one is the more intersting; the SCMF algorithm describes how we can approximate this interaction with a quasi-instantaneous field approximation.
As a result the particles do not interact directly with one another but they interact with an intermediate field instead.
The beauty of this approach is that almost all particles decouple i.e. the of other particles does not change except if the two particles are directly bonded in the graph.
For the Monte-Carlo simulations, the approach is determine a set of particles with the condition that no particles in this set are directly bonded with one another.
This makes each of these individual Monte-Carlo moves independent and we can conserve global balance (usually impossible to prove) as long as every individual move fullfills detailed balance (usually proveable).
After the particles move their positions, the interaction field is updated and the process repeats.

This high degree of independence all

---

[^1]: [Multi-architecture Monte-Carlo (MC) simulation of soft coarse-grained polymeric materials: SOft coarse grained Monte-Carlo Acceleration (SOMA), Computer Physics Communications, Volume 235, February 2019, Pages 463-476](https://doi.org/10.1016/j.cpc.2018.08.011)

[^2]: [Multi-architecture Monte-Carlo (MC) simulation of soft coarse-grained polymeric materials: SOft coarse grained Monte-Carlo Acceleration (SOMA), arXiv 1711.03828](https://arxiv.org/abs/1711.03828)

{: data-content="footnotes"}
