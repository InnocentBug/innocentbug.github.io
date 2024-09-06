---
layout: post
author: Ludwig Schneider
tags: [amso, soma]
---

# Rebranding SOMA2L AMSO

SOMA is a success, so with this refactoring of SOMA we want to continue to build on that.
One obvious solutions would be to name this project SOMA version 2, but I am not particularly happy with this: It just doesn't roll of the tongue as nicely. And writing it as SOMA2 looks a little awkward in my opionion.

## Origin of the SOMA name

Originally, we settled on using SOMA as a name for multiple reasons.
For the TU Dresden/Nvidia Hackathon we chose "soft matters" as a team name.
This was a play on the fact that for SCMF and SOMA to work and be parallelizable it is necessary for the description of the macro-molecules in a coarse-grained fashion, such that the non-bonded potentials become "soft". Soft in this context means, that the center of mass of different interaction centers can fully overlap without a diverging potential energy.
This is counter intuitive, because we tend to imagine atoms as hard spheres that bounce off one another. One of the most common potentials the Lennard-Jones potential is a hard potential, so we wanted to highlight on how we are different with this project.

Additonally, the "soft matters" is also a play on the toxic masculinity that only values "hard" men.
So, we loved the idea of show casing and naming as something contradicting toxic masculinity.

You can see this slogan reflected in the original SOMA logo designed by Max. M. Schneider.

![SOMA Logo](images/research/soma_logo.svg)

For the scientific publication, we then made SOMA an acronym: SOft coarse grained Monte-Carlo Acceleration (SOMA)
The acronym was always a little awkward as it didn't naturally contained a word starting with "O".
So we also made a second acronym: Soma Offers Monte-Carlo Acceleration.
This one is better and more fun as it a recursive acronym, as it refers to itself.
An inside programming joke we borrowed from GNU: GNU's not UNIX!
Unfortunately, this acronym is not very specific, so we decided to not use it the official publication.

## Problems with the name SOMA

We didn't do a serach to consider if SOMA is a good name for a project at the time.
As a result, there are some issues with using SOMA as a project name.
For once it is the latin transcription of the greek word σῶμα which mean body.
As such it is used in Biology to describe the cell body of neurons. It is also used as short for "South of Market" a neighborhood in San Francisco, CA.
And it is also the name of a popular video game.
And there are many more uses of the word.
So it is not particularly good brand name, and this project will have a hard time competing in search engine results.

Most importantly for this particular project is that "soma" is already taken on readthedocs.org.
Hence, it would not be possible to host the documentation with its name there.

So, I decided to rebrand SOMA version 2.
But this time, I want to put a bit more thought into picking the name.

## AMSO: Accelerated Monte-carlo: Soma Optimized | Assembling Macromolecules: Soft Objects

![AMSO Logo](images/research/amso_logo.svg)
