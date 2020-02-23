# Morphogenesis

# Acknowledgement

"2014, Hoetzlein, Rama Karl. Fast Fixed-Radius Nearest Neighbors: Interactive Million-Particle Fluids. GPU Technology Conference, 2014. San Jose, CA. 2010-2014. Online at http://fluids3.com"
 
Morphogenesis started as a minimal variant of Rama Hoetzlein's Fluids SPH, cut down from gFluidSurface in gvdb-voxels, 
https://github.com/ramakarl/gvdb-voxels, 
which was in turn developed from Fluids-v3.

## Master branch
This is the cut down version of gFluidSurface.
Dependence on gvdb-voxels library has been removed, and CMakeLists.txt has been rewritten.
New output has been written to provide ascii .ply files for viewing in MeshLab.

This code compiles and runs with cmake 3.10, Cuda 9.1 on Ubuntu 18.04 with GTX 980m, 
and on Suse Linux cluster with Cuda 9.1 and Tesla P100.

## Morphogenesis branch
A morphogenesis simulator _(in progress)_ , with soft-matter elasticity, diffusion of heat/chemicals/morphogens, epi-genetics and particle automata behaviour.
