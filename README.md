# fluids_v4
A minimal variant of Rama Hoetzlein's Fluids SPH, cut down from gFluidSurface in gvdb-voxels, 
https://github.com/ramakarl/gvdb-voxels.

Dependence on gvdb-voxels library has been removed, and CMakeLists.txt has been rewritten.
New output has been written to provide ascii .ply files for viewing in MeshLab.

This code compiles and runs with cmake 3.10, Cuda 9.1 on Ubuntu 18.04 with GTX 980m, 
and on Suse Linux cluster with Cuda 9.1 and Tesla P100.
