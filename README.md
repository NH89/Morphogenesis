# Morphogenesis

## Acknowledgement

"2014, Hoetzlein, Rama Karl. Fast Fixed-Radius Nearest Neighbors: Interactive Million-Particle Fluids. GPU Technology Conference, 2014. San Jose, CA. 2010-2014. Online at http://fluids3.com"

Morphogenesis started as a minimal variant of Rama Hoetzlein's Fluids SPH, cut down from gFluidSurface in gvdb-voxels, 
https://github.com/ramakarl/gvdb-voxels, 
which was in turn developed from Fluids-v3.

## Master branch
This is the cut down version of gFluidSurface.
Dependence on gvdb-voxels library has been removed, and CMakeLists.txt has been rewritten.
New output has been written to provide ascii .ply files for viewing in MeshLab.

This code compiles and runs with cmake 3.10, vtk-9.0, and Cuda 11.2 on Ubuntu 20.04 with GTX 980m.
and on Suse Linux cluster with Cuda 9.1 and Tesla P100.

## Morphogenesis branch
A morphogenesis simulator _(in progress)_ , with soft-matter elasticity, diffusion of heat/chemicals/morphogens, epi-genetics and particle automata behaviour.

The notes below are rough working notes, and will change with development.

### Updated (August 2026) - Cuda-13.1, Ubuntu-26.04 LTS, VTK-9.5, C++17, CMake-4.2

The code has beed updated to work with Ubuntu 26.04LTS, and therefore Cuda-13.1, VTK-9.5, C++17, CMake-4.2, all from the Ubuntu repository, no custom installs required. (Except Cuda-13.1 patch, see below.)

Morphogenesis has no OS-brand dependency, and builds with CMake plus Ninja or Make,  so it 'should' build on any OS where those libraries work with C++17.

### Environment variables + patch for Cuda 13.1

Cuda requres all the following added to your environment variables on Ubuntu :
```
    export CPATH=${CPATH}:/usr/local/cuda/include

    export C_PLUS_INCLUDE_PATH=${C_PLUS_INCLUDE_PATH}:/usr/local/cuda/include

    export LIBRARY_PATH=${LIBRARY_PATH}:/usr/local/cuda/lib64  

    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
```

This is best set in the ~/.profile file, so that they are found by you IDE as well as your terminal. 

(Nvidia's [cuda-installation-guide-linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#environment-setup) does not mention  ```CPATH```  and ```LIBRARY_PATH```, but omitting them can lead to "file not found" errors.)

#### Patch
Cuda-13.1 (in the Ubuntu 26.04LTS repository) has a bug which has been patched in Cuda 13.2 (custom install not yet supported on Ubuntu 26.04LTS).

Details of the bug, and the patch are provided in [Cuda 13.1 patch on Ubuntu 26.04 LTS.md](Patches_sometimes_required/Cuda-13.1-patch-on-Ubuntu-26.04LTS/Cuda 13.1 patch on Ubuntu 26.04 LTS.md)


### Build instructions

This project uses CMake.
Create a build subdirectory.
In the build subdirectory
```
    cmake ../
    make
    make install
```
#### Or with cmake-gui, recommended
- In your build directory
- type "``` cmake-gui ```"


- select the "build" and "src" directories
- click "configure"
- provide paths to libraries as needed
- click "configure" again, until all required libraries are found
- then click "generate"


- Quit cmake-gui
- type: "``` cmake --build . ``` "
- type: "``` cmake --install ``` "


## Running Morphogenis


### (1) make_demo
usage:

In the folder ``` Morphogenesis/data/```

**```
    make_demo  num_particles  spacing  x_dim  y_dim  z_dim  demoType  simSpace
```**

where:

1. demoType(0:free falling, 1:remodelling & actuation, 2:diffusion & epigenetics.)
    - demoType sets individual particle properties in "demo/particles_pos_vel_color100001.csv", especially epigenetic states, from 3D positions.

2. simSpace{0:regression test, 1:Tower(256,128,256), 2:Wave pool(400,200,400), 3:Small dam break(80,60,80), 4:Dual-Wave pool(200,100,30), 5: Microgravity(160,100.160) }
    - simSpace sets parameters in "SimParams." , especially gravity, and wavepool actuation.

e.g.:

    ../build/make_demo 125 1  6 6 6  0 5       // free falling

    ../build/make_demo 120 1  2 2 30  1 5
    ../build/make_demo 600 1  4 4 30  1 5      // remodelling & actuation, with fixed, bone, tendon, muscle, elastic, mesenchyme, external actuation
    ../build/make_demo 2000 1  6 6 30  1 5     //  "" , with reserve particles for growth.

    ../build/make_demo 400 1  10 10 3  1 5     // diffusion & epigenetics, with reserve particles for growth.

    ../build/make_demo 10000 1  100 10 10  1 1
    ../build/make_demo 100000 1  100 10 100  1 1
    ../build/make_demo 100000 1  100 100 10  1 1
    ../build/make_demo 1000000 1  100 100 100  1 1


### (2) check_demo

CPU-only test program to verify the ability to read and re-output models.

usage:

In the folder ``` Morphogenesis/data/```

**```
    check_demo  simulation_data_folder  output_folder 
```**

e.g.
```
    ../build/check_demo  demo  check 
```


### (3) load_sim
New launch program to load data from files, and run simulation on GPU.


usage:

In the folder ``` Morphogenesis/data/```

**```
    load_sim  simulation_data_folder  output_folder  num_files  steps_per_file  freeze_steps save_ply(y/n)  save_csv(y/n)  save_vtp(y/n)  debug(0-5) gene_activity(y/n)  remodelling(y/n)
```**

    debug: 0=full speed, 1=current special output,  2=host cout, 3=device printf, 4=SaveUintArray(), 5=save .csv after each kernel.

e.g.
```
    cd data
    ../build/load_sim   demo/  out/  10    3 1  y y y 1 n n
    ../build/load_sim   demo/  out/  10    1 6  n y y 1 n n

    ../build/load_sim   demo/  out/  1000 10 0  n y y 1 y y                   // NB Now '100' frames per timestep x02-x20 snapshot after each kernel. x91 end of timestep. x00 begining of simulation.

    ../build/load_sim   demo/  out/  1000  1 10 n y y 1 n y                   // NB now 'freeze_steps' delay the start of particle movement, while heal() forms initial bonds.

    ../build/load_sim   demo/  out/  1000  1 10 n y y 1 n n
    ../build/load_sim   demo/  out/  1000  1 10 n y y 1 y y

    ../build/load_sim   demo/  out/  200   3  1 n n y 1 y y                    // Good launch option for profiling

    ../build/load_sim   demo_10000_1_100_10_10/      out/  100 30 1 n n y
    ../build/load_sim   demo_100000_1_100_100_10/    out/  100 30 1 n n y
    ../build/load_sim   demo_1000000_1_100_100_100/  out/  100 30 1 n n y
```


## Viewing & analysing simulations

### Viewing with Meshlab - to check geometry

The .ply files output can be viewed in MeshLab.
It is recommended to select the following MeshLab options.

```
    Render - Show vertex dots
    Render - Render Mode - Wireframe
```
( NB need MeshLab-2025.07 on the Ubuntu-26.04 LTS repository. Some older version on Ubuntu-20+ do not work.)

### Viewing with Paraview - to visualize all data, and generate videos.

The .vtp files output can be viewed in Paraview. This allows visualization of all the parameters, and is especially relevant for diffusion of morphogens, epigenetic state, and material properties.

ParaView is a widely used scientific data visualization tool.
ParaView can be downloaded from [https://www.paraview.org/](https://www.paraview.org/)

**NB GPU conflict:**

It is advised to exit Paraview before launching Morphogenesis.
Sometimes other runtime errors arrise if Paraview is still running when Morphogenesis is launched.

Both Paraview and Morphogenesis will both try to use your GPU.
This may result in an **"invalid device context"** or **"There is no device supporting CUDA"** error.

This seems to happen specifically _where the computer suspends while ParaView is open._
If ParaView fails to release the GPU after being shut down, then it may be necessary to reboot.

This **does not arise** where the two programs are run on _separate machines_, as when Morphogenesis is run on a cluster.


### Loading the data into ParaView:
load the .vtp file:

- select the file in the pipeline browser

Visualize the data

- From the top menu bar, select "Filters->Common->Threshold"
- In Properties(Thresold2), in scalars, select FPARTICLE_ID.
- Set Maximum to the number of active particles in the simulation.
- click "Apply" (green button in Properties)

Zoom to data

- In the third row of the tool bar, 
- click "zoom to data" icon (four arrows pointing inwards).

   NB this is necessary, when unused particles are stored in one corner of the simulation, with FPARTICLE_ID = UINT_MAX.

Set particle / spring color

- In "Coloring" select the model parameter of interest
- Adjust the coloring scale

Set the background color:

- Select Edit->Settings
- In the pop-up window, select Color Pallette->Background
- Choose a colour, click Apply, Okay.

For Volume rendering:

- In the top menu bar, select "Filters->Point Interpolation->Point Volume Interpolator"
    - Select the new "PointVolumeInterpolator" in the pipeline browser

- In the Properties pane, select a kernel type, e.g. Gaussian Kernel or Shepard Kernel
    - In Coloring, select the model parameter of interest
    - Adjust the coloring scale

- In "Volume Refinement", "Representation", select "Volume"
    (Alternatively select these in the top menu, second row tool bar.)

- In Volume Rendering (near the bottom of the Properties pane), 
- in "Volume Rendering Mode", select "Smart" or "GPU"

- Click Apply (at the toi of the Properties pane)


Also in "Volume Rendering"
    In "Blend Mode" select between "Compostite/IsoSurface/Slice"
```
    Colour key for F_TISSUE_TYPE:        Preset colours
                                        "Cool-warm"      "Jet"          "Black-body"
    Mesenchyme/other    tissueType =0    dark blue       dark blue      black
    Tendon              tissueType =6    pink            yellow         red-orange
    Muscle              tissueType =7    orange          orange         orange
    Cartilage           tissueType =8                    red-orange     yellow
    Bone                tissueType =9                    red            pale yellow
    Elast lig           tissueType =10   dark red        dark red       white
    
    
    Note Tissue order in "Demo" from Pos.z=0 along z-axis.
    Fixed particles, 
    mesenchyme, 
    bone, 
    tendon, 
    muscle, 
    elastic 
    tissue, 
    mesenchyme, 
    actuated particles. 
```






