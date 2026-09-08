# Morphogenesis

# Acknowledgement

"2014, Hoetzlein, Rama Karl. Fast Fixed-Radius Nearest Neighbors: Interactive Million-Particle Fluids. GPU Technology Conference, 2014. San Jose, CA. 2010-2014. Online at http://fluids3.com"
 
Morphogenesis started as a minimal variant of Rama Hoetzlein's Fluids SPH, cut down from gFluidSurface in gvdb-voxels, 
https://github.com/ramakarl/gvdb-voxels, 
which was in turn developed from Fluids-v3.

## "Master" branch
This is the cut down version of gFluidSurface.
Dependence on gvdb-voxels library has been removed, and CMakeLists.txt has been rewritten.
New output has been written to provide ascii .ply files for viewing in MeshLab.

This code compiles and runs with cmake 3.10, vtk-9.0, and Cuda 11.2 on Ubuntu 20.04 with GTX 980m.
and on Suse Linux cluster with Cuda 9.1 and Tesla P100.

## "Non-Reciprocal" branch
This is the **morphogenesis simulator** (in progress), with 

* viscous flow
* anisotropic fibrous elasticity
* diffusion of heat/chemicals/morphogens
* epi-genetics
* particle automata behaviour

It can used for

* general purpose soft matter simulator
* real-time simulation (depending on model size and hardware)
* biolgical morphogenesis, especially complex mechanical anatomy 
* living tissue simulation (healing, pathology, disease processes)

It is in principle a **differentiable simulator** because gradients are available for all parameters. NB functions to access this have not _yet_ been written as of April 2021.

----
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

----
# Build instructions  
Morphogenesis depends on Cuda>=13, C++17, CMake>=3.20, and vtk-9.x.
Scripts are provided to build and install Morphogenesis.
These "should" work on individual workstations or GPU clusters.

The scripts call ccmake (cmake ncrurses text-gui, usually available in non-gui environments e.g. clusters).
Press "c" for configure, twice, then "g" for generate the Makefiles.
The script creates the folder "Morphogenesis/build/".
The software is installed to "~/apps/".

If Environment Modules are installed (e.g. on a GPU cluster), then the script installs the modules in "Morphogenesis/src/module" to "~/modules", otherwise it sets the needed environment variables directly.

## vtk-9

    Morphogenesis depends on vtk>=9.5
    See #includes in fluid_system.h

NB The user must ensure that vtk-9.x is installed _before_ building Morphogenesis.

## Morphogenesis

    cd Morphogenesis/src
    bash ./install_scripts/install_morphogenesis.sh

    . install_scripts/set_env.sh

Note the "dot space script" syntax is required for set_env.sh .
This runs the script in the current shell, so that it can set the required enviroment variables.

### Custom installation
See the details of the install scripts and module files, and adapt as needed.


## SLURM for launching batches on a cluster.

To launch a job using a Slurm script :

    sbatch <path_to_script/scriptname.slurm>

e.g.
    cd Morphogenesis/data
    sbatch ../src/slurm/Morphogenesis_test1.slurm

NB SLURM writes stdout and stderr to file :

    <working_dir>/slurm-<job_no>.out .

To check the queue :

    squeue -u <userID>

----
# Executables
Within Non-Reciprocal branch executables (so far) include:

## "make_demo2"

This is the main way of launching for testing. It reads a file called SpecificationFile.txt in the "demo" directory.

usage:

cd data

<b><pre>make_demo2   input_folder    output_folder</pre></b>

* Input_folder must contain "SpecificationFile.txt", output will be wrtitten to "output_folder_<data_time>".
* If output_folder is not given, the value from SpecificationFile.txt will be used.

e.g. 

    make_demo2 demo test

SpecificationFile.txt is generated by make_demo (see below), and contains the launch parameters for both make_demo and load_sim (see below).

make_demo_2 generates a fresh model, then runs the simulation. 
Editng SpecificationFile.txt allows all relevant parameters to be controlled in one place.
This makes iteration for development and ebugging faster.


## "make_demo"
usage:

cd data

<b><pre>make_demo</pre></b>    

<b><pre>make_demo   num_particles   spacing    x_dim  y_dim  z_dim    demoType  simSpace</pre></b>

"demoType" sets individual particle properties in "demo/particles_pos_vel_color100001.csv", especially epigenetic states, from 3D positions.

* 0 free falling,
* 1 remodelling & actuation, ...   generates multiple tissue types
* 2 diffusion & epigenetics.

"simSpace" sets parameters in "SimParams." , especially gravity, and wavepool actuation.

* 0 regression test,
* 1 Tower (256,128,256),
* 2 Wave pool (400,200,400),
* 3 Small dam break (80,60,80),
* 4 Dual-Wave pool (200,100,30),
* 5 Microgravity (160,100.160),
* 6 Morphogenesis small demo (80,50,80)
* 7 used by make_demo_2 to take params from SpecificationFile.txt

e.g.

    make_demo                               // default simulation (works best at the moment).

    make_demo 125 1  6 6 6  0 5             // free falling

    make_demo 120 1  2 2 30  1 5
    make_demo 600 1  4 4 30  1 5            // remodelling & actuation, with fixed, bone, tendon, muscle, elastic, mesenchyme, external actuation

    make_demo 2000 1  6 6 30  1 5           //  "" , with reserve particles for growth.

    make_demo 400 1  10 10 3  1 5           // diffusion & epigenetics, with reserve particles for growth.

    make_demo 10000 1  100 10 10  1 1
    make_demo 100000 1  100 10 100  1 1
    make_demo 100000 1  100 100 10  1 1
    make_demo 1000000 1  100 100 100  1 1

        // 1M particles, free falling fluid in a wavepool  ... bug: no particles



### Re epigenetics_&_tissue_types

**make_demo** lets you specify **demoType**

**make_demo2** reads **demoType** from the SpefificationFile.txt

both call

void FluidSystem::**SetupAddVolumeMorphogenesis2**(Vector3DF min, Vector3DF max, float spacing, float offs, uint **demoType** ){...


```

 // ###   Epigenetics and tissue types

                for (int i=0; i< NUM_TF; i++)    { Conc[i]   = 0 ;}                                     // morphogen & transcription factor concentrations
                for (int i=0; i< NUM_GENES; i++) { EpiGen[i] = 0 ;}                                     // epigenetic state of each gene in this particle
                uint fixedActive = INT_MAX;                                                             // FEPIGEN below INT_MAX will count down to inactivation. 
                                                                                                        // Count down is inactivated by adding INT_MAX.
                
                EpiGen[0] = fixedActive;                                                                // active, i.e. not reserve
                EpiGen[1] = fixedActive;                                                                // solid, i.e. have elastic bonds
                EpiGen[2] = fixedActive;                                                                // living/telomere, i.e. has genes
                
                if(demoType == 1){                                                                    ////// Remodelling & actuation demo
                                                                                                        // Fixed base, bone, tendon, muscle, elastic, external actuation
                    if(Pos.z <= min.z+spacing)                                EpiGen[11]=fixedActive;   // fixed particle
                    if(Pos.z >= max.z-spacing)                                EpiGen[12]=fixedActive;   // external actuation particle 
                    
                    if(Pos.z >= min.z+5*spacing && Pos.z < min.z+10*spacing)  EpiGen[9] =fixedActive;   // bone
                    if(Pos.z >= min.z+10*spacing && Pos.z < min.z+15*spacing) EpiGen[6] =fixedActive;   // tendon
                    if(Pos.z >= min.z+15*spacing && Pos.z < min.z+20*spacing) EpiGen[7] =fixedActive;   // muscle
                    if(Pos.z >= min.z+20*spacing && Pos.z < min.z+25*spacing) EpiGen[10]=fixedActive;   // elastic tissue

                }else if (demoType == 2){                                                            ////// Diffusion & epigenetics demo
                                                                                                        // Fixed base, homogeneous particles (initially) 
                    if(Pos.z == min.z) EpiGen[0]=fixedActive;                                           // fixed particle
                    EpiGen[2]=1;                                                                        // living particle NB set gene behaviour
                }                                                                                       // => (i) French flag, (ii) polartity, (iii) clock & wave front
                
                p = AddParticleMorphogenesis2 (
                                                &Pos,                       // Vector3DF*
                                                &Vel,                       // Vector3DF*
                                                Age,                        // uint
                                                Clr,                        // uint
                                                ElastIdxU,                  // uint*
                                                ElastIdxF,                  // uint*
                                                Particle_Idx,               // unit*
                                                Particle_ID,                // uint
                                                Mass_Radius,                // uint
                                                NerveIdx,                   // uint
                                                Conc,                       // float*
                                                EpiGen                      // uint*
                                            );
```


## "check_demo"
usage:

cd data

<b><pre>check_demo  simulation_data_folder  output_folder</pre></b>

e.g.
    check_demo  demo  check 

CPU-only test program to verify the ability to read and re-output models.

## "SpecfileBatchGenerator"

Generates a batch of /demo_intstiff* folders containing variations on the original SpecificationFile.txt file.  
These can then be used with a Slurm script to launch a batch of simulations.

usage: 

<b><pre>SpecfileBatchGenerator   SpecificationFile.txt  </pre></b>

e.g.

    cd data
    SpecfileBatchGenerator demo


## "load_sim"
Loads complex particle model data from .csv files, and run simulation on GPU. 

Allows the user to choose whether to run genes and/or remodelling.

usage:

<b><pre>
load_sim   simulation_data_folder   output_folder      num_files  steps_per_file  freeze_steps      save_ply(y/n)  save_csv(y/n)  save_vtp(y/n)      debug(0-5)  gene_activity(y/n)  remodelling(y/n)
</pre></b>

Freeze -> create bonds. Converts a fluid into an elastic solid. 

The "initialize_bonds" kernel sets bond rest-length and modulus, according to tissue type set by epigenetic state.

```
debug:

 0=full speed,   1=current special output,   2=host cout,   3=device printf,   4=SaveUintArray(),   5=save .csv after each kernel.


e.g.
    cd data/test
    load_sim ../demo/                           ../out/      10  3 1    n n y    1 n n          //
    load_sim ../demo/                           ../out/      10  1 6    n y y    0 n n

    load_sim ../demo/                           ../out/    1000 10 0    n n y    0 n n          // Fluid sim (no freeze->no bonds), 10 files to save, 10 timesteps per .vtp file.
    load_sim ../demo/                           ../out/    1000  1 10   n y y    0 n y          // Remodelling sim,                 10 files to save,  1 timestep  per .csv and .vtp file
    load_sim ../demo/                           ../out/    1000  1 10   n y y    0 n y          // Remodelling & genetics sim,      10 files to save,  1 timestep  per .csv and .vtp file


    load_sim ../demo/                           ../out/    1000  1 10   n n y    0 n n
    load_sim ../demo/                           ../out/    1000  1 10   n n y    0 y y

    load_sim ../demo/                           ../out/     200  3  1   n n y    1 y y          // Good launch option for profiling

    load_sim ../demo_10000_1_100_10_10/         ../out/     100 30  1   n n y    0 n n
    load_sim ../demo_100000_1_100_100_10/       ../out/     100 30  1   n n y    0 n n
    load_sim ../demo_1000000_1_100_100_100/     ../out/     100 30  1   n n y    0 n n
```


                                                                                                // NB Now '100' frames per timestep x02-x20 snapshot after each kernel.
                                                                                                //x91 end of timestep. x00 begining of simulation.

                                                                                                // NB now 'freeze_steps' delay the start of particle movement,
                                                                                                //while heal() forms initial bonds.  ... not yet in use.


----
# Viewing with Paraview

The .vtp files output can be viewed in Paraview. 
This allows visualization of all the parameters, and is especially relevant for diffusion of morphogens, epigenetic state, and material properties.

ParaView is a widely used scientific data visualization tool.
ParaView can be downloaded from https://www.paraview.org/

## NB GPU conflict:
It is advised to exit Paraview before launching Morphogenesis.
Sometimes other runtime errors arrise if Paraview is still running when Morphogenesis is launched.
    
Both Paraview and Morphogenesis will both try to use your GPU. 
This may result in an _"invalid device context"_ or _"There is no device supporting CUDA"_ error.
This seems to happen specifically where the computer suspends while ParaView is open.
If ParaView fails to release the GPU after being shut down, then it may be necessary to reboot.
    
This does not arise where the two programs are run on separate machines, as when Morphogenesis is run on a cluster.
    
## Loading the data into ParaView:
    load the .vtp file 
    select the file in the pipeline browser
    
    From the top menu bar, select "Filters->Common->Treshold"
    In Properties(Thresold2), in scalars, select FPARTICLE_ID.
    Set "Maximum" to the number of active particles in the simulation.
    click "Apply" (green button in Properties)
    In the third row of the tool bar, click "zoom to data" icon (four arrows pointing inwards).
    NB this is necessary, when unused particles are stored in one corner of the simulation, 
    with FPARTICLE_ID = UINT_MAX.
    
    In "Coloring" select the model parameter of interest
    Adjust the coloring scale
    
    Set the background colour:
    Select Edit->Settings
    In the pop-up window,
    select Color Pallette->Background
    Choose a colour, click Apply, Okay.
    
    For Volume rendering:
    In the top menu bar, select "Filters->Point Interpolation->Point Volume Interpolator"
    Select the new "PointVolumeInterpolator" in the pipeline browser
    In the Properties pane, select a kernel type, e.g. Gaussian Kernel or Shepard Kernel
    In Coloring, select the model parameter of interest
    Adjust the coloring scale
    In "Volume Refinement", "Representation", select "Volume"
    (Alternatively select these in the top menu, second row tool bar.)
    In Volume Rendering (near the bottom of the Properties pane), in "Volume Rendering Mode", 
    select "Smart" or "GPU"
    Click Apply (at the toip of the Properties pane.
    
    
    Also in "Volume Rendering"
    In "Blend Mode" select between "Compostite/IsoSurface/Slice"

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
    
