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

This code compiles and runs with cmake 3.10, vtk-9.0, and Cuda 11.2 on Ubuntu 20.04 with GTX 980m.
and on Suse Linux cluster with Cuda 9.1 and Tesla P100.

## Morphogenesis branch
A morphogenesis simulator _(in progress)_ , with soft-matter elasticity, diffusion of heat/chemicals/morphogens, epi-genetics and particle automata behaviour.

The notes below are rough working notes, and will change with development.

### Dependencies 
VTK is used for .vtp file output for visualization in Parraview.
NB vtk-dev is broken in Ubuntu 18.04 LTS. 
You may need to build VTK from source.
This code has been developed with VTK-9.0.1 .

***NB if you need to build VTK, please remember to ***

    sudo make install
    sudo ldconfig
    
so that the new library will be found at runtime.


### Build instructions 

This project uses Cmake. 
Create a build subdirectory.
In the build subdirectory

    cmake ../
    make
    make install


### Within Morphogenesis branch executables (so far) include:


#### make_demo
usage:
    
    cd data
    make_demo  num_particles  spacing  x_dim  y_dim  z_dim  demoType  simSpace
    
    where demoType(0:free falling, 1:remodelling & actuation, 2:diffusion & epigenetics.)
    
    and simSpace(0:regression test, 1:tower, 2:wavepool, 3: small dam break, 4:dual-wavepool, 5: microgravity)
    
    demoType sets individual particle properties in particles_pos_vel_color100000.csv, especially epigenetic states, from 3D positions.
    
    simSpace sets parameters in SimParams.txt , especially gravity, and wavepool actuation.
    
    
e.g.

    ../build/install/bin/make_demo 125 1  6 6 6  0 5       // free falling
    
    ../build/install/bin/make_demo 120 1  2 2 30  1 5
    ../build/install/bin/make_demo 600 1  4 4 30  1 5      // remodelling & actuation, with fixed, bone, tendon, muscle, elastic, mesenchyme, external actuation
    
    ../build/install/bin/make_demo 400 1  10 10 3  1 5     // diffusion & epigenetics, with reserve particles for growth.
    
    
    ../build/install/bin/make_demo 10000 1  100 10 10  0 1
    ../build/install/bin/make_demo 100000 1  100 100 10  0 1
    ../build/install/bin/make_demo 1000000 1  100 100 100  0 1

    
CPU-only test program to generate example **"SimParams.txt"** and **"particles_pos_vel_color100001.csv"** files for specifying models, and a **"particles_pos100001.ply"** for viewing a model in e.g. Meshlab.

#### check_demo
usage:

    cd data
    check_demo  simulation_data_folder  output_folder
    
e.g.

    ../build/install/bin/check_demo  demo  check 
    
    
CPU-only test program to verify the ability to read and re-output models.

#### load_sim
New launch program to load data from files, and run simulation on GPU.

usage:

    cd data
    ../build/install/bin/load_sim  demo  out   num_files   steps_per_file   freeze_steps   save_ply(y/n)  save_csv(y/n)  save_vtp(y/n)  debug(y/n)  gene_activity(y/n)  remodelling(y/n)
    
e.g.

    cd data/test
    ./load_sim ../demo/ ../out/  10 3 1 y y y
    ./load_sim ../demo/ ../out/  10 1 6 n y y
    
    ./load_sim ../demo/ ../out/  1000 10 0 n y y y y y                   // NB Now '100' frames per timestep x02-x20 snapshot after each kernel. x91 end of timestep. x00 begining of simulation.
    ./load_sim ../demo/ ../out/  1000 1 10 n y y y n y                  // NB now 'freeze_steps' delay the start of particle movement, while heal() forms initial bonds.
    ./load_sim ../demo/ ../out/  1000 1 10 n y y y n n
    ./load_sim ../demo/ ../out/  1000 1 10 n y y n y y
    
    ./load_sim ../demo_10000_1_100_10_10/ ../out/  100 30 1 n n y
    ./load_sim ../demo_100000_1_100_100_10/ ../out/  100 30 1 n n y
    ./load_sim ../demo_1000000_1_100_100_100/ ../out/  100 30 1 n n y


### viewing with Meshlab

The .ply files output can be viewed in MeshLab.
It is recommended to select the following MeshLab options:   NB need old MeshLab (Ubuntu18.04). Version on Ubuntu 20.04 does not work.

    Render - Show vertex dots
    Render - Render Mode - Wireframe
    
    
### viewing with Paraview

The .vtp files output can be viewed in Paraview. This allows visualization of all the parameters, and is especially relevant for diffusion of morphogens, epigenetic state, and material properties.

    ParaView is a widely used scientific data visualization tool.
    ParaView can be downloaded from https://www.paraview.org/

    NB GPU conflict:
    It is advised to exit Paraview before launching Morphogenesis.
    Sometimes other runtime errors arrise if Paraview is still running when Morphogenesis is launched.
    
    Both Paraview and Morphogenesis will both try to use your GPU. 
    This may result in an _"invalid device context"_ or _"There is no device supporting CUDA"_ error.
    This seems to happen specifically where the computer suspends while ParaView is open.
    If ParaView fails to release the GPU after being shut down, then it may be necessary to reboot.
    
    This does not arise where the two programs are run on separate machines, as when Morphogenesis is run on a cluster.
    

    Loading the data into ParaView:
    load the .vtp file 
    select the file in the pipeline browser
    
    From the top menu bar, select "Filters->Common->Treshold"
    In Properties(Thresold2), in scalars, select FPARTICLE_ID.
    Set Maximum to the number of active particles in the simulation.
    click "Apply" (green button in Properties)
    In the third row of the tool bar, click "zoom to data" icon (four arrows pointing inwards).
    NB this is necessary, when unused particles are stored in one corner of the simulation, with FPARTICLE_ID = UINT_MAX.
    
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
    In Volume Rendering (near the bottom of the Properties pane), in "Volume Rendering Mode", select "Smart" or "GPU"
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
    
    
### New data structures in Morphogenesis branch

See notes in fluid.h

#### Additional buffers - per particle

    #define FELASTIDX   14      //# uint[BONDS_PER_PARTICLE +1]  0=self UID, mass, radius. >0= modulus & particle UID
    #define FNERVEIDX   15      //# uint
    #define FCONC       16      //# float[NUM_TF]        NUM_TF = num transcription factors & morphogens
    #define FEPIGEN     17      //# uint[NUM_GENES]
    
    FELASTIDX - gives the particle ID of those particles that have elastic bonds to this particle.
    
    FNERVEIDX - gives the ID of the nerve that reads/writes to this particle.
    
    FCONC - gives the concentration in this particle of each diffusable morphogen.
    
    FEPIGEN - gives the epigenetic state of the genes of this particle.
    
    
// Buffers:

    // NB one buffer created per particle parameter (marked '#'),
    // & explicitly allocated in FluidSystem::AllocateParticles ( int cnt ).
    //
    // Need to list any new parameters here. e.g. for elasticity, diffusion, epigenetics, nerves & muscles.
    // NB Epigenetics & diffusion only in morphogenesis mode.

    // Elasticity:
    // Elasticity requires a list of [3+] attached particles, and a uid for each particle, plus material type &/or modulus
    // 32bit unsigned int for Uid - upto 4Bn (4,294,967,296) particles
    // 32bit float for modulus (could be more complex e.g. fatigue, nonlinear) (lookup type)
    // Require:
    // 128bits = 4*32 extra bits/particle (upto 3 'bonds) 
    // Could use 32bit uint - 8 bits for 256 bond types (look up modulus on exp scale), and 2^24=16,777,216 particles.
    // Use 8 bit of self-UID for 2^4=16 mass & fluid radius.
    // 
    // # if elastic force is written to both interacting particles, then the effective number of bonds doubles.
    // # i.e. each particle stores three bonds, but the average bonds per atom would be six.
    #define BONDS_PER_PARTICLE  3   // +1 for self ID, mass & radius
    
    // Sensorimotor:
    // Sensory nerve endings & muscles need type and nerve uid to write/read.
    // 32bit uint for i/d, but use 3 bits for type (0)muscle, (1)stretch, (2)pressure/vibration, (3)temp, (4-7)other..
    // => 536,870,912 nerve UIDs
    // ? 3 bits ? in particle UID => then read nerve ID from 32bit uint array.

    // Diffusion:
    // Diffusion of morphogens and accumulation of transcription factors, reqires concentration, 
    // (8bit integer 0-255, 4 per 32 bit unsigned int)
    // Need _at_least_ 8 morphogens, and 8 transcription factors => 16/4=4 uints or 8*16= 128bits.
    // + rule index (common to most cells, copied to 'local' SMP memory on GPU) => diffusion rate & breakdown rate.
    // 128bits
    #define NUM_TF              16      //  minimum 8 transcription factors + 8 diffusable morphogens
    
    // Epigenetic state requires for each gene: 
    // (i)current activation, (ii)available/silenced (bool), (iii)spread/stop (bool).
    // Could use uint[num_genes=32], and use the 1st 2 bits for booleans & 14 bits for the value (0-16383).
    // Or 2 genes per uint, with 4 boolean bits and 6-bit values (0-63).
    // If there are 32 genes, then need 32*16bits = 512bits.
    // Or 16 genes * 8bits = 128 bits
    // (NB there are about 200 cell types in the human body => minimum 8 genes)
    // 128bits
    #define NUM_GENES           16      //  >= NUM_TF NB each gene wll have a list of sensitivities to TFs & morphogens
    #define BITS_PER_EPIGENETIC_STATE 8 //i.e. 2 boolean + 6bit activation 2^6=64.

    // NB Genome defined at bottom of this file.
    
// Data costs:
    
    // Lean simulation, 16M particles, 16 TF or morphogens, 16 genes:
    // 128 + 128 + 128 = 384 additional bits for minimal morphogenesis at minimal resolution => 960 bits/particle.
    // 3bonds*(8bits log modulus + 24bit uid) + self(4bit mass + 4bit radius + 24bit uid) +
    // 16(TF + morphogens) * 8bit concentration + 16genes * (2 bool + 4bit activation) 
    
    // Elastic only simulation 
    // 128 additional bits/particle for minimal elastic simulation => 704 bits/particle.

    // Extra bonds
    // Could use 6 bonds per particle, and drop FNBRCNT,FNBRNDX  : 224 + 128 + 128 + 576 - 32 = 1024 ie:1k/particle.
    
    // Rich simulation 4Bn particles (7.5x spatial resolution), 32 genes etc. 
    // 8*64 + 256 + 256 = 1024 additional bits/particle for a 'rich' simulation => 1600 bits/particle.
    // 8bonds * 32bit uid + 8bonds*32bit modulus +  32TF*8bits + 32genes*8bits     
    
    
    // NB target architectures:
    // Tesla P100, 3584 cores, 1.Ghz, 16Gb, 732GB/sec, 10.6 TeraFLOPS single, 21.2 TeraFLOPS half precision,  copmute capability 6.0
    // RTX2080-Ti, 4352 cores, 1.545Ghz  11Gb, 616GB/sec, 11.75TFLOPS single, 23.5TFLOPS half
    // 
    // base morphogenesis -> 16GB / 960 bits * 2 fullcopy  = 8M particles on one card's memory. 
    // 'Rich'   "         -> 16GB / 1600 bits      "       = <5M
    // speed 732GB/sec  / 16GB = 45.75fps  (upper bound data speed)
    
    // NB spatial resolution:
    // 4M^(1/3) = 158.7  , 8M^(1/3) = 200 particles per side of cube.
    // 4M/card =  1,600,000,000, max Bracewell, => 1,169 particles per side of cube.

    //                                  bits    4*96 + 2*32 + 8*16 = 576bits/particle : fluid only
    
	#define FPOS		0       //# 3DF   96  particle buffers //list of meaninful defines for the array of pointers in FBufs below.
	#define FVEL		1       //# 3DF        velocity
	#define FVEVAL		2       //# 3DF        half step in velocity - used in numerical integration
	#define FFORCE		3       //# 3DF        force 
	#define FPRESS		4       //# F      32  pressure
	#define FDENSITY	5       //# F          density
	#define FAGE		6       //# Ushort 16  particle age. 
	#define FCLR		7       //# uint   16  colour
	#define FGCELL		8       //# uint       grid cell
	#define FGNDX		9       //# uint       grid index
	#define FGNEXT		10      //# uint
	#define FNBRNDX		11      //# uint       particle neighbors index (optional)
	#define FNBRCNT		12      //# uint       particle neighbors count
	#define FCLUSTER	13      //# uint
	
 // additional buffers for morphogenesis   
 
    #define FELASTIDX   14      //# uint[BONDS_PER_PARTICLE +1]  0=self UID, mass, radius. >0= modulus & particle UID
    //#define FELASTMOD         //# uint[BONDS_PER_PARTICLE +1]  modulus of bond (use a standard length) //not required
    #define FNERVEIDX   15      //# uint
    #define FCONC       16      //# float[NUM_TF]        NUM_TF = num transcription factors & morphogens
    #define FEPIGEN     17      //# uint[NUM_GENES]
    
// original buffers continued    

	#define FGRID		18		//!         uniform acceleration grid
	#define FGRIDCNT	19      //!         grid count
	#define	FGRIDOFF	20      //!         grid offset
	#define FGRIDACT	21      //!
	#define FSTATE		22      //# uint 
	#define FBRICK		23      //!
	#define FPARAMS		24		//! fluid parameters
	#define FAUXARRAY1	25		//! auxiliary arrays (prefix sums)
	#define FAUXSCAN1   26		//!
	#define FAUXARRAY2	27		//!
	#define FAUXSCAN2	28		//!
	#define MAX_BUF		29		//!

    

#### Additional global buffers


    
    
    
// Genome 

    // (common to most cells, copied to 'local' SMP memory on GPU) for each gene: 
    // (i) mutability,              - Not used during morphogenesis
    // (ii) delay/insulator,        - Needs to set an epigenetic counter 
    // (iii) sensitivity to inputs  - Transcription factors/morphogens, stress/strain cycles (stored in a TF)
    //      (a) array of sensitivities to each TF/Morphogen => increment activity of the gene (recorded in the epigenetics)
    // 
    // (iv)cell actions             - the function of the gene is called, with activation level as its parameter.
    //                              - only functions of active genes are called.
    //      (a)secrete morphogen,        - the product of the gene 
    //      (b)move,                     - exert force on neighbouring cells, according to concetration gradient of a morphogen 
    //      (c)adhere,                   - form elastic bonds - to a particular cell type.
    //      (d)divide                    - split/combine particles - NB elastic links. 
    //      (e)secrete/resorb material   - change mass, radius, modulus, viscosity of particle. 
    // NB we don't have a particle-wise viscosity param ...yet 
    
    struct FGenome{
        uint mutability[NUM_GENES];
        uint delay[NUM_GENES];
        uint sensitivity[NUM_GENES][NUM_GENES]; // for each gene, its sensitivity to each TF or morphogen
        uint difusability[NUM_GENES][2];// for each gene, the diffusion and breakdown rates of its TF.
        //uint *function[NUM_GENES];    // Hard code a case-switch that calls each gene's function iff the gene is active.
    };                                  // NB gene functions need to be in fluid_system_cuda.cu
    
    // Multi-scale particles    
    // - use radius parameter in FELASTIDX buffer, 0=self UID, mass, radius.
    // - requires a split/combine kernel
    // 1st order - all spheres, not 2nd order - elipsoids with orientatoin and angular momentum.
    // When finding particles in range for a small particle - consider (i) is the bin in range, (ii) iff particle x&y&z are in range 
    // When combining particles - (i) similar type ?, (ii) gradients, (iii) significance to simulation. 
    

### New kernels in Morphogenesis branch

New kernels are being written force


- elastic forces & breaking elastic bonds

- diffusion 

- particle automata behaviour - controlled by epigenetics 

- connecting a nervous system 

- visualization







