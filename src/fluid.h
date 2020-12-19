/*
  FLUIDS v.1 - SPH Fluid Simulator for CPU and GPU
  Copyright (C) 2008. Rama Hoetzlein, http://www.rchoetzlein.com

  ZLib license
  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

#ifndef DEF_FLUID
	#define DEF_FLUID
	
	#include <cuda.h>
	#include <curand.h>
    #include <string.h>
    #include "vector.h"
//	#include "gvdb_vec.h"
//	using namespace nvdb;

    #include "masks.h"

	typedef	unsigned int		uint;	
	typedef	unsigned short int	ushort;	

	struct NList {
		int num;
		int first;
	};
	struct Fluid {						// offset - TOTAL: 72 (must be multiple of 12)
		Vector3DF		pos;			// 0                // could borrow common/vector.h  .cpp from fluids_v3
		Vector3DF		vel;			// 12
		Vector3DF		veleval;		// 24
		Vector3DF		force;			// 36
		float			pressure;		// 48
		float			density;		// 52
		int				grid_cell;		// 56
		int				grid_next;		// 60
		uint			clr;			// 64
		uint			state;			// 68
	};

    
    
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
    #define BONDS_PER_PARTICLE  6// 6 enables triangulated cubic structure    4   // current: 4 bonds plus length and modulus of each NB written to both particles so average 8 bonds per particle //old: actually 3, [0] for self ID, mass & radius
#define DATA_PER_BOND 9 //6 : [0]current index, [1]elastic limit, [2]restlength, [3]modulus, [4]damping coeff, [5]particle ID, [6]bond index [7]stress integrator [8]change-type binary indicator
                        // previously 3 : [0]current index, [1]mod_lim, [2]particle ID.
#define BOND_DATA BONDS_PER_PARTICLE * DATA_PER_BOND
#define REST_LENGTH  1  // need to find suitable number relative to particle and bin size, plus elastic limits.

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

    #define NUM_CHANGES 9   //  lengthen/shorten/weaken/strengthen * muscle/tissue + heal : lists for calling particle modification kernels
    
    // NB Genome defined at bottom of this file.
    
    // Data costs:
    
    // Lean simulation, 16M particles, 16 TF or morphogens, 16 genes:
    // 128 + 128 + 128 = 384 additional bits for minimal morphogenesis at minimal resolution => 960 bits/particle.
    // 3bonds*(8bits log modulus + 24bit uid) + self(4bit mass + 4bit radius + 24bit uid) +
    // 16(TF + morphogens) * 8bit concentration + 16genes * (2 bool + 4bit activation) 
    
    // Elastic only simulation 
    // 128 additional bits/particle for minimal elastic simulation => 704 bits/particle.

    // Extra bonds
    // Could use 6 bonds per particle, and drop FNBRCNT,FNBRNDX  : 224 + 128 + 128 + 576 - 32 = 1024 ie:1kbit/particle = 128bytes/particle
    
    // Rich simulation 4Bn particles (7.5x spatial resolution), 32 genes etc. 
    // 8*64 + 256 + 256 = 1024 additional bits/particle for a 'rich' simulation => 1600 bits/particle  = 200bytes/particle
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
	#define FPOS		0		//# 3DF   96  particle buffers //list of meaninful defines for the array of pointers in FBufs below.
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
	#define FNBRNDX		11		//# uint       particle neighbors index (optional)
	#define FNBRCNT		12      //# uint       particle neighbors count
	#define FCLUSTER	13	    //# uint

    // additional buffers for morphogenesis   
    #define FELASTIDX   14      //# currently [0]current index, [1]elastic limit, [2]restlength, [3]modulus, [4]damping coeff, [5]particle ID, [6]bond index [7]stress integrator [8]change-type binary indicator
                                //old : BOND_DATA = BONDS_PER_PARTICLE*3 = 12 //uint[BONDS_PER_PARTICLE * 2 = 8 ]  particleID, modulus, elastic limit    
                                /* old old : 0=self UID, mass, radius. >0= modulus & particle UID */
    #define FPARTICLEIDX 29    //# uint[BONDS_PER_PARTICLE *2]  list of other particles' bonds connecting to this particle AND their indices // NB risk of overwriting race condition, when making bonds.   
    #define FPARTICLE_ID 30     //# uint original pnum, used for bonds between particles. 32bit, track upto 4Bn particles.
    #define FMASS_RADIUS 31     //# uint holding modulus 16bit and limit 16bit.      
    //#define FELASTMOD         //# uint[BONDS_PER_PARTICLE +1]  modulus of bond (use a standard length) //not required
    #define FNERVEIDX   15      //# uint
    #define FCONC       16      //# float[NUM_TF]        NUM_TF = num transcription factors & morphogens
    #define FEPIGEN     17      //# uint[NUM_GENES]  holds epigenetic state. if cast as int, let -ve values indicate inactive or silenced genes.
                                // NB if data is ordered FEPIGEN[gene][particle], then contiguious read writes by synchronous kernels are favoured. 

// additional buffers for dense lists  
    #define INITIAL_BUFFSIZE_ACTIVE_GENES 1024    // initial buffer size for densely packed lists    
                                            // NB NUM_BINS is m_GridTotal computed in FluidSystem::SetupGrid()
                                            // NB The FGRID* buffers are set up in FluidSystem::AllocateGrid()
    #define FGRIDCNT_ACTIVE_GENES    32  //# uint[NUM_BINS][NUM_GENES]  for each bin, for each gene, num active particles
    #define FGRIDOFF_ACTIVE_GENES    33  //# uint[NUM_BINS][NUM_GENES]  For each bin, for each gene, the offset of the bin in the gene's dense array.
    #define FDENSE_LIST_LENGTHS      34  //# uint[NUM_GENES]  for each gene the length of its dense list, i.e. num active particles for that gene.
    #define FDENSE_LISTS             35  //# *uint[NUM_GENES]   where each points to uint[FNUM_ACTIVE_GENES[gene]]
                                            // In AllocateParticles(...) AllocateBuffer initial size 1024 for each gene.
                                            // Each timestep, before CountingSortFull(...) need to check size & enlarge if needed.
                                            // NB need to AllocateBuffer, _if_ (new size is > old size) 
                                            // enlarge in quadruplings, starting with 1024 particles. 
                                            // VNB need to modify cleanup at exit.
    #define FDENSE_BUF_LENGTHS       36  //# uint[NUM_GENES] holds length of currently allocated buffer.
    
// original buffers continued    
	#define FGRID		18		//!         uniform acceleration grid : the bin to which a particle belongs
	#define FGRIDCNT	19      //!         grid count                : array holding num particles in each bin
	#define	FGRIDOFF	20      //!         grid offset               : array holding the offset of each bin
	#define FGRIDACT	21      //!
	#define FSTATE		22      //# uint 
	//#define FBRICK		23      //!            #Not used
	#define FPARAMS		24		//! fluid parameters
	#define FAUXARRAY1	25		//! auxiliary arrays (prefix sums)
	#define FAUXSCAN1   26		//!
	#define FAUXARRAY2	27		//!
	#define FAUXSCAN2	28		//!
	
    #define FGRIDCNT_CHANGES            38     // for packing lists for particle change kenels
    #define FGRIDOFF_CHANGES            37
    #define FDENSE_LIST_LENGTHS_CHANGES 39
    #define FDENSE_LISTS_CHANGES        40     //# *uint [NUM_CHANGES] holds pointers to change_list buffers [2][list_length] holding : particleIDx and bondIDx TODO edit buffer allocation & use  
    #define FDENSE_BUF_LENGTHS_CHANGES  41
	
	#define MAX_BUF		                42
    

	#ifdef CUDA_KERNEL                                                                   // fluid_system_cuda.cuh:37:	#define CUDA_KERNEL ,   fluid_system_cuda.cu:29:#define CUDA_KERNEL
		#define	CALLFUNC	__device__
	#else
		#define CALLFUNC
	#endif		

	// Particle & Grid Buffers
	struct FBufs {       // holds an array of pointers, and functions to access them.    // used to declare "fbuf" at top of fluid_system_cuda.cu
        // Data type sizes  see https://en.cppreference.com/w/cpp/language/types ,  
        // 64 bit Linux uses  or 4/8/8 (int is 32-bit, long and pointer are 64-bit) 
        // short int 16bit, int 32bit, long int 64bit, float 32bit, double 64bit, 
		#ifdef CUDA_KERNEL
			// on device, access data via gpu pointers 
			inline CALLFUNC Vector3DF* bufV3(int n)		{ return (Vector3DF*) mgpu[n]; }
			inline CALLFUNC float3* bufF3(int n)		{ return (float3*) mgpu[n]; }
			inline CALLFUNC float*  bufF (int n)		{ return (float*)  mgpu[n]; }
			inline CALLFUNC uint*   bufI (int n)		{ return (uint*)   mgpu[n]; }
			inline CALLFUNC char*   bufC (int n)		{ return (char*)   mgpu[n]; }
			inline CALLFUNC uint**  bufII (int n)       { return (uint**)  mgpu[n]; }        // for elastIdx[][]
			//inline CALLFUNC unsigned short* bufS (int n)		{ return (unsigned short*)   mgpu[n]; }
		#else
			// on host, access data via cpu pointers
			inline CALLFUNC Vector3DF* bufV3(int n)		{ return (Vector3DF*) mcpu[n]; }
			inline CALLFUNC float3* bufF3(int n)		{ return (float3*) mcpu[n]; }
			inline CALLFUNC float*  bufF (int n)		{ return (float*)  mcpu[n]; }
			inline CALLFUNC uint*   bufI (int n)		{ return (uint*)   mcpu[n]; }
			inline CALLFUNC char*   bufC (int n)		{ return (char*)   mcpu[n]; }
			inline CALLFUNC uint**  bufII (int n)       { return (uint**)  mcpu[n]; }        // for elastIdx[][]
			//inline CALLFUNC unsigned short* bufS (int n)		{ return (unsigned short*)   mgpu[n]; }
		#endif
		inline CALLFUNC void    setBuf (int n, char* buf )	{ mcpu[n] = buf; }			// stores pointer to buffer in mcpu[]

		char*				mcpu[ MAX_BUF ];

		#ifdef CUDA_KERNEL
			char*			mgpu[ MAX_BUF ];		// on device, pointer is local.
		#else			
			CUdeviceptr		mgpu[ MAX_BUF ];		// on host, gpu is a device pointer // an array of pointers, filled by cuMemAlloc
			CUdeviceptr		gpu (int n )	{ return mgpu[n]; }
			CUdeviceptr*	gpuptr (int n )	{ return &mgpu[n]; }		
		#endif			
	};

/*			float3*			mpos;			// particle buffers
		float3*			mvel;
		float3*			mveleval;
		float3*			mforce;
		float*			mpress;
		float*			mdensity;
		ushort*			mage;
		uint*			mclr;			
		uint*			mgcell;
		uint*			mgnext;		
		uint*			mnbrndx;
		uint*			mnbrcnt;
		uint*			mcluster;
		char*			msortbuf;		// sorting buffer
		
		uint*			mgrid;			// grid buffers
		int*			mgridcnt;
		int*			mgridoff;
		int*			mgridactive;

		char*			mstate;			// state buffer
		float*			mbrick;*/


	// Temporary sort buffer offsets
	#define BUF_POS			0
	#define BUF_VEL			(sizeof(float3))
	#define BUF_VELEVAL		(BUF_VEL + sizeof(float3))
	#define BUF_FORCE		(BUF_VELEVAL + sizeof(float3))
	#define BUF_PRESS		(BUF_FORCE + sizeof(float3))
	#define BUF_DENS		(BUF_PRESS + sizeof(float))
	#define BUF_GCELL		(BUF_DENS + sizeof(float))
	#define BUF_GNDX		(BUF_GCELL + sizeof(uint))
	#define BUF_CLR			(BUF_GNDX + sizeof(uint))

	#define OFFSET_POS		0
	#define OFFSET_VEL		12
	#define OFFSET_VELEVAL	24
	#define OFFSET_FORCE	36
	#define OFFSET_PRESS	48
	#define OFFSET_DENS		52
	#define OFFSET_CELL		56
	#define OFFSET_GCONT	60
	#define OFFSET_CLR		64	

	// Fluid Parameters (stored on both host and device)
	struct FParams {
		int				numThreads, numBlocks, threadsPerBlock;
		int				gridThreads, gridBlocks;	

		int				szPnts, szHash, szGrid;
		int				stride, pnum, pnumActive;
        bool            freeze;
        uint            frame;
		int				chk;
		float			pdist, pmass, prest_dens;
		float			pextstiff, pintstiff;
		float			pradius, psmoothradius, r2, psimscale, pvisc;
		float			pforce_min, pforce_max, pforce_freq, pground_slope;
		float			pvel_limit, paccel_limit, pdamp;
		float3			pboundmin, pboundmax, pgravity;
		float			AL, AL2, VL, VL2;

		float			d2, rd2, vterm;		// used in force calculation		 
		
		float			poly6kern, spikykern, lapkern, gausskern;

		float3			gridSize, gridDelta, gridMin, gridMax;
		int3			gridRes, gridScanMax;
		int				gridSrch, gridTotal, gridAdjCnt, gridActive;
		int				gridAdj[64];

		int3			brickRes;
		int				pemit;
	};
    
    //////////////////////
    // material parameters for bonds, used in remodelling
    struct FBondParams{             //  0=elastin, 1=collagen, 2=apatite //
        enum params{  /*triggering bond parameter changes*/elongation_threshold, elongation_factor, strength_threshold, strengthening_factor, \
                      /*triggering particle changes*/max_rest_length, min_rest_length, max_modulus, min_modulus, \
                      /*initial values for new bonds*/elastLim, default_rest_length, default_modulus, default_damping 
        };
        static float param[12];
        /*
        // triggering bond parameter changes
        static float elongation_threshold   ;// = { 0.1, 0.1, 0.1}; // stress   fraction of elastlim 
        static float elongation_factor      ;// = { 0.1, 0.1, 0.1}; // length   fraction of restlength
        static float strength_threshold     ;// = { 0.1, 0.1, 0.1}; // stress   fraction of modulus
        static float strengthening_factor   ;// = { 0.1, 0.1, 0.1}; // modulus  fraction of current modulus
        
        // triggering particle changes
        static float max_rest_length        ;// = { 0.8, 0.8, 0.8};
        static float min_rest_length        ;// = { 0.3, 0.3, 0.3};
        static float max_modulus            ;// = { 0.8, 0.8, 0.8};
        static float min_modulus            ;// = { 0.3, 0.3, 0.3};
        
        // initial values for new bonds
        static float elastLim               ;// = {2,   0.55,  0.05 };// 400%, 10%, 1%
        static float default_rest_length    ;// = { 0.5, 0.5,    0.5  };
        static float default_modulus        ;// = { 100000, 10000000, 10000000};
        static float default_damping        ;// = { 10, 100, 1000};
        */
    };
    // values from make demo
    // [1]elastLim	 [2]restLn	 [3]modulus	 [4]damping
    //    2	            0.5	        100000	    9.055386
    
    ///////////////////////
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
    
    struct FGenome{   // ## currently using fixed size genome for efficiency. NB Particle data size depends on genome size.
        uint mutability[NUM_GENES];
        uint delay[NUM_GENES];
        uint sensitivity[NUM_GENES][NUM_GENES]; // for each gene, its sensitivity to each TF or morphogen
        uint tf_diffusability[NUM_TF];  // for each transcription_factor, the diffusion and breakdown rates of its TF.
        uint tf_breakdown_rate[NUM_TF];
        // sparse lists final entry = num elem, other entries (elem_num, param)
        int secrete[NUM_GENES][2*NUM_TF+1];        // -ve secretion => active breakdown. Can be useful for pattern formation.
        int activate[NUM_GENES][2*NUM_GENES+1];
        //uint *function[NUM_GENES];    // Hard code a case-switch that calls each gene's function iff the gene is active.
        enum {elastin,collagen,apatite};
        //FBondParams fbondparams[3];   // 0=elastin, 1=collagen, 2=apatite
        
        enum params{  /*triggering bond parameter changes*/ elongation_threshold,   elongation_factor,      strength_threshold,     strengthening_factor, \
                      /*triggering particle changes*/       max_rest_length,        min_rest_length,        max_modulus,            min_modulus, \
                      /*initial values for new bonds*/      elastLim,               default_rest_length,    default_modulus,        default_damping 
        };
        float param[3][12];      // TODO update all uses of FBondParams & test.
    };                                  // NB gene functions need to be in fluid_system_cuda.cu
    
    ///////////////////////
    // Multi-scale particles    
    // - use radius parameter in FELASTIDX buffer, 0=self UID, mass, radius.
    // - requires a split/combine kernel
    // 1st order - all spheres, not 2nd order - elipsoids with orientatoin and angular momentum.
    // When finding particles in range for a small particle - consider (i) is the bin in range, (ii) iff particle x&y&z are in range 
    // When combining particles - (i) similar type ?, (ii) gradients, (iii) significance to simulation. 
    
    



#endif /*PARTICLE_H_*/
