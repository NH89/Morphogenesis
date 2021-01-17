
//-------------------------------------------------------------------
// FLUIDS v.3 - SPH Fluid Simulator for CPU and GPU
// Copyright (C) 2012-2013. Rama Hoetzlein, http://fluids3.com
//
// Attribute-ZLib license (* See additional part 4)
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.
// 4. Any published work based on this code must include public acknowledgement
//    of the origin. This includes following when applicable:
//	   - Journal/Paper publications. Credited by reference to work in text & citation.
//	   - Public presentations. Credited in at least one slide.
//	   - Distributed Games/Apps. Credited as single line in game or app credit page.	 
//	 Retaining this additional license term is required in derivative works.
//	 Acknowledgement may be provided as:
//	   Publication version:  
//	      2012-2013, Hoetzlein, Rama C. Fluids v.3 - A Large-Scale, Open Source
//	 	  Fluid Simulator. Published online at: http://fluids3.com
//	   Single line (slides or app credits):
//	      GPU Fluids: Rama C. Hoetzlein (Fluids v3 2013)
//--------------------------------------------------------------------

#ifndef DEF_FLUID_SYS
	#define DEF_FLUID_SYS

	#include <iostream>
	#include <vector>
	#include <stdio.h>
	#include <stdlib.h>
	#include <math.h>	
	#include <vtk-9.0/vtkCellArray.h>
    #include <vtk-9.0/vtkPoints.h>
    #include <vtk-9.0/vtkXMLPolyDataWriter.h>
    #include <vtk-9.0/vtkPolyData.h>
    #include <vtk-9.0/vtkSmartPointer.h>
    #include <vtk-9.0/vtkLine.h>
    #include <vtk-9.0/vtkDataSet.h>
    #include <vtk-9.0/vtkUnsignedIntArray.h>
    #include <vtk-9.0/vtkUnsignedCharArray.h>
    #include <vtk-9.0/vtkFloatArray.h>
    #include <vtk-9.0/vtkPointData.h>
    #include <vtk-9.0/vtkCellData.h>
	#include "fluid.h"

	#define MAX_PARAM			50             // used for m_Param[], m_Vec[], m_Toggle[]
	#define GRID_UCHAR			0xFF           // used in void FluidSystem::InsertParticles (){.. memset(..); ...}
	#define GRID_UNDEF			4294967295	


    // Run params : values for m_Param[PMODE]
	#define RUN_PAUSE			0
	#define RUN_SEARCH			1
	#define RUN_VALIDATE		2
	#define RUN_CPU_SLOW		3
	#define RUN_CPU_GRID		4	
	#define RUN_GPU_FULL		5
	#define RUN_PLAYBACK		6

	// Scalar params   "m_Param[]"  //  /*remove some of these lines ?*/   Need to check if/when each is used...
	#define PMODE				0
	#define PNUM				1
	#define PEXAMPLE			2   // 0=Regression test. N x N x N static grid, 1=Tower , 2=Wave pool , 3=Small dam break , 4=Dual-Wave pool , 5=Microgravity . See  FluidSystem::SetupExampleParams (). Used in void FluidSystem::Initialize ()  
	#define PSIMSIZE			3
	#define PSIMSCALE			4
	#define PGRID_DENSITY		5
	#define PGRIDSIZE			6
	#define PVISC				7
	#define PRESTDENSITY		8
	#define PMASS				9
	#define PRADIUS				10
	#define PDIST				11
	#define PSMOOTHRADIUS		12
	#define PINTSTIFF			13
	#define PEXTSTIFF			14
	#define PEXTDAMP			15
	#define PACCEL_LIMIT		16
	#define PVEL_LIMIT			17
	#define PSPACING			18
	#define PGROUND_SLOPE		19
	#define PFORCE_MIN			20
	#define PFORCE_MAX			21
	#define PMAX_FRAC			22
	#define PDRAWMODE			23
	#define PDRAWSIZE			24
	#define PDRAWGRID			25	
	#define PDRAWTEXT			26	
	#define PCLR_MODE			27
	#define PGRAV				28
	#define PSTAT_OCCUPY		29
	#define PSTAT_GRIDCNT		30
	#define PSTAT_NBR			31
	#define PSTAT_NBRMAX		32
	#define PSTAT_SRCH			33
	#define PSTAT_SRCHMAX		34
	#define PSTAT_PMEM			35
	#define PSTAT_GMEM			36
	#define PTIME_INSERT		37
	#define PTIME_SORT			38
	#define PTIME_COUNT			39
	#define PTIME_PRESS			40
	#define PTIME_FORCE			41
	#define PTIME_ADVANCE		42
	#define PTIME_RECORD		43
	#define PTIME_RENDER		44
	#define PTIME_TOGPU			45
	#define PTIME_FROMGPU		46
	#define PFORCE_FREQ			47	

	// Vector params   "m_Vec[]" 
	#define PVOLMIN				0
	#define PVOLMAX				1
	#define PBOUNDMIN			2
	#define PBOUNDMAX			3
	#define PINITMIN			4
	#define PINITMAX			5
	#define PEMIT_POS			6
	#define PEMIT_ANG			7
	#define PEMIT_DANG			8
	#define PEMIT_SPREAD		9
	#define PEMIT_RATE			10
	#define PPOINT_GRAV_POS		11	
	#define PPLANE_GRAV_DIR		12	

	// Booleans        "m_Toggle[]"
	#define PRUN				0
	#define PDEBUG				1	
	#define PUSE_CUDA			2	//not used?  /*remove this line ?*/
	#define	PUSE_GRID			3
	#define PWRAP_X				4
	#define PWALL_BARRIER		5
	#define PLEVY_BARRIER		6
	#define PDRAIN_BARRIER		7		
	#define PPLANE_GRAV_ON		11	
	#define PPROFILE			12
	#define PCAPTURE			13

	#define BFLUID				2    // not used ?  /*remove this line ?*/

	// kernel function   "m_Func[]"
	#define FUNC_INSERT			0
	#define	FUNC_COUNTING_SORT	1
	#define FUNC_QUERY			2
	#define FUNC_COMPUTE_PRESS	3
	#define FUNC_COMPUTE_FORCE	4
	#define FUNC_ADVANCE		5
	#define FUNC_EMIT			6
	#define FUNC_RANDOMIZE		7
	#define FUNC_SAMPLE			8
	#define FUNC_FPREFIXSUM		9
	#define FUNC_FPREFIXFIXUP	10
    #define FUNC_TALLYLISTS     11
    #define FUNC_COMPUTE_DIFFUSION          12
    #define FUNC_COUNT_SORT_LISTS           13
    #define FUNC_COMPUTE_GENE_ACTION        14
#define FUNC_TALLY_GENE_ACTION 35
    #define FUNC_COMPUTE_BOND_CHANGES       15
    
    #define FUNC_INSERT_CHANGES             16 //insertChanges
    #define FUNC_PREFIXUP_CHANGES           17 //prefixFixupChanges
    #define FUNC_PREFIXSUM_CHANGES          18 //prefixSumChanges
    #define FUNC_TALLYLISTS_CHANGES         19 //tally_changelist_lengths
    #define FUNC_COUNTING_SORT_CHANGES      20 //countingSortChanges 
    #define FUNC_COMPUTE_NERVE_ACTION       21 //computeNerveActivation
    
    #define FUNC_COMPUTE_MUSCLE_CONTRACTION 22 //computeMuscleContraction
    #define FUNC_HEAL                       23 //heal
    #define FUNC_LENGTHEN_MUSCLE            24 //lengthen_muscle
    #define FUNC_LENGTHEN_TISSUE            25 //lengthen_tissue
    #define FUNC_SHORTEN_MUSCLE             26 //shorten_muscle
    #define FUNC_SHORTEN_TISSUE             27 //shorten_tissue
    
    #define FUNC_STRENGTHEN_MUSCLE          28 //strengthen_muscle
    #define FUNC_STRENGTHEN_TISSUE          29 //strengthen_tissue
    #define FUNC_WEAKEN_MUSCLE              30 //weaken_muscle
    #define FUNC_WEAKEN_TISSUE              31 //weaken_tissue
    
    #define FUNC_EXTERNAL_ACTUATION         32
    #define FUNC_FIXED                      33
    #define FUNC_CLEAN_BONDS                34
    
    #define FUNC_INIT_FCURAND_STATE         36
    
    #define FUNC_MAX			            37

    //  used for AllocateBuffer(  .... )
	#define GPU_OFF				0
	#define GPU_SINGLE			1
	#define GPU_TEMP			2
	#define GPU_DUAL			3
	#define CPU_OFF				4
	#define CPU_YES				5

	bool cuCheck (CUresult launch_stat, const char* method, const char* apicall, const char* arg, bool bDebug);
	
	class FluidSystem {
	public:
		FluidSystem ();
        
		void LoadKernel ( int id, std::string kname );
		void Initialize ();
        void InitializeCuda ();                             // used for load_sim

		// Particle Utilities
		void AllocateBuffer(int buf_id, int stride, int cpucnt, int gpucnt, int gpumode, int cpumode);		
        void AllocateBufferDenseLists ( int buf_id, int stride, int gpucnt, int lists );
		void TransferToTempCUDA ( int buf_id, int sz );
        void AllocateParticles ( int cnt, int gpu_mode = GPU_DUAL, int cpu_mode = CPU_YES );
        void AddNullPoints ();
        int  AddParticleMorphogenesis2(Vector3DF* Pos, Vector3DF* Vel, uint Age, uint Clr, uint *_ElastIdxU, float *_ElastIdxF, uint *_Particle_Idx, uint Particle_ID, uint Mass_Radius, uint NerveIdx, float* _Conc, uint* _EpiGen );
        
        
		//void AddEmit ( float spacing );
		int NumPoints ()				{ return mNumPoints; }
		Vector3DF* getPos ( int n )	    { return &m_Fluid.bufV3(FPOS)[n]; }
		Vector3DF* getVel ( int n )	    { return &m_Fluid.bufV3(FVEL)[n]; }
		uint* getAge ( int n )			{ return &m_Fluid.bufI(FAGE)[n]; }
		uint* getClr ( int n )			{ return &m_Fluid.bufI(FCLR)[n]; }
        uint* getElastIdx( int n )      { return &m_Fluid.bufI(FELASTIDX)[n*(BONDS_PER_PARTICLE * DATA_PER_BOND)]; }        //note #define FELASTIDX   14      
        uint* getParticle_Idx( int n )  { return &m_Fluid.bufI(FPARTICLEIDX)[n*BONDS_PER_PARTICLE*2]; } 
        uint* getParticle_ID(int n )    { return &m_Fluid.bufI(FPARTICLE_ID)[n]; }
        uint* getMass_Radius(int n )    { return &m_Fluid.bufI(FMASS_RADIUS)[n]; }
        uint* getNerveIdx( int n )      { return &m_Fluid.bufI(FNERVEIDX)[n]; }          //#define FNERVEIDX        15      //# uint
        float* getConc(int tf)        { return &m_Fluid.bufF(FCONC)[tf*mMaxPoints];}        //note #define FCONC       16      //# float[NUM_TF]        NUM_TF = num transcription factors & morphogens
        uint* getEpiGen(int gene)       { return &m_Fluid.bufI(FEPIGEN)[gene*mMaxPoints];}   //note #define FEPIGEN     17    //# uint[NUM_GENES] // used in savePoints... 
                                                                                             //NB int mMaxPoints is set even if FluidSetupCUDA(..) isn't called, e.g. in makedemo ..
		// Setup
		void SetupKernels ();
		void SetupDefaultParams ();
		void SetupExampleParams (uint simSpace);
        void SetupExampleGenome();
		void SetupSpacing ();
        void SetupAddVolumeMorphogenesis2(Vector3DF min, Vector3DF max, float spacing, float offs, uint demoType );  // NB ony used in WriteDemoSimParams()
		void SetupGrid ( Vector3DF min, Vector3DF max, float sim_scale, float cell_size, float border );		
		void AllocateGrid ();
        void AllocateGrid(int gpu_mode, int cpu_mode);

		// Simulation
		void Run ();	
        void Run( const char * relativePath, int frame, bool debug, bool gene_activity, bool remodelling );
        void Freeze ();
        void Freeze (const char * relativePath, int frame, bool debug, bool gene_activity, bool remodelling);
		void AdvanceTime ();
		
		void Exit ();
		void TransferToCUDA ();
		void TransferFromCUDA ();
		double GetDT()		{ return m_DT; }

		// Debugging
		int GetSelected ()		{ return mSelected; }
		
		// Acceleration Grid
		Vector3DF GetGridRes ()		{ return m_GridRes; }
		Vector3DF GetGridMin ()		{ return m_GridMin; }
		Vector3DF GetGridMax ()		{ return m_GridMax; }
		Vector3DF GetGridDelta ()	{ return m_GridDelta; }

		void FluidSetupCUDA ( int num, int gsrch, int3 res, float3 size, float3 delta, float3 gmin, float3 gmax, int total, int chk );
		void FluidParamCUDA ( float ss, float sr, float pr, float mass, float rest, float3 bmin, float3 bmax, float estiff, float istiff, float visc, float damp, float fmin, float fmax, float ffreq, float gslope, float gx, float gy, float gz, float al, float vl, int emit );

        void Init_FCURAND_STATE_CUDA ();
		void InsertParticlesCUDA ( uint* gcell, uint* ccell, uint* gcnt );	
		void PrefixSumCellsCUDA ( int zero_offsets );		
		void CountingSortFullCUDA ( Vector3DF* gpos );
        
        void InsertChangesCUDA ( /*uint* gcell, uint* gndx, uint* gcnt*/ );
        void PrefixSumChangesCUDA ( int zero_offsets );
        void CountingSortChangesCUDA ( );
        
		void ComputePressureCUDA ();
		void ComputeDiffusionCUDA();
		void ComputeForceCUDA ();
        void ComputeGenesCUDA ();
        void ComputeBondChangesCUDA ();
        void ComputeParticleChangesCUDA ();
        void CleanBondsCUDA ();                                         // Should this functionality be rolled into countingSortFull() ? OR should it be kept separate? 
        
		void TransferPosVelVeval ();                                    // Called B4 1st timestep, & B4 AdvanceCuda thereafter. 
        void AdvanceCUDA ( float time, float dt, float ss );            // Writes to ftemp 
        void SpecialParticlesCUDA (float tm, float dt, float ss);       // Reads fbuf, writes to ftemp, corects AdvanceCUDA().
		void EmitParticlesCUDA ( float time, int cnt );
        
		// I/O Files
        void SaveUintArray( uint* array, int numElem1, const char * relativePath );
        void SavePointsVTP2 ( const char * relativePath, int frame );
        void SavePointsCSV2 ( const char * relativePath, int frame );
        void ReadSimParams ( const char * relativePath );    // path to folder containing simparams and .csv files
        void WriteDemoSimParams ( const char * relativePath, uint num_particles, float spacing, float x_dim, float y_dim, float z_dim, uint demoType, uint simSpace); // Write standard demo to file, as demonstration of file format. 
        void WriteSimParams ( const char * relativePath );
        void ReadPointsCSV2 ( const char * relativePath, int gpu_mode, int cpu_mode);
        
        // timers
      //  std::chrono::time_point< std::chrono::_V2::steady_clock, std::chrono::duration< long int, std::ratio< 1, 1000000000 > > > begin;

        // Genome for Morphogenesis
        void UpdateGenome ();
        void SetGenome ( FGenome newGenome );
        void ReadGenome( const char * relativePath);
        void WriteGenome( const char * relativePath);
        
		// Parameters
		void UpdateParams ();
		void SetParam (int p, float v );
		void SetParam (int p, int v )		{ m_Param[p] = (float) v; }
		float GetParam ( int p )			{ return (float) m_Param[p]; }
		float* getParamPtr ( int p )		{ return &m_Param[p]; }
		float SetParam ( int p, float v, float mn, float mx )	{ m_Param[p] = v ; if ( m_Param[p] > mx ) m_Param[p] = mn; return m_Param[p];}
		float IncParam ( int p, float v, float mn, float mx )	{ 
			m_Param[p] += v; 
			if ( m_Param[p] < mn ) m_Param[p] = mn; 
			if ( m_Param[p] > mx ) m_Param[p] = mn; 
			return m_Param[p];
		}
		void IncVec ( int p, Vector3DF v )	{ m_Vec[p] += v; }
		Vector3DF GetVec ( int p )			{ return m_Vec[p]; }
		void SetVec ( int p, Vector3DF v );
		void Toggle ( int p )				{ m_Toggle[p] = !m_Toggle[p]; }		
		bool GetToggle ( int p )			{ return m_Toggle[p]; }
		std::string		getSceneName ()		{ return mSceneName; }

		void SetupMode ( bool* cmds, Vector3DI range, std::string inf, std::string outf, std::string wpath, Vector3DI res, int brickres, float th);
		std::string getResolvedName ( bool bIn, int frame );

		CUdeviceptr getBufferGPU ( int id )	{ return m_Fluid.gpu(id); }

		void SetDebug(uint b) { m_debug=b; m_FParams.debug=b; /*mbDebug = (bool)b;*/ }
	
	private:
		Vector3DI					m_FrameRange;
		Vector3DI					m_VolRes;
		int							m_BrkRes;
		std::string					m_InFile;
		std::string					m_OutFile;
		std::string					m_WorkPath;
		float						m_Thresh;
		bool						mbDebug;
        uint                        m_debug;            // 0=full speed, 1=current special output,  2=host cout, 3=device printf, 4=SaveUintArray, 5=save csv after each kernel

		std::string					mSceneName;

		// Time
		int							m_Frame;		
		float						m_DT;
		float						m_Time;	

		// CUDA Kernels
		CUmodule					m_Module;
		CUfunction					m_Func[ FUNC_MAX ];
        
		// Simulation Parameters                                //  NB MAX_PARAM = 50 
		float						m_Param [ MAX_PARAM ];	    // 0-47 used.  see defines above. NB m_Param[1] = maximum number of points.
		Vector3DF					m_Vec   [ MAX_PARAM ];      // 0-12 used 
		bool						m_Toggle[ MAX_PARAM ];		// 0-13 used. 

		// SPH Kernel functions
		float						m_R2, m_Poly6Kern, m_LapKern, m_SpikyKern;		

		// Particle Buffers
		int						mNumPoints;
		int						mMaxPoints;
		int						mActivePoints;
		FBufs					m_Fluid;				// Fluid buffers - NB this is an array of pointers (in mPackBuf ?)
		FBufs					m_FluidTemp;			// Fluid buffers (temporary)
		FParams					m_FParams;				// Fluid parameters struct - that apply to all particles 
		FGenome					m_FGenome;				// Genome struct of arrays for genne params

		CUdeviceptr				cuFBuf;					// GPU pointer containers
		CUdeviceptr				cuFTemp;
		CUdeviceptr				cuFParams;
		CUdeviceptr				cuFGenome;

		// Acceleration Grid		
		int						m_GridTotal;			// total # cells
		Vector3DI				m_GridRes;				// resolution in each axis
		Vector3DF				m_GridMin;				// volume of grid (may not match domain volume exactly)
		Vector3DF				m_GridMax;		
		Vector3DF				m_GridSize;				// physical size in each axis
		Vector3DF				m_GridDelta;
		int						m_GridSrch;
		int						m_GridAdjCnt;
		int						m_GridAdj[216];         // 216 => up to 8 particles per cell

		// Acceleration Neighbor Table
		int						m_NeighborNum;
		int						m_NeighborMax;
		int*					m_NeighborTable;
		float*					m_NeighborDist;

		char*					mPackBuf;               // pointer to array holding the particles ?  - not used ? 
		int*					mPackGrid;

		int						mVBO[3];

		// Record/Playback
		bool					mbRecord;		
		bool					mbRecordBricks;
		int						mSpherePnts;
		int						mTex[1];		

		// Selected particle
		int						mSelected;


		// Saved results (for algorithm validation)
		uint*					mSaveNdx;
		uint*					mSaveCnt;
		uint*					mSaveNeighbors;		
	};	

	

#endif
