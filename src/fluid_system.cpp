
#include <assert.h>
#include <iostream>//<stdio.h>
#include <cuda.h>
#include "cutil_math.h"
#include "fluid_system.h"

    #include <stdlib.h>
    #include <unistd.h>
#include <vtk-9.0/vtkFloatArray.h>
#include <vtk-9.0/vtkPointData.h>



//#include <GL/glew.h>

extern bool gProfileRend;

#define EPSILON			0.00001f			// for collision detection

#define SCAN_BLOCKSIZE		512				// must match value in fluid_system_cuda.cu

bool cuCheck (CUresult launch_stat, const char* method, const char* apicall, const char* arg, bool bDebug)
{
    CUresult kern_stat = CUDA_SUCCESS;

    if (bDebug) {
        kern_stat = cuCtxSynchronize();
    }
    if (kern_stat != CUDA_SUCCESS || launch_stat != CUDA_SUCCESS) {
        const char* launch_statmsg = "";
        const char* kern_statmsg = "";
        cuGetErrorString(launch_stat, &launch_statmsg);
        cuGetErrorString(kern_stat, &kern_statmsg);
        std::cout << "FLUID SYSTEM, CUDA ERROR:\t";
        std::cout << " Launch status: "<< launch_statmsg <<"\t";
        std::cout << " Kernel status: "<< kern_statmsg <<"\t";
        std::cout << " Caller: FluidSystem::"<<  method <<"\t";
        std::cout << " Call:   "<< apicall <<"\t";
        std::cout << " Args:   "<< arg <<"\n";

        if (bDebug) {
            std::cout << "  Generating assert to examine call stack.\n" ;
            assert(0);		// debug - trigger break (see call stack)
        }
        else {
            std::cout << "fluid_system.cpp line 40, 'nverror()' ";
            //nverror();		// exit - return 0
        }
        return false;
    }
    return true;

}
//////////////////////////////////////////////////
FluidSystem::FluidSystem ()
{
    mNumPoints = 0;
    mMaxPoints = 0;
/*remove this line ?*/mPackBuf = 0x0;   // pointer to the array of particles ?  not used ?
    mPackGrid = 0x0;
    mbRecord = false;
    mbRecordBricks = false;
    mSelected = -1;
    m_Frame = 0;
    m_Thresh = 0;
    m_NeighborTable = 0x0;
    m_NeighborDist = 0x0;
    for (int n=0; n < FUNC_MAX; n++ ) m_Func[n] = (CUfunction) -1;
    m_Toggle [ PDEBUG ]		=	false;
    m_Toggle [ PUSE_GRID ]	=	false;
    m_Toggle [ PPROFILE ]	=	false;
    m_Toggle [ PCAPTURE ]   =	false;
}

void FluidSystem::LoadKernel ( int fid, std::string func )
{
    char cfn[512];
    strcpy ( cfn, func.c_str() );

    if ( m_Func[fid] == (CUfunction) -1 )
        cuCheck ( cuModuleGetFunction ( &m_Func[fid], m_Module, cfn ), "LoadKernel", "cuModuleGetFunction", cfn, mbDebug );
}

// Must have a CUDA context to initialize
void FluidSystem::Initialize ()     // used for CPU only for "check_demo".
{
    std::cout << "FluidSystem::Initialize () \n";
    /// debug chk : what directory are we in ?
    system("pwd");

    // An FBufs struct holds an array of pointers.
    std::cout << "Chk1.3 \n";
    // Clear all buffers
    memset ( &m_Fluid, 0,		sizeof(FBufs) );
    memset ( &m_FluidTemp, 0,	sizeof(FBufs) );
    memset ( &m_FParams, 0,		sizeof(FParams) );
    memset ( &m_FGenome, 0,		sizeof(FGenome) );

    std::cout << "Chk1.4 \n";
    // Allocate the sim parameters
    AllocateBuffer ( FPARAMS,		sizeof(FParams),	0,	1,	 GPU_OFF,     CPU_YES );//AllocateBuffer ( int buf_id, int stride,     int cpucnt, int gpucnt,    int gpumode,    int cpumode )
    std::cout << "Chk1.5 \n";
    m_Time = 0;
    ClearNeighborTable ();
    mNumPoints = 0;			// reset count
    std::cout << "Chk1.6 \n";
}

// /home/nick/Programming/Cuda/Morphogenesis/build/install/ptx/objects/fluid_systemPTX/fluid_system_cuda.ptx
void FluidSystem::InitializeCuda ()         // used for load_sim  /home/nick/Programming/Cuda/Morphogenesis/build/install/ptx/objects-Debug/fluid_systemPTX/fluid_system_cuda.ptx
{
    std::cout << "FluidSystem::InitializeCuda () \n";
    cuCheck ( cuModuleLoad ( &m_Module, /*"fluid_system_cuda.ptx"*/ "fluid_system_cuda.ptx" ), "LoadKernel", "cuModuleLoad", "fluid_system_cuda.ptx", mbDebug);  
    // loads the file "fluid_system_cuda.ptx" as a module with pointer  m_Module.

    std::cout << "Chk1.1 \n";
    LoadKernel ( FUNC_INSERT,			"insertParticles" );
    LoadKernel ( FUNC_COUNTING_SORT,	"countingSortFull" );
//    LoadKernel ( FUNC_SORT_BONDIDX,	    "countingSortBondIDX" );
    LoadKernel ( FUNC_QUERY,			"computeQuery" );
    LoadKernel ( FUNC_COMPUTE_PRESS,	"computePressure" );
    LoadKernel ( FUNC_COMPUTE_FORCE,	"computeForce" );
    LoadKernel ( FUNC_ADVANCE,			"advanceParticles" );
    LoadKernel ( FUNC_EMIT,				"emitParticles" );
    LoadKernel ( FUNC_RANDOMIZE,		"randomInit" );
    LoadKernel ( FUNC_SAMPLE,			"sampleParticles" );
    LoadKernel ( FUNC_FPREFIXSUM,		"prefixSum" );
    LoadKernel ( FUNC_FPREFIXFIXUP,		"prefixFixup" );
    LoadKernel ( FUNC_FREEZE,		    "freeze" );
    LoadKernel ( FUNC_COMPUTE_DIFFUSION,"computeDiffusion");

    std::cout << "Chk1.2 \n";
    size_t len = 0;
    cuCheck ( cuModuleGetGlobal ( &cuFBuf, &len,		m_Module, "fbuf" ),		"LoadKernel", "cuModuleGetGlobal", "cuFBuf", mbDebug);      // Returns a global pointer (cuFBuf) from a module  (m_Module), see line 81.
    cuCheck ( cuModuleGetGlobal ( &cuFTemp, &len,		m_Module, "ftemp" ),	"LoadKernel", "cuModuleGetGlobal", "cuFTemp", mbDebug);     // fbuf, ftemp, fparam are defined at top of fluid_system_cuda.cu,
    cuCheck ( cuModuleGetGlobal ( &cuFParams, &len,	m_Module, "fparam" ),		"LoadKernel", "cuModuleGetGlobal", "cuFParams", mbDebug);   // based on structs "FParams", "FBufs", "FGenome" defined in fluid.h
    cuCheck ( cuModuleGetGlobal ( &cuFGenome, &len,	m_Module, "fgenome" ),		"LoadKernel", "cuModuleGetGlobal", "cuFGenome", mbDebug);   // NB defined differently in kernel vs cpu code.
    // An FBufs struct holds an array of pointers.
    std::cout << "Chk1.3 \n";
    // Clear all buffers
    memset ( &m_Fluid, 0,		sizeof(FBufs) );
    memset ( &m_FluidTemp, 0,	sizeof(FBufs) );
    memset ( &m_FParams, 0,		sizeof(FParams) );
    memset ( &m_FGenome, 0,		sizeof(FGenome) );

    std::cout << "Chk1.4 \n";
    // Allocate the sim parameters
    AllocateBuffer ( FPARAMS,		sizeof(FParams),	0,	1,	 GPU_SINGLE,     CPU_OFF );//AllocateBuffer ( int buf_id, int stride,     int cpucnt, int gpucnt,    int gpumode,    int cpumode )
    std::cout << "Chk1.5 \n";
    m_Time = 0;
    ClearNeighborTable ();
    mNumPoints = 0;			// reset count
    std::cout << "Chk1.6 \n";
}




void FluidSystem::Start ( int num )     // #### creates the particles ####
{
    std::cout << "FluidSystem::Start ( "<< num <<" ) \n";
std::cout << "Chk2.0\n"<<std::flush;
#ifdef TEST_PREFIXSUM
    TestPrefixSum ( 16*1024*1024 );
    exit(-2);
#endif

    m_Time = 0;
    ClearNeighborTable ();
    mNumPoints = 0;			// reset count
std::cout << "Chk2.1\n"<<std::flush;
    SetupDefaultParams ();
std::cout << "Chk2.2\n"<<std::flush;
    SetupExampleParams ();
std::cout << "Chk2.3\n"<<std::flush;
    m_Param[PNUM]      = (float) num;	// maximum number of points
    mMaxPoints         = num;
    m_Param [PGRIDSIZE] = 2*m_Param[PSMOOTHRADIUS] / m_Param[PGRID_DENSITY];

    // Setup stuff
std::cout << "Chk2.4\n"<<std::flush;
/*remove this line ?*/    AllocatePackBuf ();   // allocates mPackBuf (particle buffer ?) ...  _not_used_. Particles are in m_Fluid.bufC(pos/vel/veval/force/press/density/clr)
std::cout << "Chk2.5\n"<<std::flush;
    SetupKernels ();
std::cout << "Chk2.6\n"<<std::flush;
    SetupSpacing ();
std::cout << "Chk2.7\n"<<std::flush;
    SetupGrid ( m_Vec[PVOLMIN], m_Vec[PVOLMAX], m_Param[PSIMSCALE], m_Param[PGRIDSIZE], 1.0f );	// Setup grid
std::cout << "Chk2.8\n"<<std::flush;
    FluidSetupCUDA ( mMaxPoints, m_GridSrch, *(int3*)& m_GridRes, *(float3*)& m_GridSize, *(float3*)& m_GridDelta, *(float3*)& m_GridMin, *(float3*)& m_GridMax, m_GridTotal, 0 );
std::cout << "Chk2.9\n"<<std::flush;
    UpdateParams();            //  sends simulation params to device.
std::cout << "Chk2.10\n"<<std::flush;
    UpdateGenome();            //  sends genome to device.              // NB need to initialize genome from file, or something.
std::cout << "Chk2.11\n"<<std::flush;
    // Allocate data
    AllocateParticles( mMaxPoints );  // allocates cpu, gpu and gpu_temp buffers for particles
std::cout << "Chk2.12\n"<<std::flush;
    AllocateGrid();
std::cout << "Chk2.13\n"<<std::flush;
    // Create the particles (after allocate)
    SetupAddVolume(m_Vec[PINITMIN], m_Vec[PINITMAX], m_Param[PSPACING], 0.1f, (int)m_Param[PNUM]);		// increases mNumPoints, defines array of particles in 3D space
std::cout << "Chk2.14\n"<<std::flush;
    TransferToCUDA ();		 // Initial transfer
    std::cout << "FluidSystem::Start()  finished \n"<<std::flush;
}



/////////////////////////////////////////////////////////////////
void FluidSystem::UpdateGenome ()   // Update Genome on GPU
{
    cuCheck ( cuMemcpyHtoD ( cuFGenome,	&m_FGenome,		sizeof(FGenome) ), "FluidGenomeCUDA", "cuMemcpyHtoD", "cuFGenome", mbDebug);
}

void FluidSystem::SetGenome (FGenome newGenome )
{
    for(int i=0; i< NUM_GENES; i++) m_FGenome.mutability[i] = newGenome.mutability[i];
    for(int i=0; i< NUM_GENES; i++) m_FGenome.delay[i] = newGenome.delay[i];
    for(int i=0; i< NUM_GENES; i++) for(int j=0; j< NUM_GENES; j++) {
            m_FGenome.sensitivity[i][j] = newGenome.sensitivity[i][j];
        }
    for(int i=0; i< NUM_GENES; i++) {
        m_FGenome.difusability[i][0] = newGenome.difusability[i][0];
        m_FGenome.difusability[i][1] = newGenome.difusability[i][1];
    }
}


void FluidSystem::UpdateParams ()
{
    // Update Params on GPU
    Vector3DF grav = m_Vec[PPLANE_GRAV_DIR] * m_Param[PGRAV];
    FluidParamCUDA (  m_Param[PSIMSCALE], m_Param[PSMOOTHRADIUS], m_Param[PRADIUS], m_Param[PMASS], m_Param[PRESTDENSITY],
                      *(float3*)& m_Vec[PBOUNDMIN], *(float3*)& m_Vec[PBOUNDMAX], m_Param[PEXTSTIFF], m_Param[PINTSTIFF],
                      m_Param[PVISC], m_Param[PEXTDAMP], m_Param[PFORCE_MIN], m_Param[PFORCE_MAX], m_Param[PFORCE_FREQ],
                      m_Param[PGROUND_SLOPE], grav.x, grav.y, grav.z, m_Param[PACCEL_LIMIT], m_Param[PVEL_LIMIT],
                      (int) m_Vec[PEMIT_RATE].x );
}

void FluidSystem::SetParam (int p, float v )
{
    m_Param[p] = v;
    UpdateParams ();
}

void FluidSystem::SetVec ( int p, Vector3DF v )
{
    m_Vec[p] = v;
    UpdateParams ();
}

void FluidSystem::Exit ()
{
    std::cout << "starting FluidSystem::Exit (),  freeing buffers,  MAX_BUF = " << MAX_BUF << std::flush;
    // Free fluid buffers
    for (int n=0; n < MAX_BUF; n++ ) {
        std::cout << "\n n = " << n << std::flush;
        if ( m_Fluid.bufC(n) != 0x0 )
            free ( m_Fluid.bufC(n) );
    }

    //cudaExit ();
}

void FluidSystem::AllocateBuffer ( int buf_id, int stride, int cpucnt, int gpucnt, int gpumode, int cpumode )   // mallocs a buffer - called by FluidSystem::Initialize(), AllocateParticles, and AllocateGrid()
//also called by WriteDemoSimParams(..)
{
std::cout<<"\nAllocateBuffer ( int buf_id="<<buf_id<<", int stride="<<stride<<", int cpucnt="<<cpucnt<<", int gpucnt="<<gpucnt<<", int "<<gpumode<<", int "<<cpumode<<" )\t"<<std::flush;
    if (cpumode == CPU_YES) {
std::cout<<" cpumode == CPU_YES \t"<<std::flush;
        char* src_buf = m_Fluid.bufC(buf_id);
        char* dest_buf = (char*) malloc(cpucnt*stride);                   //  ####  malloc the buffer   ####
std::cout<<"  , dest_buf="<<static_cast<void*>(dest_buf)<<"\t"<<std::flush; // src_buf="<<src_buf<<"
        if (src_buf != 0x0) {
            memcpy(dest_buf, src_buf, cpucnt*stride);
            free(src_buf);
        }
        m_Fluid.setBuf(buf_id, dest_buf);                                 // stores pointer to buffer in mcpu[buf_id]
std::cout<<" mcpu["<<buf_id<<"]="<<static_cast<void*>(m_Fluid.mcpu[buf_id])<<"\t"<<std::flush;
    }

    if (gpumode == GPU_SINGLE || gpumode == GPU_DUAL )	{
std::cout<<"cuMemFree(m_Fluid.gpu("<<buf_id<<"))\t"<<std::flush;
        if (m_Fluid.gpuptr(buf_id) != 0x0) cuCheck(cuMemFree(m_Fluid.gpu(buf_id)), "AllocateBuffer", "cuMemFree", "Fluid.gpu", mbDebug);
std::cout<<"cuMemAlloc(m_Fluid.gpuptr("<<buf_id<<"), "<<stride*gpucnt<<")\t"<<std::flush;
        cuCheck( cuMemAlloc(m_Fluid.gpuptr(buf_id), stride*gpucnt), "AllocateBuffer", "cuMemAlloc", "Fluid.gpu", mbDebug);         //  ####  cuMemAlloc the buffer, stores pointer to buffer in   m_Fluid.mgpu[buf_id]
    }
    if (gpumode == GPU_TEMP || gpumode == GPU_DUAL ) {
std::cout<<"cuMemFree(m_FluidTemp.gpu("<<buf_id<<"))\t"<<std::flush;
        if (m_FluidTemp.gpuptr(buf_id) != 0x0) cuCheck(cuMemFree(m_FluidTemp.gpu(buf_id)), "AllocateBuffer", "cuMemFree", "FluidTemp.gpu", mbDebug);
        cuCheck( cuMemAlloc(m_FluidTemp.gpuptr(buf_id), stride*gpucnt), "AllocateBuffer", "cuMemAlloc", "FluidTemp.gpu", mbDebug); //  ####  cuMemAlloc the buffer, stores pointer to buffer in   m_FluidTemp.mgpu[buf_id]
    }
}

// Allocate particle memory
void FluidSystem::AllocateParticles ( int cnt, int gpu_mode, int cpu_mode ) // calls AllocateBuffer(..) for each buffer.  
// Defaults in header : int gpu_mode = GPU_DUAL, int cpu_mode = CPU_YES
// Called by FluidSystem::ReadPointsCSV(..), and FluidSystem::WriteDemoSimParams(...), cnt = mMaxPoints.
// Called with default values by FluidSystem::Start(..), cnt = mMaxPoints.
{
std::cout<<"AllocateParticles ( int cnt="<<cnt<<", int "<<gpu_mode<<", int "<<cpu_mode<<" )\n"<<std::flush;
std::cout<<" GPU_OFF=0, GPU_SINGLE=1, GPU_TEMP=2, GPU_DUAL=3, CPU_OFF=4, CPU_YES=5"<<std::flush;
    AllocateBuffer ( FPOS,		sizeof(Vector3DF),	cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );//AllocateBuffer ( int buf_id, int stride,     int cpucnt, int gpucnt,    int gpumode,    int cpumode )
std::cout<<" m_Fluid.gpu(FPOS)="<< m_Fluid.gpu(FPOS)<<"\tm_Fluid.bufC(FPOS)="<< static_cast<void*>(m_Fluid.bufC(FPOS))<<"\n"<<std::flush;
    AllocateBuffer ( FCLR,		sizeof(uint),		cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FVEL,		sizeof(Vector3DF),	cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );// NB gpu_mode means create buffers for _both_  "m_Fluid.mgpu[buf_id]" and "m_FluidTemp.mgpu[buf_id]"
    AllocateBuffer ( FVEVAL,	sizeof(Vector3DF),	cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FAGE,		sizeof(uint/*unsigned short*/), cnt,m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FPRESS,	sizeof(float),		cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FDENSITY,	sizeof(float),		cnt, 	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FFORCE,	sizeof(Vector3DF),	cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FCLUSTER,	sizeof(uint),		cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FGCELL,	sizeof(uint),		cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FGNDX,		sizeof(uint),		cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FGNEXT,	sizeof(uint),		cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FNBRNDX,	sizeof(uint),		cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FNBRCNT,	sizeof(uint),		cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FSTATE,	sizeof(uint),		cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    // extra buffers for morphogenesis
std::cout<<"chl2.11.1\n"<<std::flush;
    AllocateBuffer ( FELASTIDX,	sizeof(uint[BOND_DATA]),             cnt,   m_FParams.szPnts,	gpu_mode, cpu_mode );  // used to be [BONDS_PER_PARTICLE * 2]
    AllocateBuffer ( FPARTICLEIDX,	sizeof(uint[BONDS_PER_PARTICLE *2]),cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FPARTICLE_ID,	sizeof(uint),		             cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FMASS_RADIUS,	sizeof(uint),		             cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    
    AllocateBuffer ( FNERVEIDX,	sizeof(uint),		                 cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FCONC,	    sizeof(float[NUM_TF]),		         cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FEPIGEN,	sizeof(uint[NUM_GENES]),	         cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
std::cout<<"chl2.11.2\n"<<std::flush;

    // Update GPU access pointers
    if (gpu_mode != GPU_OFF ) {
        cuCheck( cuMemcpyHtoD(cuFBuf, &m_Fluid, sizeof(FBufs)),			"AllocateParticles", "cuMemcpyHtoD", "cuFBuf", mbDebug);
        cuCheck( cuMemcpyHtoD(cuFTemp, &m_FluidTemp, sizeof(FBufs)),	"AllocateParticles", "cuMemcpyHtoD", "cuFTemp", mbDebug);
        cuCheck( cuMemcpyHtoD(cuFParams, &m_FParams, sizeof(FParams)),  "AllocateParticles", "cuMemcpyHtoD", "cuFParams", mbDebug);
        cuCheck( cuMemcpyHtoD(cuFGenome, &m_FGenome, sizeof(FGenome)),  "AllocateParticles", "cuMemcpyHtoD", "cuFGenome", mbDebug);
        cuCheck(cuCtxSynchronize(), "AllocateParticles", "cuCtxSynchronize", "", mbDebug );
    }

    m_Param[PSTAT_PMEM] = 68.0f * 2 * cnt;

    // Allocate auxiliary buffers (prefix sums)
    int blockSize = SCAN_BLOCKSIZE << 1;
    int numElem1 = m_GridTotal;
    int numElem2 = int ( numElem1 / blockSize ) + 1;
    int numElem3 = int ( numElem2 / blockSize ) + 1;

    if (gpu_mode != GPU_OFF ) {
        AllocateBuffer ( FAUXARRAY1,	sizeof(uint),		0,	numElem2, GPU_SINGLE, CPU_OFF );
        AllocateBuffer ( FAUXSCAN1,	    sizeof(uint),		0,	numElem2, GPU_SINGLE, CPU_OFF );
        AllocateBuffer ( FAUXARRAY2,	sizeof(uint),		0,	numElem3, GPU_SINGLE, CPU_OFF );
        AllocateBuffer ( FAUXSCAN2,	    sizeof(uint),		0,	numElem3, GPU_SINGLE, CPU_OFF );
    }
}


void FluidSystem::AllocateGrid()
{
    // Allocate grid
    int cnt = m_GridTotal;
    m_FParams.szGrid = (m_FParams.gridBlocks * m_FParams.gridThreads);
    AllocateBuffer ( FGRID,		sizeof(uint),		mMaxPoints,	m_FParams.szPnts,	GPU_SINGLE, CPU_YES );    // # grid elements = number of points
    AllocateBuffer ( FGRIDCNT,	sizeof(uint),		cnt,	m_FParams.szGrid,	GPU_SINGLE, CPU_YES );
    AllocateBuffer ( FGRIDOFF,	sizeof(uint),		cnt,	m_FParams.szGrid,	GPU_SINGLE, CPU_YES );
    AllocateBuffer ( FGRIDACT,	sizeof(uint),		cnt,	m_FParams.szGrid,	GPU_SINGLE, CPU_YES );

    // Update GPU access pointers
    cuCheck(cuMemcpyHtoD(cuFBuf, &m_Fluid, sizeof(FBufs)), "AllocateGrid", "cuMemcpyHtoD", "cuFBuf", mbDebug);
    cuCheck(cuCtxSynchronize(), "AllocateParticles", "cuCtxSynchronize", "", mbDebug);
}


void FluidSystem::AllocateGrid(int gpu_mode, int cpu_mode)
{
    // Allocate grid
    int cnt = m_GridTotal;
    m_FParams.szGrid = (m_FParams.gridBlocks * m_FParams.gridThreads);
    AllocateBuffer ( FGRID,		sizeof(uint),		mMaxPoints,	m_FParams.szPnts,	gpu_mode, cpu_mode );    // # grid elements = number of points
    AllocateBuffer ( FGRIDCNT,	sizeof(uint),		cnt,	m_FParams.szGrid,	gpu_mode, cpu_mode );
    AllocateBuffer ( FGRIDOFF,	sizeof(uint),		cnt,	m_FParams.szGrid,	gpu_mode, cpu_mode );
    AllocateBuffer ( FGRIDACT,	sizeof(uint),		cnt,	m_FParams.szGrid,	gpu_mode, cpu_mode );

    // Update GPU access pointers
    if (gpu_mode != GPU_OFF ) {
        cuCheck(cuMemcpyHtoD(cuFBuf, &m_Fluid, sizeof(FBufs)), "AllocateGrid", "cuMemcpyHtoD", "cuFBuf", mbDebug);
        cuCheck(cuCtxSynchronize(), "AllocateParticles", "cuCtxSynchronize", "", mbDebug);
    }
}


int FluidSystem::AddParticle ()
{
    if ( mNumPoints >= mMaxPoints ) return -1;
    int n = mNumPoints;
    (m_Fluid.bufV3(FPOS) + n)->Set ( 0,0,0 );
    (m_Fluid.bufV3(FVEL) + n)->Set ( 0,0,0 );
    (m_Fluid.bufV3(FVEVAL) + n)->Set ( 0,0,0 );
    (m_Fluid.bufV3(FFORCE) + n)->Set ( 0,0,0 );
    *(m_Fluid.bufF(FPRESS) + n) = 0;
    *(m_Fluid.bufF(FDENSITY) + n) = 0;
    *(m_Fluid.bufI(FGNEXT) + n) = -1;
    *(m_Fluid.bufI(FCLUSTER)  + n) = -1;
    *(m_Fluid.bufF(FSTATE) + n ) = (float) rand();
    *(m_Fluid.bufI(FPARTICLE_ID)+n) = n+1; 
    mNumPoints++;
    return n;
}

int FluidSystem::AddParticle (Vector3DF* Pos, Vector3DF* Vel)
{
    if ( mNumPoints >= mMaxPoints ) return -1;
    int n = mNumPoints;
    (m_Fluid.bufV3(FPOS) + n)->Set ( Pos->x,Pos->y,Pos->z );
    (m_Fluid.bufV3(FVEL) + n)->Set ( Vel->x,Vel->y,Vel->z );
    (m_Fluid.bufV3(FVEVAL) + n)->Set ( 0,0,0 );
    (m_Fluid.bufV3(FFORCE) + n)->Set ( 0,0,0 );
    *(m_Fluid.bufF(FPRESS) + n) = 0;
    *(m_Fluid.bufF(FDENSITY) + n) = 0;
    *(m_Fluid.bufI(FGNEXT) + n) = -1;
    *(m_Fluid.bufI(FCLUSTER)  + n) = -1;
    *(m_Fluid.bufF(FSTATE) + n ) = (float) rand();
    *(m_Fluid.bufI(FPARTICLE_ID)+n) = n+1;
    mNumPoints++;
    return n;
}

int FluidSystem::AddParticleMorphogenesis ()
{
    //std::cout << "\n AddParticleMorphogenesis () " << std::flush ;
    
    if ( mNumPoints >= mMaxPoints ) return -1;
    int n = mNumPoints;
    (m_Fluid.bufV3(FPOS) + n)->Set ( 0,0,0 );
    (m_Fluid.bufV3(FVEL) + n)->Set ( 0,0,0 );
    (m_Fluid.bufV3(FVEVAL) + n)->Set ( 0,0,0 );
    (m_Fluid.bufV3(FFORCE) + n)->Set ( 0,0,0 );
    *(m_Fluid.bufF(FPRESS) + n) = 0;
    *(m_Fluid.bufF(FDENSITY) + n) = 0;
    *(m_Fluid.bufI(FAGE) + n) = 0;                  //# Ushort 16  particle age
    *(m_Fluid.bufI(FGNEXT) + n) = -1;
    *(m_Fluid.bufI(FCLUSTER)  + n) = -1;
    *(m_Fluid.bufF(FSTATE) + n ) = (float) rand();

    //#define FELASTIDX   14      //# uint[BONDS_PER_PARTICLE +1]  0=self UID, mass, radius. >0= modulus & particle UID
    //#define FNERVEIDX   15      //# uint
    //#define FCONC       16      //# float[NUM_TF]        NUM_TF = num transcription factors & morphogens
    //#define FEPIGEN     17      //# uint[NUM_GENES]

    //*(m_Fluid.bufI(FAGE) + n) = 0;
    *(m_Fluid.bufI(FCLR) + n) = 0;

    uint* ElastIdx = (m_Fluid.bufI(FELASTIDX) + n * BOND_DATA);
    ElastIdx[0] = n;                                        // ElastIdx[0] = particle ID
    for(int j=1; j<(BOND_DATA); j++) {
        ElastIdx[j] = 0;
    }

    *(m_Fluid.bufI(FNERVEIDX) + n) = 0;

    float* Conc = (m_Fluid.bufF(FCONC) + n * NUM_TF);
    for(int j=0; j<(NUM_TF); j++) {
        Conc[j] = 0.0f;
    }

    uint* EpiGen = (m_Fluid.bufI(FEPIGEN) + n * NUM_GENES);
    for(int j=0; j<(NUM_GENES); j++) {
        EpiGen[j]= 0;
    }
    *(m_Fluid.bufI(FPARTICLE_ID)+n) = n+1;
    
    mNumPoints++;
    return n;
}


int FluidSystem::AddParticleMorphogenesis (Vector3DF* Pos, Vector3DF* Vel, uint Age, uint Clr, uint *_ElastIdx, uint NerveIdx, /* Particle_ID, mass, radius, */ float* _Conc, uint* _EpiGen )
{
    if ( mNumPoints >= mMaxPoints ) return -1;
    int n = mNumPoints;
    (m_Fluid.bufV3(FPOS) + n)->Set ( Pos->x,Pos->y,Pos->z );
    (m_Fluid.bufV3(FVEL) + n)->Set ( Vel->x,Vel->y,Vel->z );
    (m_Fluid.bufV3(FVEVAL) + n)->Set ( 0,0,0 );
    (m_Fluid.bufV3(FFORCE) + n)->Set ( 0,0,0 );
    *(m_Fluid.bufF(FPRESS) + n) = 0;
    *(m_Fluid.bufF(FDENSITY) + n) = 0;
    *(m_Fluid.bufI(FGNEXT) + n) = -1;
    *(m_Fluid.bufI(FCLUSTER)  + n) = -1;
    *(m_Fluid.bufF(FSTATE) + n ) = (float) rand();

    *(m_Fluid.bufI(FAGE) + n) = Age;
    *(m_Fluid.bufI(FCLR) + n) = Clr;

    uint* ElastIdx = (m_Fluid.bufI(FELASTIDX) + n * BOND_DATA );
    for(int j=0; j<(BOND_DATA); j++) {
        ElastIdx[j] = _ElastIdx[j] ;
    }
    
    //fprintf(fp, "\t%u, \t", *NerveIdx);
    *(m_Fluid.bufI(FNERVEIDX) + n) = NerveIdx;

    float* Conc = (m_Fluid.bufF(FCONC) + n * NUM_TF);
    for(int j=0; j<(NUM_TF); j++) {
        Conc[j] = _Conc[j];
    }
    uint* EpiGen = (m_Fluid.bufI(FEPIGEN) + n * NUM_GENES);
    for(int j=0; j<(NUM_GENES); j++) { EpiGen[j]= _EpiGen[j];}
    
    *(m_Fluid.bufI(FPARTICLE_ID)+n) = n+1;

    mNumPoints++;
    return n;
}


int FluidSystem::AddParticleMorphogenesis2 (Vector3DF* Pos, Vector3DF* Vel, uint Age, uint Clr, float *_ElastIdx, uint *_Particle_Idx, uint Particle_ID, uint Mass_Radius, uint NerveIdx, float* _Conc, uint* _EpiGen )  // called by :ReadPointsCSV2 (...) where :    uint Particle_Idx[BONDS_PER_PARTICLE * 2];  AND SetupAddVolumeMorphogenesis2(....)
{
    if ( mNumPoints >= mMaxPoints ) return -1;
    int n = mNumPoints;
    (m_Fluid.bufV3(FPOS) + n)->Set ( Pos->x,Pos->y,Pos->z );
    (m_Fluid.bufV3(FVEL) + n)->Set ( Vel->x,Vel->y,Vel->z );
    (m_Fluid.bufV3(FVEVAL) + n)->Set ( 0,0,0 );
    (m_Fluid.bufV3(FFORCE) + n)->Set ( 0,0,0 );
    *(m_Fluid.bufF(FPRESS) + n) = 0;
    *(m_Fluid.bufF(FDENSITY) + n) = 0;
    *(m_Fluid.bufI(FGNEXT) + n) = -1;
    *(m_Fluid.bufI(FCLUSTER)  + n) = -1;
    *(m_Fluid.bufF(FSTATE) + n ) = (float) rand();
    *(m_Fluid.bufI(FAGE) + n) = Age;
    *(m_Fluid.bufI(FCLR) + n) = Clr;

    uint* ElastIdx = (m_Fluid.bufI(FELASTIDX) + n * BOND_DATA );
    for(int j=0; j<(BOND_DATA); j++) {
        ElastIdx[j] = _ElastIdx[j] ;
    }
    uint* Particle_Idx = (m_Fluid.bufI(FPARTICLEIDX) + n * BONDS_PER_PARTICLE *2 );     // index of incoming bonds
    for(int j=0; j<(BONDS_PER_PARTICLE *2); j++) {
        Particle_Idx[j] = _Particle_Idx[j] ;
        std::cout<<"."<<Particle_Idx[j];
    }std::cout<<std::flush;
    *(m_Fluid.bufI(FPARTICLE_ID) + n)   = Particle_ID;                                  // permanent ID of particle 
    *(m_Fluid.bufI(FMASS_RADIUS) + n)   = Mass_Radius;
    *(m_Fluid.bufI(FNERVEIDX) + n)      = NerveIdx;

    float* Conc = (m_Fluid.bufF(FCONC) + n * NUM_TF);
    for(int j=0; j<(NUM_TF); j++) {
        Conc[j] = _Conc[j];
    }
    uint* EpiGen = (m_Fluid.bufI(FEPIGEN) + n * NUM_GENES);
    for(int j=0; j<(NUM_GENES); j++) {
        EpiGen[j]= _EpiGen[j];
    }
    mNumPoints++;
//std::cout << ", Particle_ID="<<Particle_ID<<" " << std::flush ;
    return n;
}


void FluidSystem::SetupAddVolume ( Vector3DF min, Vector3DF max, float spacing, float offs, int total )
{
    Vector3DF pos;
    int p;
    float dx, dy, dz;
    int cntx, cntz;
    cntx = (int) ceil( (max.x-min.x-offs) / spacing );
    cntz = (int) ceil( (max.z-min.z-offs) / spacing );
    int cnt = cntx * cntz;
    int c2;

    min += offs;
    max -= offs;

    dx = max.x-min.x;
    dy = max.y-min.y;
    dz = max.z-min.z;

    Vector3DF rnd;

    c2 = cnt/2;
    for (pos.y = min.y; pos.y <= max.y; pos.y += spacing ) {
        for (int xz=0; xz < cnt; xz++ ) {

            pos.x = min.x + (xz % int(cntx))*spacing;
            pos.z = min.z + (xz / int(cntx))*spacing;
            p = AddParticle ();

            if ( p != -1 ) {
                rnd.Random ( 0, spacing, 0, spacing, 0, spacing );
                *(m_Fluid.bufV3(FPOS)+p) = pos + rnd;

                Vector3DF clr ( (pos.x-min.x)/dx, 0, (pos.z-min.z)/dz );
                clr *= 0.8;
                clr += 0.2;
                clr.Clamp (0, 1.0);
                m_Fluid.bufI(FCLR) [p] = COLORA( clr.x, clr.y, clr.z, 1);
                // = COLORA( 0.25, +0.25 + (y-min.y)*.75/dy, 0.25 + (z-min.z)*.75/dz, 1);  // (x-min.x)/dx
            }
        }
    }
}

void FluidSystem::SetupAddVolumeMorphogenesis(Vector3DF min, Vector3DF max, float spacing, float offs, int total )  // NB ony used in WriteDemoSimParams()
{
    Vector3DF pos;
    int p;
    float dx, dy, dz;
    int cntx, cntz;
    cntx = (int) ceil( (max.x-min.x-offs) / spacing );
    cntz = (int) ceil( (max.z-min.z-offs) / spacing );
    int cnt = cntx * cntz;
    int c2;

    min += offs;
    max -= offs;

    dx = max.x-min.x;
    dy = max.y-min.y;
    dz = max.z-min.z;

    Vector3DF rnd;

    c2 = cnt/2;
    
    std::cout << " chk1.2.1 " << std::flush ;
    for (pos.y = min.y; pos.y <= max.y; pos.y += spacing ) {
        
        for (int xz=0; xz < cnt; xz++ ) {
            
            pos.x = min.x + (xz % int(cntx))*spacing;
            pos.z = min.z + (xz / int(cntx))*spacing;
            p = AddParticleMorphogenesis (); // AddParticle (); //

            if ( p != -1 ) {
                rnd.Random ( 0, spacing, 0, spacing, 0, spacing );
                *(m_Fluid.bufV3(FPOS)+p) = pos + rnd;

                Vector3DF clr ( (pos.x-min.x)/dx, 0, (pos.z-min.z)/dz );
                clr *= 0.8;
                clr += 0.2;
                clr.Clamp (0, 1.0);
                m_Fluid.bufI(FCLR) [p] = COLORA( clr.x, clr.y, clr.z, 1);
                // = COLORA( 0.25, +0.25 + (y-min.y)*.75/dy, 0.25 + (z-min.z)*.75/dz, 1);  // (x-min.x)/dx
            }
        }
    }
}

void FluidSystem::SetupAddVolumeMorphogenesis2(Vector3DF min, Vector3DF max, float spacing, float offs, int total )  // NB ony used in WriteDemoSimParams() called by make_demo.cpp . Creates a cuboid with all particle values definable.
{
std::cout << "\n SetupAddVolumeMorphogenesis2 \t" << std::flush ;
    Vector3DF pos;
    int p;
    float dx, dy, dz;
    int cntx, cntz;
    cntx = (int) ceil( (max.x-min.x-offs) / spacing );
    cntz = (int) ceil( (max.z-min.z-offs) / spacing );
    int cnt = cntx * cntz;
    int c2;

    min += offs;
    max -= offs;

    dx = max.x-min.x;
    dy = max.y-min.y;
    dz = max.z-min.z;

    Vector3DF rnd;

    c2 = cnt/2;
    
    Vector3DF Pos, Vel; 
    uint Age, Clr, Particle_ID, Mass_Radius, /*mass, radius,*/ NerveIdx;
    float ElastIdx[BOND_DATA];
    uint Particle_Idx[BONDS_PER_PARTICLE*2]; // FPARTICLE_IDX : other particles with incoming bonds attaching here. 
    float Conc[NUM_TF];
    uint EpiGen[NUM_GENES];
    
    Particle_ID = 0; // NB Particle_ID=0 means "no particle" in ElastIdx.
    
    for (Pos.x = min.x; Pos.x <= max.x; Pos.x += spacing ) {
        for (Pos.y = min.y; Pos.y <= max.y; Pos.y += spacing){
            for (Pos.z = min.z; Pos.z <= max.z; Pos.z += spacing){     //for (int xz=0; xz < cnt; xz++ ) {
                Particle_ID ++;  // NB AddParticleMorphogenesis2(...) checks not to exceed max num particles
                Vel.x=0; Vel.y=0; Vel.z=0; 
                Age =  0; 
                
                // Colour of particles
                Vector3DF clr ( (pos.x-min.x)/dx, 0, (pos.z-min.z)/dz );
                clr *= 0.8;
                clr += 0.2;
                clr.Clamp (0, 1.0);
                Clr = COLORA( clr.x, clr.y, clr.z, 1);
                
                // Modulus & length of elastic bonds
                // 8bits log modulus + 24bit uid, with fixed length // but for now 16bit modulus and radius
                uint modulus, length, mod_len;
                modulus = uint(m_Param [ PINTSTIFF ]) ; // m_Param [ PINTSTIFF ] =		1.0f;
                length = uint(1000 * m_Param [ PSMOOTHRADIUS ]); // m_Param [ PSMOOTHRADIUS ] =	0.015f;	// m // related to spacing, but also max particle range i.e. ....
                mod_len = ( modulus <<16 | length ); // NB should mask length to prevent it exceeding 16bits, i.e. 255*255
                
                for (int i = 0; i<BONDS_PER_PARTICLE;i++){ for (int j = 0; j< DATA_PER_BOND; j++){ ElastIdx[i*DATA_PER_BOND +j] = 0; } }
                //NB #define DATA_PER_BOND 6 //6 : [0]current index, [1]elastic limit, [2]restlength, [3]modulus, [4]damping coeff, [5]particle ID, [6]bond index
                
                for (int i = 0; i<BONDS_PER_PARTICLE*2;i++){ Particle_Idx[i] = UINT_MAX; }
                
                // nerve connections to particles
                if (Particle_ID % 10 == 0){NerveIdx = Particle_ID/10;} else {NerveIdx = 0;} // Every 10th particle has nerve connection
                
                // Mass & radius of particles
                // 4bit mass + 4bit radius + 24bit uid // but for now, 16bit mass & radius
                // Note m_params[] is set in "FluidSystem::SetupDefaultParams ()" and "FluidSystem::SetupExampleParams ()"
                // mass = m_Param[PMASS]; // 0.00020543f; // kg
                // radius = m_Param[PRADIUS]; // 0.015f; // m
                Mass_Radius =  ( (uint(m_Param[PMASS]*255.0f*255.0f)<<16) | uint(m_Param[PRADIUS]*255.0f*255.0f) ) ; // mass=>13, radius=>975
                
                // morphogen & transcription factor concentrations
                for (int i=0; i< NUM_TF; i++){ Conc[i] = 1 ;}
                
                // epigenetic state of each gene in this particle
                for (int i=0; i< NUM_GENES; i++){ EpiGen[i] = i ;}
            
                p = AddParticleMorphogenesis2 (
                /* Vector3DF* */ &Pos, 
                /* Vector3DF* */ &Vel, 
                /* uint */ Age, 
                /* uint */ Clr, 
                /* uint *_*/ ElastIdx, 
                /* unit * */ Particle_Idx,
                /* uint */ Particle_ID, 
                /* uint */ Mass_Radius, 
                /* uint */ NerveIdx, 
                /* float* */ Conc,
                /* uint* */ EpiGen 
                );

                if(p==-1){std::cout << "\n SetupAddVolumeMorphogenesis2 exited on p==-1, Pos=("<<Pos.x<<","<<Pos.y<<","<<Pos.z<<"), Particle_ID="<<Particle_ID<<" \n " << std::flush ; return;}
            }
        }
    }
    std::cout << "\n SetupAddVolumeMorphogenesis2 finished \n" << std::flush ;
}


////////////////////////////////////////////////////////////////



void FluidSystem::AddEmit ( float spacing )
{
    int p;
    Vector3DF dir;
    Vector3DF pos;
    float ang_rand, tilt_rand;
    float rnd = m_Vec[PEMIT_RATE].y * 0.15f;
    int x = (int) sqrt(m_Vec[PEMIT_RATE].y);

    for ( int n = 0; n < m_Vec[PEMIT_RATE].y; n++ ) {
        ang_rand = (float(rand()*2.0f/RAND_MAX) - 1.0f) * m_Vec[PEMIT_SPREAD].x;
        tilt_rand = (float(rand()*2.0f/RAND_MAX) - 1.0f) * m_Vec[PEMIT_SPREAD].y;
        dir.x = cos ( ( m_Vec[PEMIT_ANG].x + ang_rand) * DEGtoRAD ) * sin( ( m_Vec[PEMIT_ANG].y + tilt_rand) * DEGtoRAD ) * m_Vec[PEMIT_ANG].z;
        dir.y = sin ( ( m_Vec[PEMIT_ANG].x + ang_rand) * DEGtoRAD ) * sin( ( m_Vec[PEMIT_ANG].y + tilt_rand) * DEGtoRAD ) * m_Vec[PEMIT_ANG].z;
        dir.z = cos ( ( m_Vec[PEMIT_ANG].y + tilt_rand) * DEGtoRAD ) * m_Vec[PEMIT_ANG].z;
        pos = m_Vec[PEMIT_POS];
        pos.x += spacing * (n/x);
        pos.y += spacing * (n%x);

        p = AddParticle ();
        *(m_Fluid.bufV3(FPOS)+n) = pos;
        *(m_Fluid.bufV3(FVEL)+n) = dir;
        *(m_Fluid.bufV3(FVEVAL)+n) = dir;
        *(m_Fluid.bufI(FAGE)+n) = 0;
        *(m_Fluid.bufI(FCLR)+n) = COLORA ( m_Time/10.0, m_Time/5.0, m_Time /4.0, 1 );
    }
}


void FluidSystem::EmitParticles ()
{
    if ( m_Vec[PEMIT_RATE].x > 0 && (++m_Frame) % (int) m_Vec[PEMIT_RATE].x == 0 ) {
        float ss = m_Param [ PDIST ] / m_Param[ PSIMSCALE ];		// simulation scale (not Schutzstaffel)
        AddEmit ( ss );
    }
}


///////////////////////////////////////////////////////////////////
void FluidSystem::Run ()
{
std::cout << "\tFluidSystem::Run (),  "<<std::flush;  
    //case RUN_GPU_FULL:					// Full CUDA pathway, GRID-accelerted GPU, /w deep copy sort
//TransferFromCUDA ();
//std::cout << "\n\n Chk1 \n"<<std::flush;
    InsertParticlesCUDA ( 0x0, 0x0, 0x0 );
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After InsertParticlesCUDA", mbDebug);
//TransferFromCUDA ();
//std::cout << "\n\n Chk2 \n"<<std::flush;
    PrefixSumCellsCUDA ( 0x0, 1 );
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After PrefixSumCellsCUDA", mbDebug);
//TransferFromCUDA ();
//std::cout << "\n\n Chk3 \n"<<std::flush;
    CountingSortFullCUDA ( 0x0 );
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After CountingSortFullCUDA", mbDebug);
//TransferFromCUDA ();
//std::cout << "\n\n Chk4 \n"<<std::flush;
    
    ComputePressureCUDA();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After ComputePressureCUDA", mbDebug); 
//TransferFromCUDA ();
//std::cout << "\n\n Chk5 \n"<<std::flush;
    // FreezeCUDA ();                                   // makes the system plastic, ie the bonds keep reforming
//std::cout << "\n\n Chk6 \n"<<std::flush;
    ComputeForceCUDA ();                                // now includes the function of freeze 
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After ComputeForceCUDA", mbDebug);
//TransferFromCUDA ();
//std::cout << "\n\n Chk6 \n"<<std::flush;

    // I believe this is where we're supposed to call it, TODO check if correct
    ComputeDiffusionCUDA();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After ComputeDiffusionCUDA", mbDebug);

    AdvanceCUDA ( m_Time, m_DT, m_Param[PSIMSCALE] );
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After AdvanceCUDA", mbDebug);    
//TransferFromCUDA ();
    //EmitParticlesCUDA ( m_Time, (int) m_Vec[PEMIT_RATE].x );
    TransferFromCUDA ();	// return for rendering
//std::cout << "\n\n Chk7 \n"<<std::flush;

    AdvanceTime ();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After AdvanceTime", mbDebug);  
//std::cout << " finished \n";
}

 
void FluidSystem::Run (const char * relativePath, int frame )       // version to save data after each kernel
{
    m_FParams.frame = frame;                 // used by computeForceCuda( .. Args)
std::cout << "\tFluidSystem::Run (const char * relativePath, int frame ) "<<std::flush;
    //case RUN_GPU_FULL:					// Full CUDA pathway, GRID-accelerted GPU, /w deep copy sort
//TransferFromCUDA ();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "begin Run", mbDebug); 
    TransferFromCUDA ();
    SavePointsCSV2 (  relativePath, frame );
std::cout << "\n\nRun(relativePath,frame) Chk1, saved "<< frame <<".csv At start of Run(...) \n"<<std::flush;

    InsertParticlesCUDA ( 0x0, 0x0, 0x0 );
//TransferFromCUDA ();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After InsertParticlesCUDA", mbDebug); 
    TransferFromCUDA ();
    SavePointsCSV2 (  relativePath, frame+1 );
std::cout << "\n\nRun(relativePath,frame) Chk2, saved "<< frame+1 <<".csv  After InsertParticlesCUDA\n"<<std::flush;

    PrefixSumCellsCUDA ( 0x0, 1 );
//TransferFromCUDA ();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After PrefixSumCellsCUDA", mbDebug); 
    TransferFromCUDA ();
    SavePointsCSV2 (  relativePath, frame+2 );
std::cout << "\n\nRun(relativePath,frame) Chk3, saved "<< frame+2 <<".csv  After PrefixSumCellsCUDA\n"<<std::flush;

    CountingSortFullCUDA ( 0x0 );
//TransferFromCUDA ();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After CountingSortFullCUDA", mbDebug); 
    TransferFromCUDA ();
    SavePointsCSV2 (  relativePath, frame+3 );
std::cout << "\n\nRun(relativePath,frame) Chk4, saved "<< frame+3 <<".csv  After CountingSortFullCUDA\n"<<std::flush;
    
    ComputePressureCUDA();
//TransferFromCUDA ();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After ComputePressureCUDA", mbDebug); 
    TransferFromCUDA ();
    SavePointsCSV2 (  relativePath, frame+4 );
std::cout << "\n\nRun(relativePath,frame) Chk5, saved "<< frame+4 <<".csv  After ComputePressureCUDA \n"<<std::flush;
    // FreezeCUDA ();                                   // makes the system plastic, ie the bonds keep reforming
//std::cout << "\n\n Chk6 \n"<<std::flush;


    ComputeForceCUDA ();                                // now includes the function of freeze 
//TransferFromCUDA ();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After ComputeForceCUDA", mbDebug); // stalls here on 2nd cycle of Freeze()
    TransferFromCUDA ();
    SavePointsCSV2 (  relativePath, frame+5 );
std::cout << "\n\nRun(relativePath,frame) Chk6, saved "<< frame+5 <<".csv  After ComputeForceCUDA \n"<<std::flush;

    AdvanceCUDA ( m_Time, m_DT, m_Param[PSIMSCALE] );
//TransferFromCUDA ();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After AdvanceCUDA", mbDebug); 
    TransferFromCUDA ();
    SavePointsCSV2 (  relativePath, frame+6 );
std::cout << "\n\nRun(relativePath,frame) Chk7, saved "<< frame+6 <<".csv  After AdvanceCUDA \n"<<std::flush;

    //cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After AdvanceCUDA", mbDebug);    
    //EmitParticlesCUDA ( m_Time, (int) m_Vec[PEMIT_RATE].x );
    //TransferFromCUDA ();	// return for rendering

    AdvanceTime ();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After AdvanceTime", mbDebug); 
    TransferFromCUDA ();
    SavePointsCSV2 (  relativePath, frame+7 );
std::cout << "Run(relativePath,frame) finished,  saved "<< frame+7 <<".csv  After AdvanceTime \n";
}// 0:start, 1:InsertParticles, 2:PrefixSumCellsCUDA, 3:CountingSortFull, 4:ComputePressure, 5:ComputeForce, 6:Advance, 7:AdvanceTime



void FluidSystem::Freeze ()
{
    m_FParams.freeze = true;
    Run();
    m_FParams.freeze = false;
}

void FluidSystem::Freeze (const char * relativePath, int frame)
{
    m_FParams.freeze = true;
    Run(relativePath, frame);
    m_FParams.freeze = false;
}


void FluidSystem::AdvanceTime ()
{
    //std::cout << "FluidSystem::AdvanceTime (),  m_Time = "<< m_Time <<"\n";
    //std::cout << " \n";
    m_Time += m_DT;

    m_Frame += m_FrameRange.z;

    if ( m_Frame > m_FrameRange.y && m_FrameRange.y != -1 ) {

        m_Frame = m_FrameRange.x;
        mbRecord = false;
        mbRecordBricks = false;
        m_Toggle[ PCAPTURE ] = false;
        std::cout << "Exiting.\n";
        //nvprintf ( "Exiting.\n" );
        exit ( 1 );
    }
}


///////////////////////////////////////////////////


void FluidSystem::AllocatePackBuf ()
{
    if ( mPackBuf != 0x0 ) free ( mPackBuf );
    mPackBuf = (char*) malloc ( sizeof(Fluid) * mMaxPoints );
}


/////////////////////////////////////////////////////


void FluidSystem::Advance ()
{
    Vector3DF norm, z;
    Vector3DF dir, accel;
    Vector3DF vnext;
    Vector3DF bmin, bmax;
    Vector4DF clr;
    float adj;
    float AL, AL2, SL, SL2, ss, radius;
    float stiff, damp, speed, diff;

    AL = m_Param[PACCEL_LIMIT];
    AL2 = AL*AL;
    SL = m_Param[PVEL_LIMIT];
    SL2 = SL*SL;

    stiff = m_Param[PEXTSTIFF];
    damp = m_Param[PEXTDAMP];
    radius = m_Param[PRADIUS];
    bmin = m_Vec[PBOUNDMIN];
    bmax = m_Vec[PBOUNDMAX];
    ss = m_Param[PSIMSCALE];

    // Get particle buffers
    Vector3DF*	ppos =		m_Fluid.bufV3(FPOS);
    Vector3DF*	pvel =		m_Fluid.bufV3(FVEL);
    Vector3DF*	pveleval =	m_Fluid.bufV3(FVEVAL);
    Vector3DF*	pforce =	m_Fluid.bufV3(FFORCE);
    uint*		pclr =		m_Fluid.bufI(FCLR);
    float*		ppress =	m_Fluid.bufF(FPRESS);
    float*		pdensity =	m_Fluid.bufF(FDENSITY);

    // Advance each particle
    for ( int n=0; n < NumPoints(); n++ ) {

        if ( m_Fluid.bufI(FGCELL)[n] == GRID_UNDEF) continue;

        // Compute Acceleration
        accel = *pforce;
        accel *= m_Param[PMASS];

        // Boundary Conditions
        // Y-axis walls
        diff = radius - ( ppos->y - (bmin.y+ (ppos->x-bmin.x)*m_Param[PGROUND_SLOPE] ) )*ss;
        if (diff > EPSILON ) {
            norm.Set ( -m_Param[PGROUND_SLOPE], 1.0f - m_Param[PGROUND_SLOPE], 0 );
            adj = stiff * diff - damp * (float) norm.Dot ( *pveleval );
            accel.x += adj * norm.x;
            accel.y += adj * norm.y;
            accel.z += adj * norm.z;
        }
        diff = radius - ( bmax.y - ppos->y )*ss;
        if (diff > EPSILON) {
            norm.Set ( 0, -1, 0 );
            adj = stiff * diff - damp * (float) norm.Dot ( *pveleval );
            accel.x += adj * norm.x;
            accel.y += adj * norm.y;
            accel.z += adj * norm.z;
        }

        // X-axis walls
        if ( !m_Toggle[PWRAP_X] ) {
            diff = radius - ( ppos->x - (bmin.x + (sin(m_Time*m_Param[PFORCE_FREQ])+1)*0.5f * m_Param[PFORCE_MIN]) )*ss;
            //diff = 2 * radius - ( p->pos.x - min.x + (sin(m_Time*10.0)-1) * m_Param[FORCE_XMIN_SIN] )*ss;
            if (diff > EPSILON ) {
                norm.Set ( 1.0, 0, 0 );
                adj = (m_Param[ PFORCE_MIN ]+1) * stiff * diff - damp * (float) norm.Dot ( *pveleval ) ;
                accel.x += adj * norm.x;
                accel.y += adj * norm.y;
                accel.z += adj * norm.z;
            }

            diff = radius - ( (bmax.x - (sin(m_Time*m_Param[PFORCE_FREQ])+1)*0.5f* m_Param[PFORCE_MAX]) - ppos->x )*ss;
            if (diff > EPSILON) {
                norm.Set ( -1, 0, 0 );
                adj = (m_Param[ PFORCE_MAX ]+1) * stiff * diff - damp * (float) norm.Dot ( *pveleval );
                accel.x += adj * norm.x;
                accel.y += adj * norm.y;
                accel.z += adj * norm.z;
            }
        }

        // Z-axis walls
        diff = radius - ( ppos->z - bmin.z )*ss;
        if (diff > EPSILON) {
            norm.Set ( 0, 0, 1 );
            adj = stiff * diff - damp * (float) norm.Dot ( *pveleval );
            accel.x += adj * norm.x;
            accel.y += adj * norm.y;
            accel.z += adj * norm.z;
        }
        diff = radius - ( bmax.z - ppos->z )*ss;
        if (diff > EPSILON) {
            norm.Set ( 0, 0, -1 );
            adj = stiff * diff - damp * (float) norm.Dot ( *pveleval );
            accel.x += adj * norm.x;
            accel.y += adj * norm.y;
            accel.z += adj * norm.z;
        }


        // Wall barrier
        if ( m_Toggle[PWALL_BARRIER] ) {
            diff = 2 * radius - ( ppos->x - 0 )*ss;
            if (diff < 2*radius && diff > EPSILON && fabs(ppos->y) < 3 && ppos->z < 10) {
                norm.Set ( 1.0, 0, 0 );
                adj = 2*stiff * diff - damp * (float) norm.Dot ( *pveleval ) ;
                accel.x += adj * norm.x;
                accel.y += adj * norm.y;
                accel.z += adj * norm.z;
            }
        }

        // Levy barrier
        if ( m_Toggle[PLEVY_BARRIER] ) {
            diff = 2 * radius - ( ppos->x - 0 )*ss;
            if (diff < 2*radius && diff > EPSILON && fabs(ppos->y) > 5 && ppos->z < 10) {
                norm.Set ( 1.0, 0, 0 );
                adj = 2*stiff * diff - damp * (float) norm.Dot ( *pveleval ) ;
                accel.x += adj * norm.x;
                accel.y += adj * norm.y;
                accel.z += adj * norm.z;
            }
        }
        // Drain barrier
        if ( m_Toggle[PDRAIN_BARRIER] ) {
            diff = 2 * radius - ( ppos->z - bmin.z-15 )*ss;
            if (diff < 2*radius && diff > EPSILON && (fabs(ppos->x)>3 || fabs(ppos->y)>3) ) {
                norm.Set ( 0, 0, 1);
                adj = stiff * diff - damp * (float) norm.Dot ( *pveleval );
                accel.x += adj * norm.x;
                accel.y += adj * norm.y;
                accel.z += adj * norm.z;
            }
        }

        // Plane gravity
        accel += m_Vec[PPLANE_GRAV_DIR] * m_Param[PGRAV];

        // Point gravity
        if ( m_Vec[PPOINT_GRAV_POS].x > 0 && m_Param[PGRAV] > 0 ) {
            norm.x = ( ppos->x - m_Vec[PPOINT_GRAV_POS].x );
            norm.y = ( ppos->y - m_Vec[PPOINT_GRAV_POS].y );
            norm.z = ( ppos->z - m_Vec[PPOINT_GRAV_POS].z );
            norm.Normalize ();
            norm *= m_Param[PGRAV];
            accel -= norm;
        }

        // Acceleration limiting
        speed = accel.x*accel.x + accel.y*accel.y + accel.z*accel.z;
        if ( speed > AL2 ) {
            accel *= AL / sqrt(speed);
        }

        // Velocity limiting
        speed = pvel->x*pvel->x + pvel->y*pvel->y + pvel->z*pvel->z;
        if ( speed > SL2 ) {
            speed = SL2;
            (*pvel) *= SL / sqrt(speed);
        }

        // Leapfrog Integration ----------------------------
        vnext = accel;
        vnext *= m_DT;
        vnext += *pvel;						// v(t+1/2) = v(t-1/2) + a(t) dt

        *pveleval = *pvel;
        *pveleval += vnext;
        *pveleval *= 0.5;					// v(t+1) = [v(t-1/2) + v(t+1/2)] * 0.5		used to compute forces later
        *pvel = vnext;
        vnext *= m_DT/ss;
        *ppos += vnext;						// p(t+1) = p(t) + v(t+1/2) dt

        /*if ( m_Param[PCLR_MODE]==1.0 ) {
        	adj = fabs(vnext.x)+fabs(vnext.y)+fabs(vnext.z) / 7000.0;
        	adj = (adj > 1.0) ? 1.0 : adj;
        	*pclr = COLORA( 0, adj, adj, 1 );
        }
        if ( m_Param[PCLR_MODE]==2.0 ) {
        	float v = 0.5 + ( *ppress / 1500.0);
        	if ( v < 0.1 ) v = 0.1;
        	if ( v > 1.0 ) v = 1.0;
        	*pclr = COLORA ( v, 1-v, 0, 1 );
        }*/
        if ( speed > SL2*0.1f) {
            adj = SL2*0.1f;
            clr.fromClr ( *pclr );
            clr += Vector4DF( 2/255.0f, 2/255.0f, 2/255.0f, 2/255.0f);
            clr.Clamp ( 1, 1, 1, 1);
            *pclr = clr.toClr();
        }
        if ( speed < 0.01 ) {
            clr.fromClr ( *pclr);
            clr.x -= float(1/255.0f);
            if ( clr.x < 0.2f ) clr.x = 0.2f;
            clr.y -= float(1/255.0f);
            if ( clr.y < 0.2f ) clr.y = 0.2f;
            *pclr = clr.toClr();
        }

        // Euler integration -------------------------------
        /* accel += m_Gravity;
        accel *= m_DT;
        p->vel += accel;				// v(t+1) = v(t) + a(t) dt
        p->vel_eval += accel;
        p->vel_eval *= m_DT/d;
        p->pos += p->vel_eval;
        p->vel_eval = p->vel;  */


        if ( m_Toggle[PWRAP_X] ) {
            diff = ppos->x - (m_Vec[PBOUNDMIN].x + 2);			// -- Simulates object in center of flow
            if ( diff <= 0 ) {
                ppos->x = (m_Vec[PBOUNDMAX].x - 2) + diff*2;
                ppos->z = 10;
            }
        }

        ppos++;
        pvel++;
        pveleval++;
        pforce++;
        pclr++;
        ppress++;
        pdensity++;
    }

}
///////////////////////////////////////////


void FluidSystem::ClearNeighborTable ()
{
    if ( m_NeighborTable != 0x0 )	free (m_NeighborTable);
    if ( m_NeighborDist != 0x0)		free (m_NeighborDist );
    m_NeighborTable = 0x0;
    m_NeighborDist = 0x0;
    m_NeighborNum = 0;
    m_NeighborMax = 0;
}

void FluidSystem::ResetNeighbors ()
{
    m_NeighborNum = 0;
}

// Allocate new neighbor tables, saving previous data
int FluidSystem::AddNeighbor ()
{
    if ( m_NeighborNum >= m_NeighborMax ) {
        m_NeighborMax = 2*m_NeighborMax + 1;
        int* saveTable = m_NeighborTable;
        m_NeighborTable = (int*) malloc ( m_NeighborMax * sizeof(int) );
        if ( saveTable != 0x0 ) {
            memcpy ( m_NeighborTable, saveTable, m_NeighborNum*sizeof(int) );
            free ( saveTable );
        }
        float* saveDist = m_NeighborDist;
        m_NeighborDist = (float*) malloc ( m_NeighborMax * sizeof(float) );
        if ( saveDist != 0x0 ) {
            memcpy ( m_NeighborDist, saveDist, m_NeighborNum*sizeof(int) );
            free ( saveDist );
        }
    };
    m_NeighborNum++;
    return m_NeighborNum-1;
}

void FluidSystem::ClearNeighbors ( int i )
{
    *(m_Fluid.bufI(FNBRCNT)+i) = 0;
}

int FluidSystem::AddNeighbor( int i, int j, float d )
{
    int k = AddNeighbor();
    m_NeighborTable[k] = j;
    m_NeighborDist[k] = d;
    if (*(m_Fluid.bufI(FNBRCNT)+i) == 0 ) *(m_Fluid.bufI(FNBRCNT)+i) = k;
    (*(m_Fluid.bufI(FNBRCNT)+i))++;
    return k;
}

// Ideal grid cell size (gs) = 2 * smoothing radius = 0.02*2 = 0.04
// Ideal domain size = k*gs/d = k*0.02*2/0.005 = k*8 = {8, 16, 24, 32, 40, 48, ..}
//    (k = number of cells, gs = cell size, d = simulation scale)
void FluidSystem::SetupGrid ( Vector3DF min, Vector3DF max, float sim_scale, float cell_size, float border )
{
    float world_cellsize = cell_size / sim_scale;

    m_GridMin = min;
    m_GridMax = max;
    m_GridSize = m_GridMax;
    m_GridSize -= m_GridMin;
#if 0
    m_GridRes.Set ( 6, 6, 6 );				// Fixed grid res
#else
    m_GridRes.x = (int) ceil ( m_GridSize.x / world_cellsize );		// Determine grid resolution
    m_GridRes.y = (int) ceil ( m_GridSize.y / world_cellsize );
    m_GridRes.z = (int) ceil ( m_GridSize.z / world_cellsize );
    m_GridSize.x = m_GridRes.x * cell_size / sim_scale;				// Adjust grid size to multiple of cell size
    m_GridSize.y = m_GridRes.y * cell_size / sim_scale;
    m_GridSize.z = m_GridRes.z * cell_size / sim_scale;
#endif
    m_GridDelta = m_GridRes;		// delta = translate from world space to cell #
    m_GridDelta /= m_GridSize;

    m_GridTotal = (int)(m_GridRes.x * m_GridRes.y * m_GridRes.z);

    m_Param[PSTAT_GMEM] = 12.0f * m_GridTotal;		// Grid memory used

    // Number of cells to search:
    // n = (2r / w) +1,  where n = 1D cell search count, r = search radius, w = world cell width
    //
    m_GridSrch = (int) (floor(2.0f*(m_Param[PSMOOTHRADIUS]/sim_scale) / world_cellsize) + 1.0f);
    if ( m_GridSrch < 2 ) m_GridSrch = 2;
    m_GridAdjCnt = m_GridSrch * m_GridSrch * m_GridSrch ;			// 3D search count = n^3, e.g. 2x2x2=8, 3x3x3=27, 4x4x4=64

    if ( m_GridSrch > 6 ) {
        //nvprintf ( "ERROR: Neighbor search is n > 6. \n " );
        exit(-1);
    }

    int cell = 0;
    for (int y=0; y < m_GridSrch; y++ )
        for (int z=0; z < m_GridSrch; z++ )
            for (int x=0; x < m_GridSrch; x++ )
                m_GridAdj[cell++] = ( y*m_GridRes.z + z )*m_GridRes.x +  x ;			// -1 compensates for ndx 0=empty


    /*nvprintf ( "Adjacency table (CPU) \n");
    for (int n=0; n < m_GridAdjCnt; n++ ) {
    	nvprintf ( "  ADJ: %d, %d\n", n, m_GridAdj[n] );
    }*/

    if ( mPackGrid != 0x0 ) free ( mPackGrid );
    mPackGrid = (int*) malloc ( sizeof(int) * m_GridTotal );


}


///////////////////////////////////////////////////////////////



int FluidSystem::getGridCell ( int p, Vector3DI& gc )
{
    return getGridCell ( m_Fluid.bufV3(FPOS)[p], gc );
}
int FluidSystem::getGridCell ( Vector3DF& pos, Vector3DI& gc )
{
    gc.x = (int)( (pos.x - m_GridMin.x) * m_GridDelta.x);			// Cell in which particle is located
    gc.y = (int)( (pos.y - m_GridMin.y) * m_GridDelta.y);
    gc.z = (int)( (pos.z - m_GridMin.z) * m_GridDelta.z);
    return (int)( (gc.y*m_GridRes.z + gc.z)*m_GridRes.x + gc.x);
}
Vector3DI FluidSystem::getCell ( int c )
{
    Vector3DI gc;
    int xz = m_GridRes.x*m_GridRes.z;
    gc.y = c / xz;
    c -= gc.y*xz;
    gc.z = c / m_GridRes.x;
    c -= gc.z*m_GridRes.x;
    gc.x = c;
    return gc;
}



////////////////////////////////////////////////////////


void FluidSystem::InsertParticles ()  /* CPU version. May not be used */
{
    int gs;

    // Reset all grid pointers and neighbor tables to empty 
    // Sets the first num bytes of the block of memory pointed by ptr to the specified value (interpreted as an unsigned char).
    memset ( m_Fluid.bufC(FGNEXT),		GRID_UCHAR, NumPoints()*sizeof(uint) );
    memset ( m_Fluid.bufC(FGCELL),		GRID_UCHAR, NumPoints()*sizeof(uint) );
    memset ( m_Fluid.bufC(FCLUSTER),	GRID_UCHAR, NumPoints()*sizeof(uint) );

    // Reset all grid cells to empty
    memset( m_Fluid.bufC(FGRID),		GRID_UCHAR, m_GridTotal*sizeof(uint));
    memset( m_Fluid.bufI(FGRIDCNT),		0, m_GridTotal*sizeof(uint));

    // Insert each particle into spatial grid
    Vector3DI gc;
    Vector3DF* ppos =	m_Fluid.bufV3(FPOS);
    uint* pgrid =		m_Fluid.bufI(FGCELL);
    uint* pnext =		m_Fluid.bufI(FGNEXT);
    uint* pcell =		m_Fluid.bufI(FCLUSTER);

    float poff = m_Param[PSMOOTHRADIUS] / m_Param[PSIMSCALE];

    int ns = (int) pow ( (float) m_GridAdjCnt, 1.0f/3.0f );
    register int xns, yns, zns;
    xns = m_GridRes.x - m_GridSrch;
    yns = m_GridRes.y - m_GridSrch;
    zns = m_GridRes.z - m_GridSrch;

    m_Param[ PSTAT_OCCUPY ] = 0.0;
    m_Param [ PSTAT_GRIDCNT ] = 0.0;
    uint* m_Grid = m_Fluid.bufI(FGRID);
    uint* m_GridCnt = m_Fluid.bufI(FGRIDCNT);

    for ( int n=0; n < NumPoints(); n++ ) {
        gs = getGridCell ( *ppos, gc );
        if ( gc.x >= 1 && gc.x <= xns && gc.y >= 1 && gc.y <= yns && gc.z >= 1 && gc.z <= zns ) {
            // put current particle at head of grid cell, pointing to next in list (previous head of cell)
            *pgrid = gs;
            *pnext = m_Grid[gs];
            if ( *pnext == GRID_UNDEF ) m_Param[ PSTAT_OCCUPY ] += 1.0;
            m_Grid[gs] = n;
            m_GridCnt[gs]++;
            m_Param [ PSTAT_GRIDCNT ] += 1.0;
            /* -- 1/2 cell offset search method
            gx = (int)( (-poff + ppos->x - m_GridMin.x) * m_GridDelta.x);
            if ( gx < 0 ) gx = 0;
            if ( gx > m_GridRes.x-2 ) gx = m_GridRes.x-2;
            gy = (int)( (-poff + ppos->y - m_GridMin.y) * m_GridDelta.y);
            if ( gy < 0 ) gy = 0;
            if ( gy > m_GridRes.y-2 ) gx = m_GridRes.y-2;
            gz = (int)( (-poff + ppos->z - m_GridMin.z) * m_GridDelta.z);
            if ( gz < 0 ) gz = 0;
            if ( gz > m_GridRes.z-2 ) gz = m_GridRes.z-2;
            *pcell = (int)( (gy*m_GridRes.z + gz)*m_GridRes.x + gx) ;	// Cell in which to start 2x2x2 search*/
        } else {
            Vector3DF vel, ve;
            vel = m_Fluid.bufV3(FVEL) [n];
            ve = m_Fluid.bufV3(FVEVAL) [n];
            float pr, dn;
            pr = m_Fluid.bufF(FPRESS) [n];
            dn = m_Fluid.bufF(FDENSITY) [n];
            //printf ( "WARNING: Out of Bounds: %d, P<%f %f %f>, V<%f %f %f>, prs:%f, dns:%f\n", n, ppos->x, ppos->y, ppos->z, vel.x, vel.y, vel.z, pr, dn );
            //ppos->x = -1; ppos->y = -1; ppos->z = -1;
        }
        pgrid++;
        ppos++;
        pnext++;
        pcell++;
    }

    // STATS
    /*m_Param[ PSTAT_OCCUPY ] = 0;
    m_Param[ PSTAT_GRIDCNT ] = 0;
    for (int n=0; n < m_GridTotal; n++) {
    	if ( m_GridCnt[n] > 0 )  m_Param[ PSTAT_OCCUPY ] += 1.0;
    	m_Param [ PSTAT_GRIDCNT ] += m_GridCnt[n];
    }*/
}



///////////////////////////////////////


void FluidSystem::SaveResults ()
{
    if ( mSaveNdx != 0x0 ) free ( mSaveNdx );
    if ( mSaveCnt != 0x0 ) free ( mSaveCnt );
    if ( mSaveNeighbors != 0x0 )	free ( mSaveNeighbors );

    mSaveNdx = (uint*) malloc ( sizeof(uint) * NumPoints() );
    mSaveCnt = (uint*) malloc ( sizeof(uint) * NumPoints() );
    mSaveNeighbors = (uint*) malloc ( sizeof(uint) * m_NeighborNum );
    memcpy ( mSaveNdx, m_Fluid.bufC(FNBRNDX), sizeof(uint) * NumPoints() );
    memcpy ( mSaveCnt, m_Fluid.bufC(FNBRCNT), sizeof(uint) * NumPoints() );
    memcpy ( mSaveNeighbors, m_NeighborTable, sizeof(uint) * m_NeighborNum );
}


//////////////////////////////////////////

// Compute Pressures - Using spatial grid, and also create neighbor table
void FluidSystem::ComputePressureGrid ()
{
    int i, j, cnt = 0;
    float sum, dsq, c;
    float d = m_Param[PSIMSCALE];
    float d2 = d*d;
    float radius = m_Param[PSMOOTHRADIUS] / m_Param[PSIMSCALE];

    // Get particle buffers
    Vector3DF*	ipos =		m_Fluid.bufV3(FPOS);
    float*		ipress =	m_Fluid.bufF(FPRESS);
    float*		idensity =	m_Fluid.bufF(FDENSITY);
    uint*		inbr =		m_Fluid.bufI(FNBRNDX);
    uint*		inbrcnt =	m_Fluid.bufI(FNBRCNT);

    Vector3DF	dst;
    int			nadj = (m_GridRes.z + 1)*m_GridRes.x + 1;
    uint*		m_Grid = m_Fluid.bufI(FGRID);
    uint*		m_GridCnt = m_Fluid.bufI(FGRIDCNT);

    int nbrcnt = 0;
    int srch = 0;

    for ( i=0; i < NumPoints(); i++ ) {

        sum = 0.0;

        if ( m_Fluid.bufI(FGCELL)[i] != GRID_UNDEF ) {
            for (int cell=0; cell < m_GridAdjCnt; cell++) {
                j = m_Grid [   m_Fluid.bufI(FGCELL)[i] - nadj + m_GridAdj[cell] ] ;
                while ( j != GRID_UNDEF ) {
                    if ( i==j ) {
                        j = m_Fluid.bufI(FGNEXT)[j];
                        continue;
                    }
                    dst = m_Fluid.bufV3(FPOS)[j];
                    dst -= *ipos;
                    dsq = d2*(dst.x*dst.x + dst.y*dst.y + dst.z*dst.z);
                    if ( dsq <= m_R2 ) {
                        c =  m_R2 - dsq;
                        sum += c * c * c;
                        nbrcnt++;
                        /*nbr = AddNeighbor();			// get memory for new neighbor
                        *(m_NeighborTable + nbr) = j;
                        *(m_NeighborDist + nbr) = sqrt(dsq);
                        inbr->num++;*/
                    }
                    srch++;
                    j = m_Fluid.bufI(FGNEXT)[j];
                }
            }
        }
        *idensity = sum * m_Param[PMASS] * m_Poly6Kern ;
        *ipress = ( *idensity - m_Param[PRESTDENSITY] ) * m_Param[PINTSTIFF];
        *idensity = 1.0f / *idensity;

        ipos++;
        idensity++;
        ipress++;
    }
    // Stats:
    m_Param [ PSTAT_NBR ] = float(nbrcnt);
    m_Param [ PSTAT_SRCH ] = float(srch);
    if ( m_Param[PSTAT_NBR] > m_Param [ PSTAT_NBRMAX ] ) m_Param [ PSTAT_NBRMAX ] = m_Param[PSTAT_NBR];
    if ( m_Param[PSTAT_SRCH] > m_Param [ PSTAT_SRCHMAX ] ) m_Param [ PSTAT_SRCHMAX ] = m_Param[PSTAT_SRCH];
}

// Compute Forces - Using spatial grid with saved neighbor table. Fastest.
void FluidSystem::ComputeForceGrid ()
{
    Vector3DF force;
    register float pterm, vterm, dterm;
    int i, j;
    float c, d;
    float dx, dy, dz;
    float mR, visc;

    d = m_Param[PSIMSCALE];
    mR = m_Param[PSMOOTHRADIUS];
    visc = m_Param[PVISC];

    // Get particle buffers
    Vector3DF*	ipos =		m_Fluid.bufV3(FPOS);
    Vector3DF*	iveleval =	m_Fluid.bufV3(FVEVAL);
    Vector3DF*	iforce =	m_Fluid.bufV3(FFORCE);
    float*		ipress =	m_Fluid.bufF(FPRESS);
    float*		idensity =	m_Fluid.bufF(FDENSITY);

    Vector3DF	jpos;
    float		jdist;
    float		jpress;
    float		jdensity;
    Vector3DF	jveleval;
    float		dsq;
    float		d2 = d*d;
    int			nadj = (m_GridRes.z + 1)*m_GridRes.x + 1;
    uint* m_Grid = m_Fluid.bufI(FGRID);
    uint* m_GridCnt = m_Fluid.bufI(FGRIDCNT);

    for ( i=0; i < NumPoints(); i++ ) {

        iforce->Set ( 0, 0, 0 );

        if ( m_Fluid.bufI(FGCELL)[i] != GRID_UNDEF ) {
            for (int cell=0; cell < m_GridAdjCnt; cell++) {
                j = m_Grid [  m_Fluid.bufI(FGCELL)[i] - nadj + m_GridAdj[cell] ];
                while ( j != GRID_UNDEF ) {
                    if ( i==j ) {
                        j = m_Fluid.bufI(FGNEXT)[j];
                        continue;
                    }
                    jpos = m_Fluid.bufV3(FPOS)[j];
                    dx = ( ipos->x - jpos.x);		// dist in cm
                    dy = ( ipos->y - jpos.y);
                    dz = ( ipos->z - jpos.z);
                    dsq = d2*(dx*dx + dy*dy + dz*dz);
                    if ( dsq <= m_R2 ) {

                        jdist = sqrt(dsq);

                        jpress = m_Fluid.bufF(FPRESS)[j];
                        jdensity = m_Fluid.bufF(FDENSITY)[j];
                        jveleval = m_Fluid.bufV3(FVEVAL)[j];
                        dx = ( ipos->x - jpos.x);		// dist in cm
                        dy = ( ipos->y - jpos.y);
                        dz = ( ipos->z - jpos.z);
                        c = (mR-jdist);
                        pterm = d * -0.5f * c * m_SpikyKern * ( *ipress + jpress ) / jdist;
                        dterm = c * (*idensity) * jdensity;
                        vterm = m_LapKern * visc;
                        iforce->x += ( pterm * dx + vterm * ( jveleval.x - iveleval->x) ) * dterm;
                        iforce->y += ( pterm * dy + vterm * ( jveleval.y - iveleval->y) ) * dterm;
                        iforce->z += ( pterm * dz + vterm * ( jveleval.z - iveleval->z) ) * dterm;
                    }
                    j = m_Fluid.bufI(FGNEXT)[j];
                }
            }
        }
        ipos++;
        iveleval++;
        iforce++;
        ipress++;
        idensity++;
    }
}


// Compute Forces - Using spatial grid with saved neighbor table. Fastest.
void FluidSystem::ComputeForceGridNC ()
{
    Vector3DF force;
    register float pterm, vterm, dterm;
    int i, j;
    float c, d;
    float dx, dy, dz;
    float mR, visc;

    d = m_Param[PSIMSCALE];
    mR = m_Param[PSMOOTHRADIUS];
    visc = m_Param[PVISC];

    // Get particle buffers
    Vector3DF*	ipos =		m_Fluid.bufV3(FPOS);
    Vector3DF*	iveleval =	m_Fluid.bufV3(FVEVAL);
    Vector3DF*	iforce =	m_Fluid.bufV3(FFORCE);
    float*		ipress =	m_Fluid.bufF(FPRESS);
    float*		idensity =	m_Fluid.bufF(FDENSITY);
    uint*		inbr =		m_Fluid.bufI(FNBRNDX);
    uint*		inbrcnt =	m_Fluid.bufI(FNBRCNT);

    int			jndx;
    Vector3DF	jpos;
    float		jdist;
    float		jpress;
    float		jdensity;
    Vector3DF	jveleval;

    for ( i=0; i < NumPoints(); i++ ) {

        iforce->Set ( 0, 0, 0 );

        jndx = *inbr;
        for (int nbr=0; nbr < (int) *inbrcnt; nbr++ ) {
            j = *(m_NeighborTable+jndx);
            jpos =		m_Fluid.bufV3(FPOS)[j];
            jpress =	m_Fluid.bufF(FPRESS)[j];
            jdensity =  m_Fluid.bufF(FDENSITY)[j];
            jveleval =  m_Fluid.bufV3(FVEVAL)[j];
            jdist = *(m_NeighborDist + jndx);
            dx = ( ipos->x - jpos.x);		// dist in cm
            dy = ( ipos->y - jpos.y);
            dz = ( ipos->z - jpos.z);
            c = ( mR - jdist );
            pterm = d * -0.5f * c * m_SpikyKern * ( *ipress + jpress ) / jdist;
            dterm = c * (*idensity) * jdensity;
            vterm = m_LapKern * visc;
            iforce->x += ( pterm * dx + vterm * ( jveleval.x - iveleval->x) ) * dterm;
            iforce->y += ( pterm * dy + vterm * ( jveleval.y - iveleval->y) ) * dterm;
            iforce->z += ( pterm * dz + vterm * ( jveleval.z - iveleval->z) ) * dterm;
            jndx++;
        }
        ipos++;
        iveleval++;
        iforce++;
        ipress++;
        idensity++;
        inbr++;
    }
}



////////////////////////////////////////////////////////



void FluidSystem::StartRecord ()
{
    mbRecord = !mbRecord;
}
void FluidSystem::StartRecordBricks ()
{
    mbRecordBricks = !mbRecordBricks;
}

void FluidSystem::ReadGenome( const char * relativePath, int gpu_mode, int cpu_mode)
// NB currently GPU allocation is by Allocate particles, called by ReadPointsCSV.
{
    const char * genes_file_path = relativePath;
    printf("\n## opening file %s \n", genes_file_path);
    FILE * genes_file = fopen(genes_file_path, "rb");
    if (genes_file == NULL) {
        std::cout << "\nvoid FluidSystem::ReadGenome( const char * relativePath, int gpu_mode, int cpu_mode)  Could not read file "<< genes_file_path <<"\n"<< std::flush;
        assert(0);
    }
    // find number of lines = number of particles
    int ch, number_of_lines = 0;
    while (EOF != (ch=getc(genes_file)))   if ('\n' == ch)  ++number_of_lines;

    std::fseek(genes_file, 0, SEEK_SET);
    /*if (gpu_mode != GPU_OFF){
        FluidSetupCUDA ( mMaxPoints, m_GridSrch, *(int3*)& m_GridRes, *(float3*)& m_GridSize, *(float3*)& m_GridDelta, *(float3*)& m_GridMin, *(float3*)& m_GridMax, m_GridTotal, 0 );
        UpdateParams();            //  sends simulation params to device.
        UpdateGenome();            //  sends genome to device.              // NB need to initialize genome from file, or something.
    }*/
    //AllocateParticles ( mMaxPoints, gpu_mode, cpu_mode );  // allocates only cpu buffer for particles
    int num_genes;
    std::fscanf(genes_file, "mutability,\tdelay,\tsensitivity[%i],\tdifusability[2] \n", &num_genes );

    if ((num_genes != number_of_lines-1) || (num_genes != NUM_GENES) )
    {
        std::cout << "\n! Miss-match of parameters ! ((num_genes != number_of_lines-1 || (num_genes != NUM_GENES) )\n";
        std::cout << "num_genes = " << num_genes <<"\tnumber_of_lines = "<<number_of_lines<<"\tNUM_GENES = "<<NUM_GENES<<"\n";
    }
    int i;
    for (i=0; i<number_of_lines-1; i++ ) {
        int ret = std::fscanf(genes_file,"%i,",&m_FGenome.mutability[i] );
        ret += std::fscanf(genes_file,"%i,",&m_FGenome.delay[i] );
        for(int j=0; j<NUM_GENES; j++) {
            ret += std::fscanf(genes_file,"%i,", &m_FGenome.sensitivity[i][j] );
        }
        ret += std::fscanf(genes_file,"%i,",&m_FGenome.difusability[i][0] );
        ret += std::fscanf(genes_file,"%i,",&m_FGenome.difusability[i][1] );
        ret += std::fscanf(genes_file, " \n");

        if (ret != (2 + NUM_GENES + 2) ) {
            std::cout << "\nvoid FluidSystem::ReadGenome, read failure !  gene number = " << i;
            std::cout << "\n " << std::flush;
            fclose(genes_file);
            return;
        }
        std::cout << m_FGenome.mutability[i] <<",\t";
        std::cout << m_FGenome.delay[i] <<",\t\t";
        for(int j=0; j<NUM_GENES; j++) {
            std::cout << m_FGenome.sensitivity[i][j] <<",\t";
        }
        std::cout <<"\t"<< m_FGenome.difusability[i][0] <<",\t";
        std::cout << m_FGenome.difusability[i][1] <<",\t";
        std::cout <<"\n";
    }
    std::cout << "\n" << i << " genes read.\n" << std::flush;
    fclose(genes_file);
}

void FluidSystem::WriteGenome( const char * relativePath)
{
    std::cout << "\n  FluidSystem::WriteGenome( const char * relativePath)  started \n" << std::flush;
    char buf[256];
    sprintf ( buf, "%s/genome.csv", relativePath );
    FILE* fp = fopen ( buf, "w" );
    if (fp == NULL) {
        std::cout << "\nvoid FluidSystem::WriteGenome( const char * relativePath)  Could not open file "<< fp <<"\n"<< std::flush;
        assert(0);
    }

    fprintf(fp, "mutability,\tdelay,\tsensitivity[%i],\tdifusability[2] \n", NUM_GENES );

    for(int i=0; i<NUM_GENES; i++) {

        fprintf(fp, "%i,\t", m_FGenome.mutability[i] );
        fprintf(fp, "%i,\t\t", m_FGenome.delay[i] );

        for(int j=0; j<NUM_GENES; j++) {
            fprintf(fp, "%i,\t", m_FGenome.sensitivity[i][j] );
        }
        fprintf(fp, "\t%i,\t", m_FGenome.difusability[i][0] );
        fprintf(fp, "%i,\t", m_FGenome.difusability[i][1] );
        fprintf(fp, " \n" );
    }
}


void FluidSystem::SavePoints ( int frame )
{
    char buf[256];
    sprintf ( buf, "jet%04d.pts", frame );
    FILE* fp = fopen ( buf, "wb" );

    int numpnt = NumPoints();
    int numfield = 3;
    int ftype;		// 0=char, 1=int, 2=float, 3=double
    int fcnt;
    fwrite ( &numpnt, sizeof(int), 1, fp );
    fwrite ( &numfield, sizeof(int), 1, fp );

    // write positions
    ftype = 2;
    fcnt = 3;		// float, 3 channel
    fwrite ( &ftype, sizeof(int), 1, fp );
    fwrite ( &fcnt,  sizeof(int), 1, fp );
    fwrite ( m_Fluid.bufC(FPOS),  numpnt*sizeof(Vector3DF), 1, fp );

    // write velocities
    ftype = 2;
    fcnt = 3;		// float, 3 channel
    fwrite ( &ftype, sizeof(int), 1, fp );
    fwrite ( &fcnt,  sizeof(int), 1, fp );
    fwrite ( m_Fluid.bufC(FVEL),  numpnt*sizeof(Vector3DF), 1, fp );

    // write colors
    ftype = 0;
    fcnt = 4;		// char, 4 channel
    fwrite ( &ftype, sizeof(int), 1, fp );
    fwrite ( &fcnt,  sizeof(int), 1, fp );
    fwrite ( m_Fluid.bufC(FCLR),  numpnt*sizeof(unsigned char)*4, 1, fp );

    fclose ( fp );

    fflush ( fp );
}

void FluidSystem::SavePointsVTP ( const char * relativePath, int frame ){
    std::cout << "\n  SavePointsVTP ( const char * relativePath = "<< relativePath << ", int frame = "<< frame << " );  started \n" << std::flush;
    char buf[256];
    frame += 100000;    // ensures numerical and alphabetic order match
    sprintf ( buf, "%s/particles_pos_vel_color%04d.vtp", relativePath, frame );
    FILE* fp = fopen ( buf, "w" );
    if (fp == NULL) {
        std::cout << "\nvoid FluidSystem::SavePointsVTP ( const char * relativePath, int frame )  Could not open file "<< fp <<"\n"<< std::flush;
        assert(0);
    }

    uint numpnt = NumPoints();
    uint numlines = numpnt * BONDS_PER_PARTICLE;
    Vector3DF* Pos;
    Vector3DF* Vel;
    uint* Age, *Clr, *NerveIdx, *ElastIdx, *Particle_Idx, *Particle_ID, *Mass_Radius, *Conc, *EpiGen;                  // Q: why are these pointers? A: they get dereferenced below.
    uint mass, radius;
    float *ElastIdxPtr;
//// file header    
    fprintf(fp, "x coord, y coord, z coord\t\t x vel, y vel, z vel\t\t age,  color\t\t FELASTIDX[%u*%u]", BONDS_PER_PARTICLE, DATA_PER_BOND);  // This system inserts commas to align header with csv data
    for (int i=0; i<BONDS_PER_PARTICLE; i++)fprintf(fp, ",[0]curIdx, [1]elastLim, [2]restLn, [3]modulus, [4]damping, [5]partID, [6]bond index,,  ");
    fprintf(fp, "\t"); 
    fprintf(fp, "\tParticle_ID, mass, radius, FNERVEIDX,\t\t Particle_Idx[%u*2]", BONDS_PER_PARTICLE);    
    for (int i=0; i<BONDS_PER_PARTICLE*3; i++)fprintf(fp, ", ");
    fprintf(fp, "\t\tFCONC[%u]", NUM_TF);
    for (int i=0; i<NUM_TF; i++)fprintf(fp, ", ");
    fprintf(fp, "\t\tFEPIGEN[%u] \n", NUM_GENES);
    
//// vtp xml file header
/*
<VTKFile type=”ImageData” version=”0.1” byte_order=”LittleEndian”>
...
</VTKFile>
*/
/*
type            — The type of the file (the bulleted items in the previous list)..
version         — File version number in “major.minor” format.
byte_order      — Machine byte order in which data are stored. This is either “BigEndian” or “LittleEndian”.
compressor      — Some data in the file may be compressed. This specifies the subclass of vtkDataCompressor that was used to compress the data.
*/
/*
<VTKFile type=”PolyData”  version=”0.1” byte_order=”LittleEndian”>
<PolyData>
<Piece
 NumberOfPoints=”#”
 NumberOfVerts=”#”
 NumberOfLines=”#”
 NumberOfStrips=”#” 
 NumberOfPolys=”#”>
<PointData>...</PointData>
<CellData>...</CellData>
<Points>...</Points>
<Verts>...</Verts>
<Lines>...</Lines>
<Strips>...</Strips>
<Polys>...</Polys>
</Piece>
</PolyData>
</VTKFile>
*/
fprintf(fp, "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">" );
fprintf(fp, "\n\t<PolyData>" );
fprintf(fp, "\n\t\t<Piece NumberOfPoints=\"%u\" NumberOfLines=\"%u\"  =\" \">",numpnt, numlines  );

// pointData
fprintf(fp, "\n\t<pointData> =\" \"  =\" \"  =\" \" ");// each DataArray holds data for different (sub)block of CSV file. 
fprintf(fp, "\n\t\t<DataArray type=\"Float32\" Name=\"vectors\" NumberOfComponents=\"3\" format=\"appended\" offset=\"0\"/>");

fprintf(fp, "\n\t\t<DataArray type=\"Float32\" Name=\"scalars\" format=\"binary\">");
// for loop : write data
fprintf(fp, "\n\t\t</DataArray>");

fprintf(fp, "\n\t\t<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">");
// for loop : write data
fprintf(fp, "\n\t\t</DataArray>");

fprintf(fp, "\n\t</pointData>");

//points
fprintf(fp, "\n\t<points> =\" \"  =\" \"  =\" \" ");// holds data for FPOS of each particle
fprintf(fp, "\n\t\t<Coordinates>");
fprintf(fp, "\n\t\t<DataArray type=\"Float32\" format=\"ascii\">");
          // for loop : write x-coord data
fprintf(fp, "\n\t\t</DataArray>");//x
fprintf(fp, "\n\t\t<DataArray type=\"Float32\" .../>");//y
fprintf(fp, "\n\t\t<DataArray type=\"Float32\" .../>");//z
fprintf(fp, "\n\t\t</Coordinates>");
fprintf(fp, "</points>");

/*
Points — The Points element explicitly defines coordinates for each point individually. It contains one
DataArray element describing an array with three components per value, each specifying the coordinates of one
point.
<Points>
<DataArray NumberOfComponents=\"3\" .../>
</Points>

Coordinates — The Coordinates element defines point coordinates for an extent by specifying the ordinate
along each axis for each integer value in the extent’s range. It contains three DataArray elements describing the
ordinates along the x-y-z axes, respectively.
<Coordinates>
<DataArray .../>
<DataArray .../>
<DataArray .../>
</Coordinates>
*/

fprintf(fp, "\n\t<lines> =\" \"  =\" \"  =\" \"");// holds indices for self and other particle for each bond, i.e. edges of .ply file.
fprintf(fp, "\n\t<DataArray type=\"UInt32\" Name=\"connectivity\" .../>" );
fprintf(fp, "\n\t<DataArray type=\"UInt32\" Name=\"offsets\" .../>" );
fprintf(fp, "\n\t<\\lines>");

fprintf(fp, "\n\t\t</Piece>" );
fprintf(fp, "\n\t</PolyData>" );
fprintf(fp, "\n</VTKFile>" );



    
// need some whole sim data - re num particles, particles per bond etc     


/// vtp file content
// NB points, lines (springs as visible edges), data arrays for each   block of the CSV line. NB which are uint vs float32  
    

    
//// file content    
//std::cout << "\n SavePointsCSV2: 1 \n" << std::flush;    
    for(int i=0; i<numpnt; i++) {       // nb need get..() accessors for private data.
//std::cout << " "<<i<<" " << std::flush;    
        Pos = getPos(i);                // e.g.  Vector3DF* getPos ( int n )	{ return &m_Fluid.bufV3(FPOS)[n]; }
        Vel = getVel(i);
        Age = getAge(i);
        Clr = getClr(i);
        ElastIdx = getElastIdx(i);      // NB [BONDS_PER_PARTICLE]
        ElastIdxPtr = (float*)ElastIdx; // #############packing floats and uints into the same array - should replace with a struct.#################
        Particle_Idx = getParticle_Idx(i);
        
        Particle_ID = getParticle_ID(i);//# uint  original pnum, used for bonds between particles. 32bit, track upto 4Bn particles.
//std::cout << " Particle_ID ="<<*Particle_ID<<",\t"<< std::flush;
        if(*Particle_ID==0){
         std::cout << "Particle_ID = pointer not assigned. i="<<i<<". \t" << std::flush;
         return;
        }
        // ? should I be splitting mass_radius with bitshift etc  OR just use two uit arrays .... where are/will these used anyway ?
        Mass_Radius = getMass_Radius(i);//# uint holding modulus 16bit and limit 16bit.
        if(*Mass_Radius==0){   mass = 0; }else{  mass = *Mass_Radius; }    // modulus          // '&' bitwise AND is bit masking. ;
        radius = mass >> 16;
        mass = mass & TWO_POW_16_MINUS_1;
        
        NerveIdx = getNerveIdx(i);      //# uint
        Conc = getConc(i);              //# uint[NUM_TF]        NUM_TF = num transcription factors & morphogens
        EpiGen = getEpiGen(i);          //# uint[NUM_GENES]
        
        fprintf(fp, "%f,%f,%f,\t%f,%f,%f,\t %u, %u,, \t", Pos->x, Pos->y,Pos->z, Vel->x,Vel->y,Vel->z, *Age, *Clr );
        
        for(int j=0; j<(BOND_DATA); j+=DATA_PER_BOND) { 
            fprintf(fp, "%u, %f, %f, %f, %f, %u, %u, ", ElastIdx[j], ElastIdxPtr[j+1], ElastIdxPtr[j+2], ElastIdxPtr[j+3], ElastIdxPtr[j+4], ElastIdx[j+5], ElastIdx[j+6] );
                // print as int   [0]current index, [5]particle ID, [6]bond index 
                // print as float [1]elastic limit, [2]restlength, [3]modulus, [4]damping coeff,   
            fprintf(fp, "\t\t");
        }

        fprintf(fp, " \t%u, %u, %u, %u, \t\t", *Particle_ID, mass, radius, *NerveIdx );
        for(int j=0; j<(BONDS_PER_PARTICLE*2); j+=2)   { fprintf(fp, "%u, %u,, ",  Particle_Idx[j], Particle_Idx[j+1] );}  fprintf(fp, "\t\t"); // NB index of other particle AND other particle's index of the bond
        for(int j=0; j<(NUM_TF); j++)               { fprintf(fp, "%u, ",  Conc[j] ); }         fprintf(fp, "\t\t");    
        for(int j=0; j<(NUM_GENES); j++)            { fprintf(fp, "%u, ",  EpiGen[j] );}        fprintf(fp, " \n");
    }
    fclose ( fp );
    fflush ( fp );
    
}

void FluidSystem::SavePointsVTP2 ( const char * relativePath, int frame ){// uses vtk library to write binary vtp files
    // based on VtpWrite(....)demo at https://vtk.org/Wiki/Write_a_VTP_file  (30 April 2009)
    // and on https://lorensen.github.io/VTKExamples/site/Cxx/IO/WriteVTP/   (post vtk-8.90.9)

    std::cout << "\n  SavePointsVTP2 ( const char * relativePath = "<< relativePath << ", int frame = "<< frame << " );  started \n" << std::flush;
    char buf[256];
    frame += 100000;    // ensures numerical and alphabetic order match
    sprintf ( buf, "%s/particles_pos_vel_color%04d.vtp", relativePath, frame );
    
    // Header information:  ?? how can this be added ??
    //  A) fparams & fgenome
    //  B) header of the.csv file, giving sizes of arrays.
    
    
    // points, vertices & lines
    // points & vertices = FPOS 3df
    vtkSmartPointer<vtkPoints> points3D = vtkSmartPointer<vtkPoints>::New();                           // Points3D
	vtkSmartPointer<vtkCellArray> Vertices = vtkSmartPointer<vtkCellArray>::New();                     // Vertices

    for ( unsigned int i = 0; i < NumPoints(); ++i )
	{	
		vtkIdType pid[1];
		//Point P = Model.Points[i];
        Vector3DF* Pos = getPos(i); 
		pid[0] = points3D->InsertNextPoint(Pos->x, Pos->y, Pos->z);
		Vertices->InsertNextCell(1,pid);
	}
    
    // edges = FELASTIDX [0]current index uint                                                         // Lines
    vtkSmartPointer<vtkCellArray> Lines = vtkSmartPointer<vtkCellArray>::New();
    uint *ElastIdx;
    for ( unsigned int i = 0; i < NumPoints(); ++i )
	{	
        ElastIdx = getElastIdx(i);   
        for(int j=0; j<(BONDS_PER_PARTICLE ); j++) { 
            int secondParticle = ElastIdx[j * DATA_PER_BOND];
            int bond = ElastIdx[j * DATA_PER_BOND +1];          // NB [0]current index, [1]elastic limit, [2]restlength, [3]modulus, [4]damping coeff, [5]particle ID, [6]bond index 
            if (bond==0) secondParticle = i;                    // i.e. if [1]elastic limit, then bond is broken, therefore bond to self.
            vtkSmartPointer<vtkLine> line = vtkSmartPointer<vtkLine>::New();
            line->GetPointIds()->SetId(0,i);
            line->GetPointIds()->SetId(1,secondParticle);
            Lines->InsertNextCell(line);
        }
	}

    ///////////////////////////////////////////////////////////////////////////////////////////////////// Particle Data 
    
    // FELASTIDX bond data, float and uint vtkDataArrays, stored in particles
    vtkSmartPointer<vtkUnsignedIntArray> BondsUIntData = vtkSmartPointer<vtkUnsignedIntArray>::New();
    BondsUIntData->SetNumberOfComponents(3);
	BondsUIntData->SetName("curr_idx, particle ID, bond index");
    
    vtkSmartPointer<vtkFloatArray> BondsFloatData = vtkSmartPointer<vtkFloatArray>::New();
    BondsFloatData->SetNumberOfComponents(4);
	BondsFloatData->SetName("elastic limit, restlength, modulus, damping coeff");
    
    float *ElastIdxPtr;
    for ( unsigned int i = 0; i < NumPoints(); ++i )
	{
        ElastIdx = getElastIdx(i);                          // FELASTIDX[BONDS_PER_PARTICLE]  [0]current index uint, [5]particle ID uint, [6]bond index uint
        ElastIdxPtr = (float*)ElastIdx;                     // FELASTIDX[BONDS_PER_PARTICLE]  [1]elastic limit float, [2]restlength float, [3]modulus float, [4]damping coeff float,
        for(int j=0; j<(BOND_DATA); j+=DATA_PER_BOND) { 
            BondsUIntData->InsertNextTuple3(ElastIdx[j], ElastIdx[j+5], ElastIdx[j+6]);
            BondsFloatData->InsertNextTuple4(ElastIdxPtr[j+1], ElastIdxPtr[j+2], ElastIdxPtr[j+3], ElastIdxPtr[j+4]);
        }
    }
    BondsUIntData->SetNumberOfComponents(BONDS_PER_PARTICLE *3);
    BondsFloatData->SetNumberOfComponents(BONDS_PER_PARTICLE *4); 
    

    // FVEL 3df, 
    Vector3DF* Vel;
    vtkSmartPointer<vtkFloatArray> fvel = vtkSmartPointer<vtkFloatArray>::New();
    fvel->SetNumberOfComponents(3);
	fvel->SetName("FVEL");
    for(unsigned int i=0;i<NumPoints();i++){
        Vel = getVel(i);
        fvel->InsertNextTuple3(Vel->x,Vel->y,Vel->z);
    }
    fvel->SetNumberOfComponents(BONDS_PER_PARTICLE *3);
    
    
/*    // FVEVAL 3df, 
    Vector3DF* Veval;
    vtkSmartPointer<vtkFloatArray> fveval = vtkSmartPointer<vtkFloatArray>::New();
    fvel->SetNumberOfComponents(3);
	fvel->SetName("FVEVAL");
    for(unsigned int i=0;i<NumPoints();i++){
        Veval = getVeval(i);
        fveval->InsertNextTuple3(Veval->x,Veval->y,Veval->z);
    }
    fveval->SetNumberOfComponents(BONDS_PER_PARTICLE *3);
*/
    
/*    // FFORCE 3df, 
    Vector3DF* Force;
    vtkSmartPointer<vtkFloatArray> fforce = vtkSmartPointer<vtkFloatArray>::New();
    fforce->SetNumberOfComponents(3);
	fforce->SetName("FFORCE");
    for(unsigned int i=0;i<NumPoints();i++){
        Force = getForce(i);
        fforce->InsertNextTuple3(Force->x,Force->y,Force->z);
    }
    fforce->SetNumberOfComponents(BONDS_PER_PARTICLE *3);
*/
    
    
/*    // FPRESS f,
    float* Pres;
    vtkSmartPointer<vtkFloatArray> fpres = vtkSmartPointer<vtkFloatArray>::New();
    fpres->SetNumberOfComponents(1);
	fpres->SetName("FPRESS");
    for(unsigned int i=0;i<NumPoints();i++){
        Pres = getPres(i);
        fpres->InsertNextTuple(Pres);
    }
*/     
    
/*    // FDENSITY f, 
    float* Dens;
    vtkSmartPointer<vtkFloatArray> fdens = vtkSmartPointer<vtkFloatArray>::New();
    fdens->SetNumberOfComponents(1);
	fdens->SetName("FDENSITY");
    for(unsigned int i=0;i<NumPoints();i++){
        Dens = getDensity(i);
        fdens->InsertNextTuple(Dens);
    }
*/
    
    // FAGE ushort, 
    unsigned int* age = getAge(0);
    vtkSmartPointer<vtkUnsignedIntArray> fage = vtkSmartPointer<vtkUnsignedIntArray>::New();
    fage->SetNumberOfComponents(1);
	fage->SetName("FAGE");
    for(unsigned int i=0;i<NumPoints();i++){
        fage->InsertNextValue(age[i]);
    }
    
    // FCLR uint, 
    unsigned int* color = getClr(0);
    vtkSmartPointer<vtkUnsignedIntArray> fcolor = vtkSmartPointer<vtkUnsignedIntArray>::New();
    fcolor->SetNumberOfComponents(1);
	fcolor->SetName("FCLR");
    for(unsigned int i=0;i<NumPoints();i++){
        fcolor->InsertNextValue(color[i]);
    }
    
    // FGCELL	uint, 
    
    // FPARTICLEIDX uint[BONDS_PER_PARTICLE *2],  
    
    // FPARTICLE_ID  uint, 
    unsigned int* pid = getParticle_ID(0);
    vtkSmartPointer<vtkUnsignedIntArray> fpid = vtkSmartPointer<vtkUnsignedIntArray>::New();
    fpid->SetNumberOfComponents(1);
	fpid->SetName("FPARTICLE_ID");
    for(unsigned int i=0;i<NumPoints();i++){
        fpid->InsertNextValue(pid[i]);
    }
    
    // FMASS_RADIUS uint (holding modulus 16bit and limit 16bit.),    
    unsigned int* Mass_Radius = getMass_Radius(0);
    uint mass, radius;
    vtkSmartPointer<vtkUnsignedIntArray> fmass_radius = vtkSmartPointer<vtkUnsignedIntArray>::New();
    fmass_radius->SetNumberOfComponents(2);
	fmass_radius->SetName("FMASS_RADIUS");
    for(unsigned int i=0;i<NumPoints();i++){
        if(Mass_Radius[i]==0){   mass = 0; }else{  mass = Mass_Radius[i]; } 
        radius = mass >> 16;
        mass = mass & TWO_POW_16_MINUS_1;
        fmass_radius->InsertNextTuple2(mass,radius);
    }
    
    // FNERVEIDX uint, 
    unsigned int* nidx = getNerveIdx(0);
    vtkSmartPointer<vtkUnsignedIntArray> fnidx = vtkSmartPointer<vtkUnsignedIntArray>::New();
    fnidx->SetNumberOfComponents(1);
	fnidx->SetName("FNERVEIDX");
    for(unsigned int i=0;i<NumPoints();i++){
        fnidx->InsertNextValue(nidx[i]);
    }
    
/*    // FCONC float[NUM_TF].                                                                                       // commented out until Matt's edit FCONC uint->foat is merged
    vtkSmartPointer<vtkFloatArray> fconc = vtkSmartPointer<vtkFloatArray>::New();
    fconc->SetNumberOfComponents(1);
	fconc->SetName("FCONC");
    float dummy = 0.1;
    float *conc = &dummy;//getConc(0);                   //#####  change FCONC to float, as Matt did. 
    for ( unsigned int i = 0, k=0; i < NumPoints(); ++i )
	{
        for(int j=0; j<NUM_TF; j++, k++)    fconc->InsertNextValue(conc[k]);
    }
    fconc->SetNumberOfComponents(NUM_TF); 
*/    
    
    // FEPIGEN uint[NUM_GENES]
    vtkSmartPointer<vtkUnsignedIntArray> fepigen = vtkSmartPointer<vtkUnsignedIntArray>::New();
    fepigen->SetNumberOfComponents(1);
	fepigen->SetName("FEPIGEN");
    unsigned int *epigen = getEpiGen(0);                   //#####  change FCONC to float, as Matt did. 
    for ( unsigned int i = 0, k=0; i < NumPoints(); ++i )
	{
        for(int j=0; j<NUM_GENES; j++, k++)    fepigen->InsertNextValue(epigen[k]);
    }
    fepigen->SetNumberOfComponents(NUM_GENES); 
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // POLYDATA
	vtkSmartPointer<vtkPolyData> polydata = vtkPolyData::New();                                        // polydata
	polydata->SetPoints(points3D);
	polydata->SetVerts(Vertices);
    polydata->SetLines(Lines);
    
    cout << "\nStarting writing bond data to polydata\n" << std::flush;

    polydata->GetPointData()->AddArray(BondsUIntData);
    polydata->GetPointData()->AddArray(BondsFloatData);
    polydata->GetPointData()->AddArray(fage);
    polydata->GetPointData()->AddArray(fcolor);
    polydata->GetPointData()->AddArray(fpid);
    polydata->GetPointData()->AddArray(fmass_radius);
    polydata->GetPointData()->AddArray(fnidx);
    //polydata->GetPointData()->AddArray(fconc);
    polydata->GetPointData()->AddArray(fepigen);
    
    
    cout << "\nFinished writing bond data to polydata\n" << std::flush;
    
    // WRITER  
	vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();       // writer
	writer->SetFileName(buf /*Filename.c_str()*/);
	writer->SetInputData(polydata);
    writer->SetDataModeToAscii();    
	writer->Write();
    
	cout << "\nFinished writing vtp file " << buf << "." << endl;
	cout << "\tNumPoints: " << NumPoints() << endl;

	//write finshed ok  
    
/*    
bool VtpWrite(ModelClass &Model, const string &Filename)
{
    if(Model.Colors.size() == 0)
	{
		for(unsigned int i = 0; i < Model.Points.size(); i++)
			Model.Colors.push_back(Red());
	}
	
	
	vtkSmartPointer<vtkPoints> points3D = vtkSmartPointer<vtkPoints>::New();                           // points3D
	vtkSmartPointer<vtkCellArray> Vertices = vtkSmartPointer<vtkCellArray>::New();                     // Vertices

	for ( unsigned int i = 0; i < Model.Points.size(); ++i )
	{	
		vtkIdType pid[1];
		Point P = Model.Points[i];
		pid[0] = points3D->InsertNextPoint(P.x, P.y, P.z);
		Vertices->InsertNextCell(1,pid);
	}

	
	vtkSmartPointer<vtkUnsignedCharArray> Colors = vtkSmartPointer<vtkUnsignedCharArray>::New();       // Colors

	Colors->SetNumberOfComponents(3);
	Colors->SetName("Colors");

	for ( unsigned int i = 0; i < Model.Colors.size(); ++i )
	{
		Color Color = Model.Colors[i];
		unsigned char ColorArray[3];
		CharArray(Color, ColorArray);
		Colors->InsertNextTupleValue(ColorArray);
	}

	
	bool HasTriangles;
	if(Model.VertexList.size() > 0)
		HasTriangles = true;
	else
		HasTriangles = false;

    
	vtkSmartPointer<vtkCellArray> triangles = vtkSmartPointer<vtkCellArray>::New();                    // triangles
	if(HasTriangles)
	{
		for(unsigned int i = 0; i < Model.VertexList.size(); i++)
		{
			vector<int> vlist = Model.VertexList[i];
			vtkSmartPointer<vtkTriangle> triangle = vtkSmartPointer<vtkTriangle>::New();
			triangle->GetPointIds()->SetId(0,vlist[0]);
			triangle->GetPointIds()->SetId(1,vlist[1]);
			triangle->GetPointIds()->SetId(2,vlist[2]);
			triangles->InsertNextCell(triangle);
		}
	}

	
	vtkSmartPointer<vtkPolyData> polydata = vtkPolyData::New();                                        // polydata

	polydata->SetPoints(points3D);
	polydata->SetVerts(Vertices);
	
	if(HasTriangles)
		polydata->SetPolys(triangles);

	polydata->GetPointData()->SetVectors(Colors);

	
	
	vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();       // writer
	writer->SetFileName(Filename.c_str());
	writer->SetInput(polydata);
	writer->Write();	

	
	
	cout << "Finished writing vtp file " << Filename << "." << endl;
	cout << "NumPoints: " << Model.Points.size() << endl;
	cout << "NumColors: " << Model.Colors.size() << endl;

	return true;//write finshed ok  
}
*/    
}

void FluidSystem::SavePointsCSV ( const char * relativePath, int frame )
{
    std::cout << "\n  SavePointsCSV ( const char * relativePath, int frame );  started \n" << std::flush;
    char buf[256];
    frame += 100000;    // ensures numerical and alphabetic order match
    sprintf ( buf, "%s/particles_pos_vel_color%04d.csv", relativePath, frame );
    FILE* fp = fopen ( buf, "w" );
    if (fp == NULL) {
        std::cout << "\nvoid FluidSystem::SavePointsCSV ( const char * relativePath, int frame )  Could not open file "<< fp <<"\n"<< std::flush;
        assert(0);
    }
    int numpnt = NumPoints();
    Vector3DF* Pos;
    Vector3DF* Vel;
    float *Conc;
    uint* Age, *Clr, *NerveIdx, *ElastIdx, *EpiGen;

    //#define FELASTIDX   14      //# uint[BONDS_PER_PARTICLE +1]  0=self UID, mass, radius. >0= modulus & particle UID
    //#define FNERVEIDX   15      //# uint
    //#define FCONC       16      //# float[NUM_TF]        NUM_TF = num transcription factors & morphogens
    //#define FEPIGEN     17      //# uint[NUM_GENES]

    fprintf(fp, "x coord, y coord, z coord,\t x vel, y vel, z vel,\t age,  color, \tFELASTIDX[%u], \tFNERVEIDX\t, FCONC[%u], \tFEPIGEN[%u] \n",
            (BOND_DATA), NUM_TF, NUM_GENES );

    for(int i=0; i<numpnt; i++) {
        Pos = getPos(i);
        Vel = getVel(i);
        Age = getAge(i);
        Clr = getClr(i);
        ElastIdx = getElastIdx(i);
        NerveIdx = getNerveIdx(i);
        Conc = getConc(i);
        EpiGen = getEpiGen(i);
        fprintf(fp, "%f,%f,%f,\t%f,%f,%f,\t %u, %u, \t", Pos->x, Pos->y,Pos->z, Vel->x,Vel->y,Vel->z, *Age, *Clr );

        for(int j=0; j<(BOND_DATA); j++) {
            fprintf(fp, "%u, ",  ElastIdx[j] );
        }
        fprintf(fp, "\t%u, \t", *NerveIdx);

        for(int j=0; j<(NUM_TF); j++) {
            fprintf(fp, "%f, ",  Conc[j] );
        }
        fprintf(fp, "\t");

        for(int j=0; j<(NUM_GENES); j++) {
            fprintf(fp, "%u, ",  EpiGen[j] );
        }
        fprintf(fp, " \n");
    }
    fclose ( fp );
    fflush ( fp );
}

void FluidSystem::SavePointsCSV2 ( const char * relativePath, int frame )
{
    std::cout << "\n  SavePointsCSV2 ( const char * relativePath = "<< relativePath << ", int frame = "<< frame << " );  started \n" << std::flush;
    char buf[256];
    frame += 100000;    // ensures numerical and alphabetic order match
    sprintf ( buf, "%s/particles_pos_vel_color%04d.csv", relativePath, frame );
    FILE* fp = fopen ( buf, "w" );
    if (fp == NULL) {
        std::cout << "\nvoid FluidSystem::SavePointsCSV ( const char * relativePath, int frame )  Could not open file "<< fp <<"\n"<< std::flush;
        assert(0);
    }

    int numpnt = NumPoints();
    Vector3DF* Pos;
    Vector3DF* Vel;
    float *Conc;
    uint* Age, *Clr, *NerveIdx, *ElastIdx, *Particle_Idx, *Particle_ID, *Mass_Radius, *EpiGen;                  // Q: why are these pointers? A: they get dereferenced below.
    uint mass, radius;
    float *ElastIdxPtr;
    
    fprintf(fp, "x coord, y coord, z coord\t\t x vel, y vel, z vel\t\t age,  color\t\t FELASTIDX[%u*%u]", BONDS_PER_PARTICLE, DATA_PER_BOND);  // This system inserts commas to align header with csv data
    for (int i=0; i<BONDS_PER_PARTICLE; i++)fprintf(fp, ",[0]curIdx, [1]elastLim, [2]restLn, [3]modulus, [4]damping, [5]partID, [6]bond index,,  ");
    fprintf(fp, "\t"); 
    fprintf(fp, "\tParticle_ID, mass, radius, FNERVEIDX,\t\t Particle_Idx[%u*2]", BONDS_PER_PARTICLE);    
    for (int i=0; i<BONDS_PER_PARTICLE*3; i++)fprintf(fp, ", ");
    fprintf(fp, "\t\tFCONC[%u]", NUM_TF);
    for (int i=0; i<NUM_TF; i++)fprintf(fp, ", ");
    fprintf(fp, "\t\tFEPIGEN[%u] \n", NUM_GENES);

//std::cout << "\n SavePointsCSV2: 1 \n" << std::flush;    

    for(int i=0; i<numpnt; i++) {       // nb need get..() accessors for private data.
//std::cout << " "<<i<<" " << std::flush;    
        Pos = getPos(i);                // e.g.  Vector3DF* getPos ( int n )	{ return &m_Fluid.bufV3(FPOS)[n]; }
        Vel = getVel(i);
        Age = getAge(i);
        Clr = getClr(i);
        ElastIdx = getElastIdx(i);      // NB [BONDS_PER_PARTICLE]
        ElastIdxPtr = (float*)ElastIdx; // #############packing floats and uints into the same array - should replace with a struct.#################
        Particle_Idx = getParticle_Idx(i);
        
        Particle_ID = getParticle_ID(i);//# uint  original pnum, used for bonds between particles. 32bit, track upto 4Bn particles.
//std::cout << " Particle_ID ="<<*Particle_ID<<",\t"<< std::flush;
        if(*Particle_ID==0){
         std::cout << "Particle_ID = pointer not assigned. i="<<i<<". \t" << std::flush;
         return;
        }
        // ? should I be splitting mass_radius with bitshift etc  OR just use two uit arrays .... where are/will these used anyway ?
        Mass_Radius = getMass_Radius(i);//# uint holding modulus 16bit and limit 16bit.
        if(*Mass_Radius==0){   mass = 0; }else{  mass = *Mass_Radius; }    // modulus          // '&' bitwise AND is bit masking. ;
        radius = mass >> 16;
        mass = mass & TWO_POW_16_MINUS_1;
        
        NerveIdx = getNerveIdx(i);      //# uint
        Conc = getConc(i);              //# float[NUM_TF]        NUM_TF = num transcription factors & morphogens
        EpiGen = getEpiGen(i);          //# uint[NUM_GENES]
        
        fprintf(fp, "%f,%f,%f,\t%f,%f,%f,\t %u, %u,, \t", Pos->x, Pos->y,Pos->z, Vel->x,Vel->y,Vel->z, *Age, *Clr );
        
        for(int j=0; j<(BOND_DATA); j+=DATA_PER_BOND) { 
            fprintf(fp, "%u, %f, %f, %f, %f, %u, %u, ", ElastIdx[j], ElastIdxPtr[j+1], ElastIdxPtr[j+2], ElastIdxPtr[j+3], ElastIdxPtr[j+4], ElastIdx[j+5], ElastIdx[j+6] );
           // if ((j%DATA_PER_BOND==0)||((j+1)%DATA_PER_BOND==0))  fprintf(fp, "%u, ",  ElastIdx[j] );  // print as int   [0]current index, [5]particle ID, [6]bond index 
           // else  fprintf(fp, "%f, ",  ElastIdxPtr[j] );                                              // print as float [1]elastic limit, [2]restlength, [3]modulus, [4]damping coeff, 
           //  if((j+1)%DATA_PER_BOND==0)  
            fprintf(fp, "\t\t");
        }
/*        
        for(int j=0; j<(BONDS_PER_PARTICLE); j++)            { 
            fprintf(fp, "%u, ",  ElastIdx[j] );
            for(int k=1;k<(DATA_PER_BOND-1);k++){
                fprintf(fp, "%f, ",  ElastIdxPtr[j+k] );       // save floats for [1]elastic limit, [2]restlength, [3]modulus, [4]damping coeff,
            }
            fprintf(fp, "%u, ",  ElastIdx[j+(BONDS_PER_PARTICLE-1)] );
            fprintf(fp, "\t\t"); // if((j+1)%DATA_PER_BOND==0)
        }
*/
        fprintf(fp, " \t%u, %u, %u, %u, \t\t", *Particle_ID, mass, radius, *NerveIdx );
        for(int j=0; j<(BONDS_PER_PARTICLE*2); j+=2)   { fprintf(fp, "%u, %u,, ",  Particle_Idx[j], Particle_Idx[j+1] );}  fprintf(fp, "\t\t"); // NB index of other particle AND other particle's index of the bond
        for(int j=0; j<(NUM_TF); j++)               { fprintf(fp, "%f, ",  Conc[j] ); }         fprintf(fp, "\t\t");
        for(int j=0; j<(NUM_GENES); j++)            { fprintf(fp, "%u, ",  EpiGen[j] );}        fprintf(fp, " \n");
    }
    fclose ( fp );
    fflush ( fp );
}

void FluidSystem::ReadPointsCSV ( const char * relativePath, int gpu_mode, int cpu_mode)     // NB allocates buffers as well.
{
    const char * points_file_path = relativePath;
    printf("\n## opening file %s ", points_file_path);
    FILE * points_file = fopen(points_file_path, "rb");
    if (points_file == NULL) {
        std::cout << "\nvoid FluidSystem::ReadPointsCSV ( const char * relativePath, int gpu_mode, int cpu_mode )  Could not read file "<< points_file_path <<"\n"<< std::flush;
        assert(0);
    }
    // find number of lines = number of particles
    int ch, number_of_lines = 0;
    while (EOF != (ch=getc(points_file)))   if ('\n' == ch)  ++number_of_lines;

    // Allocate buffers for points
    m_Param [PNUM] = number_of_lines;
    mMaxPoints = m_Param [PNUM];
    m_Param [PGRIDSIZE] = 2*m_Param[PSMOOTHRADIUS] / m_Param[PGRID_DENSITY];

    SetupKernels ();
    SetupSpacing ();
    SetupGrid ( m_Vec[PVOLMIN]/*bottom corner*/, m_Vec[PVOLMAX]/*top corner*/, m_Param[PSIMSCALE], m_Param[PGRIDSIZE], 1.0f );

    if (gpu_mode != GPU_OFF) {
        FluidSetupCUDA ( mMaxPoints, m_GridSrch, *(int3*)& m_GridRes, *(float3*)& m_GridSize, *(float3*)& m_GridDelta, *(float3*)& m_GridMin, *(float3*)& m_GridMax, m_GridTotal, 0 );
        UpdateParams();            //  sends simulation params to device.
        UpdateGenome();            //  sends genome to device.              // NB need to initialize genome from file, or something.
    }

    AllocateParticles ( mMaxPoints, gpu_mode, cpu_mode );  // allocates only cpu buffer for particles
    AllocateGrid(gpu_mode, cpu_mode);

    uint Clr, Age;
    Vector3DF Pos, Vel, PosMin, PosMax;
    uint ElastIdx[BOND_DATA];
    uint NerveIdx;
    float Conc[NUM_TF];
    uint EpiGen[NUM_GENES];

    float vel_lim = GetParam ( PVEL_LIMIT );
    PosMin = GetVec ( PBOUNDMIN );
    PosMax = GetVec ( PBOUNDMAX );

    std::fseek(points_file, 0, SEEK_SET);
    //std::fscanf(points_file, "x coord, y coord, z coord, x vel, y vel, z vel,  color \n");
    //fprintf(fp, "x coord, y coord, z coord,\t x vel, y vel, z vel,\t age,  color, \tFELASTIDX[%u], \tFNERVEIDX\t, FCONC[%u], \tFEPIGEN[%u] \n",
    //        (BONDS_PER_PARTICLE +1), NUM_TF, NUM_GENES );
    uint bonds_per_particle, num_TF, num_genes;
    std::fscanf(points_file, "x coord, y coord, z coord,\t x vel, y vel, z vel,\t age,  color, \tFELASTIDX[%u], \tFNERVEIDX\t, FCONC[%u], \tFEPIGEN[%u] \n",
                &bonds_per_particle, &num_TF, &num_genes );

    if (bonds_per_particle != (BOND_DATA) || num_TF != NUM_TF || num_genes != NUM_GENES)   // ## later replace #defines with variables throughout.
    {
        std::cout << "\n! Miss-match of parameters !  (bonds_per_particle != (BOND_DATA) || num_TF != NUM_TF || num_genes != NUM_GENES) \n";
        std::cout << "\n In .csv file  bonds_per_particle = " << bonds_per_particle << ".  In program (BOND_DATA) = " << (BOND_DATA) ;
        std::cout << "\n num_TF  = " << num_TF << ",  NUM_TF = " << NUM_TF ;
        std::cout << "\n num_genes  = " << num_genes << ",  NUM_GENES = " << NUM_GENES ;
    }
    ////////////////////
    //Vector3DF* Pos;
    //Vector3DF* Vel;
    //uint /* *Age, *Clr, */ *NerveIdx, *ElastIdx, *Conc, *EpiGen;
    int i;
    for (i=1; i<number_of_lines; i++ ) {
        // transcribe particle data from file to Pos, Vel and Clr
        //int ret = std::fscanf(points_file, "%f,%f,%f,%f,%f,%f, %u \n", &Pos.x, &Pos.y, &Pos.z, &Vel.x, &Vel.y, &Vel.z, &Clr);

        int ret = std::fscanf(points_file, "%f,%f,%f,\t%f,%f,%f,\t %u, %u, \t", &Pos.x, &Pos.y, &Pos.z, &Vel.x, &Vel.y, &Vel.z, &Age, &Clr );

        for(int j=0; j<(BOND_DATA); j++) {
            ret += std::fscanf(points_file, "%u, ",  &ElastIdx[j] );
        }
        ret += std::fscanf(points_file, "\t%u, \t", &NerveIdx);

        for(int j=0; j<(NUM_TF); j++) {
            ret += std::fscanf(points_file, "%f, ",  &Conc[j] );
        }
        ret += std::fscanf(points_file, "\t");

        for(int j=0; j<(NUM_GENES); j++) {
            ret += std::fscanf(points_file, "%u, ",  &EpiGen[j] );
        }
        ret += std::fscanf(points_file, " \n");
        /////////////////////////////

        if (ret != (8 + (BOND_DATA) + 1 + NUM_TF + NUM_GENES) ) {
            std::cout << "\nvoid FluidSystem::ReadPointsCSV, read failure !  particle number = " << i;
            std::cout << "\n " << std::flush;
            fclose(points_file);
            return;
        }
        // check particle is within simulation bounds
        if (Pos.x < PosMin.x || Pos.y < PosMin.y || Pos.z < PosMin.z
                || Pos.x > PosMax.x   || Pos.y > PosMax.y || Pos.z > PosMax.z
                || (Vel.x * Vel.x + Vel.y * Vel.y + Vel.z * Vel.z) > vel_lim * vel_lim )
        {
            std::cout << "\n void FluidSystem::ReadPointsCSV, out of bounds !  particle number = " << i;
            std::cout << "\n Pos.x = " << Pos.x << "  Pos.y = " << Pos.y << "  Pos.z = " << Pos.z;
            std::cout << "\n PosMax.x = " << PosMax.x << "  PosMax.y = " << PosMax.y << "  PosMax.z = " << PosMax.z;
            std::cout << "\n PosMin.x = " << PosMin.x << "  PosMin.y = " << PosMin.y << "  PosMin.z = " << PosMin.z;
            std::cout << "\n velocity = " << sqrt(Vel.x * Vel.x + Vel.y * Vel.y + Vel.z * Vel.z) << "   vel_lim = " << vel_lim;
            std::cout << "\n " << std::flush;
            fclose(points_file);
            return;
        }
        AddParticleMorphogenesis (&Pos, &Vel, Age, Clr, ElastIdx, NerveIdx, Conc, EpiGen );
    }
    std::cout << "\n" << i-1 << " particles read.\n" << std::flush;
    fclose(points_file);

    if (gpu_mode != GPU_OFF) {
        TransferToCUDA ();		 // Initial transfer
    }

}


void FluidSystem::ReadPointsCSV2 ( const char * relativePath, int gpu_mode, int cpu_mode)     // NB allocates buffers as well.
{
    std::cout << "\n  ReadPointsCSV2 ( const char * relativePath, int gpu_mode, int cpu_mode);  started \n" << std::flush;
    const char * points_file_path = relativePath;
    printf("\n## opening file %s ", points_file_path);
    FILE * points_file = fopen(points_file_path, "rb");
    if (points_file == NULL) {
        std::cout << "\nvoid FluidSystem::ReadPointsCSV ( const char * relativePath, int gpu_mode, int cpu_mode )  Could not read file "<< points_file_path <<"\n"<< std::flush;
        assert(0);
    }
    // find number of lines = number of particles
    int ch, number_of_lines = 0;
    while (EOF != (ch=getc(points_file)))   if ('\n' == ch)  ++number_of_lines;

    // Allocate buffers for points
    m_Param [PNUM] = number_of_lines -1;                                    // NB there is a line of text above the particles, hence -1.
    mMaxPoints = m_Param [PNUM];
    m_Param [PGRIDSIZE] = 2*m_Param[PSMOOTHRADIUS] / m_Param[PGRID_DENSITY];

    SetupKernels ();
    SetupSpacing ();
    SetupGrid ( m_Vec[PVOLMIN]/*bottom corner*/, m_Vec[PVOLMAX]/*top corner*/, m_Param[PSIMSCALE], m_Param[PGRIDSIZE], 1.0f );
//std::cout<<"\nchk ReadPointsCSV2 1.0\n"<<std::flush;
    if (gpu_mode != GPU_OFF) {
        FluidSetupCUDA ( mMaxPoints, m_GridSrch, *(int3*)& m_GridRes, *(float3*)& m_GridSize, *(float3*)& m_GridDelta, *(float3*)& m_GridMin, *(float3*)& m_GridMax, m_GridTotal, 0 );
        UpdateParams();            //  sends simulation params to device.
        UpdateGenome();            //  sends genome to device.              // NB need to initialize genome from file, or something.
    }
//std::cout<<"\nchk ReadPointsCSV2 1.1\n"<<std::flush;
    AllocateParticles ( mMaxPoints, gpu_mode, cpu_mode );  // allocates only cpu buffer for particles
//std::cout<<"\nchk ReadPointsCSV2 1.2\n"<<std::flush;
    AllocateGrid(gpu_mode, cpu_mode);
//std::cout<<"\nchk ReadPointsCSV2 1.3\n"<<std::flush;
    uint Clr, Age;
    Vector3DF Pos, Vel, PosMin, PosMax;
    float ElastIdx[BOND_DATA];
    uint Particle_Idx[BONDS_PER_PARTICLE * 2];
    uint Particle_ID, mass, radius, Mass_Radius, NerveIdx;
    float Conc[NUM_TF];
    uint EpiGen[NUM_GENES];

    float vel_lim = GetParam ( PVEL_LIMIT );
    PosMin = GetVec ( PBOUNDMIN );
    PosMax = GetVec ( PBOUNDMAX );

    std::fseek(points_file, 0, SEEK_SET);
    //std::fscanf(points_file, "x coord, y coord, z coord, x vel, y vel, z vel,  color \n");
    //fprintf(fp, "x coord, y coord, z coord,\t x vel, y vel, z vel,\t age,  color, \tFELASTIDX[%u], \tFNERVEIDX\t, FCONC[%u], \tFEPIGEN[%u] \n",
    //        (BONDS_PER_PARTICLE +1), NUM_TF, NUM_GENES );
    uint bond_data=999, data_per_bond=999, bonds_per_particle=999, num_TF=999, num_genes=999;
    int result=-2;
    result = std::fscanf(points_file, "x coord, y coord, z coord\t\t x vel, y vel, z vel\t\t age,  color\t\t FELASTIDX[%u*%u]", &bond_data, &data_per_bond);
    
    for (int i=0; i<data_per_bond; i++)std::fscanf(points_file, ",[0]curIdx, [1]elastLim, [2]restLn, [3]modulus, [4]damping, [5]partID, [6]bond index,,  ");
    bond_data = bond_data * data_per_bond;
    fscanf(points_file, "\t");
//    printf("\nresult=%i, bond_data=%u",result,bond_data);
    
    result = std::fscanf(points_file, "\tParticle_ID, mass, radius, FNERVEIDX,\t\t Particle_Idx[%u*2]", &bonds_per_particle);
    for (int i=0; i<BONDS_PER_PARTICLE*3; i++)fscanf(points_file, ", ");
//    printf("\nresult=%i, bonds_per_particle=%u",result,bonds_per_particle);
    
    result = std::fscanf(points_file, "\t\tFCONC[%u]",&num_TF);
    for (int i=0; i<NUM_TF; i++)fscanf(points_file, ", ");
//    printf("\nresult=%i, num_TF=%u",result,num_TF);
    
    result = std::fscanf(points_file, "\t\tFEPIGEN[%u] \n", &num_genes );
//    printf("\nresult=%i, num_genes=%u",result,num_genes);

    /*
    fprintf(fp, "x coord, y coord, z coord\t\t x vel, y vel, z vel\t\t age,  color\t\t FELASTIDX[%u]", BOND_DATA);  // This system inserts commas to align header with csv data
    for (int i=0; i<BOND_DATA ; i++)fprintf(fp, ", ");
    for (int i=0; i<BONDS_PER_PARTICLE ; i++)fprintf(fp, "\t\t"); 
    fprintf(fp, "\tParticle_ID, mass, radius, FNERVEIDX,\t\t Particle_Idx[%u]", BONDS_PER_PARTICLE);
    for (int i=0; i<BONDS_PER_PARTICLE; i++)fprintf(fp, ", ");
    fprintf(fp, "\t\tFCONC[%u]", NUM_TF);
    for (int i=0; i<NUM_TF; i++)fprintf(fp, ", ");
    fprintf(fp, "\t\tFEPIGEN[%u] \n", NUM_GENES);
    */

    if (bonds_per_particle != BONDS_PER_PARTICLE || bond_data != BOND_DATA || num_TF != NUM_TF || num_genes != NUM_GENES)   // ## later replace #defines with variables throughout.
    {
        std::cout << "\n! Miss-match of parameters !  (bonds_per_particle != BONDS_PER_PARTICLE || bond_data != BOND_DATA || num_TF != NUM_TF || num_genes != NUM_GENES) \n";
        std::cout << "\n In .csv file  bonds_per_particle = " << bonds_per_particle << ".  In program BONDS_PER_PARTICLE = " << BONDS_PER_PARTICLE ;
        std::cout << "\n bond_data  = " << bond_data << ",  BOND_DATA = " << BOND_DATA ;
        std::cout << "\n num_TF  = "    << num_TF    << ",  NUM_TF = "    << NUM_TF ;
        std::cout << "\n num_genes  = " << num_genes << ",  NUM_GENES = " << NUM_GENES << std::flush;
    }
//    std::cout << "\n" << std::flush;
    ////////////////////
    //Vector3DF* Pos;
    //Vector3DF* Vel;
    //uint /* *Age, *Clr, */ *NerveIdx, *ElastIdx, *Conc, *EpiGen;
    int i;
    for (i=1; i<number_of_lines; i++ ) {
        // transcribe particle data from file to Pos, Vel and Clr
        //int ret = std::fscanf(points_file, "%f,%f,%f,%f,%f,%f, %u \n", &Pos.x, &Pos.y, &Pos.z, &Vel.x, &Vel.y, &Vel.z, &Clr);
        int ret = std::fscanf(points_file, "%f,%f,%f,\t%f,%f,%f,\t %u, %u,, \t", &Pos.x, &Pos.y, &Pos.z, &Vel.x, &Vel.y, &Vel.z, &Age, &Clr );
//        std::cout<<"ret="<<ret<<"\t"<< std::flush;
        
        for(int j=0; j<(BOND_DATA); j++) {// BONDS_PER_PARTICLE * DATA_PER_BOND
            ret += std::fscanf(points_file, "%f, ",  &ElastIdx[j] );
        }
//        std::cout<<"ret="<<ret<<"\t"<< std::flush;
        //fprintf(fp, " \t%u, %u, %u, %u, \t\t", *Particle_ID, mass, radius, *NerveIdx );
        ret += std::fscanf(points_file, " \t%u, %u, %u, %u, \t\t", &Particle_ID, &mass, &radius, &NerveIdx);
        Mass_Radius = mass + (radius << 16);                                    // pack two 16bit uint  into one 32bit uint.
//        std::cout<<"ret="<<ret<<"\t"<< std::flush;

        //for(int j=0; j<(BONDS_PER_PARTICLE); j++)   { fprintf(fp, "%u, ",  Particle_Idx[j] ); }
        std::cout<<"\t(i="<<i<<",j=";
        for(int j=0; j<(BONDS_PER_PARTICLE*2); j+=2) {
            ret += std::fscanf(points_file, "%u, %u,, ",  &Particle_Idx[j], &Particle_Idx[j+1] );
            std::cout<<Particle_Idx[j]<<", ";
        }
        std::cout<<")"<<std::flush;
//        std::cout<<"ret="<<ret<<"\t"<< std::flush;
        
        for(int j=0; j<(NUM_TF); j++)       {    ret += std::fscanf(points_file, "%f, ",  &Conc[j] );   } ret += std::fscanf(points_file, "\t");
//        std::cout<<"ret="<<ret<<"\t"<< std::flush;
        for(int j=0; j<(NUM_GENES); j++)    {    ret += std::fscanf(points_file, "%u, ",  &EpiGen[j] ); } ret += std::fscanf(points_file, " \n");
//        std::cout<<"ret="<<ret<<"\t"<< std::flush;

        if (ret != (8 + BOND_DATA + 4 + BONDS_PER_PARTICLE*2 + NUM_TF + NUM_GENES) ) {  
            std::cout << "\nvoid FluidSystem::ReadPointsCSV, read failure !  particle number = " << i;
            std::cout << "\nret=" << ret;
            std::cout << "\n " << std::flush;
            fclose(points_file);
            return;
        } // ret=8 ret=32 ret=36 ret=48 ret=64 ret=80 

        // check particle is within simulation bounds
        if (Pos.x < PosMin.x || Pos.y < PosMin.y || Pos.z < PosMin.z
                || Pos.x > PosMax.x   || Pos.y > PosMax.y || Pos.z > PosMax.z
                || (Vel.x * Vel.x + Vel.y * Vel.y + Vel.z * Vel.z) > vel_lim * vel_lim )
        {
            std::cout << "\n void FluidSystem::ReadPointsCSV, out of bounds !  particle number = " << i;
            std::cout << "\n Pos.x = " << Pos.x << "  Pos.y = " << Pos.y << "  Pos.z = " << Pos.z;
            std::cout << "\n PosMax.x = " << PosMax.x << "  PosMax.y = " << PosMax.y << "  PosMax.z = " << PosMax.z;
            std::cout << "\n PosMin.x = " << PosMin.x << "  PosMin.y = " << PosMin.y << "  PosMin.z = " << PosMin.z;
            std::cout << "\n velocity = " << sqrt(Vel.x * Vel.x + Vel.y * Vel.y + Vel.z * Vel.z) << "   vel_lim = " << vel_lim;
            std::cout << "\n " << std::flush;
            fclose(points_file);
            return;
        }
        AddParticleMorphogenesis2 (&Pos, &Vel, Age, Clr, ElastIdx, Particle_Idx, Particle_ID, Mass_Radius,  NerveIdx, Conc, EpiGen );
        std::cout << ",\t" << std::flush;
    }
    std::cout << "\n" << i-1 << " particles read.\n" << std::flush;
    fclose(points_file);

    if (gpu_mode != GPU_OFF) {
        TransferToCUDA ();		 // Initial transfer
    }
}


void FluidSystem::ReadSimParams ( const char * relativePath ) // transcribe SimParams from file to fluid_system object.
{
    const char * SimParams_file_path = relativePath;
    printf ( "\n## opening file %s ", SimParams_file_path );
    FILE * SimParams_file = fopen ( SimParams_file_path, "rb" );
    if ( SimParams_file == NULL ) {
        std::cout << "\nvoid FluidSystem::ReadSimParams (const char * relativePath )  Could not read file "<< SimParams_file_path <<"\n"<< std::flush;
        assert ( 0 );
    }
    // find number of lines
    int ch, number_of_lines = 0;
    while ( EOF != ( ch=getc ( SimParams_file ) ) )   if ( '\n' == ch )  ++number_of_lines; // chk num lines
    std::cout << "\nNumber of lines in SimParams_file = " << number_of_lines << std::flush;

    Vector3DF point_grav_pos, pplane_grav_dir, pemit_pos, pemit_rate, pemit_ang, pemit_dang, pvolmin, pvolmax, pinitmin, pinitmax;
    int pwrapx, pwall_barrier, plevy_barrier, pdrain_barrier, prun;

    std::fseek(SimParams_file, 0, SEEK_SET);
    int ret = std::fscanf ( SimParams_file, " m_Time = %f\n ", &m_Time );
    ret += std::fscanf ( SimParams_file, "m_DT = %f\n ", &m_DT );
    ret += std::fscanf ( SimParams_file, "m_Param [ PSIMSCALE ] = %f\n ", &m_Param [ PSIMSCALE ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PGRID_DENSITY ] = %f\n ", &m_Param [ PGRID_DENSITY ] ); // added
    ret += std::fscanf ( SimParams_file, "m_Param [ PVISC ] = %f\n ", &m_Param [ PVISC ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PRESTDENSITY ] = %f\n ", &m_Param [ PRESTDENSITY ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PSPACING ] = %f\n ", &m_Param [ PSPACING ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PMASS ] = %f\n ", &m_Param [ PMASS ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PRADIUS ] = %f\n ", &m_Param [ PRADIUS ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PDIST ] = %f\n ", &m_Param [ PDIST ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PSMOOTHRADIUS ] = %f\n ", &m_Param [ PSMOOTHRADIUS ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PINTSTIFF ] = %f\n ", &m_Param [ PINTSTIFF ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PEXTSTIFF ] = %f\n ", &m_Param [ PEXTSTIFF ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PEXTDAMP ] = %f\n ", &m_Param [ PEXTDAMP ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PACCEL_LIMIT ] = %f\n ", &m_Param [ PACCEL_LIMIT ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PVEL_LIMIT ] = %f\n ", &m_Param [ PVEL_LIMIT ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PMAX_FRAC ] = %f\n ", &m_Param [ PMAX_FRAC ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PGRAV ] = %f\n ", &m_Param [ PGRAV ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PGROUND_SLOPE ] = %f\n ", &m_Param [ PGROUND_SLOPE ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PFORCE_MIN ] = %f\n ", &m_Param [ PFORCE_MIN ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PFORCE_MAX ] = %f\n ", &m_Param [ PFORCE_MAX ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PFORCE_FREQ ] = %f\n ", &m_Param [ PFORCE_FREQ ] );
    ret += std::fscanf ( SimParams_file, "m_Toggle [ PWRAP_X ] = %i\n ", &pwrapx );
    ret += std::fscanf ( SimParams_file, "m_Toggle [ PWALL_BARRIER ] = %i\n ", &pwall_barrier );
    ret += std::fscanf ( SimParams_file, "m_Toggle [ PLEVY_BARRIER ] = %i\n ", &plevy_barrier );
    ret += std::fscanf ( SimParams_file, "m_Toggle [ PDRAIN_BARRIER ] = %i\n ", &pdrain_barrier );
    ret += std::fscanf ( SimParams_file, "m_Param [ PSTAT_NBRMAX ] = %f\n ", &m_Param [ PSTAT_NBRMAX ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PSTAT_SRCHMAX ] = %f\n ", &m_Param [ PSTAT_SRCHMAX ] );
    ret += std::fscanf ( SimParams_file, "m_Vec [ PPOINT_GRAV_POS ].Set ( %f, %f, %f )\n ", &point_grav_pos.x, &point_grav_pos.y, &point_grav_pos.z );
    ret += std::fscanf ( SimParams_file, "m_Vec [ PPLANE_GRAV_DIR ].Set ( %f, %f, %f )\n ", &pplane_grav_dir.x, &pplane_grav_dir.y, &pplane_grav_dir.z );
    ret += std::fscanf ( SimParams_file, "m_Vec [ PEMIT_POS ].Set ( %f, %f, %f )\n ", &pemit_pos.x, &pemit_pos.y, &pemit_pos.z );
    ret += std::fscanf ( SimParams_file, "m_Vec [ PEMIT_RATE ].Set ( %f, %f, %f )\n ", &pemit_rate.x, &pemit_rate.y, &pemit_rate.z );
    ret += std::fscanf ( SimParams_file, "m_Vec [ PEMIT_ANG ].Set ( %f, %f, %f )\n ", &pemit_ang.x, &pemit_ang.y, &pemit_ang.z );
    ret += std::fscanf ( SimParams_file, "m_Vec [ PEMIT_DANG ].Set ( %f, %f, %f )\n ", &pemit_dang.x, &pemit_dang.y, &pemit_dang.z );
    std::fscanf ( SimParams_file, "// Default sim config\n ");
    ret += std::fscanf ( SimParams_file, "m_Toggle [ PRUN ] = %i\n ", &prun );
    ret += std::fscanf ( SimParams_file, "m_Param [ PGRIDSIZE ] = %f\n ", &m_Param [ PGRIDSIZE ] );
    ret += std::fscanf ( SimParams_file, "m_Vec [ PVOLMIN ].Set ( %f, %f, %f )\n ", &pvolmin.x, &pvolmin.y, &pvolmin.z );
    ret += std::fscanf ( SimParams_file, "m_Vec [ PVOLMAX ].Set ( %f, %f, %f )\n ", &pvolmax.x, &pvolmax.y, &pvolmax.z );
    ret += std::fscanf ( SimParams_file, "m_Vec [ PINITMIN ].Set ( %f, %f, %f )\n ", &pinitmin.x, &pinitmin.y, &pinitmin.z );
    ret += std::fscanf ( SimParams_file, "m_Vec [ PINITMAX ].Set ( %f, %f, %f )\n ", &pinitmax.x, &pinitmax.y, &pinitmax.z );
    ret += std::fscanf ( SimParams_file, "m_Param [ PFORCE_MIN ] = %f\n ", &m_Param [ PFORCE_MIN ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PFORCE_FREQ ] = %f\n ", &m_Param [ PFORCE_FREQ ] );
    ret += std::fscanf ( SimParams_file, "m_Param [ PGROUND_SLOPE ] = %f\n ", &m_Param [ PGROUND_SLOPE ] );

    if ( ret != 63 ) {
        std::cout << "\nvoid FluidSystem::ReadSimParams(..), read failure ! ret = " << ret;
        std::cout << std::flush;
        fclose ( SimParams_file );
        return;
    }

    m_Vec [ PPOINT_GRAV_POS ].Set ( point_grav_pos.x, point_grav_pos.y, point_grav_pos.z );
    m_Vec [ PPLANE_GRAV_DIR ].Set ( pplane_grav_dir.x, pplane_grav_dir.y, pplane_grav_dir.z );
    m_Vec [ PEMIT_POS ].Set ( pemit_pos.x, pemit_pos.y, pemit_pos.z );
    m_Vec [ PEMIT_RATE ].Set ( pemit_rate.x, pemit_rate.y, pemit_rate.z );
    m_Vec [ PEMIT_ANG ].Set ( pemit_ang.x, pemit_ang.y, pemit_ang.z );
    m_Vec [ PEMIT_DANG ].Set ( pemit_dang.x, pemit_dang.y, pemit_dang.z );
    // Default sim config
    m_Vec [ PVOLMIN ].Set ( pvolmin.x, pvolmin.y, pvolmin.z );
    m_Vec [ PVOLMAX ].Set ( pvolmax.x, pvolmax.y, pvolmax.z );
    m_Vec [ PINITMIN ].Set ( pinitmin.x, pinitmin.y, pinitmin.z );
    m_Vec [ PINITMAX ].Set ( pinitmax.x, pinitmax.y, pinitmax.z );

    m_Toggle [ PWRAP_X ] = ( bool ) pwrapx;
    m_Toggle [ PWALL_BARRIER ] = ( bool ) pwall_barrier;
    m_Toggle [ PLEVY_BARRIER ] = ( bool ) plevy_barrier;
    m_Toggle [ PDRAIN_BARRIER ] = ( bool ) pdrain_barrier;
    m_Toggle [ PRUN ] = ( bool ) prun;

    std::cout << "\nvoid FluidSystem::ReadSimParams(..), read success !  ret = " << ret;
    std::cout << "\n" << std::flush;
    fclose ( SimParams_file );
    return;
}


void FluidSystem::WriteSimParams ( const char * relativePath )
{
    Vector3DF point_grav_pos, pplane_grav_dir, pemit_pos, pemit_rate, pemit_ang, pemit_dang, pvolmin, pvolmax, pinitmin, pinitmax;

    int pwrapx, pwall_barrier, plevy_barrier, pdrain_barrier, prun;

    point_grav_pos = m_Vec [ PPOINT_GRAV_POS ];
    pplane_grav_dir = m_Vec [ PPLANE_GRAV_DIR ];
    pemit_pos = m_Vec [ PEMIT_POS ];
    pemit_rate = m_Vec [ PEMIT_RATE ];
    pemit_ang = m_Vec [ PEMIT_ANG ];
    pemit_dang = m_Vec [ PEMIT_DANG ];
    pvolmin = m_Vec [ PVOLMIN ];
    pvolmax = m_Vec [ PVOLMAX ];
    pinitmin = m_Vec [ PINITMIN ];
    pinitmax = m_Vec [ PINITMAX ];

    pwrapx = m_Toggle [ PWRAP_X ] ;
    pwall_barrier =  m_Toggle [ PWALL_BARRIER ];
    plevy_barrier = m_Toggle [ PLEVY_BARRIER ];
    pdrain_barrier = m_Toggle [ PDRAIN_BARRIER ];
    prun = m_Toggle [ PRUN ];

    // open file to write SimParams to
    char SimParams_file_path[256];
    sprintf ( SimParams_file_path, "%s/SimParams.txt", relativePath );
    printf("\n## opening file %s ", SimParams_file_path);
    FILE* SimParams_file = fopen ( SimParams_file_path, "w" );
    if (SimParams_file == NULL) {
        std::cout << "\nvoid FluidSystem::WriteSimParams (const char * relativePath )  Could not open file "<< SimParams_file_path <<"\n"<< std::flush;
        assert(0);
    }

    int ret = std::fprintf(SimParams_file,
                           " m_Time = %f\n m_DT = %f\n m_Param [ PSIMSCALE ] = %f\n m_Param [ PGRID_DENSITY ] = %f\n m_Param [ PVISC ] = %f\n m_Param [ PRESTDENSITY ] = %f\n m_Param [ PSPACING ] = %f\n m_Param [ PMASS ] = %f\n m_Param [ PRADIUS ] = %f\n m_Param [ PDIST ] = %f\n m_Param [ PSMOOTHRADIUS ] = %f\n m_Param [ PINTSTIFF ] = %f\n m_Param [ PEXTSTIFF ] = %f\n m_Param [ PEXTDAMP ] = %f\n m_Param [ PACCEL_LIMIT ] = %f\n m_Param [ PVEL_LIMIT ] = %f\n m_Param [ PMAX_FRAC ] = %f\n m_Param [ PGRAV ] = %f\n m_Param [ PGROUND_SLOPE ] = %f\n m_Param [ PFORCE_MIN ] = %f\n m_Param [ PFORCE_MAX ] = %f\n m_Param [ PFORCE_FREQ ] = %f\n m_Toggle [ PWRAP_X ] = %i\n m_Toggle [ PWALL_BARRIER ] = %i\n m_Toggle [ PLEVY_BARRIER ] = %i\n m_Toggle [ PDRAIN_BARRIER ] = %i\n m_Param [ PSTAT_NBRMAX ] = %f\n m_Param [ PSTAT_SRCHMAX ] = %f\n m_Vec [ PPOINT_GRAV_POS ].Set ( %f, %f, %f )\n m_Vec [ PPLANE_GRAV_DIR ].Set ( %f, %f, %f )\n m_Vec [ PEMIT_POS ].Set ( %f, %f, %f )\n m_Vec [ PEMIT_RATE ].Set ( %f, %f, %f )\n m_Vec [ PEMIT_ANG ].Set ( %f, %f, %f )\n m_Vec [ PEMIT_DANG ].Set ( %f, %f, %f )\n // Default sim config\n m_Toggle [ PRUN ] = %i\n m_Param [ PGRIDSIZE ] = %f\n m_Vec [ PVOLMIN ].Set ( %f, %f, %f )\n m_Vec [ PVOLMAX ].Set ( %f, %f, %f )\n m_Vec [ PINITMIN ].Set ( %f, %f, %f )\n m_Vec [ PINITMAX ].Set ( %f, %f, %f )\n m_Param [ PFORCE_MIN ] = %f\n m_Param [ PFORCE_FREQ ] = %f\n m_Param [ PGROUND_SLOPE ] = %f\n ",
                           m_Time,
                           m_DT,
                           m_Param [ PSIMSCALE ],
                           m_Param [ PGRID_DENSITY ],
                           m_Param [ PVISC ],
                           m_Param [ PRESTDENSITY ],
                           m_Param [ PSPACING ],
                           m_Param [ PMASS ],
                           m_Param [ PRADIUS ],
                           m_Param [ PDIST ],
                           m_Param [ PSMOOTHRADIUS ],
                           m_Param [ PINTSTIFF ],
                           m_Param [ PEXTSTIFF ],
                           m_Param [ PEXTDAMP ],
                           m_Param [ PACCEL_LIMIT ],
                           m_Param [ PVEL_LIMIT ],
                           m_Param [ PMAX_FRAC ],
                           m_Param [ PGRAV ],
                           m_Param [ PGROUND_SLOPE ],
                           m_Param [ PFORCE_MIN ],
                           m_Param [ PFORCE_MAX ],
                           m_Param [ PFORCE_FREQ ],
                           pwrapx, pwall_barrier, plevy_barrier, pdrain_barrier,
                           m_Param [ PSTAT_NBRMAX ],
                           m_Param [ PSTAT_SRCHMAX ],
                           point_grav_pos.x, point_grav_pos.y, point_grav_pos.z,
                           pplane_grav_dir.x, pplane_grav_dir.y, pplane_grav_dir.z,
                           pemit_pos.x, pemit_pos.y, pemit_pos.z,
                           pemit_rate.x, pemit_rate.y, pemit_rate.z,
                           pemit_ang.x, pemit_ang.y, pemit_ang.z,
                           pemit_dang.x, pemit_dang.y, pemit_dang.z,
                           // Default sim config
                           prun,
                           m_Param [ PGRIDSIZE ],
                           pvolmin.x, pvolmin.y, pvolmin.z,
                           pvolmax.x, pvolmax.y, pvolmax.z,
                           pinitmin.x, pinitmin.y, pinitmin.z,
                           pinitmax.x, pinitmax.y, pinitmax.z,
                           m_Param [ PFORCE_MIN ],
                           m_Param [ PFORCE_FREQ ],
                           m_Param [ PGROUND_SLOPE ]
                          );

    std::cout << "\nvoid FluidSystem::WriteSimParams (const char * relativePath ) wrote file "<< SimParams_file_path <<"\t"<<
              "ret = " << ret << "\n" << std::flush;
    fclose(SimParams_file);
    return;
}


void FluidSystem::SavePoints_asciiPLY ( const char * relativePath, int frame )
{
//    std::cout << " FluidSystem::SavePoints_asciiPLY () \n";
//    std::cout << "Chk4.0\n";
    char buf[256];
    frame += 100000;    // ensures numerical and alphabetic order match
    sprintf ( buf, "%s/particles_pos%04d.ply", relativePath, frame );
    FILE* fp = fopen ( buf, "w" );

    if(!fp){std::cout<<"failed to open file "<< buf <<"\n";}
    
//    std::cout << "Chk4.1\n";
    
    int numpnt = NumPoints();
    int numfield = 3;
    int ftype;         // 0=char, 1=int, 2=float, 3=double
    int fcnt;

    Vector3DF* Pos;
    Vector3DF* Vel;
    uint* Clr;
    
//    std::cout << "Chk4.1.1\n";

    fprintf(fp, "ply \n format ascii 1.0\n comment particle cloud from Fluids_v4\n element vertex %i\n", numpnt );
    fprintf(fp, "property float x\nproperty float y\nproperty float z\n");
    fprintf(fp, "end_header\n");

//    std::cout << "Chk4.2\n";
    
    for(int i=0; i<numpnt; i++) {
        Pos = getPos(i);
        Vel = getVel(i);
        Clr = getClr(i);
        fprintf(fp, "%f %f %f\n", Pos->x, Pos->y,Pos->z);
    }
    
//    std::cout << "Chk4.3\n";
    
    fclose ( fp );
    fflush ( fp );
//    std::cout << "FluidSystem::SavePoints_asciiPLY ()   finished   \n";
}

void FluidSystem::SavePoints_asciiPLY_with_edges ( const char * relativePath, int frame )
{
//    std::cout << " FluidSystem::SavePoints_asciiPLY () \n";
//    std::cout << "Chk4.0\n";
    char buf[256];
    frame += 100000;    // ensures numerical and alphabetic order match
    sprintf ( buf, "%s/particles_pos%04d.ply", relativePath, frame );
    FILE* fp = fopen ( buf, "w" );

    if(!fp){std::cout<<"failed to open file "<< buf <<"\n";}
    
//    std::cout << "Chk4.1\n";
    
    int numpnt = NumPoints();
    //int numfield = 3;
    //int ftype;         // 0=char, 1=int, 2=float, 3=double
    //int fcnt;

    Vector3DF* Pos;
    Vector3DF* Vel;
    uint* Clr, *ElastIdx;
    
//    std::cout << "Chk4.1.1\n";

    fprintf(fp, "ply \n format ascii 1.0\n comment particle cloud from Fluids_v4\n element vertex %i\n", numpnt  + 4); // +4 for axes
    fprintf(fp, "property float x\nproperty float y\nproperty float z\n");
    fprintf(fp,"element edge %i\n",numpnt*BONDS_PER_PARTICLE + 3);    // +3 for axes
    fprintf(fp,"property int vertex1\nproperty int vertex2\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n");
    fprintf(fp, "end_header\n");

//    std::cout << "Chk4.2\n";
    
    for(int i=0; i<numpnt; i++) {
        Pos = getPos(i);
        Vel = getVel(i);
        Clr = getClr(i);
        fprintf(fp, "%f %f %f\n", Pos->x, Pos->y,Pos->z);
    }
    Vector3DF axes = m_Vec[PINITMIN];
    float len = sqrt(m_FParams.rd2);
    fprintf(fp, "%f %f %f\n", axes.x, axes.y, axes.z);                             // points for axes placed at the origin of the particles created my make_demo
    fprintf(fp, "%f %f %f\n", len + axes.x, axes.y, axes.z);
    fprintf(fp, "%f %f %f\n", axes.x, len + axes.y, axes.z);
    fprintf(fp, "%f %f %f\n", axes.x, axes.y, len + axes.z);
    
    // make index to look up particle position in list from particle ID
    // first allocate a buffer ?
    
    for(int i=0; i<numpnt; i++) {
        ElastIdx = getElastIdx(i);                              // NB .ply requires particle position in list, not ID
        for(int j=0; j<(BONDS_PER_PARTICLE ); j++) { 
            int secondParticle = ElastIdx[j * DATA_PER_BOND];
            int bond = ElastIdx[j * DATA_PER_BOND +1];          // NB [0]current index, [1]elastic limit, [2]restlength, [3]modulus, [4]damping coeff, [5]particle ID, [6]bond index 
            if (bond==0) secondParticle = i;                    // i.e. check the bond is not broken
// printf(" (bond=%i, secondParticle=%i)",bond,secondParticle);
            fprintf(fp, "%u %u 255 255 0\n", i, secondParticle ); 
            
        }
    }
    fprintf(fp, "%u %u 255 0 0\n", NumPoints(), NumPoints()+1 );      // axes
    fprintf(fp, "%u %u 0 255 0\n", NumPoints(), NumPoints()+2 ); 
    fprintf(fp, "%u %u 0 0 255\n", NumPoints(), NumPoints()+3 ); 
    
//    std::cout << "Chk4.3\n";
    
    fclose ( fp );
    fflush ( fp );
//    std::cout << "FluidSystem::SavePoints_asciiPLY ()   finished   \n";
}


// Commented out hdf5 due to library path clashes for <hdf5.h> between Ubuntu and SUSE.
////////////////////////
// adapted from Example 1 of http://web.mit.edu/fwtools_v3.1.0/www/Intro/IntroExamples.html#CreateExample
/*#include <hdf5/serial/hdf5.h>	//hdf5/serial/
#include <stdio.h>
#include <stdlib.h>

#define DATASETNAME "Vec3DF_Array"
#define NX     5                      / * dataset dimensions * /
#define NY     3
#define RANK   2

int FluidSystem::WriteParticlesToHDF5File (int filenum)
{
    std::cout << "WriteParticlesToHDF5File \n" << std::flush;
    hid_t       file, dataset;         / * file and dataset handles * /
    hid_t       datatype, dataspace;   / * handles * /
    hsize_t     dimsf[2];              / * dataset dimensions * /
    herr_t      status;

    const int NX =  NumPoints();

    int         i, j;
    float **    data = new float*[NX];  / * allocate data to write * /
    for(i=0; i<NX; ++i)
	data[i] =  new float[NY];

    if (data == nullptr) {
	std::cout << "Error: memory could not be allocated";
	return -1;
    }

    / * Data  and output buffer initialization.  * /
    for (j = 0; j < NX; j++) {
	for (i = 0; i < NY; i++)
	    data[j][i] = (float)(i + j);
    }

    edit filename
    char filename[256];
    filenum += 100000;    // ensures numerical and alphabetic order match
	sprintf ( filename, "particles_pos_%04d.h5", filenum );

    / * Create a new file using H5F_ACC_TRUNC access,
     * default file creation properties, and default file
     * access properties. * /
    file = H5Fcreate(/ *FILE2 * /filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    / * Describe the size of the array and create the data space for fixed
     * size dataset. * /
    dimsf[0] = NX;
    dimsf[1] = NY;
    dataspace = H5Screate_simple(RANK, dimsf, NULL);

    / * Define datatype for the data in the file.
     * We will store little endian INT numbers.* /
    datatype = H5Tcopy(H5T_IEEE_F64LE / *H5T_NATIVE_INT   * /);
    status = H5Tset_order(datatype, H5T_ORDER_LE);

    / * Create a new dataset within the file using defined dataspace and
     * datatype and default dataset creation properties. * /
    dataset = H5Dcreate(file, DATASETNAME, datatype, dataspace,
			H5P_DEFAULT,
			H5P_DEFAULT,
			H5P_DEFAULT);

    / * Write the data to the dataset using default transfer properties.   * /
    status = H5Dwrite(dataset, H5T_IEEE_F64LE / * H5T_NATIVE_INT  * /, H5S_ALL, H5S_ALL,
		      H5P_DEFAULT, m_Fluid.bufV3(FPOS) / * data  * /);

    / * Close/release resources. * /
    H5Sclose(dataspace);
    H5Tclose(datatype);
    H5Dclose(dataset);
    H5Fclose(file);

    return 0;
} */

/* hdf5 notes
// #include <iostream>
// #include <string>
// #include <hdf5>
// #include <hdf5/serial/H5Cpp.h>//"H5Cpp.h"
// using namespace H5;
//
// const H5std_string      FILE_NAME("h5tutr_dset.h5");
// const H5std_string      DATASET_NAME("dset");
// const int        NX = 4;                     // dataset dimensions
// const int        NY = 6;
// const int        RANK = 2;
//
// int main (void)
// {
//     Try block to detect exceptions raised by any of the calls inside it
//     try
//     {
//         Turn off the auto-printing when failure occurs so that we can
//         handle the errors appropriately
//         Exception::dontPrint();
//
//         Create a new file using the default property lists.
//         H5File file(FILE_NAME, H5F_ACC_TRUNC);
//
//         Create the data space for the dataset.
//         hsize_t dims[2];               // dataset dimensions
//         dims[0] = NX;
//         dims[1] = NY;
//         DataSpace dataspace(RANK, dims);
//
//         Create the dataset.
//         DataSet dataset = file.createDataSet(DATASET_NAME, PredType::STD_I32BE, dataspace);
//
//     }  // end of try block
//
//     catch failure caused by the H5File operations
//     catch(FileIException error)
//     {
//         error.printErrorStack();
//         return -1;
//     }
//
//     catch failure caused by the DataSet operations
//     catch(DataSetIException error)
//     {
//         error.printErrorStack();
//         return -1;
//     }
//
//     catch failure caused by the DataSpace operations
//     catch(DataSpaceIException error)
//     {
//         error.printErrorStack();
//         return -1;
//     }
//
//     return 0;  // successfully terminated
// }
*/



void FluidSystem::WriteDemoSimParams ( const char * relativePath, uint num_particles, float spacing, float x_dim, float y_dim, float z_dim )
{
    //Initialize ();
    memset ( &m_Fluid, 0,		sizeof(FBufs) );
    memset ( &m_FluidTemp, 0,	sizeof(FBufs) );
    memset ( &m_FParams, 0,		sizeof(FParams) );
    memset ( &m_FGenome, 0,		sizeof(FGenome) );
    
    m_Param[PEXAMPLE] = 2;      // wave pool example.
    m_Param[PGRID_DENSITY] = 2.0;
    m_Param[PNUM] = num_particles;// 1000000;    //1000 = minimal simulation, 1000000 = large simulation

    AllocateBuffer ( FPARAMS, sizeof(FParams), 1,0,/*0, 1,*/ GPU_OFF, CPU_YES );//AllocateBuffer ( int buf_id, int stride, int cpucnt, int gpucnt, int gpumode, int cpumode ) 

    m_Time = 0;
    ClearNeighborTable();   // ###### Is this needed ? #######
    mNumPoints = 0;			// reset count

    // set up the standard demo
    SetupDefaultParams();
    SetupExampleParams();
    SetupExampleGenome();

    mMaxPoints = m_Param[PNUM];    
    
    m_Param[PGRIDSIZE] = 2*m_Param[PSMOOTHRADIUS] / m_Param[PGRID_DENSITY];
    SetupKernels();
    SetupSpacing();
    SetupGrid ( m_Vec[PVOLMIN], m_Vec[PVOLMAX], m_Param[PSIMSCALE], m_Param[PGRIDSIZE], 1.0f );	// Setup grid

std::cout << " chk1.0 " << std::flush ;
    
    AllocateParticles ( mMaxPoints, GPU_OFF, CPU_YES );  // allocates only cpu buffer for particles
    
std::cout << " chk1.1 " << std::flush ;
    AllocateGrid(GPU_OFF, CPU_YES);
std::cout << " chk1.2 " << std::flush ;
    
    //SetupAddVolume(m_Vec[PINITMIN], m_Vec[PINITMAX], m_Param[PSPACING], 0.1f, (int)m_Param[PNUM]);
    Vector3DF pinit_max = {x_dim,y_dim,z_dim};
    pinit_max += m_Vec[PINITMIN];
    SetupAddVolumeMorphogenesis2(m_Vec[PINITMIN], pinit_max, spacing, 0.1f, (int)num_particles);  //  Particles created here //
std::cout << " chk1.3 " << std::flush ;
    
    // write data to file
    WriteSimParams ( relativePath );
std::cout << "\n WriteSimParams ( relativePath );  completed \n" << std::flush ;
    WriteGenome ( relativePath);
std::cout << "\n WriteGenome ( relativePath );  completed \n" << std::flush ;
    SavePointsCSV2 ( relativePath, 1 );                                                  //SavePointsCSV ( relativePath, 1 );
std::cout << "\n SavePointsCSV ( relativePath, 1 );  completed \n" << std::flush ;
    SavePoints_asciiPLY ( relativePath, 1 );
std::cout << "\n SavePoints_asciiPLY ( relativePath, 1 );  completed \n" << std::flush ;
}


/////////////////////////////////////////////////////////////


void FluidSystem::SetupKernels ()
{
    m_Param [ PDIST ] = pow ( (float) m_Param[PMASS] / m_Param[PRESTDENSITY], 1.0f/3.0f );
    m_R2 = m_Param [PSMOOTHRADIUS] * m_Param[PSMOOTHRADIUS];
    m_Poly6Kern = 315.0f / (64.0f * 3.141592f * pow( m_Param[PSMOOTHRADIUS], 9.0f) );	// Wpoly6 kernel (denominator part) - 2003 Muller, p.4
    m_SpikyKern = -45.0f / (3.141592f * pow( m_Param[PSMOOTHRADIUS], 6.0f) );			// Laplacian of viscocity (denominator): PI h^6
    m_LapKern = 45.0f / (3.141592f * pow( m_Param[PSMOOTHRADIUS], 6.0f) );
}



void FluidSystem::SetupDefaultParams ()
{
    //  Range = +/- 10.0 * 0.006 (r) =	   0.12			m (= 120 mm = 4.7 inch)
    //  Container Volume (Vc) =			   0.001728		m^3
    //  Rest Density (D) =				1000.0			kg / m^3
    //  Particle Mass (Pm) =			   0.00020543	kg						(mass = vol * density)
    //  Number of Particles (N) =		4000.0
    //  Water Mass (M) =				   0.821		kg (= 821 grams)
    //  Water Volume (V) =				   0.000821     m^3 (= 3.4 cups, .21 gals)
    //  Smoothing Radius (R) =             0.02			m (= 20 mm = ~3/4 inch)
    //  Particle Radius (Pr) =			   0.00366		m (= 4 mm  = ~1/8 inch)
    //  Particle Volume (Pv) =			   2.054e-7		m^3	(= .268 milliliters)
    //  Rest Distance (Pd) =			   0.0059		m
    //
    //  Given: D, Pm, N
    //    Pv = Pm / D			0.00020543 kg / 1000 kg/m^3 = 2.054e-7 m^3
    //    Pv = 4/3*pi*Pr^3    cuberoot( 2.054e-7 m^3 * 3/(4pi) ) = 0.00366 m
    //     M = Pm * N			0.00020543 kg * 4000.0 = 0.821 kg
    //     V =  M / D              0.821 kg / 1000 kg/m^3 = 0.000821 m^3
    //     V = Pv * N			 2.054e-7 m^3 * 4000 = 0.000821 m^3
    //    Pd = cuberoot(Pm/D)    cuberoot(0.00020543/1000) = 0.0059 m
    //
    // Ideal grid cell size (gs) = 2 * smoothing radius = 0.02*2 = 0.04
    // Ideal domain size = k*gs/d = k*0.02*2/0.005 = k*8 = {8, 16, 24, 32, 40, 48, ..}
    //    (k = number of cells, gs = cell size, d = simulation scale)

    // "The viscosity coefficient is the dynamic viscosity, visc > 0 (units Pa.s),
    // and to include a reasonable damping contribution, it should be chosen
    // to be approximately a factor larger than any physical correct viscosity
    // coefficient that can be looked up in the literature. However, care should
    // be taken not to exaggerate the viscosity coefficient for fluid materials.
    // If the contribution of the viscosity force density is too large, the net effect
    // of the viscosity term will introduce energy into the system, rather than
    // draining the system from energy as intended."
    //    Actual visocity of water = 0.001 Pa.s    // viscosity of water at 20 deg C.

    m_Time = 0.0f;							// Start at T=0
    m_DT = 0.003f;

    m_Param [ PSIMSCALE ] =		0.005f;			// unit size
    m_Param [ PVISC ] =			0.50f;			// pascal-second (Pa.s) = 1 kg m^-1 s^-1  (see wikipedia page on viscosity)
    m_Param [ PRESTDENSITY ] =	400.0f;			// kg / m^3
    m_Param [ PSPACING ]	=	0.0f;			// spacing will be computed automatically from density in most examples (set to 0 for autocompute)
    m_Param [ PMASS ] =			0.00020543f;		// kg
    m_Param [ PRADIUS ] =		0.015f;			// m
    m_Param [ PDIST ] =			0.0059f;			// m
    m_Param [ PSMOOTHRADIUS ] =	0.015f;			// m
    m_Param [ PINTSTIFF ] =		1.0f;
    m_Param [ PEXTSTIFF ] =		50000.0f;
    m_Param [ PEXTDAMP ] =		100.0f;
    m_Param [ PACCEL_LIMIT ] =	150.0f;			// m / s^2
    m_Param [ PVEL_LIMIT ] =	3.0f;			// m / s
    m_Param [ PMAX_FRAC ] =		1.0f;
    m_Param [ PGRAV ] =			1.0f;

    m_Param [ PGROUND_SLOPE ] = 0.0f;
    m_Param [ PFORCE_MIN ] =	0.0f;
    m_Param [ PFORCE_MAX ] =	0.0f;
    m_Param [ PFORCE_FREQ ] =	16.0f;
    m_Toggle [ PWRAP_X ] = false;
    m_Toggle [ PWALL_BARRIER ] = false;
    m_Toggle [ PLEVY_BARRIER ] = false;
    m_Toggle [ PDRAIN_BARRIER ] = false;

    m_Param [ PSTAT_NBRMAX ] = 0 ;
    m_Param [ PSTAT_SRCHMAX ] = 0 ;

    m_Vec [ PPOINT_GRAV_POS ].Set ( 0, 0, 0 );
    m_Vec [ PPLANE_GRAV_DIR ].Set ( 0, -9.8f, 0 );
    m_Vec [ PEMIT_POS ].Set ( 0, 0, 0 );
    m_Vec [ PEMIT_RATE ].Set ( 0, 0, 0 );
    m_Vec [ PEMIT_ANG ].Set ( 0, 90, 1.0f );
    m_Vec [ PEMIT_DANG ].Set ( 0, 0, 0 );

    // Default sim config
    m_Toggle [ PRUN ] = true;				// Run integrator
    m_Param [PGRIDSIZE] = m_Param[PSMOOTHRADIUS] * 2;
    m_Param [PDRAWMODE] = 1;				// Sprite drawing
    m_Param [PDRAWGRID] = 0;				// No grid
    m_Param [PDRAWTEXT] = 0;				// No text

}



void FluidSystem::SetupExampleParams ()
{
    Vector3DF pos;
    Vector3DF min, max;

    switch ( (int) m_Param[PEXAMPLE] ) {

    case 0:	{	// Regression test. N x N x N static grid

        int k = (int) ceil ( pow ( (float) m_Param[PNUM], (float) 1.0f/3.0f ) );
        m_Vec [ PVOLMIN ].Set ( 0, 0, 0 );
        m_Vec [ PVOLMAX ].Set ( 2.0f+(k/2), 2.0f+(k/2), 2.0f+(k/2) );
        m_Vec [ PINITMIN ].Set ( 1.0f, 1.0f, 1.0f );
        m_Vec [ PINITMAX ].Set ( 1.0f+(k/2), 1.0f+(k/2), 1.0f+(k/2) );

        m_Param [ PGRAV ] = 0.0;
        m_Vec [ PPLANE_GRAV_DIR ].Set ( 0.0, 0.0, 0.0 );
        m_Param [ PSPACING ] = 0.5;				// Fixed spacing		Dx = x-axis density
        m_Param [ PSMOOTHRADIUS ] =	m_Param [PSPACING];		// Search radius
        m_Toggle [ PRUN ] = false;				// Do NOT run sim. Neighbors only.
        m_Param [PDRAWMODE] = 1;				// Point drawing
        m_Param [PDRAWGRID] = 1;				// Grid drawing
        m_Param [PDRAWTEXT] = 1;				// Text drawing
        m_Param [PSIMSCALE ] = 1.0f;

    }
    break;
    case 1:		// Tower
        m_Vec [ PVOLMIN ].Set (   0,   0,   0 );
        m_Vec [ PVOLMAX ].Set (  256, 128, 256 );
        m_Vec [ PINITMIN ].Set (  5,   5,  5 );
        m_Vec [ PINITMAX ].Set ( 256*0.3, 128*0.9, 256*0.3 );
        break;
    case 2:		// Wave pool
        m_Vec [ PVOLMIN ].Set (   0,   0,   0 );
        m_Vec [ PVOLMAX ].Set (  400, 200, 400 );
        m_Vec [ PINITMIN ].Set ( 100, 80,  100 );
        m_Vec [ PINITMAX ].Set ( 300, 190, 300 );
        m_Param [ PFORCE_MIN ] = 100.0f;
        m_Param [ PFORCE_FREQ ] = 6.0f;
        m_Param [ PGROUND_SLOPE ] = 0.10f;
        break;
    case 3:		// Small dam break
        m_Vec [ PVOLMIN ].Set ( -40, 0, -40  );
        m_Vec [ PVOLMAX ].Set ( 40, 60, 40 );
        m_Vec [ PINITMIN ].Set ( 0, 8, -35 );
        m_Vec [ PINITMAX ].Set ( 35, 55, 35 );
        m_Param [ PFORCE_MIN ] = 0.0f;
        m_Param [ PFORCE_MAX ] = 0.0f;
        m_Vec [ PPLANE_GRAV_DIR ].Set ( 0.0f, -9.8f, 0.0f );
        break;
    case 4:		// Dual-Wave pool
        m_Vec [ PVOLMIN ].Set ( -100, 0, -15 );
        m_Vec [ PVOLMAX ].Set ( 100, 100, 15 );
        m_Vec [ PINITMIN ].Set ( -80, 8, -10 );
        m_Vec [ PINITMAX ].Set ( 80, 90, 10 );
        m_Param [ PFORCE_MIN ] = 20.0;
        m_Param [ PFORCE_MAX ] = 20.0;
        m_Vec [ PPLANE_GRAV_DIR ].Set ( 0.0f, -9.8f, 0.0f );
        break;
    case 5:		// Microgravity
        m_Vec [ PVOLMIN ].Set ( -80, 0, -80 );
        m_Vec [ PVOLMAX ].Set ( 80, 100, 80 );
        m_Vec [ PINITMIN ].Set ( -60, 40, -60 );
        m_Vec [ PINITMAX ].Set ( 60, 80, 60 );
        m_Vec [ PPLANE_GRAV_DIR ].Set ( 0, -1, 0 );
        m_Param [ PGROUND_SLOPE ] = 0.1f;
        break;
    }

}

void FluidSystem::SetupExampleGenome()  // need to set up a demo genome
{
    // Null genome
    for(int i=0; i< NUM_GENES; i++) m_FGenome.mutability[i] = 0;
    for(int i=0; i< NUM_GENES; i++) m_FGenome.delay[i] = 1;
    for(int i=0; i< NUM_GENES; i++) for(int j=0; j< NUM_GENES; j++)
        {
            m_FGenome.sensitivity[i][j] = j;
        }
    for(int i=0; i< NUM_GENES; i++) {
        m_FGenome.difusability[i][0] = 0;
        m_FGenome.difusability[i][1] = 1;
    }
}

//////////////////////////////////////////////////////



void FluidSystem::SetupSpacing ()
{
    m_Param [ PSIMSIZE ] = m_Param [ PSIMSCALE ] * (m_Vec[PVOLMAX].z - m_Vec[PVOLMIN].z);

    if ( m_Param[PSPACING] == 0 ) {
        // Determine spacing from density
        m_Param [PDIST] = pow ( (float) m_Param[PMASS] / m_Param[PRESTDENSITY], 1/3.0f );
        m_Param [PSPACING] = m_Param [ PDIST ]*0.87f / m_Param[ PSIMSCALE ];
    } else {
        // Determine density from spacing
        m_Param [PDIST] = m_Param[PSPACING] * m_Param[PSIMSCALE] / 0.87f;
        m_Param [PRESTDENSITY] = m_Param[PMASS] / pow ( (float) m_Param[PDIST], 3.0f );
    }
    //nvprintf ( "Add Particles. Density: %f, Spacing: %f, PDist: %f\n", m_Param[PRESTDENSITY], m_Param [ PSPACING ], m_Param[ PDIST ] );

    // Particle Boundaries
    m_Vec[PBOUNDMIN] = m_Vec[PVOLMIN];
    m_Vec[PBOUNDMIN] += 2.0*(m_Param[PGRIDSIZE] / m_Param[PSIMSCALE]);
    m_Vec[PBOUNDMAX] = m_Vec[PVOLMAX];
    m_Vec[PBOUNDMAX] -= 2.0*(m_Param[PGRIDSIZE] / m_Param[PSIMSCALE]);
}



int iDivUp (int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void computeNumBlocks (int numPnts, int minThreads, int &numBlocks, int &numThreads)
{
    numThreads = min( minThreads, numPnts );
    numBlocks = (numThreads==0) ? 1 : iDivUp ( numPnts, numThreads );
}

/*
 * void FluidSystem::cudaExit ()
{
	cudaDeviceReset();
}
*/


void FluidSystem::TransferToTempCUDA ( int buf_id, int sz )
{
    cuCheck ( cuMemcpyDtoD ( m_FluidTemp.gpu(buf_id), m_Fluid.gpu(buf_id), sz ), "TransferToTempCUDA", "cuMemcpyDtoD", "m_FluidTemp", mbDebug);
}


///////////////////////////////////////////////


void FluidSystem::FluidSetupCUDA ( int num, int gsrch, int3 res, float3 size, float3 delta, float3 gmin, float3 gmax, int total, int chk )
{
    m_FParams.pnum = num;
    m_FParams.freeze = false;
    m_FParams.gridRes = res;
    m_FParams.gridSize = size;
    m_FParams.gridDelta = delta;
    m_FParams.gridMin = gmin;
    m_FParams.gridMax = gmax;
    m_FParams.gridTotal = total;
    m_FParams.gridSrch = gsrch;
    m_FParams.gridAdjCnt = gsrch*gsrch*gsrch;
    m_FParams.gridScanMax = res;
    m_FParams.gridScanMax -= make_int3( m_FParams.gridSrch, m_FParams.gridSrch, m_FParams.gridSrch );
    m_FParams.chk = chk;

    // Build Adjacency Lookup
    int cell = 0;
    for (int y=0; y < gsrch; y++ )
        for (int z=0; z < gsrch; z++ )
            for (int x=0; x < gsrch; x++ )
                m_FParams.gridAdj [ cell++]  = ( y * m_FParams.gridRes.z+ z )*m_FParams.gridRes.x +  x ;

    // Compute number of blocks and threads
    int threadsPerBlock = 512;

    computeNumBlocks ( m_FParams.pnum, threadsPerBlock, m_FParams.numBlocks, m_FParams.numThreads);				// particles
    computeNumBlocks ( m_FParams.gridTotal, threadsPerBlock, m_FParams.gridBlocks, m_FParams.gridThreads);		// grid cell

    // Compute particle buffer & grid dimensions
    m_FParams.szPnts = (m_FParams.numBlocks  * m_FParams.numThreads);
    //nvprintf ( "CUDA Config: \n" );
    //nvprintf ( "  Pnts: %d, t:%dx%d=%d, Size:%d\n", m_FParams.pnum, m_FParams.numBlocks, m_FParams.numThreads, m_FParams.numBlocks*m_FParams.numThreads, m_FParams.szPnts);
    //nvprintf ( "  Grid: %d, t:%dx%d=%d, bufGrid:%d, Res: %dx%dx%d\n", m_FParams.gridTotal, m_FParams.gridBlocks, m_FParams.gridThreads, m_FParams.gridBlocks*m_FParams.gridThreads, m_FParams.szGrid, (int) m_FParams.gridRes.x, (int) m_FParams.gridRes.y, (int) m_FParams.gridRes.z );

    // Initialize random numbers
    int blk = int(num/16)+1;
    //randomInit<<< blk, 16 >>> ( rand(), gFluidBufs., num );

}


/////////////////////////////


void FluidSystem::FluidParamCUDA ( float ss, float sr, float pr, float mass, float rest, float3 bmin, float3 bmax, float estiff, float istiff, float visc, float damp, float fmin, float fmax, float ffreq, float gslope, float gx, float gy, float gz, float al, float vl, int emit )
{
    m_FParams.psimscale = ss;
    m_FParams.psmoothradius = sr;
    m_FParams.pradius = pr;
    m_FParams.r2 = sr * sr;
    m_FParams.pmass = mass;
    m_FParams.prest_dens = rest;
    m_FParams.pboundmin = bmin;
    m_FParams.pboundmax = bmax;
    m_FParams.pextstiff = estiff;
    m_FParams.pintstiff = istiff;
    m_FParams.pvisc = visc;
    m_FParams.pdamp = damp;
    m_FParams.pforce_min = fmin;
    m_FParams.pforce_max = fmax;
    m_FParams.pforce_freq = ffreq;
    m_FParams.pground_slope = gslope;
    m_FParams.pgravity = make_float3( gx, gy, gz );
    m_FParams.AL = al;
    m_FParams.AL2 = al * al;
    m_FParams.VL = vl;
    m_FParams.VL2 = vl * vl;
    m_FParams.pemit = emit;

    m_FParams.pdist = pow ( m_FParams.pmass / m_FParams.prest_dens, 1/3.0f );
    m_FParams.poly6kern = 315.0f / (64.0f * 3.141592f * pow( sr, 9.0f) );
    m_FParams.spikykern = -45.0f / (3.141592f * pow( sr, 6.0f) );
    m_FParams.lapkern = 45.0f / (3.141592f * pow( sr, 6.0f) );
    m_FParams.gausskern = 1.0f / pow(3.141592f * 2.0f*sr*sr, 3.0f/2.0f);

    m_FParams.d2 = m_FParams.psimscale * m_FParams.psimscale;
    m_FParams.rd2 = m_FParams.r2 / m_FParams.d2;
    m_FParams.vterm = m_FParams.lapkern * m_FParams.pvisc;

    // Transfer sim params to device
    cuCheck ( cuMemcpyHtoD ( cuFParams,	&m_FParams,		sizeof(FParams) ), "FluidParamCUDA", "cuMemcpyHtoD", "cuFParams", mbDebug);
}


void FluidSystem::TransferToCUDA ()
{
std::cout<<"\nTransferToCUDA ()\n"<<std::flush;
    // Send particle buffers
//std::cout<<"FPOS\n"<<std::flush;
    cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FPOS), m_Fluid.bufC(FPOS),			mNumPoints *sizeof(float) * 3),	"TransferToCUDA", "cuMemcpyHtoD", "FPOS", mbDebug);
std::cout<<" m_Fluid.gpu(FPOS)="<< m_Fluid.gpu(FPOS)<<"\tm_Fluid.bufC(FPOS)="<< static_cast<void*>(m_Fluid.bufC(FPOS))<<"\n"<<std::flush;


//std::cout<<"FVEL\n"<<std::flush;   
    cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FVEL),	m_Fluid.bufC(FVEL),			mNumPoints *sizeof(float)*3 ),	"TransferToCUDA", "cuMemcpyHtoD", "FVEL", mbDebug);
//std::cout<<"FVEVAL\n"<<std::flush; 
    cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FVEVAL),	m_Fluid.bufC(FVEVAL),	mNumPoints *sizeof(float)*3 ),  "TransferToCUDA", "cuMemcpyHtoD", "FVELAL", mbDebug);
//std::cout<<"FFORCE\n"<<std::flush; 
    cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FFORCE),	m_Fluid.bufC(FFORCE),	mNumPoints *sizeof(float)*3 ),  "TransferToCUDA", "cuMemcpyHtoD", "FFORCE", mbDebug);
//std::cout<<"FPRESS\n"<<std::flush; 
    cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FPRESS),	m_Fluid.bufC(FPRESS),	mNumPoints *sizeof(float) ),	"TransferToCUDA", "cuMemcpyHtoD", "FPRESS", mbDebug);
//std::cout<<"FDENSITY\n"<<std::flush; 
    cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FDENSITY), m_Fluid.bufC(FDENSITY),	mNumPoints *sizeof(float) ),	"TransferToCUDA", "cuMemcpyHtoD", "FDENSITY", mbDebug);
//std::cout<<"FCLR\n"<<std::flush; 
    cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FCLR),	m_Fluid.bufC(FCLR),			mNumPoints *sizeof(uint) ),		"TransferToCUDA", "cuMemcpyHtoD", "FCLR", mbDebug);
//std::cout<<" m_Fluid.gpu(FCLR)="<< m_Fluid.gpu(FCLR)<<"\tm_Fluid.bufC(FCLR)="<< *m_Fluid.bufC(FCLR)<<"\n"<<std::flush;
    
    uint colours[10]; 
    /////colours = static_cast<void*>(m_Fluid.bufC(FCLR));

//std::cout<<"FELASTIDX\n"<<std::flush;     
    // add extra data for morphogenesis
    cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FELASTIDX), m_Fluid.bufC(FELASTIDX),	mNumPoints *sizeof(uint[BOND_DATA]) ),	"TransferToCUDA", "cuMemcpyHtoD", "FELASTIDX", mbDebug);
//std::cout<<" m_Fluid.gpu(FELASTIDX)="<< m_Fluid.gpu(FELASTIDX)<<"\tm_Fluid.bufC(FELASTIDX)="<< *m_Fluid.bufC(FELASTIDX)<<"\n"<<std::flush;
    cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FPARTICLEIDX), m_Fluid.bufC(FPARTICLEIDX),	mNumPoints *sizeof(uint[BONDS_PER_PARTICLE *2]) ),	"TransferToCUDA", "cuMemcpyHtoD", "FPARTICLEIDX", mbDebug);
    
    cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FPARTICLE_ID),	m_Fluid.bufC(FPARTICLE_ID),			mNumPoints *sizeof(uint) ),		"TransferToCUDA", "cuMemcpyHtoD", "FPARTICLE_ID", mbDebug);
    cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FMASS_RADIUS),	m_Fluid.bufC(FMASS_RADIUS),			mNumPoints *sizeof(uint) ),		"TransferToCUDA", "cuMemcpyHtoD", "FMASS_RADIUS", mbDebug);
    

    cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FNERVEIDX), m_Fluid.bufC(FNERVEIDX),	mNumPoints *sizeof(uint) ),	"TransferToCUDA", "cuMemcpyHtoD", "FNERVEIDX", mbDebug);
//std::cout<<" m_Fluid.gpu(FNERVEIDX)="<< m_Fluid.gpu(FNERVEIDX)<<"\tm_Fluid.bufC(FNERVEIDX)="<< *m_Fluid.bufC(FNERVEIDX)<<"\n"<<std::flush;

    cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FCONC), m_Fluid.bufC(FCONC),	mNumPoints *sizeof(float[NUM_TF]) ),	"TransferToCUDA", "cuMemcpyHtoD", "FCONC", mbDebug);
//std::cout<<"FEPIGEN\n"<<std::flush;
    cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FEPIGEN), m_Fluid.bufC(FEPIGEN),	mNumPoints *sizeof(uint[NUM_GENES]) ),	"TransferToCUDA", "cuMemcpyHtoD", "FEPIGEN", mbDebug);
std::cout<<"TransferToCUDA ()  finished\n"<<std::flush;

}

void FluidSystem::TransferFromCUDA ()
{
std::cout<<"\nTransferFromCUDA () \n"<<std::flush;    
    // Return particle buffers

    cuCheck( cuMemcpyDtoH ( m_Fluid.bufC(FPOS),	m_Fluid.gpu(FPOS),	mNumPoints *sizeof(float)*3 ), "TransferFromCUDA", "cuMemcpyDtoH", "FPOS", mbDebug);
//std::cout<<" m_Fluid.gpu(FPOS)="<< m_Fluid.gpu(FPOS)<<"\tm_Fluid.bufC(FPOS)="<< static_cast<void*>(m_Fluid.bufC(FPOS))<<"\n"<<std::flush; // m_Fluid.bufV3(FPOS) + n)


    cuCheck( cuMemcpyDtoH ( m_Fluid.bufC(FVEL),	m_Fluid.gpu(FVEL),	mNumPoints *sizeof(float)*3 ), "TransferFromCUDA", "cuMemcpyDtoH", "FVEL", mbDebug);
    cuCheck( cuMemcpyDtoH ( m_Fluid.bufC(FCLR),	m_Fluid.gpu(FCLR),	mNumPoints *sizeof(uint) ),	"TransferFromCUDA", "cuMemcpyDtoH", "FCLR", mbDebug);
//std::cout<<" m_Fluid.gpu(FCLR)="<< m_Fluid.gpu(FCLR)<<"\tm_Fluid.bufC(FCLR)="<< static_cast<void*>(m_Fluid.bufC(FCLR))<<"\n"<<std::flush;

    // add extra data for morphogenesis
    cuCheck( cuMemcpyDtoH ( m_Fluid.bufC(FELASTIDX),	m_Fluid.gpu(FELASTIDX),	mNumPoints *sizeof(uint[BOND_DATA]) ),	"TransferFromCUDA", "cuMemcpyDtoH", "FELASTIDX", mbDebug);
//std::cout<<" m_Fluid.gpu(FELASTIDX)="<< m_Fluid.gpu(FELASTIDX)<<"\tm_Fluid.bufC(FELASTIDX)="<< static_cast<void*>(m_Fluid.bufC(FELASTIDX))<<"\n"<<std::flush;   
    
    cuCheck( cuMemcpyDtoH ( m_Fluid.bufC(FPARTICLEIDX),	m_Fluid.gpu(FPARTICLEIDX),	mNumPoints *sizeof(uint[BONDS_PER_PARTICLE *2]) ),	"TransferFromCUDA", "cuMemcpyDtoH", "FPARTICLEIDX", mbDebug);

    cuCheck( cuMemcpyDtoH ( m_Fluid.bufC(FPARTICLE_ID),	m_Fluid.gpu(FPARTICLE_ID),	mNumPoints *sizeof(uint) ),	"TransferFromCUDA", "cuMemcpyDtoH", "FPARTICLE_ID", mbDebug);
    cuCheck( cuMemcpyDtoH ( m_Fluid.bufC(FMASS_RADIUS),	m_Fluid.gpu(FMASS_RADIUS),	mNumPoints *sizeof(uint) ),	"TransferFromCUDA", "cuMemcpyDtoH", "FMASS_RADIUS", mbDebug);
    
    cuCheck( cuMemcpyDtoH ( m_Fluid.bufC(FNERVEIDX),	m_Fluid.gpu(FNERVEIDX),	mNumPoints *sizeof(uint) ),	"TransferFromCUDA", "cuMemcpyDtoH", "FNERVEIDX", mbDebug);
//std::cout<<" m_Fluid.gpu(FNERVEIDX)="<< m_Fluid.gpu(FNERVEIDX)<<"\tm_Fluid.bufC(FNERVEIDX)="<< static_cast<void*>(m_Fluid.bufC(FNERVEIDX))<<"\n"<<std::flush;    
    
    cuCheck( cuMemcpyDtoH ( m_Fluid.bufC(FCONC),	m_Fluid.gpu(FCONC),	mNumPoints *sizeof(float[NUM_TF]) ),	"TransferFromCUDA", "cuMemcpyDtoH", "FCONC", mbDebug);
//    cuCheck( cuMemcpyDtoH ( m_Fluid.bufC(FEPIGEN),	m_Fluid.gpu(FEPIGEN),	mNumPoints *sizeof(uint[NUM_GENES]) ),	"TransferFromCUDA", "cuMemcpyDtoH", "FEPIGEN", mbDebug);

}


void FluidSystem::InsertParticlesCUDA ( uint* gcell, uint* gndx, uint* gcnt )
{
    cuCheck ( cuMemsetD8 ( m_Fluid.gpu(FGRIDCNT), 0,	m_GridTotal*sizeof(int) ), "InsertParticlesCUDA", "cuMemsetD8", "FGRIDCNT", mbDebug );
    cuCheck ( cuMemsetD8 ( m_Fluid.gpu(FGRIDOFF), 0,	m_GridTotal*sizeof(int) ), "InsertParticlesCUDA", "cuMemsetD8", "FGRIDOFF", mbDebug );

    void* args[1] = { &mNumPoints };
    cuCheck(cuLaunchKernel(m_Func[FUNC_INSERT], m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL),
            "InsertParticlesCUDA", "cuLaunch", "FUNC_INSERT", mbDebug);

    // Transfer data back if requested (for validation)
    if (gcell != 0x0) {
        cuCheck( cuMemcpyDtoH ( gcell,	m_Fluid.gpu(FGCELL),		mNumPoints *sizeof(uint) ), "InsertParticlesCUDA", "cuMemcpyDtoH", "FGCELL", mbDebug );
        cuCheck( cuMemcpyDtoH ( gndx,		m_Fluid.gpu(FGNDX),		mNumPoints *sizeof(uint) ), "InsertParticlesCUDA", "cuMemcpyDtoH", "FGNDX", mbDebug);
        cuCheck( cuMemcpyDtoH ( gcnt,		m_Fluid.gpu(FGRIDCNT),	m_GridTotal*sizeof(uint) ), "InsertParticlesCUDA", "cuMemcpyDtoH", "FGRIDCNT", mbDebug);
        cuCtxSynchronize ();
    }
}


// #define CPU_SUMS

void FluidSystem::PrefixSumCellsCUDA ( uint* goff, int zero_offsets )
{
#ifdef CPU_SUMS

    PERF_PUSH ( "PrefixSum (CPU)" );
    int numCells = m_GridTotal;

    cuCheck(cuMemcpyDtoH( m_Fluid.bufC(FGRIDCNT), m_Fluid.gpu(FGRIDCNT), numCells*sizeof(int)), "DtoH mgridcnt");
    cuCheck( cuCtxSynchronize(), "cuCtxSync(PrefixSum)" );

    uint* mgcnt = m_Fluid.bufI(FGRIDCNT);
    uint* mgoff = m_Fluid.bufI(FGRIDOFF);
    int sum = 0;
    for (int n=0; n < numCells; n++) {
        mgoff[n] = sum;
        sum += mgcnt[n];
    }
    cuCheck(cuMemcpyHtoD(m_Fluid.gpu(FGRIDOFF), m_Fluid.bufI(FGRIDOFF), numCells*sizeof(int)), "HtoD mgridoff");
    cuCheck( cuCtxSynchronize(), "cuCtxSync(PrefixSum)" );
    PERF_POP ();

    if ( goff != 0x0 ) {
        memcpy ( goff, mgoff, numCells*sizeof(uint) );
    }

#else

    // Prefix Sum - determine grid offsets
    int blockSize = SCAN_BLOCKSIZE << 1;
    int numElem1 = m_GridTotal;
    int numElem2 = int ( numElem1 / blockSize ) + 1;
    int numElem3 = int ( numElem2 / blockSize ) + 1;
    int threads = SCAN_BLOCKSIZE;
    int zon=1;

    CUdeviceptr array1  = m_Fluid.gpu(FGRIDCNT);		// input
    CUdeviceptr scan1   = m_Fluid.gpu(FGRIDOFF);		// output
    CUdeviceptr array2  = m_Fluid.gpu(FAUXARRAY1);
    CUdeviceptr scan2   = m_Fluid.gpu(FAUXSCAN1);
    CUdeviceptr array3  = m_Fluid.gpu(FAUXARRAY2);
    CUdeviceptr scan3   = m_Fluid.gpu(FAUXSCAN2);

#ifndef xlong
    typedef unsigned long long	xlong;		// 64-bit integer
#endif


    if ( numElem1 > SCAN_BLOCKSIZE*xlong(SCAN_BLOCKSIZE)*SCAN_BLOCKSIZE) {
        //nvprintf ( "ERROR: Number of elements exceeds prefix sum max. Adjust SCAN_BLOCKSIZE.\n" );
    }

    void* argsA[5] = {&array1, &scan1, &array2, &numElem1, &zero_offsets }; // sum array1. output -> scan1, array2
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXSUM], numElem2, 1, 1, threads, 1, 1, 0, NULL, argsA, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXSUM", mbDebug);

    void* argsB[5] = { &array2, &scan2, &array3, &numElem2, &zon }; // sum array2. output -> scan2, array3
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXSUM], numElem3, 1, 1, threads, 1, 1, 0, NULL, argsB, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXSUM", mbDebug);

    if ( numElem3 > 1 ) {
        CUdeviceptr nptr = {0};
        void* argsC[5] = { &array3, &scan3, &nptr, &numElem3, &zon };	// sum array3. output -> scan3
        cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXSUM], 1, 1, 1, threads, 1, 1, 0, NULL, argsC, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);

        void* argsD[3] = { &scan2, &scan3, &numElem2 };	// merge scan3 into scan2. output -> scan2
        cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXFIXUP], numElem3, 1, 1, threads, 1, 1, 0, NULL, argsD, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);
    }

    void* argsE[3] = { &scan1, &scan2, &numElem1 };		// merge scan2 into scan1. output -> scan1
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXFIXUP], numElem2, 1, 1, threads, 1, 1, 0, NULL, argsE, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);

    // Transfer data back if requested
    if ( goff != 0x0 ) {
        cuCheck( cuMemcpyDtoH ( goff,		m_Fluid.gpu(FGRIDOFF),	numElem1*sizeof(int) ), "PrefixSumCellsCUDA", "cuMemcpyDtoH", "FGRIDOFF", mbDebug);
        cuCtxSynchronize ();
    }
#endif
}


void FluidSystem::CountingSortFullCUDA ( Vector3DF* ppos )
{
    // Transfer particle data to temp buffers
    //  (gpu-to-gpu copy, no sync needed)
    TransferToTempCUDA ( FPOS,		mNumPoints *sizeof(Vector3DF) );
    TransferToTempCUDA ( FVEL,		mNumPoints *sizeof(Vector3DF) );
    TransferToTempCUDA ( FVEVAL,	mNumPoints *sizeof(Vector3DF) );
    TransferToTempCUDA ( FFORCE,	mNumPoints *sizeof(Vector3DF) );
    TransferToTempCUDA ( FPRESS,	mNumPoints *sizeof(float) );
    TransferToTempCUDA ( FDENSITY,	mNumPoints *sizeof(float) );
    TransferToTempCUDA ( FCLR,		mNumPoints *sizeof(uint) );
    TransferToTempCUDA ( FGCELL,	mNumPoints *sizeof(uint) );
    TransferToTempCUDA ( FGNDX,		mNumPoints *sizeof(uint) );
    
    // extra data for morphogenesis
    TransferToTempCUDA ( FELASTIDX,		mNumPoints *sizeof(uint[BOND_DATA]) );
    TransferToTempCUDA ( FPARTICLEIDX,	mNumPoints *sizeof(uint[BONDS_PER_PARTICLE *2]) );
    TransferToTempCUDA ( FPARTICLE_ID,	mNumPoints *sizeof(uint) );
    TransferToTempCUDA ( FMASS_RADIUS,	mNumPoints *sizeof(uint) );
    TransferToTempCUDA ( FNERVEIDX,		mNumPoints *sizeof(uint) );
    TransferToTempCUDA ( FCONC,		    mNumPoints *sizeof(float[NUM_TF]) );
    TransferToTempCUDA ( FEPIGEN,	    mNumPoints *sizeof(uint[NUM_GENES]) );

    // reset bonds and forces in fbuf FELASTIDX, FPARTICLEIDX and FFORCE, required to prevent interference between time steps, 
    // because these are not necessarily overwritten by the FUNC_COUNTING_SORT kernel.
/*
    cuCheck ( cuMemsetD8 ( m_Fluid.gpu(FGRIDOFF), 0,	m_GridTotal*sizeof(int) ), "InsertParticlesCUDA", "cuMemsetD8", "FGRIDOFF", mbDebug );
    cuCheck( cuMemcpyDtoH ( m_Fluid.bufC(FELASTIDX),	m_Fluid.gpu(FELASTIDX),	mNumPoints *sizeof(uint[BOND_DATA]) ),	"TransferFromCUDA", "cuMemcpyDtoH", "FELASTIDX", mbDebug);
    cuCheck( cuMemcpyDtoH ( m_Fluid.bufC(FPARTICLEIDX),	m_Fluid.gpu(FPARTICLEIDX),	mNumPoints *sizeof(uint[BONDS_PER_PARTICLE]) ),	"TransferFromCUDA", "cuMemcpyDtoH", "FPARTICLEIDX", mbDebug);
*/

    cuCtxSynchronize ();    // needed to prevent colision with previous operations
    cuCheck ( cuMemsetD32 ( m_Fluid.gpu(FELASTIDX),    UINT_MAX,  mNumPoints * BOND_DATA              ),  "CountingSortFullCUDA", "cuMemsetD32", "FELASTIDX",    mbDebug);
    cuCheck ( cuMemsetD32 ( m_Fluid.gpu(FPARTICLEIDX), UINT_MAX,  mNumPoints * BONDS_PER_PARTICLE *2  ),  "CountingSortFullCUDA", "cuMemsetD32", "FPARTICLEIDX", mbDebug);
    cuCheck ( cuMemsetD32 ( m_Fluid.gpu(FFORCE),      (uint)0.0,  mNumPoints * 3 /* ie num elements */),  "CountingSortFullCUDA", "cuMemsetD8" , "FFORCE",       mbDebug);
    cuCtxSynchronize ();    // needed to prevent colision with previous operations
    // Reset grid cell IDs
    // cuCheck(cuMemsetD32(m_Fluid.gpu(FGCELL), GRID_UNDEF, numPoints ), "cuMemsetD32(Sort)");

    void* args[1] = { &mNumPoints };
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_COUNTING_SORT], m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL),
              "CountingSortFullCUDA", "cuLaunch", "FUNC_COUNTING_SORT", mbDebug );
/*    
 * cuCtxSynchronize ();
std::cout<<"\n\nCountingSortFullCUDA finished\n\n"<<std::flush;
    //countingSortBondIDX  FUNC_SORT_BONDIDX
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_SORT_BONDIDX], m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL),
              "countingSortBondIDX", "cuLaunch", "FUNC_SORT_BONDIDX", mbDebug );
*/    
    if ( ppos != 0x0 ) {
        cuCheck( cuMemcpyDtoH ( ppos,		m_Fluid.gpu(FPOS),	mNumPoints*sizeof(Vector3DF) ), "CountingSortFullCUDA", "cuMemcpyDtoH", "FPOS", mbDebug);
        cuCtxSynchronize ();
    }
}



void FluidSystem::ComputePressureCUDA ()
{
    void* args[1] = { &mNumPoints };
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_COMPUTE_PRESS],  m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL), "ComputePressureCUDA", "cuLaunch", "FUNC_COMPUTE_PRESS", mbDebug);
}

void FluidSystem::ComputeDiffusionCUDA(){
    std::cout << "\n\nRunning ComputeDiffusionCUDA()" << std::endl;
    void* args[1] = { &mNumPoints };
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_COMPUTE_DIFFUSION],  m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL), "ComputeDiffusionCUDA", "cuLaunch", "FUNC_COMPUTE_DIFFUSION", mbDebug);
}

void FluidSystem::ComputeForceCUDA ()
{
    //printf("\n\nFluidSystem::ComputeForceCUDA (),  m_FParams.freeze=%s",(m_FParams.freeze==true) ? "true" : "false");
    void* args[3] = { &m_FParams.pnum ,  &m_FParams.freeze, &m_FParams.frame};
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_COMPUTE_FORCE],  m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL), "ComputeForceCUDA", "cuLaunch", "FUNC_COMPUTE_FORCE", mbDebug);
}
/*
void FluidSystem::FreezeCUDA ()
{
    void* args[1] = { &m_FParams.pnum };
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_FREEZE],  m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL), "ComputeForceCUDA", "cuLaunch", "FUNC_FREEZE", mbDebug);
}
*/
void FluidSystem::AdvanceCUDA ( float tm, float dt, float ss )
{
    void* args[4] = { &tm, &dt, &ss, &m_FParams.pnum };
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_ADVANCE],  m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL), "AdvanceCUDA", "cuLaunch", "FUNC_ADVANCE", mbDebug);
}

void FluidSystem::EmitParticlesCUDA ( float tm, int cnt )
{
    void* args[3] = { &tm, &cnt, &m_FParams.pnum };
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_EMIT],  m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL), "EmitParticlesCUDA", "cuLaunch", "FUNC_EMIT", mbDebug);
}








