#include <assert.h>
#include <iostream>
#include <cuda.h>
#include "cutil_math.h"
#include "fluid_system.h"
#include <stdlib.h>
#include <unistd.h>

extern bool gProfileRend;
#define EPSILON			0.00001f			// for collision detection
#define SCAN_BLOCKSIZE		512				// must match value in fluid_system_cuda.cu

bool cuCheck (CUresult launch_stat, const char* method, const char* apicall, const char* arg, bool bDebug){
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
FluidSystem::FluidSystem (){
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

void FluidSystem::LoadKernel ( int fid, std::string func ){
    char cfn[512];
    strcpy ( cfn, func.c_str() );

    if ( m_Func[fid] == (CUfunction) -1 )
        cuCheck ( cuModuleGetFunction ( &m_Func[fid], m_Module, cfn ), "LoadKernel", "cuModuleGetFunction", cfn, mbDebug );
}

void FluidSystem::Initialize (){             // used for CPU only for "check_demo".
    std::cout << "FluidSystem::Initialize () \n";
    // An FBufs struct holds an array of pointers.
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
    mNumPoints = 0;			// reset count
    std::cout << "Chk1.6 \n";
}

// /home/nick/Programming/Cuda/Morphogenesis/build/install/ptx/objects/fluid_systemPTX/fluid_system_cuda.ptx
void FluidSystem::InitializeCuda (){         // used for load_sim  /home/nick/Programming/Cuda/Morphogenesis/build/install/ptx/objects-Debug/fluid_systemPTX/fluid_system_cuda.ptx
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
    LoadKernel ( FUNC_TALLYLISTS,       "tally_denselist_lengths");
//    LoadKernel ( FUNC_FREEZE,		    "freeze" );
    LoadKernel ( FUNC_COMPUTE_DIFFUSION,"computeDiffusion");
    LoadKernel ( FUNC_COUNT_SORT_LISTS, "countingSortDenseLists");
    LoadKernel ( FUNC_COMPUTE_GENE_ACTION, "computeGeneAction");
    LoadKernel ( FUNC_COMPUTE_BOND_CHANGES, "computeBondChanges");
    
    //LoadKernel ( FUNC_INSERT_CHANGES, "insertChanges");
    //LoadKernel ( FUNC_PREFIXUP_CHANGES, "prefixFixupChanges");
    //LoadKernel ( FUNC_PREFIXSUM_CHANGES, "prefixSumChanges");
    //LoadKernel ( FUNC_TALLYLISTS_CHANGES, "tally_changelist_lengths");
    LoadKernel ( FUNC_COUNTING_SORT_CHANGES, "countingSortChanges");
    LoadKernel ( FUNC_COMPUTE_NERVE_ACTION, "computeNerveActivation");
    
    LoadKernel ( FUNC_COMPUTE_MUSCLE_CONTRACTION, "computeMuscleContraction");
    LoadKernel ( FUNC_HEAL, "heal");
    LoadKernel ( FUNC_LENGTHEN_MUSCLE, "lengthen_muscle");
    LoadKernel ( FUNC_LENGTHEN_TISSUE, "lengthen_tissue");
    LoadKernel ( FUNC_SHORTEN_MUSCLE, "shorten_muscle");
    LoadKernel ( FUNC_SHORTEN_TISSUE, "shorten_tissue");
    
    LoadKernel ( FUNC_STRENGTHEN_MUSCLE, "strengthen_muscle");
    LoadKernel ( FUNC_STRENGTHEN_TISSUE, "strengthen_tissue");
    LoadKernel ( FUNC_WEAKEN_MUSCLE, "weaken_muscle");
    LoadKernel ( FUNC_WEAKEN_TISSUE, "weaken_tissue");
    

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
    //ClearNeighborTable ();
    mNumPoints = 0;			// reset count
    std::cout << "Chk1.6 \n";
}

/////////////////////////////////////////////////////////////////
void FluidSystem::UpdateGenome (){              // Update Genome on GPU
    cuCheck ( cuMemcpyHtoD ( cuFGenome,	&m_FGenome,		sizeof(FGenome) ), "FluidGenomeCUDA", "cuMemcpyHtoD", "cuFGenome", mbDebug);
}

void FluidSystem::SetGenome (FGenome newGenome ){   // not currently used.
    for(int i=0; i< NUM_GENES; i++) m_FGenome.mutability[i] = newGenome.mutability[i];
    for(int i=0; i< NUM_GENES; i++) m_FGenome.delay[i] = newGenome.delay[i];
    for(int i=0; i< NUM_GENES; i++) for(int j=0; j< NUM_GENES; j++) {
            m_FGenome.sensitivity[i][j] = newGenome.sensitivity[i][j];
        }
    for(int i=0; i< NUM_GENES; i++) {
        m_FGenome.tf_diffusability[i] = newGenome.tf_diffusability[i];
        m_FGenome.tf_breakdown_rate[i] = newGenome.tf_breakdown_rate[i];
    }
    for(int i=0; i< NUM_GENES; i++) for(int j=0; j< 2*NUM_TF+1; j++) m_FGenome.secrete[i][j] =newGenome.secrete[i][j];     // 1st zero arrays.
    for(int i=0; i< NUM_GENES; i++) for(int j=0; j< 2*NUM_TF+1; j++) m_FGenome.activate[i][j]=newGenome.activate[i][j]; 
    /*
    //0=elastin
    m_FGenome.fbondparams[0].elongation_threshold   = 0.1  ;
    m_FGenome.fbondparams[0].elongation_factor      = 0.1  ;
    m_FGenome.fbondparams[0].strength_threshold     = 0.1  ;
    m_FGenome.fbondparams[0].strengthening_factor   = 0.1  ;
    
    m_FGenome.fbondparams[0].max_rest_length        = 0.8  ;
    m_FGenome.fbondparams[0].min_rest_length        = 0.3  ;
    m_FGenome.fbondparams[0].max_modulus            = 0.8  ;
    m_FGenome.fbondparams[0].min_modulus            = 0.3  ;
    
    m_FGenome.fbondparams[0].elastLim               = 2  ;
    m_FGenome.fbondparams[0].default_rest_length    = 0.5  ;
    m_FGenome.fbondparams[0].default_modulus        = 100000  ;
    m_FGenome.fbondparams[0].default_damping        = 10  ;
    
    //1=collagen
    m_FGenome.fbondparams[1].elongation_threshold   = 0.1  ;
    m_FGenome.fbondparams[1].elongation_factor      = 0.1  ;
    m_FGenome.fbondparams[1].strength_threshold     = 0.1  ;
    m_FGenome.fbondparams[1].strengthening_factor   = 0.1  ;
    
    m_FGenome.fbondparams[1].max_rest_length        = 0.8  ;
    m_FGenome.fbondparams[1].min_rest_length        = 0.3  ;
    m_FGenome.fbondparams[1].max_modulus            = 0.8  ;
    m_FGenome.fbondparams[1].min_modulus            = 0.3  ;
    
    m_FGenome.fbondparams[1].elastLim               = 0.55  ;
    m_FGenome.fbondparams[1].default_rest_length    = 0.5  ;
    m_FGenome.fbondparams[1].default_modulus        = 10000000  ;
    m_FGenome.fbondparams[1].default_damping        = 100  ;
    
    //2=apatite
    m_FGenome.fbondparams[2].elongation_threshold   = 0.1  ;
    m_FGenome.fbondparams[2].elongation_factor      = 0.1  ;
    m_FGenome.fbondparams[2].strength_threshold     = 0.1  ;
    m_FGenome.fbondparams[2].strengthening_factor   = 0.1  ;
    
    m_FGenome.fbondparams[2].max_rest_length        = 0.8  ;
    m_FGenome.fbondparams[2].min_rest_length        = 0.3  ;
    m_FGenome.fbondparams[2].max_modulus            = 0.8  ;
    m_FGenome.fbondparams[2].min_modulus            = 0.3  ;
    
    m_FGenome.fbondparams[2].elastLim               = 0.05  ;
    m_FGenome.fbondparams[2].default_rest_length    = 0.5  ;
    m_FGenome.fbondparams[2].default_modulus        = 10000000  ;
    m_FGenome.fbondparams[2].default_damping        = 1000  ;
    */
}

void FluidSystem::UpdateParams (){
    // Update Params on GPU
    Vector3DF grav = m_Vec[PPLANE_GRAV_DIR] * m_Param[PGRAV];
    FluidParamCUDA (  m_Param[PSIMSCALE], m_Param[PSMOOTHRADIUS], m_Param[PRADIUS], m_Param[PMASS], m_Param[PRESTDENSITY],
                      *(float3*)& m_Vec[PBOUNDMIN], *(float3*)& m_Vec[PBOUNDMAX], m_Param[PEXTSTIFF], m_Param[PINTSTIFF],
                      m_Param[PVISC], m_Param[PEXTDAMP], m_Param[PFORCE_MIN], m_Param[PFORCE_MAX], m_Param[PFORCE_FREQ],
                      m_Param[PGROUND_SLOPE], grav.x, grav.y, grav.z, m_Param[PACCEL_LIMIT], m_Param[PVEL_LIMIT],
                      (int) m_Vec[PEMIT_RATE].x );
}

void FluidSystem::SetParam (int p, float v ){
    m_Param[p] = v;
    UpdateParams ();
}

void FluidSystem::SetVec ( int p, Vector3DF v ){
    m_Vec[p] = v;
    UpdateParams ();
}

void FluidSystem::Exit (){
    // Free fluid buffers
    for (int n=0; n < MAX_BUF; n++ ) {
        std::cout << "\n n = " << n << std::flush;
        if ( m_Fluid.bufC(n) != 0x0 )
            free ( m_Fluid.bufC(n) );
    }
    if(m_Module != 0x0) cudaDeviceReset(); // Destroy all allocations and reset all state on the current device in the current process. // must only operate if we have a cuda instance.
}

void FluidSystem::AllocateBuffer ( int buf_id, int stride, int cpucnt, int gpucnt, int gpumode, int cpumode ){   // mallocs a buffer - called by FluidSystem::Initialize(), AllocateParticles, and AllocateGrid()
//also called by WriteDemoSimParams(..)
std::cout<<"\nAllocateBuffer ( int buf_id="<<buf_id<<", int stride="<<stride<<", int cpucnt="<<cpucnt<<", int gpucnt="<<gpucnt<<", int "<<gpumode<<", int "<<cpumode<<" )\t"<<std::flush;
    if (cpumode == CPU_YES) {
        char* src_buf  = m_Fluid.bufC(buf_id);
        char* dest_buf = (char*) malloc(cpucnt*stride);                   //  ####  malloc the buffer   ####
        if (src_buf != 0x0) {
            memcpy(dest_buf, src_buf, cpucnt*stride);
            free(src_buf);
        }
        m_Fluid.setBuf(buf_id, dest_buf);                                 // stores pointer to buffer in mcpu[buf_id]
    }

    if (gpumode == GPU_SINGLE || gpumode == GPU_DUAL )	{
        if (m_Fluid.gpuptr(buf_id) != 0x0) cuCheck(cuMemFree(m_Fluid.gpu(buf_id)), "AllocateBuffer", "cuMemFree", "Fluid.gpu", mbDebug);
        cuCheck( cuMemAlloc(m_Fluid.gpuptr(buf_id), stride*gpucnt), "AllocateBuffer", "cuMemAlloc", "Fluid.gpu", mbDebug);         //  ####  cuMemAlloc the buffer, stores pointer to buffer in   m_Fluid.mgpu[buf_id]
        std::cout<<"\t\t m_Fluid.gpuptr("<<buf_id<<")'"<<m_Fluid.gpuptr(buf_id)<<",   m_Fluid.gpu("<<buf_id<<")="<<m_Fluid.gpu(buf_id)<<"\t"<<std::flush;
    }
    if (gpumode == GPU_TEMP || gpumode == GPU_DUAL ) {
        if (m_FluidTemp.gpuptr(buf_id) != 0x0) cuCheck(cuMemFree(m_FluidTemp.gpu(buf_id)), "AllocateBuffer", "cuMemFree", "FluidTemp.gpu", mbDebug);
        cuCheck( cuMemAlloc(m_FluidTemp.gpuptr(buf_id), stride*gpucnt), "AllocateBuffer", "cuMemAlloc", "FluidTemp.gpu", mbDebug); //  ####  cuMemAlloc the buffer, stores pointer to buffer in   m_FluidTemp.mgpu[buf_id]
    }
}

// Allocate particle memory
void FluidSystem::AllocateParticles ( int cnt, int gpu_mode, int cpu_mode ){ // calls AllocateBuffer(..) for each buffer.  
// Defaults in header : int gpu_mode = GPU_DUAL, int cpu_mode = CPU_YES
// Called by FluidSystem::ReadPointsCSV(..), and FluidSystem::WriteDemoSimParams(...), cnt = mMaxPoints.
std::cout<<"\nAllocateParticles ( int cnt="<<cnt<<", int "<<gpu_mode<<", int "<<cpu_mode<<" )\n"<<std::flush;
std::cout<<"\nGPU_OFF=0, GPU_SINGLE=1, GPU_TEMP=2, GPU_DUAL=3, CPU_OFF=4, CPU_YES=5"<<std::flush;
    AllocateBuffer ( FPOS,		sizeof(Vector3DF),	cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );//AllocateBuffer ( int buf_id, int stride,     int cpucnt, int gpucnt,    int gpumode,    int cpumode )
std::cout<<"\nm_Fluid.gpu(FPOS)="<< m_Fluid.gpu(FPOS)<<"\tm_Fluid.bufC(FPOS)="<< static_cast<void*>(m_Fluid.bufC(FPOS))<<"\n"<<std::flush;
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
std::cout<<"\nchl2.11.1\n"<<std::flush;
    AllocateBuffer ( FELASTIDX,	sizeof(uint[BOND_DATA]),             cnt,   m_FParams.szPnts,	gpu_mode, cpu_mode );  // used to be [BONDS_PER_PARTICLE * 2]
    AllocateBuffer ( FPARTICLEIDX,	sizeof(uint[BONDS_PER_PARTICLE *2]),cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FPARTICLE_ID,	sizeof(uint),		             cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FMASS_RADIUS,	sizeof(uint),		             cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    
    AllocateBuffer ( FNERVEIDX,	sizeof(uint),		                 cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FCONC,	    sizeof(float[NUM_TF]),		         cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FEPIGEN,	sizeof(uint[NUM_GENES]),	         cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
std::cout<<"\nchl2.11.2\n"<<std::flush;
    
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

void FluidSystem::AllocateGrid(){
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

void FluidSystem::AllocateBufferDenseLists ( int buf_id, int stride, int gpucnt, int lists ) {    // mallocs a buffer - called by FluidSystem::AllocateGrid(int gpu_mode, int cpu_mode)
// Need to save "pointers to the allocated gpu buffers" in a cpu array, AND then cuMemcpyHtoD(...) that list of pointers into the device array.   
    // also called by FluidSystem::....()  to quadruple buffer as needed.
    CUdeviceptr*  listpointer = (CUdeviceptr*) &m_Fluid.bufC(lists)[buf_id * sizeof(CUdeviceptr)] ;
    printf("\nlistpointer=%p\t", (CUdeviceptr* ) *listpointer);
    if (*listpointer != 0x0) cuCheck(cuMemFree(*listpointer), "AllocateBufferDenseLists", "cuMemFree", "*listpointer", mbDebug);
    cuCheck( cuMemAlloc( listpointer, stride*gpucnt),   "AllocateBufferDenseLists", "cuMemAlloc", "listpointer", mbDebug);         
}

void FluidSystem::AllocateGrid(int gpu_mode, int cpu_mode){ // NB void FluidSystem::AllocateBuffer (int buf_id, int stride, int cpucnt, int gpucnt, int gpumode, int cpumode) 
    // Allocate grid
    int cnt = m_GridTotal;
    m_FParams.szGrid = (m_FParams.gridBlocks * m_FParams.gridThreads);
    AllocateBuffer ( FGRID,		sizeof(uint),		mMaxPoints,	m_FParams.szPnts,	gpu_mode, cpu_mode );    // # grid elements = number of points
    AllocateBuffer ( FGRIDCNT,	sizeof(uint),		cnt,	    m_FParams.szGrid,	gpu_mode, cpu_mode );
    AllocateBuffer ( FGRIDOFF,	sizeof(uint),		cnt,	    m_FParams.szGrid,	gpu_mode, cpu_mode );
    AllocateBuffer ( FGRIDACT,	sizeof(uint),		cnt,	    m_FParams.szGrid,	gpu_mode, cpu_mode );
    // extra buffers for dense lists
    AllocateBuffer ( FGRIDCNT_ACTIVE_GENES,  sizeof(uint[cnt]),       NUM_GENES,   m_FParams.szGrid,	gpu_mode, cpu_mode );
    AllocateBuffer ( FGRIDOFF_ACTIVE_GENES,  sizeof(uint[cnt]),       NUM_GENES,   m_FParams.szGrid,	gpu_mode, cpu_mode );
    AllocateBuffer ( FDENSE_LIST_LENGTHS,	 sizeof(uint),		      NUM_GENES,   NUM_GENES,	        gpu_mode, cpu_mode );
    AllocateBuffer ( FDENSE_LISTS,	         sizeof(CUdeviceptr),     NUM_GENES,   NUM_GENES,           gpu_mode, cpu_mode );
    AllocateBuffer ( FDENSE_BUF_LENGTHS,	 sizeof(uint),            NUM_GENES,   NUM_GENES,           gpu_mode, cpu_mode );
    
    AllocateBuffer ( FGRIDCNT_CHANGES,               sizeof(uint[cnt]),       NUM_CHANGES,   m_FParams.szGrid,	    gpu_mode, cpu_mode );
    AllocateBuffer ( FGRIDOFF_CHANGES,               sizeof(uint[cnt]),       NUM_CHANGES,   m_FParams.szGrid,	    gpu_mode, cpu_mode );
    AllocateBuffer ( FDENSE_LIST_LENGTHS_CHANGES,	 sizeof(uint),		      NUM_CHANGES,   NUM_CHANGES,	        gpu_mode, cpu_mode );
    AllocateBuffer ( FDENSE_LISTS_CHANGES,	         sizeof(CUdeviceptr),     NUM_CHANGES,   NUM_CHANGES,           gpu_mode, cpu_mode );
    AllocateBuffer ( FDENSE_BUF_LENGTHS_CHANGES,	 sizeof(uint),            NUM_CHANGES,   NUM_CHANGES,           gpu_mode, cpu_mode );

    if(gpu_mode == GPU_SINGLE || gpu_mode == GPU_DUAL )for(int i=0; i<NUM_GENES; i++){ //for each gene allocate intial buffer, write pointer and size to FDENSE_LISTS and FDENSE_LIST_LENGTHS
        CUdeviceptr*  _listpointer = (CUdeviceptr*) &m_Fluid.bufC(FDENSE_LISTS)[i * sizeof(CUdeviceptr)] ;
        *_listpointer = 0x0;
        AllocateBufferDenseLists( i, sizeof(uint), INITIAL_BUFFSIZE_ACTIVE_GENES, FDENSE_LISTS);  // AllocateBuffer writes pointer to  m_Fluid.gpuptr(buf_id). 
        m_Fluid.bufI(FDENSE_LIST_LENGTHS)[i] = 0;
        m_Fluid.bufI(FDENSE_BUF_LENGTHS)[i]  = INITIAL_BUFFSIZE_ACTIVE_GENES;
    }
    
    if(gpu_mode == GPU_SINGLE || gpu_mode == GPU_DUAL )for(int i=0; i<NUM_CHANGES; i++){ //Same for the changes lists
        CUdeviceptr*  _listpointer = (CUdeviceptr*) &m_Fluid.bufC(FDENSE_LISTS_CHANGES)[i * sizeof(CUdeviceptr)] ;
        *_listpointer = 0x0; 
        AllocateBufferDenseLists( i, 2*sizeof(uint), INITIAL_BUFFSIZE_ACTIVE_GENES, FDENSE_LISTS_CHANGES); // NB buf[2][list_length] holding : particleIdx, bondIdx
        m_Fluid.bufI(FDENSE_LIST_LENGTHS_CHANGES)[i] = 0;
        m_Fluid.bufI(FDENSE_BUF_LENGTHS_CHANGES)[i]  = INITIAL_BUFFSIZE_ACTIVE_GENES;
    }
    
    cuMemcpyHtoD(m_Fluid.gpu(FDENSE_LISTS), m_Fluid.bufC(FDENSE_LISTS),  NUM_GENES * sizeof(CUdeviceptr)  );
    // Update GPU access pointers
    if (gpu_mode != GPU_OFF ) {
        cuCheck(cuMemcpyHtoD(cuFBuf, &m_Fluid, sizeof(FBufs)), "AllocateGrid", "cuMemcpyHtoD", "cuFBuf", mbDebug);
        cuCheck(cuCtxSynchronize(), "AllocateParticles", "cuCtxSynchronize", "", mbDebug);
    }
}

int FluidSystem::AddParticleMorphogenesis2 (Vector3DF* Pos, Vector3DF* Vel, uint Age, uint Clr, float *_ElastIdx, uint *_Particle_Idx, uint Particle_ID, uint Mass_Radius, uint NerveIdx, float* _Conc, uint* _EpiGen ){  // called by :ReadPointsCSV2 (...) where :    uint Particle_Idx[BONDS_PER_PARTICLE * 2];  AND SetupAddVolumeMorphogenesis2(....)
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
        ElastIdx[j] = _ElastIdx[j] ;                                                    // ## implicit cast float-> uint, what happens ? 
    }
    uint* Particle_Idx = (m_Fluid.bufI(FPARTICLEIDX) + n * BONDS_PER_PARTICLE *2 );     // index of incoming bonds
    for(int j=0; j<(BONDS_PER_PARTICLE *2); j++) {
        Particle_Idx[j] = _Particle_Idx[j] ;
    }
    *(m_Fluid.bufI(FPARTICLE_ID) + n)   = Particle_ID;                                  // permanent ID of particle 
    *(m_Fluid.bufI(FMASS_RADIUS) + n)   = Mass_Radius;
    *(m_Fluid.bufI(FNERVEIDX) + n)      = NerveIdx;
    
    for(int j=0; j<(NUM_TF); j++) {
        float* Conc = getConc(j);
        Conc[n] = _Conc[j];
    }
    for(int j=0; j<(NUM_GENES); j++) {
        uint* EpiGen = getEpiGen(j);
        EpiGen[n]= _EpiGen[j];                                                          // NB  data order  FEPIGEN[gene][particle]
    }
    mNumPoints++;
    return n;
}

void FluidSystem::AddNullPoints (){// fills unallocated particles with null data upto mMaxPoints. These can then be used to "create" new particles.
    std::cout<<"\n AddNullPoints ()\n"<<std::flush;
    Vector3DF Pos, Vel;
    uint Age, Clr;
    float ElastIdx[BOND_DATA];
    uint Particle_Idx[2*BONDS_PER_PARTICLE];
    uint Particle_ID, Mass_Radius, NerveIdx;
    float Conc[NUM_TF];
    uint EpiGen[NUM_GENES];
    
    Pos.x = m_FParams.pboundmax.x; 
    Pos.y = m_FParams.pboundmax.y; 
    Pos.z = m_FParams.pboundmax.z;
    Vel.x = 0; 
    Vel.y = 0; 
    Vel.z = 0;
    Age   = 0; 
    Clr   = 0; 
    for (int j=0;j<BOND_DATA;j++)               ElastIdx[j]     = UINT_MAX;
    for (int j=0;j<2*BONDS_PER_PARTICLE;j++)    Particle_Idx[j] = UINT_MAX;
    Particle_ID = UINT_MAX;
    Mass_Radius = 0;
    NerveIdx    = UINT_MAX;
    for (int j=0;j<NUM_TF;j++)      Conc[j]     = 0;
    for (int j=0;j<NUM_GENES;j++)   EpiGen[j]   = 0;
    
    // TODO FPARTICLE_ID   // should equal mNumPoints when created
    std::cout<<"\n AddNullPoints (): mNumPoints="<<mNumPoints<<", mMaxPoints="<<mMaxPoints<<"\n"<<std::flush;
    while (mNumPoints < mMaxPoints){
        AddParticleMorphogenesis2 (&Pos, &Vel, Age, Clr, ElastIdx, Particle_Idx, Particle_ID, Mass_Radius,  NerveIdx, Conc, EpiGen );
        std::cout<<"\n AddNullPoints (): mNumPoints="<<mNumPoints<<", mMaxPoints="<<mMaxPoints<<"\n"<<std::flush;
    }
}

void FluidSystem::SetupAddVolumeMorphogenesis2(Vector3DF min, Vector3DF max, float spacing, float offs, int total ){  // NB ony used in WriteDemoSimParams() called by make_demo.cpp . Creates a cuboid with all particle values definable.
std::cout << "\n SetupAddVolumeMorphogenesis2 \t" << std::flush ;
    Vector3DF pos;
    float dx, dy, dz;
    int cntx, cntz, p, c2;
    cntx = (int) ceil( (max.x-min.x-offs) / spacing );
    cntz = (int) ceil( (max.z-min.z-offs) / spacing );
    int cnt = cntx * cntz;
    min += offs;
    max -= offs;
    dx = max.x-min.x;
    dy = max.y-min.y;
    dz = max.z-min.z;
    Vector3DF rnd;
    c2 = cnt/2;
    Vector3DF Pos, Vel; 
    uint Age, Clr, Particle_ID, Mass_Radius, NerveIdx;
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
                for (int i = 0; i<BONDS_PER_PARTICLE*2;i++) { Particle_Idx[i] = UINT_MAX; }
                if (Particle_ID % 10 == 0){NerveIdx = Particle_ID/10;} else {NerveIdx = 0;} // Every 10th particle has nerve connection
                
                // Mass & radius of particles
                // 4bit mass + 4bit radius + 24bit uid // but for now, 16bit mass & radius
                // Note m_params[] is set in "FluidSystem::SetupDefaultParams ()" and "FluidSystem::SetupExampleParams ()"
                // mass = m_Param[PMASS]; // 0.00020543f; // kg
                // radius = m_Param[PRADIUS]; // 0.015f; // m
                Mass_Radius =  ( (uint(m_Param[PMASS]*255.0f*255.0f)<<16) | uint(m_Param[PRADIUS]*255.0f*255.0f) ) ; // mass=>13, radius=>975
                for (int i=0; i< NUM_TF; i++)    { Conc[i] = 1 ;}       // morphogen & transcription factor concentrations
                for (int i=0; i< NUM_GENES; i++) { EpiGen[i] = i ;}     // epigenetic state of each gene in this particle
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
    AddNullPoints ();                                                           // If spare particles remain, fill with null points. NB these can be used to "create" particles.
    std::cout << "\n SetupAddVolumeMorphogenesis2 finished \n" << std::flush ;
}

///////////////////////////////////////////////////////////////////
void FluidSystem::Run (){
std::cout << "\tFluidSystem::Run (),  "<<std::flush;  
    //case RUN_GPU_FULL:					// Full CUDA pathway, GRID-accelerted GPU, /w deep copy sort
//TransferFromCUDA ();
//std::cout << "\n\n Chk1 \n"<<std::flush;
    InsertParticlesCUDA ( 0x0, 0x0, 0x0 );
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After InsertParticlesCUDA", mbDebug);
//TransferFromCUDA ();
std::cout << "\n\n Chk2 \n"<<std::flush;
    PrefixSumCellsCUDA ( 1 );
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After PrefixSumCellsCUDA", mbDebug);
//TransferFromCUDA ();
std::cout << "\n\n Chk3 \n"<<std::flush;
    CountingSortFullCUDA ( 0x0 );
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After CountingSortFullCUDA", mbDebug);
//TransferFromCUDA ();
std::cout << "\n\n Chk4 \n"<<std::flush;
    
    ComputePressureCUDA();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After ComputePressureCUDA", mbDebug); 
//TransferFromCUDA ();
std::cout << "\n\n Chk5 \n"<<std::flush;
    // FreezeCUDA ();                                   // makes the system plastic, ie the bonds keep reforming
//std::cout << "\n\n Chk6 \n"<<std::flush;
    ComputeForceCUDA ();                                // now includes the function of freeze 
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After ComputeForceCUDA", mbDebug);

    // TODO compute nerve activation ? 
    
    
    // TODO compute muscle action ?
    
std::cout << "\n\n Chk6 \n"<<std::flush;    
    ComputeDiffusionCUDA();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After ComputeDiffusionCUDA", mbDebug);

std::cout << "\n\n Chk7 \n"<<std::flush;    
    ComputeGenesCUDA();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After ComputeGenesCUDA", mbDebug);
    
std::cout << "\n\n Chk8 \n"<<std::flush;
    ComputeBondChangesCUDA ();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After ComputeBondChangesCUDA", mbDebug);
    
    //  make dense lists of particle changes
    // insert changes
    // prefix sum changes, inc tally_changelist_lengths
    // counting sort changes
    //InsertChangesCUDA ( ); // done by ComputeBondChanges() above
    //cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After InsertChangesCUDA", mbDebug);

std::cout << "\n\n Chk9 \n"<<std::flush;
//#    PrefixSumChangesCUDA ( 1 );
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After PrefixSumChangesCUDA", mbDebug);
    
std::cout << "\n\n Chk10 \n"<<std::flush;
//#    CountingSortChangesCUDA (  );
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After CountingSortChangesCUDA", mbDebug);
    
    
    //  execute particle changes // _should_ be able to run concurrently => no cuCtxSynchronize()
    // => single fn ComputeParticleChangesCUDA ()
std::cout << "\n\n Chk11 \n"<<std::flush;
//#    ComputeParticleChangesCUDA ();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After ComputeParticleChangesCUDA", mbDebug);


std::cout << "\n\n Chk12 \n"<<std::flush;
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

void FluidSystem::Run (const char * relativePath, int frame ){       // version to save data after each kernel
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

    PrefixSumCellsCUDA ( 1 );
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

void FluidSystem::Freeze (){
    m_FParams.freeze = true;
    Run();
    m_FParams.freeze = false;
}

void FluidSystem::Freeze (const char * relativePath, int frame){
    m_FParams.freeze = true;
    Run(relativePath, frame);
    m_FParams.freeze = false;
}

void FluidSystem::AdvanceTime () {  // may need to prune unused details from this fn.
    m_Time += m_DT;

    m_Frame += m_FrameRange.z;

    if ( m_Frame > m_FrameRange.y && m_FrameRange.y != -1 ) {

        m_Frame = m_FrameRange.x;
        mbRecord = false;
        mbRecordBricks = false;
        m_Toggle[ PCAPTURE ] = false;
        std::cout << "Exiting.\n";
        exit ( 1 );
    }
}

///////////////////////////////////////////////////
// Ideal grid cell size (gs) = 2 * smoothing radius = 0.02*2 = 0.04
// Ideal domain size = k*gs/d = k*0.02*2/0.005 = k*8 = {8, 16, 24, 32, 40, 48, ..}
//    (k = number of cells, gs = cell size, d = simulation scale)
void FluidSystem::SetupGrid ( Vector3DF min, Vector3DF max, float sim_scale, float cell_size, float border ){
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

    if ( mPackGrid != 0x0 ) free ( mPackGrid );
    mPackGrid = (int*) malloc ( sizeof(int) * m_GridTotal );
}

///////////////////////////////////////////////////////////////
void FluidSystem::ReadGenome( const char * relativePath){
    // NB currently GPU allocation is by Allocate particles, called by ReadPointsCSV.
    const char * genes_file_path = relativePath;
    printf("\n## opening file %s \n", genes_file_path);
    FILE * genes_file = fopen(genes_file_path, "rb");
    if (genes_file == NULL) {
        std::cout << "\nvoid FluidSystem::ReadGenome( const char * relativePath, int gpu_mode, int cpu_mode)  Could not read file "<< genes_file_path <<"\n"<< std::flush;
        assert(0);
    }
    int num_genes1, num_genes2, num_tf;
    std::fscanf(genes_file, "num genes = %i,\tnum transcription factors = %i \n", &num_genes1, &num_tf);
    std::fscanf(genes_file, "mutability,\tdelay,\tsensitivity[%i],\tdifusability[2] \n", &num_genes2 );

    if ((num_genes1 != num_genes2) || (num_genes1 != NUM_GENES) || (num_tf != NUM_TF )){
        std::cout << "\n! Miss-match of parameters ! ((num_genes1 != num_genes2) || (num_genes1 != NUM_GENES) || (num_tf != NUM_TF ) )\n";
        std::cout << "num_genes = " << num_genes1 <<"\tnum_genes2 = "<<num_genes2<<"\t NUM_GENES = "<<NUM_GENES<<"\tnum_tf = "<<num_tf<<"\n";
    }
    int i, j, ret=0;
    for (i=0; i<num_genes1; i++ ) {
        ret = std::fscanf(genes_file,"%i,%i,",&m_FGenome.mutability[i],&m_FGenome.delay[i] );
        for(int j=0; j<NUM_GENES; j++)  ret += std::fscanf(genes_file,"%i,", &m_FGenome.sensitivity[i][j] );
        for(int j=0; j<NUM_TF; j++) ret += std::fscanf(genes_file, "%i,%i,\t", &m_FGenome.secrete[i][j*2], &m_FGenome.secrete[i][j*2 + 1] );//(elemID, secretion_rate)
        ret += std::fscanf(genes_file, "%i,\t\t", &m_FGenome.secrete[i][2*NUM_TF] );            //num active elements, NB sparse list, kernel will only read active elems.
        for(int j=0; j<NUM_GENES; j++) ret += std::fscanf(genes_file, "%i,%i,\t", &m_FGenome.activate[i][j*2], &m_FGenome.activate[i][j*2 + 1] );//(elemID, other_geneID)
        ret += std::fscanf(genes_file, "%i,\t\t \n", &m_FGenome.activate[i][2*NUM_GENES] );        //num active elements,
        if (ret != (2 + NUM_GENES + NUM_TF*2 + 1 + NUM_GENES*2 + 1) ) {
            std::cout << "\nvoid FluidSystem::ReadGenome, read failure !  gene number = " << i << ", ret = "<< ret <<"\n " << std::flush;
            fclose(genes_file);
            return;
        }
        for(int j=0; j<NUM_GENES; j++)  std::cout << m_FGenome.sensitivity[i][j] <<",";
        std::cout <<"\n";
    }
    std::cout << "\n" << i << " genes read.\n" << std::flush;
    
    ret=0;
    std::fscanf(genes_file,"\nTranscription Factors (tf_difusibility,tf_breakdown_rate \n" );
    for(i=0; i<num_tf; i++) { 
        ret += std::fscanf(genes_file,"\t%u,\t",&m_FGenome.tf_diffusability[i] );
        ret += std::fscanf(genes_file,"%u,\t",&m_FGenome.tf_breakdown_rate[i] );
        std::cout <<"\t("<< m_FGenome.tf_diffusability[i] <<"," << m_FGenome.tf_breakdown_rate[i] <<"),"<< std::flush;
    }
    if (ret != NUM_TF*2) std::cout << "\nvoid FluidSystem::ReadGenome, Transcription Factor read failure !  gene number = " << i << ", ret = "<< ret <<"\n " << std::flush;
    std::cout << "\n" << i << " transcription factors read.\n" << std::flush;
    
    ret=0;
    std::fscanf(genes_file, "\nRemodelling parameters \n" );
    for(i=0; i<3; i++){
        for(j=0; j<12;j++) ret += std::fscanf(genes_file, "\t%f,\t", &m_FGenome.param[i][j] ); 
        std::fscanf(genes_file, "\n");
    }
    std::cout << "\n" << i <<"*"<< j << " remodelling parameters read. ret = "<< ret <<"\n" << std::flush;
    
    fclose(genes_file);
}

void FluidSystem::WriteGenome( const char * relativePath){
    std::cout << "\n  FluidSystem::WriteGenome( const char * relativePath)  started \n" << std::flush;
    char buf[256];
    sprintf ( buf, "%s/genome.csv", relativePath );
    FILE* fp = fopen ( buf, "w" );
    if (fp == NULL) {
        std::cout << "\nvoid FluidSystem::WriteGenome( const char * relativePath)  Could not open file "<< fp <<"\n"<< std::flush;
        assert(0);
    }
    fprintf(fp, "num genes = %i,\tnum transcription factors = %i \n", NUM_GENES, NUM_TF );
    fprintf(fp, "mutability,\tdelay,\tsensitivity[%i],\tdifusability[2] \n", NUM_GENES );
    for(int i=0; i<NUM_GENES; i++) {
        fprintf(fp, "%i,\t", m_FGenome.mutability[i] );
        fprintf(fp, "%i,\t\t", m_FGenome.delay[i] );
        for(int j=0; j<NUM_GENES; j++) fprintf(fp, "%i,\t", m_FGenome.sensitivity[i][j] );
        for(int j=0; j<NUM_TF; j++) fprintf(fp, "%i,%i,\t", m_FGenome.secrete[i][j*2], m_FGenome.secrete[i][j*2 + 1] );//secretion_rate
        fprintf(fp, "%i,\t\t", m_FGenome.secrete[i][2*NUM_TF] );    //num active elements, NB sparse list, kernel will only read active elems.
        for(int j=0; j<NUM_GENES; j++) fprintf(fp, "%i,%i,\t", m_FGenome.activate[i][j*2], m_FGenome.activate[i][j*2 + 1] );
        fprintf(fp, "%i,\t\t", m_FGenome.activate[i][2*NUM_GENES] );  
        fprintf(fp, " \n" );
    }
    fprintf(fp, "\nTranscription Factors (tf_difusibility,tf_breakdown_rate \n" );
    for(int i=0; i<NUM_TF; i++) {    
        fprintf(fp, "\t%i,\t", m_FGenome.tf_diffusability[i] );
        fprintf(fp, "%i,\t", m_FGenome.tf_breakdown_rate[i] );
    }
    fprintf(fp, "\nRemodelling parameters \n" );
    for(int i=0; i<3; i++){
        for(int j=0; j<12;j++) fprintf(fp, "\t%f,\t", m_FGenome.param[i][j]);
        fprintf(fp, "\n");
    }
    fclose(fp);
}

void FluidSystem::SavePointsVTP2 ( const char * relativePath, int frame ){// uses vtk library to write binary vtp files
    // based on VtpWrite(....)demo at https://vtk.org/Wiki/Write_a_VTP_file  (30 April 2009)
    // and on https://lorensen.github.io/VTKExamples/site/Cxx/IO/WriteVTP/   (post vtk-8.90.9)

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
    BondsFloatData->SetNumberOfComponents(6);
	BondsFloatData->SetName("elastic limit, restlength, modulus, damping coeff, stress integrator");
    
    float *ElastIdxPtr;
    for ( unsigned int i = 0; i < NumPoints(); ++i )
	{
        ElastIdx = getElastIdx(i);                     // FELASTIDX[BONDS_PER_PARTICLE]  [0]current index uint, [5]particle ID uint, [6]bond index uint
        ElastIdxPtr = (float*)ElastIdx;                // FELASTIDX[BONDS_PER_PARTICLE]  [1]elastic limit float, [2]restlength float, [3]modulus float, [4]damping coeff float,
        for(int j=0; j<(BOND_DATA); j+=DATA_PER_BOND) { 
            BondsUIntData->InsertNextTuple3(ElastIdx[j], ElastIdx[j+5], ElastIdx[j+6]);
            BondsFloatData->InsertNextTuple6(ElastIdxPtr[j+1], ElastIdxPtr[j+2], ElastIdxPtr[j+3], ElastIdxPtr[j+4], ElastIdxPtr[j+7], 0);
        }
    }
    //BondsUIntData->SetNumberOfComponents(BONDS_PER_PARTICLE *3);
    //BondsFloatData->SetNumberOfComponents(BONDS_PER_PARTICLE *4); 
    

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
    
    // FCONC float[NUM_TF].                                                                                     // commented out until Matt's edit FCONC uint->foat is merged
    vtkSmartPointer<vtkFloatArray> fconc[NUM_TF];
    char buf_conc[256];
    for (int a=0; a<NUM_GENES; a++){ 
        fconc[a] = vtkSmartPointer<vtkFloatArray>::New();
        fconc[a]->SetNumberOfComponents(1);
        sprintf ( buf_conc, "FCONC_%i",a);
        fconc[a]->SetName(buf_conc);
    }
    float *conc;
    for ( unsigned int i = 0; i < NUM_GENES; ++i ){
        conc = getConc(i);                   
        for(int j=0; j<NumPoints(); j++)    fconc[i]->InsertNextValue(conc[j]);                              // now have one array for each column of fepigen
    }
    
    // FEPIGEN uint[NUM_GENES] ... make an array of arrays
    vtkSmartPointer<vtkUnsignedIntArray> fepigen[NUM_GENES];
    char buf_epigen[256];
    for (int a=0; a<NUM_GENES; a++){ 
        fepigen[a] = vtkSmartPointer<vtkUnsignedIntArray>::New();
        fepigen[a]->SetNumberOfComponents(1);
        sprintf ( buf_epigen, "FEPIGEN_%i",a);
        fepigen[a]->SetName(buf_epigen);
    }
    unsigned int *epigen;
    for ( unsigned int i = 0; i < NUM_GENES; ++i ){
        epigen = getEpiGen(i);                   
        for(int j=0; j<NumPoints(); j++)    fepigen[i]->InsertNextValue(epigen[j]);                              // now have one array for each column of fepigen
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // POLYDATA
	vtkSmartPointer<vtkPolyData> polydata = vtkPolyData::New();                                        // polydata
	polydata->SetPoints(points3D);
	//polydata->SetVerts(Vertices);
    polydata->SetLines(Lines);
    
    
    //cout << "\nStarting writing bond data to polydata\n" << std::flush;
    polydata->GetCellData()->AddArray(BondsUIntData);
    polydata->GetCellData()->AddArray(BondsFloatData);
    //polydata->GetPointData()->AddArray(BondsUIntData);
    //polydata->GetPointData()->AddArray(BondsFloatData);
    polydata->GetPointData()->AddArray(fage);
    polydata->GetPointData()->AddArray(fcolor);
    polydata->GetPointData()->AddArray(fpid);
    polydata->GetPointData()->AddArray(fmass_radius);
    polydata->GetPointData()->AddArray(fnidx);
    
    for(int i=0;i<NUM_TF; i++)      polydata->GetPointData()->AddArray(fconc[i]);
    for(int i=0;i<NUM_GENES; i++)   polydata->GetPointData()->AddArray(fepigen[i]);
    
    //cout << "\nFinished writing bond data to polydata\n" << std::flush;
    
    // WRITER  
	vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();       // writer
    char buf[256];
    frame += 100000;                                                                                              // ensures numerical and alphabetic order match of filenames
    sprintf ( buf, "%s/particles_pos_vel_color%04d.vtp", relativePath, frame );
	writer->SetFileName(buf);
	writer->SetInputData(polydata);
    //writer->SetDataModeToAscii();   
    writer->SetDataModeToAppended();    // prefered, produces a human readable header followed by a binary blob.
    //writer->SetDataModeToBinary();
	writer->Write();
    
	//cout << "\nFinished writing vtp file " << buf << "." << endl;
	//cout << "\tNumPoints: " << NumPoints() << endl;
}

void FluidSystem::SavePointsCSV2 ( const char * relativePath, int frame ){
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
    for (int i=0; i<BONDS_PER_PARTICLE; i++)fprintf(fp, ",[0]curIdx, [1]elastLim, [2]restLn, [3]modulus, [4]damping, [5]partID, [6]bond index, [7]stress integrator,,  ");
    fprintf(fp, "\t"); 
    fprintf(fp, "\tParticle_ID, mass, radius, FNERVEIDX,\t\t Particle_Idx[%u*2]", BONDS_PER_PARTICLE);    
    for (int i=0; i<BONDS_PER_PARTICLE*3; i++)fprintf(fp, ", ");
    fprintf(fp, "\t\tFCONC[%u]", NUM_TF);
    for (int i=0; i<NUM_TF; i++)fprintf(fp, ", ");
    fprintf(fp, "\t\tFEPIGEN[%u] \n", NUM_GENES);
  

    for(int i=0; i<numpnt; i++) {       // nb need get..() accessors for private data.
        Pos = getPos(i);                // e.g.  Vector3DF* getPos ( int n )	{ return &m_Fluid.bufV3(FPOS)[n]; }
        Vel = getVel(i);
        Age = getAge(i);
        Clr = getClr(i);
        ElastIdx = getElastIdx(i);      // NB [BONDS_PER_PARTICLE]
        ElastIdxPtr = (float*)ElastIdx; // #############packing floats and uints into the same array - should replace with a struct.#################
        Particle_Idx = getParticle_Idx(i);
        Particle_ID = getParticle_ID(i);//# uint  original pnum, used for bonds between particles. 32bit, track upto 4Bn particles.
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
        //Conc = getConc(i);              //# float[NUM_TF]        NUM_TF = num transcription factors & morphogens
        //EpiGen = getEpiGen(i);          //# uint[NUM_GENES]  see below.
        
        fprintf(fp, "%f,%f,%f,\t%f,%f,%f,\t %u, %u,, \t", Pos->x, Pos->y,Pos->z, Vel->x,Vel->y,Vel->z, *Age, *Clr );
        
        for(int j=0; j<(BOND_DATA); j+=DATA_PER_BOND) { 
            fprintf(fp, "%u, %f, %f, %f, %f, %u, %u, %f, %u, ", ElastIdx[j], ElastIdxPtr[j+1], ElastIdxPtr[j+2], ElastIdxPtr[j+3], ElastIdxPtr[j+4], ElastIdx[j+5], ElastIdx[j+6], ElastIdxPtr[j+7], ElastIdx[j+8] );
           /*
            // if ((j%DATA_PER_BOND==0)||((j+1)%DATA_PER_BOND==0))  fprintf(fp, "%u, ",  ElastIdx[j] );  // print as int   [0]current index, [5]particle ID, [6]bond index 
           // else  fprintf(fp, "%f, ",  ElastIdxPtr[j] );                                              // print as float [1]elastic limit, [2]restlength, [3]modulus, [4]damping coeff, 
           //  if((j+1)%DATA_PER_BOND==0)  
            */
            fprintf(fp, "\t\t");
        }
        fprintf(fp, " \t%u, %u, %u, %u, \t\t", *Particle_ID, mass, radius, *NerveIdx );
        for(int j=0; j<(BONDS_PER_PARTICLE*2); j+=2)   { fprintf(fp, "%u, %u,, ",  Particle_Idx[j], Particle_Idx[j+1] );}  fprintf(fp, "\t\t"); // NB index of other particle AND other particle's index of the bond
        
        for(int j=0; j<(NUM_TF); j++)               { 
            Conc = getConc(j);
            fprintf(fp, "%f, ",  Conc[i] ); 
        }fprintf(fp, "\t\t");
        
        for(int j=0; j<(NUM_GENES); j++)            { 
            EpiGen = getEpiGen(j);
            fprintf(fp, "%u, ",  EpiGen[i] );   // NB FEPIGEN[gene][particle], for memory efficiency on the device. ? Need to test.
        }fprintf(fp, " \n");
    }
    fclose ( fp );
    fflush ( fp );
}

void FluidSystem::ReadPointsCSV2 ( const char * relativePath, int gpu_mode, int cpu_mode){ // NB allocates buffers as well.
    //std::cout << "\n  ReadPointsCSV2 ( const char * relativePath, int gpu_mode, int cpu_mode);  started \n" << std::flush;
    const char * points_file_path = relativePath;
    printf("\n## opening file %s ", points_file_path);
    FILE * points_file = fopen(points_file_path, "rb");
    if (points_file == NULL) {
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
    if (gpu_mode != GPU_OFF) {     // create CUDA instance etc.. 
        FluidSetupCUDA ( mMaxPoints, m_GridSrch, *(int3*)& m_GridRes, *(float3*)& m_GridSize, *(float3*)& m_GridDelta, *(float3*)& m_GridMin, *(float3*)& m_GridMax, m_GridTotal, 0 );
        UpdateParams();            //  sends simulation params to device.
        UpdateGenome();            //  sends genome to device.              // NB need to initialize genome from file, or something.
    }
    AllocateParticles ( mMaxPoints, gpu_mode, cpu_mode );  // allocates only cpu buffer for particles
    AllocateGrid(gpu_mode, cpu_mode);
    //////////////////////////////////////// 
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
    uint bond_data=999, data_per_bond=999, bonds_per_particle=999, num_TF=999, num_genes=999;
    int result=-2;
    result = std::fscanf(points_file, "x coord, y coord, z coord\t\t x vel, y vel, z vel\t\t age,  color\t\t FELASTIDX[%u*%u]", &bond_data, &data_per_bond);
    
    for (int i=0; i<data_per_bond; i++)std::fscanf(points_file, ",[0]curIdx, [1]elastLim, [2]restLn, [3]modulus, [4]damping, [5]partID, [6]bond index, [7]stress integrator,,  ");
    bond_data = bond_data * data_per_bond;
    fscanf(points_file, "\t");
    
    result = std::fscanf(points_file, "\tParticle_ID, mass, radius, FNERVEIDX,\t\t Particle_Idx[%u*2]", &bonds_per_particle);
    for (int i=0; i<BONDS_PER_PARTICLE*3; i++)fscanf(points_file, ", ");
    
    result = std::fscanf(points_file, "\t\tFCONC[%u]",&num_TF);
    for (int i=0; i<NUM_TF; i++)fscanf(points_file, ", ");
    
    result = std::fscanf(points_file, "\t\tFEPIGEN[%u] \n", &num_genes );
std::cout<<"\n\n ReadPointsCSV2() starting loop: number_of_lines="<<number_of_lines<<"\n"<<std::flush;
    ////////////////////
    int i;
    for (i=1; i<number_of_lines; i++ ) {
        // transcribe particle data from file to Pos, Vel and Clr
        int ret = std::fscanf(points_file, "%f,%f,%f,\t%f,%f,%f,\t %u, %u,, \t", &Pos.x, &Pos.y, &Pos.z, &Vel.x, &Vel.y, &Vel.z, &Age, &Clr );
std::cout<<"\n ReadPointsCSV2() row="<< i <<", (line 1259, ret="<<ret<<"),\t"<<std::flush;
        for(int j=0; j<(BOND_DATA); j++) {// BONDS_PER_PARTICLE * DATA_PER_BOND
            ret += std::fscanf(points_file, "%f, ",  &ElastIdx[j] );
        }
std::cout<<"(line 1263, ret="<<ret<<")\t"<<std::flush;
        ret += std::fscanf(points_file, " \t%u, %u, %u, %u, \t\t", &Particle_ID, &mass, &radius, &NerveIdx);
        Mass_Radius = mass + (radius << 16);                                    // pack two 16bit uint  into one 32bit uint.
std::cout<<"(ReadPointsCSV2() line 1266, ret="<<ret<<"),\t"<<std::flush;
        for(int j=0; j<(BONDS_PER_PARTICLE*2); j+=2) {
            ret += std::fscanf(points_file, "%u, %u,, ",  &Particle_Idx[j], &Particle_Idx[j+1] );
        }
std::cout<<"(ReadPointsCSV2() line 1270, ret="<<ret<<"),\t"<<std::flush;        
        for(int j=0; j<(NUM_TF); j++)       {    ret += std::fscanf(points_file, "%f, ",  &Conc[j] );   } ret += std::fscanf(points_file, "\t");
std::cout<<"(ReadPointsCSV2() line 1272, ret="<<ret<<"),\t"<<std::flush;
        for(int j=0; j<(NUM_GENES); j++)    {    ret += std::fscanf(points_file, "%u, ",  &EpiGen[j] ); } ret += std::fscanf(points_file, " \n");
std::cout<<"(ReadPointsCSV2() line 1274, ret="<<ret<<"),\t"<<std::flush;
        if (ret != (8 + BOND_DATA + 4 + BONDS_PER_PARTICLE*2 + NUM_TF + NUM_GENES) ) {  
            std::cout<<"\n ReadPointsCSV2() fail line 1276, ret="<<ret<<"\n"<<std::flush;
            fclose(points_file);
            return;
        } // ret=8 ret=32 ret=36 ret=48 ret=64 ret=80 

        // check particle is within simulation bounds
        if (Pos.x < PosMin.x || Pos.y < PosMin.y || Pos.z < PosMin.z
                || Pos.x > PosMax.x   || Pos.y > PosMax.y || Pos.z > PosMax.z
                || (Vel.x * Vel.x + Vel.y * Vel.y + Vel.z * Vel.z) > vel_lim * vel_lim )
        {
            //std::cout << "\n void FluidSystem::ReadPointsCSV, out of bounds !  particle number = " << i;
            //std::cout << "\n Pos.x = " << Pos.x << "  Pos.y = " << Pos.y << "  Pos.z = " << Pos.z;
            //std::cout << "\n PosMax.x = " << PosMax.x << "  PosMax.y = " << PosMax.y << "  PosMax.z = " << PosMax.z;
            //std::cout << "\n PosMin.x = " << PosMin.x << "  PosMin.y = " << PosMin.y << "  PosMin.z = " << PosMin.z;
            //std::cout << "\n velocity = " << sqrt(Vel.x * Vel.x + Vel.y * Vel.y + Vel.z * Vel.z) << "   vel_lim = " << vel_lim;
            //std::cout << "\n " << std::flush;
            fclose(points_file);
            return;
        }
        AddParticleMorphogenesis2 (&Pos, &Vel, Age, Clr, ElastIdx, Particle_Idx, Particle_ID, Mass_Radius,  NerveIdx, Conc, EpiGen );
    }
    std::cout<<"\n ReadPointsCSV2() finished reading points. i="<<i<<"\n"<<std::flush;
    fclose(points_file);
    AddNullPoints ();                                   // add null particles up to mMaxPoints
    if (gpu_mode != GPU_OFF) TransferToCUDA ();         // Initial transfer
    std::cout<<"\n ReadPointsCSV2() finished extra functions.\n"<<std::flush;
}

void FluidSystem::ReadSimParams ( const char * relativePath ) { // transcribe SimParams from file to fluid_system object.
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

void FluidSystem::WriteSimParams ( const char * relativePath ){
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

void FluidSystem::WriteDemoSimParams ( const char * relativePath, uint num_particles, float spacing, float x_dim, float y_dim, float z_dim ){
    memset ( &m_Fluid, 0,		sizeof(FBufs) );
    memset ( &m_FluidTemp, 0,	sizeof(FBufs) );
    memset ( &m_FParams, 0,		sizeof(FParams) );
    memset ( &m_FGenome, 0,		sizeof(FGenome) );
    m_Param[PEXAMPLE] = 2;          // wave pool example.
    m_Param[PGRID_DENSITY] = 2.0;
    m_Param[PNUM] = num_particles;  // 1000000;    //1000 = minimal simulation, 1000000 = large simulation
    AllocateBuffer ( FPARAMS, sizeof(FParams), 1,0, GPU_OFF, CPU_YES ); 
    m_Time = 0;
    mNumPoints = 0;			        // reset count
    SetupDefaultParams();           // set up the standard demo
    SetupExampleParams();
    SetupExampleGenome();
    mMaxPoints = m_Param[PNUM];    
    m_Param[PGRIDSIZE] = 2*m_Param[PSMOOTHRADIUS] / m_Param[PGRID_DENSITY];
    SetupKernels();
    SetupSpacing();
    SetupGrid ( m_Vec[PVOLMIN], m_Vec[PVOLMAX], m_Param[PSIMSCALE], m_Param[PGRIDSIZE], 1.0f );  std::cout << " chk1.0 " << std::flush ;
    AllocateParticles ( mMaxPoints, GPU_OFF, CPU_YES );                                          std::cout << " chk1.1 " << std::flush ;
    AllocateGrid(GPU_OFF, CPU_YES);                                                              std::cout << " chk1.2 " << std::flush ;
    Vector3DF pinit_max = {x_dim,y_dim,z_dim};
    pinit_max += m_Vec[PINITMIN];
    SetupAddVolumeMorphogenesis2(m_Vec[PINITMIN], pinit_max, spacing, 0.1f, (int)num_particles); std::cout << " chk1.3 " << std::flush ;
    WriteSimParams ( relativePath );    std::cout << "\n WriteSimParams ( relativePath );  completed \n" << std::flush ;  // write data to file
    WriteGenome ( relativePath);        std::cout << "\n WriteGenome ( relativePath );  completed \n" << std::flush ;
    SavePointsCSV2 ( relativePath, 1 ); std::cout << "\n SavePointsCSV ( relativePath, 1 );  completed \n" << std::flush ;
}

/////////////////////////////////////////////////////////////
void FluidSystem::SetupKernels (){
    m_Param [ PDIST ] = pow ( (float) m_Param[PMASS] / m_Param[PRESTDENSITY], 1.0f/3.0f );
    m_R2 = m_Param [PSMOOTHRADIUS] * m_Param[PSMOOTHRADIUS];
    m_Poly6Kern = 315.0f / (64.0f * 3.141592f * pow( m_Param[PSMOOTHRADIUS], 9.0f) );	// Wpoly6 kernel (denominator part) - 2003 Muller, p.4
    m_SpikyKern = -45.0f / (3.141592f * pow( m_Param[PSMOOTHRADIUS], 6.0f) );			// Laplacian of viscocity (denominator): PI h^6
    m_LapKern = 45.0f / (3.141592f * pow( m_Param[PSMOOTHRADIUS], 6.0f) );
}

void FluidSystem::SetupDefaultParams (){
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

void FluidSystem::SetupExampleParams (){
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

void FluidSystem::SetupExampleGenome()  {   // need to set up a demo genome
    // Null genome
    for(int i=0; i< NUM_GENES; i++) m_FGenome.mutability[i] = 0;
    for(int i=0; i< NUM_GENES; i++) m_FGenome.delay[i] = 1;
    for(int i=0; i< NUM_GENES; i++) for(int j=0; j< NUM_GENES; j++) m_FGenome.sensitivity[i][j] = j;
        
    for(int i=0; i< NUM_TF/2; i++)      m_FGenome.tf_diffusability[i]    = 0;           // 1st half of TFs are non-diffusible.
    for(int i=NUM_TF/2; i< NUM_TF; i++) m_FGenome.tf_diffusability[i]    = 1;
    for(int i=0; i< NUM_TF; i++)        m_FGenome.tf_breakdown_rate[i]  = 1;
    
    for(int i=0; i< NUM_GENES; i++) for(int j=0; j< 2*NUM_TF+1; j++) m_FGenome.secrete[i][j]=0;     // 1st zero arrays.
    for(int i=0; i< NUM_GENES; i++) for(int j=0; j< 2*NUM_TF+1; j++) m_FGenome.activate[i][j]=0;
    
    m_FGenome.secrete[0][2*NUM_TF] = 2; // gene [0] secretes TF 1 & 3, at rates 1 & 4. // minimal test case.
    m_FGenome.secrete[0][2*0] = 1;
    m_FGenome.secrete[0][2*0+1] = 1;
    m_FGenome.secrete[0][2*1] = 3;
    m_FGenome.secrete[0][2*1+1] = 4;
    
    m_FGenome.activate[0][2*NUM_TF] = 1; // gene [0] activates TF 5, at rates 6. // minimal test case.
    m_FGenome.activate[0][2*0] = 5;
    m_FGenome.activate[0][2*0+1] = 6;
    
    //FBondParams *params_ =  &m_FGenome.fbondparams[0];
    //0=elastin
    m_FGenome.param[0][m_FGenome.elongation_threshold]   = 0.1  ;
    m_FGenome.param[0][m_FGenome.elongation_factor]      = 0.1  ;
    m_FGenome.param[0][m_FGenome.strength_threshold]     = 0.1  ;
    m_FGenome.param[0][m_FGenome.strengthening_factor]   = 0.1  ;
    
    m_FGenome.param[0][m_FGenome.max_rest_length]        = 0.8  ;
    m_FGenome.param[0][m_FGenome.min_rest_length]        = 0.3  ;
    m_FGenome.param[0][m_FGenome.max_modulus]            = 0.8  ;
    m_FGenome.param[0][m_FGenome.min_modulus]            = 0.3  ;
    
    m_FGenome.param[0][m_FGenome.elastLim]               = 2  ;
    m_FGenome.param[0][m_FGenome.default_rest_length]    = 0.5  ;
    m_FGenome.param[0][m_FGenome.default_modulus]        = 100000  ;
    m_FGenome.param[0][m_FGenome.default_damping]        = 10  ;
    
    //1=collagen
    m_FGenome.param[1][m_FGenome.elongation_threshold]   = 0.1  ;
    m_FGenome.param[1][m_FGenome.elongation_factor]      = 0.1  ;
    m_FGenome.param[1][m_FGenome.strength_threshold]     = 0.1  ;
    m_FGenome.param[1][m_FGenome.strengthening_factor]   = 0.1  ;
    
    m_FGenome.param[1][m_FGenome.max_rest_length]        = 0.8  ;
    m_FGenome.param[1][m_FGenome.min_rest_length]        = 0.3  ;
    m_FGenome.param[1][m_FGenome.max_modulus]            = 0.8  ;
    m_FGenome.param[1][m_FGenome.min_modulus]            = 0.3  ;
    
    m_FGenome.param[1][m_FGenome.elastLim]               = 0.55  ;
    m_FGenome.param[1][m_FGenome.default_rest_length]    = 0.5  ;
    m_FGenome.param[1][m_FGenome.default_modulus]        = 10000000  ;
    m_FGenome.param[1][m_FGenome.default_damping]        = 100  ;
    
    //2=apatite
    m_FGenome.param[2][m_FGenome.elongation_threshold]   = 0.1  ;
    m_FGenome.param[2][m_FGenome.elongation_factor]      = 0.1  ;
    m_FGenome.param[2][m_FGenome.strength_threshold]     = 0.1  ;
    m_FGenome.param[2][m_FGenome.strengthening_factor]   = 0.1  ;
    
    m_FGenome.param[2][m_FGenome.max_rest_length]        = 0.8  ;
    m_FGenome.param[2][m_FGenome.min_rest_length]        = 0.3  ;
    m_FGenome.param[2][m_FGenome.max_modulus]            = 0.8  ;
    m_FGenome.param[2][m_FGenome.min_modulus]            = 0.3  ;
    
    m_FGenome.param[2][m_FGenome.elastLim]               = 0.05  ;
    m_FGenome.param[2][m_FGenome.default_rest_length]    = 0.5  ;
    m_FGenome.param[2][m_FGenome.default_modulus]        = 10000000  ;
    m_FGenome.param[2][m_FGenome.default_damping]        = 1000  ;
}

//////////////////////////////////////////////////////
void FluidSystem::SetupSpacing (){
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

void computeNumBlocks (int numPnts, int minThreads, int &numBlocks, int &numThreads){
    numThreads = min( minThreads, numPnts );
    numBlocks = (numThreads==0) ? 1 : iDivUp ( numPnts, numThreads );
}

void FluidSystem::TransferToTempCUDA ( int buf_id, int sz ){
    cuCheck ( cuMemcpyDtoD ( m_FluidTemp.gpu(buf_id), m_Fluid.gpu(buf_id), sz ), "TransferToTempCUDA", "cuMemcpyDtoD", "m_FluidTemp", mbDebug);
}

void FluidSystem::FluidSetupCUDA ( int num, int gsrch, int3 res, float3 size, float3 delta, float3 gmin, float3 gmax, int total, int chk ){
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
    m_FParams.threadsPerBlock = 512;                    //TODO probe hardware to set m_FParams.threadsPerBlock

    computeNumBlocks ( m_FParams.pnum, m_FParams.threadsPerBlock, m_FParams.numBlocks, m_FParams.numThreads);				// particles
    computeNumBlocks ( m_FParams.gridTotal, m_FParams.threadsPerBlock, m_FParams.gridBlocks, m_FParams.gridThreads);		// grid cell

    // Compute particle buffer & grid dimensions
    m_FParams.szPnts = (m_FParams.numBlocks  * m_FParams.numThreads);
}

void FluidSystem::FluidParamCUDA ( float ss, float sr, float pr, float mass, float rest, float3 bmin, float3 bmax, float estiff, float istiff, float visc, float damp, float fmin, float fmax, float ffreq, float gslope, float gx, float gy, float gz, float al, float vl, int emit ){
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

void FluidSystem::TransferToCUDA (){
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

void FluidSystem::TransferFromCUDA (){
//std::cout<<"\nTransferFromCUDA () \n"<<std::flush;    
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

void FluidSystem::InsertParticlesCUDA ( uint* gcell, uint* gndx, uint* gcnt ){   // first zero the counters
    cuCheck ( cuMemsetD8 ( m_Fluid.gpu(FGRIDCNT), 0,	m_GridTotal*sizeof(int) ), "InsertParticlesCUDA", "cuMemsetD8", "FGRIDCNT", mbDebug );
    cuCheck ( cuMemsetD8 ( m_Fluid.gpu(FGRIDOFF), 0,	m_GridTotal*sizeof(int) ), "InsertParticlesCUDA", "cuMemsetD8", "FGRIDOFF", mbDebug );
    
    cuCheck ( cuMemsetD8 ( m_Fluid.gpu(FGRIDCNT_ACTIVE_GENES), 0,	m_GridTotal *sizeof(uint[NUM_GENES]) ), "InsertParticlesCUDA", "cuMemsetD8", "FGRIDCNT", mbDebug );
    cuCheck ( cuMemsetD8 ( m_Fluid.gpu(FGRIDOFF_ACTIVE_GENES), 0,	m_GridTotal *sizeof(uint[NUM_GENES]) ), "InsertParticlesCUDA", "cuMemsetD8", "FGRIDOFF", mbDebug );
    
    // launch kernel "InsertParticles"
    void* args[1] = { &mNumPoints };
    cuCheck(cuLaunchKernel(m_Func[FUNC_INSERT], m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL),
            "InsertParticlesCUDA", "cuLaunch", "FUNC_INSERT", mbDebug);

    // Transfer data back if requested (for validation)
    if (gcell != 0x0) {
        cuCheck( cuMemcpyDtoH ( gcell,	m_Fluid.gpu(FGCELL),	mNumPoints *sizeof(uint) ), "InsertParticlesCUDA", "cuMemcpyDtoH", "FGCELL", mbDebug );
        cuCheck( cuMemcpyDtoH ( gndx,	m_Fluid.gpu(FGNDX),		mNumPoints *sizeof(uint) ), "InsertParticlesCUDA", "cuMemcpyDtoH", "FGNDX", mbDebug);
        cuCheck( cuMemcpyDtoH ( gcnt,	m_Fluid.gpu(FGRIDCNT),	m_GridTotal*sizeof(uint) ), "InsertParticlesCUDA", "cuMemcpyDtoH", "FGRIDCNT", mbDebug);
        cuCtxSynchronize ();
    }
}
/*
void FluidSystem::InsertChangesCUDA ( /_*uint* gcell, uint* gndx, uint* gcnt*_/ ){ // NB This is now done by ComputeBondChangesCUDA(..) 
    cuCheck ( cuMemsetD8 ( m_Fluid.gpu(FGRIDCNT_CHANGES), 0,	m_GridTotal *sizeof(uint[NUM_CHANGES]) ), "InsertParticlesCUDA", "cuMemsetD8", "FGRIDCNT", mbDebug );
    cuCheck ( cuMemsetD8 ( m_Fluid.gpu(FGRIDOFF_CHANGES), 0,	m_GridTotal *sizeof(uint[NUM_CHANGES]) ), "InsertParticlesCUDA", "cuMemsetD8", "FGRIDOFF", mbDebug );
    
    // launch kernel "InsertParticles"
    void* args[1] = { &mNumPoints };
    cuCheck(cuLaunchKernel(m_Func[FUNC_INSERT_CHANGES], m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL),   
            "InsertParticlesCUDA", "cuLaunch", "FUNC_INSERT", mbDebug);
}
*/
void FluidSystem::PrefixSumCellsCUDA ( int zero_offsets ){
/*
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
*/
    // Prefix Sum - determine grid offsets
    int blockSize = SCAN_BLOCKSIZE << 1;                // NB 1024 = 512 << 1.  NB SCAN_BLOCKSIZE is the number of threads per block
    int numElem1 = m_GridTotal;                         // tot num bins, computed in SetupGrid() 
    int numElem2 = int ( numElem1 / blockSize ) + 1;    // num sheets of bins? NB not spatial, but just dividing the linear array of bins, by a factor of 512*2
    int numElem3 = int ( numElem2 / blockSize ) + 1;    // num rows of bins?
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
    if ( numElem1 > SCAN_BLOCKSIZE*xlong(SCAN_BLOCKSIZE)*SCAN_BLOCKSIZE) { printf ( "\nERROR: Number of elements exceeds prefix sum max. Adjust SCAN_BLOCKSIZE.\n" );  }

    void* argsA[5] = {&array1, &scan1, &array2, &numElem1, &zero_offsets };     // sum array1. output -> scan1, array2.         i.e. FGRIDCNT -> FGRIDOFF, FAUXARRAY1
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXSUM], numElem2, 1, 1, threads, 1, 1, 0, NULL, argsA, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXSUM", mbDebug);

    void* argsB[5] = { &array2, &scan2, &array3, &numElem2, &zon };             // sum array2. output -> scan2, array3.         i.e. FAUXARRAY1 -> FAUXSCAN1, FAUXARRAY2
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXSUM], numElem3, 1, 1, threads, 1, 1, 0, NULL, argsB, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXSUM", mbDebug);

    if ( numElem3 > 1 ) {
        CUdeviceptr nptr = {0};
        void* argsC[5] = { &array3, &scan3, &nptr, &numElem3, &zon };	        // sum array3. output -> scan3                  i.e. FAUXARRAY2 -> FAUXSCAN2, &nptr
        cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXSUM], 1, 1, 1, threads, 1, 1, 0, NULL, argsC, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);

        void* argsD[3] = { &scan2, &scan3, &numElem2 };	                        // merge scan3 into scan2. output -> scan2      i.e. FAUXSCAN2, FAUXSCAN1 -> FAUXSCAN1
        cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXFIXUP], numElem3, 1, 1, threads, 1, 1, 0, NULL, argsD, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);
    }
    void* argsE[3] = { &scan1, &scan2, &numElem1 };		                        // merge scan2 into scan1. output -> scan1      i.e. FAUXSCAN1, FGRIDOFF -> FGRIDOFF
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXFIXUP], numElem2, 1, 1, threads, 1, 1, 0, NULL, argsE, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);
    
    cuCheck( cuMemcpyDtoH ( &mNumPoints,  m_Fluid.gpu(FGRIDOFF)+(m_GridTotal-1)*sizeof(int), sizeof(int) ), "PrefixSumCellsCUDA", "cuMemcpyDtoH", "FGRIDOFF", mbDebug);
    cuCtxSynchronize ();
    std::cout<<"\nPrefixSumCellsCUDA(): mNumPoints="<<mNumPoints<<"\n"<<std::flush;
    

    // Loop to PrefixSum the Dense Lists - NB by doing one gene at a time, we reuse the FAUX* arrays & scans.
    // For each gene, input FGRIDCNT_ACTIVE_GENES[gene*m_GridTotal], output FGRIDOFF_ACTIVE_GENES[gene*m_GridTotal]
    CUdeviceptr array0  = m_Fluid.gpu(FGRIDCNT_ACTIVE_GENES);
    CUdeviceptr scan0   = m_Fluid.gpu(FGRIDOFF_ACTIVE_GENES);

    for(int gene=0;gene<NUM_GENES;gene++){
        array1  = array0 + gene*numElem1*sizeof(int); //m_Fluid.gpu(FGRIDCNT_ACTIVE_GENES);//[gene*numElem1]   ;///
        scan1   = scan0 + gene*numElem1*sizeof(int);

        //cuCheck ( cuMemsetD8 ( array1, 0,	numElem1*sizeof(int) ), "PrefixSumCellsCUDA", "cuMemsetD8", "FGRIDCNT", mbDebug );
        cuCheck ( cuMemsetD8 ( scan1,  0,	numElem1*sizeof(int) ), "PrefixSumCellsCUDA", "cuMemsetD8", "FGRIDCNT", mbDebug );
        
        cuCheck ( cuMemsetD8 ( array2, 0,	numElem2*sizeof(int) ), "PrefixSumCellsCUDA", "cuMemsetD8", "FGRIDCNT", mbDebug );
        cuCheck ( cuMemsetD8 ( scan2,  0,	numElem2*sizeof(int) ), "PrefixSumCellsCUDA", "cuMemsetD8", "FGRIDCNT", mbDebug );
        
        cuCheck ( cuMemsetD8 ( array3, 0,	numElem3*sizeof(int) ), "PrefixSumCellsCUDA", "cuMemsetD8", "FGRIDCNT", mbDebug );
        cuCheck ( cuMemsetD8 ( scan3,  0,	numElem3*sizeof(int) ), "PrefixSumCellsCUDA", "cuMemsetD8", "FGRIDCNT", mbDebug );
        
        void* argsA[5] = {&array1, &scan1, &array2, &numElem1, &zero_offsets };     // sum array1. output -> scan1, array2.         i.e. FGRIDCNT -> FGRIDOFF, FAUXARRAY1
        cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXSUM], numElem2, 1, 1, threads, 1, 1, 0, NULL, argsA, NULL ), 
                  "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXSUM", mbDebug);

        void* argsB[5] = { &array2, &scan2, &array3, &numElem2, &zon };             // sum array2. output -> scan2, array3.         i.e. FAUXARRAY1 -> FAUXSCAN1, FAUXARRAY2
        cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXSUM], numElem3, 1, 1, threads, 1, 1, 0, NULL, argsB, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXSUM", mbDebug);

        if ( numElem3 > 1 ) {
            CUdeviceptr nptr = {0};
            void* argsC[5] = { &array3, &scan3, &nptr, &numElem3, &zon };	        // sum array3. output -> scan3                  i.e. FAUXARRAY2 -> FAUXSCAN2, &nptr
            cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXSUM], 1, 1, 1, threads, 1, 1, 0, NULL, argsC, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);

            void* argsD[3] = { &scan2, &scan3, &numElem2 };	                        // merge scan3 into scan2. output -> scan2      i.e. FAUXSCAN2, FAUXSCAN1 -> FAUXSCAN1
            cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXFIXUP], numElem3, 1, 1, threads, 1, 1, 0, NULL, argsD, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);
        }

        void* argsE[3] = { &scan1, &scan2, &numElem1 };		                        // merge scan2 into scan1. output -> scan1      i.e. FAUXSCAN1, FGRIDOFF -> FGRIDOFF
        cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXFIXUP], numElem2, 1, 1, threads, 1, 1, 0, NULL, argsE, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);
    }
    int num_lists = NUM_GENES, length = FDENSE_LIST_LENGTHS, fgridcnt = FGRIDCNT_ACTIVE_GENES, fgridoff = FGRIDOFF_ACTIVE_GENES;
    void* argsF[4] = {&num_lists, &length,&fgridcnt,&fgridoff};
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_TALLYLISTS], NUM_GENES, 1, 1, NUM_GENES, 1, 1, 0, NULL, argsF, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug); //256 threads launched
    cuCheck( cuMemcpyDtoH ( m_Fluid.bufI(FDENSE_LIST_LENGTHS), m_Fluid.gpu(FDENSE_LIST_LENGTHS),	sizeof(uint[NUM_GENES]) ), "PrefixSumCellsCUDA", "cuMemcpyDtoH", "FDENSE_LIST_LENGTHS", mbDebug);
                                                                                    //if active particles for gene > existing buff, then enlarge buff.
    for(int gene=0;gene<NUM_GENES;gene++){                                          // Note this calculation could be done by a kernel, 
        uint * densebuff_len = m_Fluid.bufI(FDENSE_BUF_LENGTHS);                    // and only m_Fluid.bufI(FDENSE_LIST_LENGTHS); copied to host.
        uint * denselist_len = m_Fluid.bufI(FDENSE_LIST_LENGTHS);                   // For each gene allocate intial buffer, 
        if (denselist_len[gene] > densebuff_len[gene]) {                            // write pointer and size to FDENSE_LISTS and FDENSE_LIST_LENGTHS 
            while(denselist_len[gene] >  densebuff_len[gene]) densebuff_len[gene] *=4;                  // m_Fluid.bufI(FDENSE_BUF_LENGTHS)[i]
            AllocateBufferDenseLists( gene, sizeof(uint), m_Fluid.gpuptr(FDENSE_LIST_LENGTHS)[gene], FDENSE_LISTS );   // NB frees previous buffer &=> clears data
        }
    }
    cuMemcpyHtoD(m_Fluid.gpu(FDENSE_LISTS), m_Fluid.bufC(FDENSE_LISTS),  NUM_GENES * sizeof(CUdeviceptr)  );  // update pointers to lists on device

std::cout << "\nChk: PrefixSumCellsCUDA 4"<<std::flush;
for(int gene=0;gene<NUM_GENES;gene++){    std::cout<<"\nlist_length["<<gene<<"]="<<m_Fluid.bufI(FDENSE_LIST_LENGTHS)[gene]<<"\t"<<std::flush;}
//#endif
}

void FluidSystem::PrefixSumChangesCUDA ( int zero_offsets ){
    // Prefix Sum - determine grid offsets
    int blockSize = SCAN_BLOCKSIZE << 1;                // NB 1024 = 512 << 1.  NB SCAN_BLOCKSIZE is the number of threads per block
    int numElem1 = m_GridTotal;                         // tot num bins, computed in SetupGrid() 
    int numElem2 = int ( numElem1 / blockSize ) + 1;    // num sheets of bins? NB not spatial, but just dividing the linear array of bins, by a factor of 512*2
    int numElem3 = int ( numElem2 / blockSize ) + 1;    // num rows of bins?
    int threads = SCAN_BLOCKSIZE;
    int zon=1;
    CUdeviceptr array1  ;		// input
    CUdeviceptr scan1   ;		// output
    CUdeviceptr array2  = m_Fluid.gpu(FAUXARRAY1);
    CUdeviceptr scan2   = m_Fluid.gpu(FAUXSCAN1);
    CUdeviceptr array3  = m_Fluid.gpu(FAUXARRAY2);
    CUdeviceptr scan3   = m_Fluid.gpu(FAUXSCAN2);
#ifndef xlong
    typedef unsigned long long	xlong;		// 64-bit integer
#endif
    // Loop to PrefixSum the Dense Lists - NB by doing one change_list at a time, we reuse the FAUX* arrays & scans.
    // For each change_list, input FGRIDCNT_ACTIVE_GENES[change_list*m_GridTotal], output FGRIDOFF_ACTIVE_GENES[change_list*m_GridTotal]
    CUdeviceptr array0  = m_Fluid.gpu(FGRIDCNT_CHANGES);
    CUdeviceptr scan0   = m_Fluid.gpu(FGRIDOFF_CHANGES);

    for(int change_list=0;change_list<NUM_CHANGES;change_list++){
        array1  = array0 + change_list*numElem1*sizeof(int); //m_Fluid.gpu(FGRIDCNT_ACTIVE_GENES);//[change_list*numElem1]   ;///
        scan1   = scan0 + change_list*numElem1*sizeof(int);
        cuCheck ( cuMemsetD8 ( scan1,  0,	numElem1*sizeof(int) ), "PrefixSumChangesCUDA", "cuMemsetD8", "FGRIDCNT", mbDebug );        
        cuCheck ( cuMemsetD8 ( array2, 0,	numElem2*sizeof(int) ), "PrefixSumChangesCUDA", "cuMemsetD8", "FGRIDCNT", mbDebug );
        cuCheck ( cuMemsetD8 ( scan2,  0,	numElem2*sizeof(int) ), "PrefixSumChangesCUDA", "cuMemsetD8", "FGRIDCNT", mbDebug );
        cuCheck ( cuMemsetD8 ( array3, 0,	numElem3*sizeof(int) ), "PrefixSumChangesCUDA", "cuMemsetD8", "FGRIDCNT", mbDebug );
        cuCheck ( cuMemsetD8 ( scan3,  0,	numElem3*sizeof(int) ), "PrefixSumChangesCUDA", "cuMemsetD8", "FGRIDCNT", mbDebug );
        
        void* argsA[5] = {&array1, &scan1, &array2, &numElem1, &zero_offsets };     // sum array1. output -> scan1, array2.         i.e. FGRIDCNT -> FGRIDOFF, FAUXARRAY1
        cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXSUM], numElem2, 1, 1, threads, 1, 1, 0, NULL, argsA, NULL ), 
                  "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXSUM", mbDebug);
        void* argsB[5] = { &array2, &scan2, &array3, &numElem2, &zon };             // sum array2. output -> scan2, array3.         i.e. FAUXARRAY1 -> FAUXSCAN1, FAUXARRAY2
        cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXSUM], numElem3, 1, 1, threads, 1, 1, 0, NULL, argsB, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXSUM", mbDebug);
        if ( numElem3 > 1 ) {
            CUdeviceptr nptr = {0};
            void* argsC[5] = { &array3, &scan3, &nptr, &numElem3, &zon };	        // sum array3. output -> scan3                  i.e. FAUXARRAY2 -> FAUXSCAN2, &nptr
            cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXSUM], 1, 1, 1, threads, 1, 1, 0, NULL, argsC, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);
            void* argsD[3] = { &scan2, &scan3, &numElem2 };	                        // merge scan3 into scan2. output -> scan2      i.e. FAUXSCAN2, FAUXSCAN1 -> FAUXSCAN1
            cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXFIXUP], numElem3, 1, 1, threads, 1, 1, 0, NULL, argsD, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);
        }
        void* argsE[3] = { &scan1, &scan2, &numElem1 };		                        // merge scan2 into scan1. output -> scan1      i.e. FAUXSCAN1, FGRIDOFF -> FGRIDOFF
        cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXFIXUP], numElem2, 1, 1, threads, 1, 1, 0, NULL, argsE, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);
    }
    int num_lists = NUM_CHANGES, length = FDENSE_LIST_LENGTHS_CHANGES, fgridcnt = FGRIDCNT_CHANGES, fgridoff = FGRIDOFF_CHANGES;
    void* argsF[4] = {&num_lists, &length,&fgridcnt,&fgridoff};
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_TALLYLISTS], NUM_CHANGES, 1, 1, NUM_CHANGES, 1, 1, 0, NULL, argsF, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug); //256 threads launched
    cuCheck( cuMemcpyDtoH ( m_Fluid.bufI(FDENSE_LIST_LENGTHS_CHANGES), m_Fluid.gpu(FDENSE_LIST_LENGTHS_CHANGES),	sizeof(uint[NUM_CHANGES]) ), "PrefixSumCellsCUDA", "cuMemcpyDtoH", "FDENSE_LIST_LENGTHS_CHANGES", mbDebug);
                                                                                                                    // If active particles for change_list > existing buff, then enlarge buff.
    for(int change_list=0;change_list<NUM_CHANGES;change_list++){                                                   // Note this calculation could be done by a kernel, 
        uint * densebuff_len = m_Fluid.bufI(FDENSE_BUF_LENGTHS_CHANGES);                                            // and only m_Fluid.bufI(FDENSE_LIST_LENGTHS); copied to host.
        uint * denselist_len = m_Fluid.bufI(FDENSE_LIST_LENGTHS_CHANGES);                                           // For each change_list allocate intial buffer, 
        if (denselist_len[change_list] > densebuff_len[change_list]) {                                              // write pointer and size to FDENSE_LISTS and FDENSE_LIST_LENGTHS 
            while(denselist_len[change_list] >  densebuff_len[change_list]) densebuff_len[change_list] *=4;         // m_Fluid.bufI(FDENSE_BUF_LENGTHS)[i]
            AllocateBufferDenseLists( change_list, 2*sizeof(uint), m_Fluid.gpuptr(FDENSE_LIST_LENGTHS_CHANGES)[change_list], FDENSE_LISTS_CHANGES );// NB frees previous buffer &=> clears data
        }                                                                                                           // NB buf[2][list_length] holding : particleIdx, bondIdx
    }
    cuMemcpyHtoD(m_Fluid.gpu(FDENSE_LISTS_CHANGES), m_Fluid.bufC(FDENSE_LISTS_CHANGES),  NUM_CHANGES * sizeof(CUdeviceptr)  );                      // update pointers to lists on device
    std::cout << "\nChk: PrefixSumChangesCUDA 4"<<std::flush;
    for(int change_list=0;change_list<NUM_CHANGES;change_list++){    std::cout<<"\nlist_length["<<change_list<<"]="<<m_Fluid.bufI(FDENSE_LIST_LENGTHS_CHANGES)[change_list]<<"\t"<<std::flush;}
}

void FluidSystem::CountingSortFullCUDA ( Vector3DF* ppos ){

    m_FParams.pnumActive = mNumPoints;
    cuCheck ( cuMemcpyHtoD ( cuFParams,	&m_FParams,		sizeof(FParams) ), "FluidParamCUDA", "cuMemcpyHtoD", "cuFParams", mbDebug); // seems the safest way to update fparam.pnumActive on device.
    
    std::cout << "\nCountingSortFullCUDA :  mNumPoints="<<mNumPoints<<",\tmAvailablePoints="<<mAvailablePoints<<".\n"<<std::flush;

    // Transfer particle data to temp buffers
    //  (gpu-to-gpu copy, no sync needed)
    TransferToTempCUDA ( FPOS,		mMaxPoints *sizeof(Vector3DF) );    // NB if some points have been removed, then the existing list is no longer dense,  
    TransferToTempCUDA ( FVEL,		mMaxPoints *sizeof(Vector3DF) );    // hence must use mMaxPoints, not mNumPoints
    TransferToTempCUDA ( FVEVAL,	mMaxPoints *sizeof(Vector3DF) );    // { Could potentially use (old_mNumPoints + mNewPoints) instead of mMaxPoints}
    TransferToTempCUDA ( FFORCE,	mMaxPoints *sizeof(Vector3DF) );    // NB buffers are declared and initialized on mMaxPoints.
    TransferToTempCUDA ( FPRESS,	mMaxPoints *sizeof(float) );
    TransferToTempCUDA ( FDENSITY,	mMaxPoints *sizeof(float) );
    TransferToTempCUDA ( FCLR,		mMaxPoints *sizeof(uint) );
    TransferToTempCUDA ( FGCELL,	mMaxPoints *sizeof(uint) );
    TransferToTempCUDA ( FGNDX,		mMaxPoints *sizeof(uint) );
    
    // extra data for morphogenesis
    TransferToTempCUDA ( FELASTIDX,		mMaxPoints *sizeof(uint[BOND_DATA]) );
    TransferToTempCUDA ( FPARTICLEIDX,	mMaxPoints *sizeof(uint[BONDS_PER_PARTICLE *2]) );
    TransferToTempCUDA ( FPARTICLE_ID,	mMaxPoints *sizeof(uint) );
    TransferToTempCUDA ( FMASS_RADIUS,	mMaxPoints *sizeof(uint) );
    TransferToTempCUDA ( FNERVEIDX,		mMaxPoints *sizeof(uint) );
    TransferToTempCUDA ( FCONC,		    mMaxPoints *sizeof(float[NUM_TF]) );
    TransferToTempCUDA ( FEPIGEN,	    mMaxPoints *sizeof(uint[NUM_GENES]) );

    // reset bonds and forces in fbuf FELASTIDX, FPARTICLEIDX and FFORCE, required to prevent interference between time steps, 
    // because these are not necessarily overwritten by the FUNC_COUNTING_SORT kernel.
    cuCtxSynchronize ();    // needed to prevent colision with previous operations
    cuCheck ( cuMemsetD32 ( m_Fluid.gpu(FELASTIDX),    UINT_MAX,  mMaxPoints * BOND_DATA              ),  "CountingSortFullCUDA", "cuMemsetD32", "FELASTIDX",    mbDebug);
    cuCheck ( cuMemsetD32 ( m_Fluid.gpu(FPARTICLEIDX), UINT_MAX,  mMaxPoints * BONDS_PER_PARTICLE *2  ),  "CountingSortFullCUDA", "cuMemsetD32", "FPARTICLEIDX", mbDebug);
    cuCheck ( cuMemsetD32 ( m_Fluid.gpu(FFORCE),      (uint)0.0,  mMaxPoints * 3 /* ie num elements */),  "CountingSortFullCUDA", "cuMemsetD32" , "FFORCE",       mbDebug);
    cuCtxSynchronize ();    // needed to prevent colision with previous operations
    // Reset grid cell IDs
    // cuCheck(cuMemsetD32(m_Fluid.gpu(FGCELL), GRID_UNDEF, numPoints ), "cuMemsetD32(Sort)");

    void* args[1] = { &mMaxPoints };
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_COUNTING_SORT], m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL),
              "CountingSortFullCUDA", "cuLaunch", "FUNC_COUNTING_SORT", mbDebug );
  
    if ( ppos != 0x0 ) {
        cuCheck( cuMemcpyDtoH ( ppos, m_Fluid.gpu(FPOS), mMaxPoints*sizeof(Vector3DF) ), "CountingSortFullCUDA", "cuMemcpyDtoH", "FPOS", mbDebug);
        cuCtxSynchronize ();
    }
    
std::cout<<"\n CountingSortFullCUDA : FUNC_COUNT_SORT_LISTS\n"<<std::flush;
    // countingSortDenseLists ( int pnum ) // NB launch on bins not particles.
    int blockSize = SCAN_BLOCKSIZE/2 << 1; 
    int numElem1 = m_GridTotal;  
    int numElem2 = int ( numElem1 / blockSize ) + 1;  
    int threads = SCAN_BLOCKSIZE/2;
    cuCtxSynchronize ();
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_COUNT_SORT_LISTS], /*m_FParams.numBlocks*/ numElem2, 1, 1, /*m_FParams.numThreads/2*/ threads , 1, 1, 0, NULL, args, NULL),
              "CountingSortFullCUDA", "cuLaunch", "FUNC_COUNT_SORT_LISTS", mbDebug );                                   // NB threads/2 required on GTX970m
    cuCtxSynchronize ();
}

void FluidSystem::CountingSortChangesCUDA ( ){
    int blockSize = SCAN_BLOCKSIZE/2 << 1; 
    int numElem1 = m_GridTotal;  
    int numElem2 = int ( numElem1 / blockSize ) + 1;  
    int threads = SCAN_BLOCKSIZE/2;
    void* args[1] = { &mNumPoints };
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_COUNTING_SORT_CHANGES], numElem2, 1, 1, threads , 1, 1, 0, NULL, args, NULL),
              "CountingSortChangesCUDA", "cuLaunch", "FUNC_COUNTING_SORT_CHANGES", mbDebug );    
}

void FluidSystem::ComputePressureCUDA (){
    void* args[1] = { &mNumPoints };
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_COMPUTE_PRESS],  m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL), "ComputePressureCUDA", "cuLaunch", "FUNC_COMPUTE_PRESS", mbDebug);
}

void FluidSystem::ComputeDiffusionCUDA(){
    //std::cout << "\n\nRunning ComputeDiffusionCUDA()" << std::endl;
    void* args[1] = { &mNumPoints };
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_COMPUTE_DIFFUSION],  m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL), "ComputeDiffusionCUDA", "cuLaunch", "FUNC_COMPUTE_DIFFUSION", mbDebug);
}

void FluidSystem::ComputeForceCUDA (){
    //printf("\n\nFluidSystem::ComputeForceCUDA (),  m_FParams.freeze=%s",(m_FParams.freeze==true) ? "true" : "false");
    void* args[3] = { &m_FParams.pnum ,  &m_FParams.freeze, &m_FParams.frame};
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_COMPUTE_FORCE],  m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL), "ComputeForceCUDA", "cuLaunch", "FUNC_COMPUTE_FORCE", mbDebug);
}

void FluidSystem::ComputeGenesCUDA (){  // for each gene, call a kernel wih the dese list for that gene
    for (int gene=0;gene<NUM_GENES;gene++) {
        uint list_length = m_Fluid.bufI(FDENSE_LIST_LENGTHS)[gene];
        void* args[3] = { &m_FParams.pnum ,  &gene, &list_length};
        int numBlocks, numThreads;
        computeNumBlocks ( list_length , m_FParams.threadsPerBlock, numBlocks, numThreads);
        
        std::cout<<"\nComputeGenesCUDA (): gene ="<<gene<<", list_length="<<list_length<<", m_FParams.threadsPerBlock="<<m_FParams.threadsPerBlock<<", numBlocks="<<numBlocks<<",  numThreads="<<numThreads<<". args={mNumPoints="<<mNumPoints<<", list_length="<<list_length<<", gene ="<<gene<<"}\n"<<std::flush;
        
        if( numBlocks>0 && numThreads>0){
            std::cout<<"\nCalling m_Func[FUNC_COMPUTE_GENE_ACTION]\n"<<std::flush;
            
            cuCheck ( cuLaunchKernel ( m_Func[FUNC_COMPUTE_GENE_ACTION],  numBlocks, 1, 1, numThreads, 1, 1, 0, NULL, args, NULL), "ComputeGenesCUDA", "cuLaunch", "FUNC_COMPUTE_GENE_ACTION", mbDebug);
        }
    }
}

void FluidSystem::ComputeBondChangesCUDA (){// Given the action of the genes, compute the changes to particle properties & splitting/combining  NB also "inserts changes" 
                                            //NB list for all living cells. (non senescent) = FEPIGEN[2]
    cuCheck ( cuMemsetD8 ( m_Fluid.gpu(FGRIDCNT_CHANGES), 0,	m_GridTotal *sizeof(uint[NUM_CHANGES]) ), "InsertParticlesCUDA", "cuMemsetD8", "FGRIDCNT", mbDebug );
    cuCheck ( cuMemsetD8 ( m_Fluid.gpu(FGRIDOFF_CHANGES), 0,	m_GridTotal *sizeof(uint[NUM_CHANGES]) ), "InsertParticlesCUDA", "cuMemsetD8", "FGRIDOFF", mbDebug );

    uint list_length = m_Fluid.bufI(FDENSE_LIST_LENGTHS)[2];    // call for dense list of living cells (gene'2'living/telomere (has genes))
    void* args[2] = { &mNumPoints, &list_length};
    int numBlocks, numThreads;
    computeNumBlocks ( list_length , m_FParams.threadsPerBlock, numBlocks, numThreads);
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_COMPUTE_BOND_CHANGES],  m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL), "computeBondChanges", "cuLaunch", "FUNC_COMPUTE_BOND_CHANGES", mbDebug);
}

void FluidSystem::ComputeParticleChangesCUDA (){// Call each for dense list to execute particle changes. NB Must run concurrently without interfering => no cuCtxSynchronize()
    //for (int change_list = 0; change_list<NUM_CHANGES;change_list++){
    int change_list = 0; // TODO debug, chk one kernel at a time
        uint list_length = m_Fluid.bufI(FDENSE_LIST_LENGTHS_CHANGES)[change_list];  // num blocks and threads by list length
        void* args[3] = { &mNumPoints, &list_length, &change_list};
        int numThreads = 1;//m_FParams.threadsPerBlock;
        int numBlocks  = 1;//iDivUp ( list_length, numThreads );
        //computeNumBlocks ( list_length , m_FParams.threadsPerBlock, numBlocks, numThreads);
        
        std::cout<<"\nComputeParticleChangesCUDA (): change_list ="<<change_list<<", list_length="<<list_length<<", m_FParams.threadsPerBlock="<<m_FParams.threadsPerBlock<<", numBlocks="<<numBlocks<<",  numThreads="<<numThreads<<". args={mNumPoints="<<mNumPoints<<", list_length="<<list_length<<", change_list="<<change_list<<"}\n"<<std::flush;
        
        if( (list_length>0) && (numBlocks>0) && (numThreads>0)){
            std::cout<<"\nCalling m_Func[FUNC_HEAL+"<<change_list<<"], list_length="<<list_length<<", numBlocks="<<numBlocks<<", numThreads="<<numThreads<<"\n"<<std::flush;
            
            cuCheck ( cuLaunchKernel ( m_Func[FUNC_HEAL+change_list], numBlocks, 1, 1, numThreads, 1, 1, 0, NULL, args, NULL), 
                  "ComputeParticleChangesCUDA", "cuLaunch", "FUNC_HEAL+change_list", mbDebug);
        }
        cuCheck(cuCtxSynchronize(), "ComputeParticleChangesCUDA", "cuCtxSynchronize", "In ComputeParticleChangesCUDA", mbDebug);
    //}
    std::cout<<"\nFinished ComputeParticleChangesCUDA ()\n"<<std::flush;
}

void FluidSystem::AdvanceCUDA ( float tm, float dt, float ss ){
    void* args[4] = { &tm, &dt, &ss, &m_FParams.pnum };
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_ADVANCE],  m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL), "AdvanceCUDA", "cuLaunch", "FUNC_ADVANCE", mbDebug);
}

void FluidSystem::EmitParticlesCUDA ( float tm, int cnt ){
    void* args[3] = { &tm, &cnt, &m_FParams.pnum };
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_EMIT],  m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL), "EmitParticlesCUDA", "cuLaunch", "FUNC_EMIT", mbDebug);
}

