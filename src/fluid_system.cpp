#include <assert.h>
#include <iostream>
#include <cuda.h>
#include <stdlib.h>
#include <unistd.h>
#include <curand_kernel.h>
#include <chrono>
#include <cstring>
#include "cutil_math.h"
#include "fluid_system.h"


//bool cuCheck (CUresult launch_stat, const char* method, const char* apicall, const char* arg, bool bDebug);
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

        std::cout << "\nFLUID SYSTEM, CUDA ERROR:\t";
        std::cout << " Launch status: "<< launch_statmsg <<"\t";
        std::cout << " Kernel status: "<< kern_statmsg <<"\t";
        std::cout << " Caller: FluidSystem::"<<  method <<"\t";
        std::cout << " Call:   "<< apicall <<"\t";
        std::cout << " Args:   "<< arg <<"\t";

        if (bDebug) {
            std::cout << "  Generating assert to examine call stack.\n" ;
            assert(0);		// debug - trigger break (see call stack)
        }
        else {
            std::cout << "fluid_system.cpp cuCheck(..), 'nverror()' \n";
            //nverror();		// exit - return 0
        }
        return false;
    }
    return true;

}
//////////////////////////////////////////////////
FluidSystem::FluidSystem (){
    cout<<"\n\nFluidSystem ()"<<std::flush;
    memset ( &m_Fluid, 0,		sizeof(FBufs) );
    memset ( &m_FluidTemp, 0,	sizeof(FBufs) );
    memset ( &m_FParams, 0,		sizeof(FParams) );
    memset ( &m_FGenome, 0,		sizeof(FGenome) );
    mNumPoints = 0;
    mMaxPoints = 0;
    mPackGrid = 0x0;
    m_Frame = 0;
    for (int n=0; n < FUNC_MAX; n++ ) m_Func[n] = (CUfunction) -1;
}

bool FluidSystem::cuCheck (CUresult launch_stat, const char* method, const char* apicall, const char* arg, bool bDebug){
    CUresult kern_stat = CUDA_SUCCESS;

    if (bDebug) {
        kern_stat = cuCtxSynchronize();
    }
    if (kern_stat != CUDA_SUCCESS || launch_stat != CUDA_SUCCESS) {
        const char* launch_statmsg = "";
        const char* kern_statmsg = "";
        cuGetErrorString(launch_stat, &launch_statmsg);
        cuGetErrorString(kern_stat, &kern_statmsg);

        std::cout << "\nFLUID SYSTEM, CUDA ERROR:\t";
        std::cout << " Launch status: "<< launch_statmsg <<"\t";
        std::cout << " Kernel status: "<< kern_statmsg <<"\t";
        std::cout << " Caller: FluidSystem::"<<  method <<"\t";
        std::cout << " Call:   "<< apicall <<"\t";
        std::cout << " Args:   "<< arg <<"\t";

        if (bDebug) {
            std::cout << "  Generating assert to examine call stack.\n" ;
            assert(0);		// debug - trigger break (see call stack)
        }
        else {
            std::cout << "fluid_system.cpp cuCheck(..), 'nverror()' \n";
            //nverror();		// exit - return 0
        }
        Exit();
        return false;
    }
    return true;

}

void FluidSystem::LoadKernel ( int fid, std::string func ){
    char cfn[512];
    strcpy ( cfn, func.c_str() );

    if ( m_Func[fid] == (CUfunction) -1 )
        cuCheck ( cuModuleGetFunction ( &m_Func[fid], m_Module, cfn ), "LoadKernel", "cuModuleGetFunction", cfn, mbDebug );
}

void FluidSystem::Initialize (){             // used for CPU only for "check_demo".
    if (m_FParams.debug>1)std::cout << "FluidSystem::Initialize () \n";
    // An FBufs struct holds an array of pointers.
    // Clear all buffers
    memset ( &m_Fluid, 0,		sizeof(FBufs) );
    memset ( &m_FluidTemp, 0,	sizeof(FBufs) );
    memset ( &m_FParams, 0,		sizeof(FParams) );
    memset ( &m_FGenome, 0,		sizeof(FGenome) );

    if (m_FParams.debug>1)std::cout << "Chk1.4 \n";
    // Allocate the sim parameters
    AllocateBuffer ( FPARAMS,		sizeof(FParams),	0,	1,	 GPU_OFF,     CPU_YES );//AllocateBuffer ( int buf_id, int stride,     int cpucnt, int gpucnt,    int gpumode,    int cpumode )
    if (m_FParams.debug>1)std::cout << "Chk1.5 \n";
    m_Time = 0;
    mNumPoints = 0;			// reset count
    if (m_FParams.debug>1)std::cout << "Chk1.6 \n";
}

// /home/nick/Programming/Cuda/Morphogenesis/build/install/ptx/objects/fluid_systemPTX/fluid_system_cuda.ptx
void FluidSystem::InitializeCuda (){         // used for load_sim  /home/nick/Programming/Cuda/Morphogenesis/build/install/ptx/objects-Debug/fluid_systemPTX/fluid_system_cuda.ptx
    if (m_FParams.debug>1)std::cout << "FluidSystem::InitializeCuda () \n";
    char* morphogenesis_ptx = std::getenv("MORPHOGENESIS_HOME");
    
    //Find release-type & path to ptx file
    sprintf( morphogenesis_ptx, "%s/ptx", morphogenesis_ptx);
    DIR *dir = opendir(morphogenesis_ptx);
    const char *name;
    const char* names[5]= { "objects", "objects-Debug", "objects-Release", "objects-RelWithDebInfo", "objects-MinSizeRel" };
    struct dirent *ent;
    int entry_num = 0;
    while((ent = readdir(dir)) != NULL) {
        if (m_FParams.debug>1)std::cout << "\nInitializeCuda: chk 6, ent->d_name="<<ent->d_name<<", entry_num="<< entry_num <<" \n";
        for (int i=0; i<5; i++){
            if (m_FParams.debug>1)std::cout << "i="<<i<<", std::strcmp("<< ent->d_name <<", "<< names[i] <<") == " << std::strcmp(ent->d_name, names[i]) << "\n";
            if (std::strcmp(ent->d_name, names[i]) == 0 ){
                if (m_FParams.debug>1)std::cout << "Match ent->d_name = names["<<i<<"]\n";
                name = names[i];
                break;
            }
        }
        if (name != NULL) break;
        entry_num++;
    }
    sprintf( morphogenesis_ptx, "%s/%s/fluid_systemPTX/fluid_system_cuda.ptx", morphogenesis_ptx, name);  
    if (m_FParams.debug>1)std::cout<<"\n release type = "<<name<<"\t ptx path = "<<morphogenesis_ptx<<std::flush;
    cuCheck ( cuModuleLoad ( &m_Module, morphogenesis_ptx), "LoadKernel", "cuModuleLoad", morphogenesis_ptx, mbDebug);  
    // loads the file "fluid_system_cuda.ptx" as a module with pointer  m_Module.

    if (m_FParams.debug>1)std::cout << "Chk1.1 \n";
    LoadKernel ( FUNC_INSERT,                           "insertParticles" );
    LoadKernel ( FUNC_COUNTING_SORT,                    "countingSortFull" );
    LoadKernel ( FUNC_QUERY,                            "computeQuery" );
    LoadKernel ( FUNC_COMPUTE_PRESS,                    "computePressure" );
    LoadKernel ( FUNC_COMPUTE_FORCE,                    "computeForce" );
    LoadKernel ( FUNC_ADVANCE,                          "advanceParticles" );
    LoadKernel ( FUNC_EMIT,                             "emitParticles" );
    LoadKernel ( FUNC_RANDOMIZE,                        "randomInit" );
    LoadKernel ( FUNC_SAMPLE,                           "sampleParticles" );
    LoadKernel ( FUNC_FPREFIXSUM,                       "prefixSum" );
    LoadKernel ( FUNC_FPREFIXFIXUP,                     "prefixFixup" );
    LoadKernel ( FUNC_TALLYLISTS,                       "tally_denselist_lengths");
    LoadKernel ( FUNC_COMPUTE_DIFFUSION,                "computeDiffusion");
    LoadKernel ( FUNC_COUNT_SORT_LISTS,                 "countingSortDenseLists");
    LoadKernel ( FUNC_COMPUTE_GENE_ACTION,              "computeGeneAction");
    LoadKernel ( FUNC_TALLY_GENE_ACTION,                "tallyGeneAction");
    LoadKernel ( FUNC_COMPUTE_BOND_CHANGES,             "computeBondChanges");
    LoadKernel ( FUNC_COUNTING_SORT_CHANGES,            "countingSortChanges");
    LoadKernel ( FUNC_COMPUTE_NERVE_ACTION,             "computeNerveActivation");
    LoadKernel ( FUNC_COMPUTE_MUSCLE_CONTRACTION,       "computeMuscleContraction");
    LoadKernel ( FUNC_CLEAN_BONDS,                      "cleanBonds");
    LoadKernel ( FUNC_HEAL,                             "heal");
    LoadKernel ( FUNC_LENGTHEN_MUSCLE,                  "lengthen_muscle");
    LoadKernel ( FUNC_LENGTHEN_TISSUE,                  "lengthen_tissue");
    LoadKernel ( FUNC_SHORTEN_MUSCLE,                   "shorten_muscle");
    LoadKernel ( FUNC_SHORTEN_TISSUE,                   "shorten_tissue");
    LoadKernel ( FUNC_STRENGTHEN_MUSCLE,                "strengthen_muscle");
    LoadKernel ( FUNC_STRENGTHEN_TISSUE,                "strengthen_tissue");
    LoadKernel ( FUNC_WEAKEN_MUSCLE,                    "weaken_muscle");
    LoadKernel ( FUNC_WEAKEN_TISSUE,                    "weaken_tissue");
    LoadKernel ( FUNC_EXTERNAL_ACTUATION,               "externalActuation");
    LoadKernel ( FUNC_FIXED,                            "fixedParticles");
    LoadKernel ( FUNC_INIT_FCURAND_STATE,               "initialize_FCURAND_STATE");
    LoadKernel ( FUNC_ASSEMBLE_MUSCLE_FIBRES_OUTGOING,  "assembleMuscleFibresOutGoing");
    LoadKernel ( FUNC_ASSEMBLE_MUSCLE_FIBRES_INCOMING,  "assembleMuscleFibresInComing");
    LoadKernel ( FUNC_INITIALIZE_BONDS,                 "initialize_bonds");

    if (m_FParams.debug>1)std::cout << "Chk1.2 \n";
    size_t len = 0;
    cuCheck ( cuModuleGetGlobal ( &cuFBuf,    &len,	m_Module, "fbuf" ),		"LoadKernel", "cuModuleGetGlobal", "cuFBuf",    mbDebug);   // Returns a global pointer (cuFBuf) from a module  (m_Module), see line 81.
    cuCheck ( cuModuleGetGlobal ( &cuFTemp,   &len,	m_Module, "ftemp" ),	"LoadKernel", "cuModuleGetGlobal", "cuFTemp",   mbDebug);   // fbuf, ftemp, fparam are defined at top of fluid_system_cuda.cu,
    cuCheck ( cuModuleGetGlobal ( &cuFParams, &len,	m_Module, "fparam" ),	"LoadKernel", "cuModuleGetGlobal", "cuFParams", mbDebug);   // based on structs "FParams", "FBufs", "FGenome" defined in fluid.h
    cuCheck ( cuModuleGetGlobal ( &cuFGenome, &len,	m_Module, "fgenome" ),	"LoadKernel", "cuModuleGetGlobal", "cuFGenome", mbDebug);   // NB defined differently in kernel vs cpu code.
    // An FBufs struct holds an array of pointers.
    if (m_FParams.debug>1)std::cout << "Chk1.3 \n";

    // Allocate the sim parameters
    AllocateBuffer ( FPARAMS,		sizeof(FParams),	0,	1,	 GPU_SINGLE,     CPU_OFF );
    //AllocateBuffer ( int buf_id, int stride,     int cpucnt, int gpucnt,    int gpumode,    int cpumode )
    if (m_FParams.debug>1)std::cout << "Chk1.4 \n";
    m_Time = 0;
    //ClearNeighborTable ();
    mNumPoints = 0;			// reset count
    if (m_FParams.debug>1)std::cout << "Chk1.5 \n";
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
                      m_Param[PVISC], m_Param[PSURFACE_TENSION], m_Param[PEXTDAMP], m_Param[PFORCE_MIN], m_Param[PFORCE_MAX], m_Param[PFORCE_FREQ],
                      m_Param[PGROUND_SLOPE], grav.x, grav.y, grav.z, m_Param[PACCEL_LIMIT], m_Param[PVEL_LIMIT] ); 
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
        if (m_FParams.debug>1)std::cout << "\n n = " << n << std::flush;
        if ( m_Fluid.bufC(n) != 0x0 )
            free ( m_Fluid.bufC(n) );
    }
    size_t   free1, free2, total;
    cudaMemGetInfo(&free1, &total);
    //if (m_FParams.debug>1)printf("\nCuda Memory, before cudaDeviceReset(): free=%lu, total=%lu.\t",free1,total);
    cuCheck(cuCtxSynchronize(), "Exit ", "cuCtxSynchronize", "before cudaDeviceReset()", mbDebug);  
    if(m_Module != 0x0){
        if (m_FParams.debug>1)printf("\ncudaDeviceReset()\n");
        cudaDeviceReset(); // Destroy all allocations and reset all state on the current device in the current process. // must only operate if we have a cuda instance.
    }
    
    cudaMemGetInfo(&free2, &total);
    //if (m_FParams.debug>1)printf("\nAfter cudaDeviceReset(): free=%lu, total=%lu, released=%lu.\n",free2,total,(free2-free1) );
}

void FluidSystem::AllocateBuffer ( int buf_id, int stride, int cpucnt, int gpucnt, int gpumode, int cpumode ){   // mallocs a buffer - called by FluidSystem::Initialize(), AllocateParticles, and AllocateGrid()
//also called by WriteDemoSimParams(..)
    bool rtn = true;
    if (m_FParams.debug>1)std::cout<<"\nAllocateBuffer ( int buf_id="<<buf_id<<", int stride="<<stride<<", int cpucnt="<<cpucnt<<", int gpucnt="<<gpucnt<<", int "<<gpumode<<", int "<<cpumode<<" )\t"<<std::flush;
    if (cpumode == CPU_YES) {
        char* src_buf  = m_Fluid.bufC(buf_id);
        char* dest_buf = (char*) malloc(cpucnt*stride);                   //  ####  malloc the buffer   ####
        if (src_buf != 0x0) {
            memcpy(dest_buf, src_buf, cpucnt*stride);
            free(src_buf);
        }
        m_Fluid.setBuf(buf_id, dest_buf);                                 // stores pointer to buffer in mcpu[buf_id]
    }
    if(gpumode == GPU_SINGLE || gpumode == GPU_DUAL || gpumode == GPU_TEMP){
        cuCheck(cuCtxSynchronize(), "AllocateBuffer ", "cuCtxSynchronize", "before 1st cudaMemGetInfo(&free1, &total)", mbDebug);  
        size_t   free1, free2, total;
        cudaMemGetInfo(&free1, &total);
        //if (m_FParams.debug>1)printf("\nCuda Memory: free=%lu, total=%lu.\t",free1,total);
    
        if (gpumode == GPU_SINGLE || gpumode == GPU_DUAL )	{
            if (m_Fluid.gpuptr(buf_id) != 0x0) cuCheck(cuMemFree(m_Fluid.gpu(buf_id)), "AllocateBuffer", "cuMemFree", "Fluid.gpu", mbDebug);
            rtn = cuCheck( cuMemAlloc(m_Fluid.gpuptr(buf_id), stride*gpucnt), "AllocateBuffer", "cuMemAlloc", "Fluid.gpu", mbDebug);         //  ####  cuMemAlloc the buffer, stores pointer to buffer in   m_Fluid.mgpu[buf_id]
            if (m_FParams.debug>1)std::cout<<"\t\t m_Fluid.gpuptr("<<buf_id<<")'"<<m_Fluid.gpuptr(buf_id)<<",   m_Fluid.gpu("<<buf_id<<")="<<m_Fluid.gpu(buf_id)<<"\t"<<std::flush;
            if(rtn == false)FluidSystem::Exit();
        }
        if (gpumode == GPU_TEMP || gpumode == GPU_DUAL ) {
            if (m_FluidTemp.gpuptr(buf_id) != 0x0) cuCheck(cuMemFree(m_FluidTemp.gpu(buf_id)), "AllocateBuffer", "cuMemFree", "FluidTemp.gpu", mbDebug);
            rtn = cuCheck( cuMemAlloc(m_FluidTemp.gpuptr(buf_id), stride*gpucnt), "AllocateBuffer", "cuMemAlloc", "FluidTemp.gpu", mbDebug); //  ####  cuMemAlloc the buffer, stores pointer to buffer in   m_FluidTemp.mgpu[buf_id]
            if(rtn == false)FluidSystem::Exit();
        }
        cuCheck(cuCtxSynchronize(), "AllocateBuffer ", "cuCtxSynchronize", "before 2nd cudaMemGetInfo(&free2, &total)", mbDebug);  
        cudaMemGetInfo(&free2, &total);
        if (m_FParams.debug>1)printf("\nAfter allocation: free=%lu, total=%lu, this buffer=%lu.\n",free2,total,(free1-free2) );
    }
}

// Allocate particle memory
void FluidSystem::AllocateParticles ( int cnt, int gpu_mode, int cpu_mode ){ // calls AllocateBuffer(..) for each buffer.  
// Defaults in header : int gpu_mode = GPU_DUAL, int cpu_mode = CPU_YES
// Called by FluidSystem::ReadPointsCSV(..), and FluidSystem::WriteDemoSimParams(...), cnt = mMaxPoints.
if (m_FParams.debug>1)std::cout<<"\n\nAllocateParticles ( int cnt="<<cnt<<", int "<<gpu_mode<<", int "<<cpu_mode<<" ), debug="<<m_FParams.debug<<", launchParams.debug="<<launchParams.debug<<"\t";//<<std::flush;
if (m_FParams.debug>1)std::cout<<"\tGPU_OFF=0, GPU_SINGLE=1, GPU_TEMP=2, GPU_DUAL=3, CPU_OFF=4, CPU_YES=5"<<std::flush;
    AllocateBuffer ( FPOS,		sizeof(Vector3DF),	cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FCLR,		sizeof(uint),		cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FVEL,		sizeof(Vector3DF),	cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FVEVAL,	sizeof(Vector3DF),	cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FAGE,		sizeof(uint),       cnt,    m_FParams.szPnts,	gpu_mode, cpu_mode );
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
    AllocateBuffer ( FELASTIDX,	    sizeof(uint[BOND_DATA]),             cnt,   m_FParams.szPnts,	gpu_mode, cpu_mode ); 
    AllocateBuffer ( FPARTICLEIDX,	sizeof(uint[BONDS_PER_PARTICLE *2]), cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FPARTICLE_ID,	sizeof(uint),		                 cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FMASS_RADIUS,	sizeof(uint),		                 cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FNERVEIDX,	    sizeof(uint),		                 cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FCONC,	        sizeof(float[NUM_TF]),		         cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FEPIGEN,	    sizeof(uint[NUM_GENES]),	         cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FCURAND_STATE,	sizeof(curandState_t),	             cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FCURAND_SEED,	sizeof(unsigned long long),	         cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    
    // Update GPU access pointers
    if (gpu_mode != GPU_OFF ) {
        cuCheck( cuMemcpyHtoD(cuFBuf, &m_Fluid, sizeof(FBufs)),			"AllocateParticles", "cuMemcpyHtoD", "cuFBuf", mbDebug);
        cuCheck( cuMemcpyHtoD(cuFTemp, &m_FluidTemp, sizeof(FBufs)),	"AllocateParticles", "cuMemcpyHtoD", "cuFTemp", mbDebug);
        cuCheck( cuMemcpyHtoD(cuFParams, &m_FParams, sizeof(FParams)),  "AllocateParticles", "cuMemcpyHtoD", "cuFParams", mbDebug);
        cuCheck( cuMemcpyHtoD(cuFGenome, &m_FGenome, sizeof(FGenome)),  "AllocateParticles", "cuMemcpyHtoD", "cuFGenome", mbDebug);
        cuCheck(cuCtxSynchronize(), "AllocateParticles", "cuCtxSynchronize", "", mbDebug );
    }

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

void FluidSystem::AllocateBufferDenseLists ( int buf_id, int stride, int gpucnt, int lists ) {    // mallocs a buffer - called by FluidSystem::AllocateGrid(int gpu_mode, int cpu_mode)
// Need to save "pointers to the allocated gpu buffers" in a cpu array, AND then cuMemcpyHtoD(...) that list of pointers into the device array.   
    // also called by FluidSystem::....()  to quadruple buffer as needed.
    cuCheck(cuCtxSynchronize(), "AllocateBufferDenseLists ", "cuCtxSynchronize", "before 1st cudaMemGetInfo(&free1, &total)", mbDebug);  
    size_t   free1, free2, total;
    cudaMemGetInfo(&free1, &total);
    if (m_FParams.debug>1)printf("\nCuda Memory: free=%lu, total=%lu.\t",free1,total);
    
    CUdeviceptr*  listpointer = (CUdeviceptr*) &m_Fluid.bufC(lists)[buf_id * sizeof(CUdeviceptr)] ;
    
    //CUdeviceptr  listpointer2 = m_Fluid.gpuptr(lists)[buf_id]  ;
    if (m_FParams.debug>1)printf("\n*listpointer=%p, listpointer=%p,  lists=%i, buf_id=%i, \t", (CUdeviceptr* ) *listpointer, listpointer,  /*listpointer2,*/ lists, buf_id);/*listpointer2=%llu,*/
    //if (m_FParams.debug>1)cout<<"\n listpointer is an:"<< typeid(listpointer).name()<<" *listpointer is an:"<< typeid(*listpointer).name()<<" listpointer2 is an:"<< typeid(listpointer2).name()<<" .  "<<std::flush;//" *listpointer2 is an:"<< typeid(*listpointer2).name()<<
    
    if (m_FParams.debug>1)printf("\nAllocateBufferDenseLists: buf_id=%i, stride=%i, gpucnt=%i, lists=%i,  .\t", buf_id, stride, gpucnt, lists);
    if (*listpointer != 0x0) cuCheck(cuMemFree(*listpointer), "AllocateBufferDenseLists1", "cuMemFree", "*listpointer", mbDebug);
    bool result = cuCheck( cuMemAlloc( listpointer, stride*gpucnt),   "AllocateBufferDenseLists2", "cuMemAlloc", "listpointer", mbDebug);    
    
    cuCheck(cuCtxSynchronize(), "AllocateBufferDenseLists ", "cuCtxSynchronize", "before 2nd cudaMemGetInfo(&free2, &total)", mbDebug);  
    cudaMemGetInfo(&free2, &total);
    if (m_FParams.debug>1)printf("\nAfter allocation: free=%lu, total=%lu, this buffer=%lu.\n",free2,total,(free1-free2) );
    if(result==false)Exit();
}

void FluidSystem::AllocateGrid(int gpu_mode, int cpu_mode){ // NB void FluidSystem::AllocateBuffer (int buf_id, int stride, int cpucnt, int gpucnt, int gpumode, int cpumode) 
    // Allocate grid
    int cnt = m_GridTotal;
    m_FParams.szGrid = (m_FParams.gridBlocks * m_FParams.gridThreads);
    cout<<"\nAllocateGrid: m_FParams.szGrid = ("<<m_FParams.gridBlocks<<" * "<<m_FParams.gridThreads<<")"<<std::flush;
    AllocateBuffer ( FGRID,		sizeof(uint),		mMaxPoints,	m_FParams.szPnts,	gpu_mode, cpu_mode );    // # grid elements = number of points
    AllocateBuffer ( FGRIDCNT,	sizeof(uint),		cnt,	    m_FParams.szGrid,	gpu_mode, cpu_mode );
    AllocateBuffer ( FGRIDOFF,	sizeof(uint),		cnt,	    m_FParams.szGrid,	gpu_mode, cpu_mode );
    AllocateBuffer ( FGRIDACT,	sizeof(uint),		cnt,	    m_FParams.szGrid,	gpu_mode, cpu_mode );      // ?? not used ?? ... active bins i.e. containing particles ? 
    // extra buffers for dense lists
    AllocateBuffer ( FGRIDCNT_ACTIVE_GENES,  sizeof(uint[NUM_GENES]),       cnt,   m_FParams.szGrid,	gpu_mode, cpu_mode );
    AllocateBuffer ( FGRIDOFF_ACTIVE_GENES,  sizeof(uint[NUM_GENES]),       cnt,   m_FParams.szGrid,	gpu_mode, cpu_mode );
    AllocateBuffer ( FDENSE_LIST_LENGTHS,	 sizeof(uint),		      NUM_GENES,   NUM_GENES,	        gpu_mode, cpu_mode );
    AllocateBuffer ( FDENSE_LISTS,	         sizeof(CUdeviceptr),     NUM_GENES,   NUM_GENES,           gpu_mode, cpu_mode );
    AllocateBuffer ( FDENSE_BUF_LENGTHS,	 sizeof(uint),            NUM_GENES,   NUM_GENES,           gpu_mode, cpu_mode );
    
    AllocateBuffer ( FGRIDCNT_CHANGES,               sizeof(uint[NUM_CHANGES]),       cnt,   m_FParams.szGrid,	    gpu_mode, cpu_mode );
    AllocateBuffer ( FGRIDOFF_CHANGES,               sizeof(uint[NUM_CHANGES]),       cnt,   m_FParams.szGrid,	    gpu_mode, cpu_mode );
    AllocateBuffer ( FDENSE_LIST_LENGTHS_CHANGES,	 sizeof(uint),		      NUM_CHANGES,   NUM_CHANGES,	        gpu_mode, cpu_mode );
    AllocateBuffer ( FDENSE_LISTS_CHANGES,	         sizeof(CUdeviceptr),     NUM_CHANGES,   NUM_CHANGES,           gpu_mode, cpu_mode );
    AllocateBuffer ( FDENSE_BUF_LENGTHS_CHANGES,	 sizeof(uint),            NUM_CHANGES,   NUM_CHANGES,           gpu_mode, cpu_mode );

    if (gpu_mode != GPU_OFF ) {
        /*if(gpu_mode == GPU_SINGLE || gpu_mode == GPU_DUAL )*/
        for(int i=0; i<NUM_GENES; i++){ //for each gene allocate intial buffer, write pointer and size to FDENSE_LISTS and FDENSE_LIST_LENGTHS
            CUdeviceptr*  _listpointer = (CUdeviceptr*) &m_Fluid.bufC(FDENSE_LISTS)[i * sizeof(CUdeviceptr)] ;
            *_listpointer = 0x0;
            AllocateBufferDenseLists( i, sizeof(uint), INITIAL_BUFFSIZE_ACTIVE_GENES, FDENSE_LISTS);  // AllocateBuffer writes pointer to  m_Fluid.gpuptr(buf_id). 
            m_Fluid.bufI(FDENSE_LIST_LENGTHS)[i] = 0;
            m_Fluid.bufI(FDENSE_BUF_LENGTHS)[i]  = INITIAL_BUFFSIZE_ACTIVE_GENES;
        }
        /*if(gpu_mode == GPU_SINGLE || gpu_mode == GPU_DUAL )*/
        for(int i=0; i<NUM_CHANGES; i++){ //Same for the changes lists
            CUdeviceptr*  _listpointer = (CUdeviceptr*) &m_Fluid.bufC(FDENSE_LISTS_CHANGES)[i * sizeof(CUdeviceptr)] ;
            *_listpointer = 0x0; 
            AllocateBufferDenseLists( i, sizeof(uint), 2*INITIAL_BUFFSIZE_ACTIVE_GENES, FDENSE_LISTS_CHANGES); // NB buf[2][list_length] holding : particleIdx, bondIdx
            m_Fluid.bufI(FDENSE_LIST_LENGTHS_CHANGES)[i] = 0;
            m_Fluid.bufI(FDENSE_BUF_LENGTHS_CHANGES)[i]  = INITIAL_BUFFSIZE_ACTIVE_GENES;
        }
        cuMemcpyHtoD(m_Fluid.gpu(FDENSE_LISTS),         m_Fluid.bufC(FDENSE_LISTS),          NUM_GENES * sizeof(CUdeviceptr)  );
        cuMemcpyHtoD(m_Fluid.gpu(FDENSE_LISTS_CHANGES), m_Fluid.bufC(FDENSE_LISTS_CHANGES),  NUM_GENES * sizeof(CUdeviceptr)  );
        cuMemcpyHtoD(m_Fluid.gpu(FDENSE_BUF_LENGTHS),   m_Fluid.bufC(FDENSE_BUF_LENGTHS),    NUM_GENES * sizeof(CUdeviceptr)  );
        cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FDENSE_BUF_LENGTHS_CHANGES), m_Fluid.bufI(FDENSE_BUF_LENGTHS_CHANGES),	sizeof(uint[NUM_CHANGES]) ), "AllocateGrid", "cuMemcpyHtoD", "FDENSE_BUF_LENGTHS_CHANGES", mbDebug);
        
    
        cuCheck(cuMemcpyHtoD(cuFBuf, &m_Fluid, sizeof(FBufs)), "AllocateGrid", "cuMemcpyHtoD", "cuFBuf", mbDebug);  // Update GPU access pointers
        cuCheck(cuCtxSynchronize(), "AllocateParticles", "cuCtxSynchronize", "", mbDebug);
    }
}

int FluidSystem::AddParticleMorphogenesis2 (Vector3DF* Pos, Vector3DF* Vel, uint Age, uint Clr, uint *_ElastIdxU, float *_ElastIdxF, uint *_Particle_Idx, uint Particle_ID, uint Mass_Radius, uint NerveIdx, float* _Conc, uint* _EpiGen ){  // called by :ReadPointsCSV2 (...) where :    uint Particle_Idx[BONDS_PER_PARTICLE * 2];  AND SetupAddVolumeMorphogenesis2(....)
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
  //if (m_FParams.debug>1)printf("m_Fluid.bufV3(FPOS)[n]=(%f,%f,%f), Pos->x=%f, Pos->y=%f, Pos->z=%f,\t",m_Fluid.bufV3(FPOS)[n].x,m_Fluid.bufV3(FPOS)[n].y,m_Fluid.bufV3(FPOS)[n].z,Pos->x,Pos->y,Pos->z);
    uint* ElastIdx = (m_Fluid.bufI(FELASTIDX) + n * BOND_DATA );
    float* ElastIdxFlt = (m_Fluid.bufF(FELASTIDX) + n * BOND_DATA );
    for (int i = 0; i<BONDS_PER_PARTICLE;i++){
  //if (m_FParams.debug>1)printf("\t%u",_ElastIdxU[i*DATA_PER_BOND+0]);
        ElastIdx[i*DATA_PER_BOND+0] = _ElastIdxU[i*DATA_PER_BOND+0] ;
        ElastIdx[i*DATA_PER_BOND+5] = _ElastIdxU[i*DATA_PER_BOND+5] ;
        ElastIdx[i*DATA_PER_BOND+6] = _ElastIdxU[i*DATA_PER_BOND+6] ;
        ElastIdx[i*DATA_PER_BOND+8] = _ElastIdxU[i*DATA_PER_BOND+8] ;
        ElastIdxFlt[i*DATA_PER_BOND+1] = _ElastIdxF[i*DATA_PER_BOND+1] ;
        ElastIdxFlt[i*DATA_PER_BOND+2] = _ElastIdxF[i*DATA_PER_BOND+2] ;
        ElastIdxFlt[i*DATA_PER_BOND+3] = _ElastIdxF[i*DATA_PER_BOND+3] ;
        ElastIdxFlt[i*DATA_PER_BOND+4] = _ElastIdxF[i*DATA_PER_BOND+4] ;
        ElastIdxFlt[i*DATA_PER_BOND+7] = _ElastIdxF[i*DATA_PER_BOND+7] ;
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
        EpiGen[n]    = _EpiGen[j];                                                      // NB 'n' is particle index, from start of this gene. Data order:  FEPIGEN[gene][particle]
    }
    mNumPoints++;
    return n;
}

void FluidSystem::AddNullPoints (){// fills unallocated particles with null data upto mMaxPoints. These can then be used to "create" new particles.
    if (m_FParams.debug>1) std::cout<<"\n AddNullPoints ()\n"<<std::flush;
    Vector3DF Pos, Vel, binSize;
    uint Age, Clr;
    uint  ElastIdxU[BOND_DATA];
    float ElastIdxF[BOND_DATA];
    uint Particle_Idx[2*BONDS_PER_PARTICLE];
    uint Particle_ID, Mass_Radius, NerveIdx;
    float Conc[NUM_TF];
    uint EpiGen[NUM_GENES];
    
    //Pos.x = m_FParams.pboundmax.x; // does not work in makeDemo because no CUDA &=> no UpdateParams. 
    //Pos.y = m_FParams.pboundmax.y; 
    //Pos.z = m_FParams.pboundmax.z;
    
    binSize.x=1.0/m_GridDelta.x; binSize.y=1.0/m_GridDelta.y; binSize.z=1.0/m_GridDelta.z;
    /*
    Pos = GetVec(PVOLMAX);        // SetupSpacing() has been called => m_Vec[PBOUNDMAX] is correctly set.  PVOLMAX - m_Param [ PRADIUS ]
    //Pos.x -= m_GridDelta.x/2; Pos.y -= m_GridDelta.y/2; Pos.z -= m_GridDelta.z/2;  // Should place particle in centre of last bin.
    Pos.x -= binSize.x*1.5; 
    Pos.y -= binSize.y*1.5; 
    Pos.z -= binSize.z*1.5;
    */
    Pos = GetVec(PBOUNDMAX); 
    
    Vel.x = 0; 
    Vel.y = 0; 
    Vel.z = 0;
    Age   = UINT_MAX; // oldest active particles have lowest "age".
    Clr   = 0; 
    for (int j=0;j<BOND_DATA;j++)               ElastIdxU[j]     = UINT_MAX;
    ElastIdxU[8] = 0;
    for (int j=0;j<BOND_DATA;j++)               ElastIdxF[j]     = 0.0;
    for (int j=0;j<2*BONDS_PER_PARTICLE;j++)    Particle_Idx[j] = UINT_MAX;
    Particle_ID = UINT_MAX;
    Mass_Radius = 0;
    NerveIdx    = UINT_MAX;
    for (int j=0;j<NUM_TF;j++)      Conc[j]     = 0;
    for (int j=0;j<NUM_GENES;j++)   EpiGen[j]   = 0;
    
    // TODO FPARTICLE_ID   // should equal mNumPoints when created
    //if (m_FParams.debug>1)std::cout<<"\n AddNullPoints (): mNumPoints="<<mNumPoints<<", mMaxPoints="<<mMaxPoints<<"\n"<<std::flush;
    while (mNumPoints < mMaxPoints){
        AddParticleMorphogenesis2 (&Pos, &Vel, Age, Clr, ElastIdxU, ElastIdxF, Particle_Idx, Particle_ID, Mass_Radius,  NerveIdx, Conc, EpiGen );
        //if (m_FParams.debug>1)std::cout<<"\n AddNullPoints (): mNumPoints="<<mNumPoints<<", mMaxPoints="<<mMaxPoints<<"\n"<<std::flush;
    }
}

void FluidSystem::SetupAddVolumeMorphogenesis2(Vector3DF min, Vector3DF max, float spacing, float offs, uint demoType ){  // NB ony used in WriteDemoSimParams() called by make_demo.cpp . Creates a cuboid with all particle values definable.
if (m_FParams.debug>1)std::cout << "\n SetupAddVolumeMorphogenesis2 \t" << std::flush ;
    Vector3DF pos;
    float dx, dy, dz;
    int cntx, cntz, p, c2;
    cntx = (int) ceil( (max.x-min.x-offs) / spacing );
    cntz = (int) ceil( (max.z-min.z-offs) / spacing );
    int cnt = cntx * cntz;
    min += offs;            // NB by default offs=0.1f, & min=m_Vec[PINITMIN], when called in WriteDemoSimParams(..)
    max -= offs;            // m_Vec[PINITMIN] is set in SetupExampleParams()
    dx = max.x-min.x;       // m_Vec[PBOUNDMIN] is set in SetupSpacing 
    dy = max.y-min.y;
    dz = max.z-min.z;
    Vector3DF rnd;
    c2 = cnt/2;
    Vector3DF Pos, Vel; 
    uint Age, Clr, Particle_ID, Mass_Radius, NerveIdx;
    uint  ElastIdxU[BOND_DATA];
    float ElastIdxF[BOND_DATA];
    uint Particle_Idx[BONDS_PER_PARTICLE*2]; // FPARTICLE_IDX : other particles with incoming bonds attaching here. 
    float Conc[NUM_TF];
    uint EpiGen[NUM_GENES]={0};
    Particle_ID = 0;                         // NB Particle_ID=0 means "no particle" in ElastIdx.           
    Vector3DF volV3DF = max-min;    
    int num_particles_to_make = 8 * 27 * int(volV3DF.x*volV3DF.y*volV3DF.z);//int(volV3DF.x*volV3DF.y*volV3DF.z / spacing*spacing*spacing);
    srand((unsigned int)time(NULL));
    cout<<"\nSetupAddVolumeMorphogenesis2: min=("<<min.x<<","<<min.y<<","<<min.z<<"), max=("<<max.x<<","<<max.y<<","<<max.z<<") "<<std::flush;
    for (int i=0; i<num_particles_to_make; i++){
        Pos.x =  min.x + (float(rand())/float((RAND_MAX)) * dx) ;
        Pos.y =  min.y + (float(rand())/float((RAND_MAX)) * dy) ;
        Pos.z =  min.z + (float(rand())/float((RAND_MAX)) * dz) ;
        
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
                
                for (int i = 0; i<BONDS_PER_PARTICLE;i++){ 
                    for (int j = 0; j< DATA_PER_BOND; j++){ ElastIdxU[i*DATA_PER_BOND +j] = UINT_MAX; ElastIdxF[i*DATA_PER_BOND +j] = 0; } 
                    ElastIdxU[i*DATA_PER_BOND +8] = 0;
                }
                //NB #define DATA_PER_BOND 6 //6 : [0]current index, [1]elastic limit, [2]restlength, [3]modulus, [4]damping coeff, [5]particle ID, [6]bond index
                for (int i = 0; i<BONDS_PER_PARTICLE*2;i++) { Particle_Idx[i] = UINT_MAX; }
                if (Particle_ID % 10 == 0){NerveIdx = Particle_ID/10;} else {NerveIdx = 0;} // Every 10th particle has nerve connection
                
                // Mass & radius of particles
                // 4bit mass + 4bit radius + 24bit uid // but for now, 16bit mass & radius
                // Note m_params[] is set in "FluidSystem::SetupDefaultParams ()" and "FluidSystem::SetupExampleParams ()"
                // mass = m_Param[PMASS]; // 0.00020543f; // kg
                // radius = m_Param[PRADIUS]; // 0.015f; // m
                Mass_Radius =  ( (uint(m_Param[PMASS]*255.0f*255.0f)<<16) | uint(m_Param[PRADIUS]*255.0f*255.0f) ) ; // mass=>13, radius=>975
                for (int i=0; i< NUM_TF; i++)    { Conc[i]   = 0 ;}     // morphogen & transcription factor concentrations
                for (int i=0; i< NUM_GENES; i++) { EpiGen[i] = 0 ;}     // epigenetic state of each gene in this particle
                uint fixedActive = INT_MAX;                             // FEPIGEN below INT_MAX will count down to inactivation. Count down is inactivated by adding INT_MAX.
                EpiGen[0] = fixedActive;                                        // active, i.e. not reserve
                EpiGen[1] = fixedActive;                                        // solid, i.e. have elastic bonds
                EpiGen[2] = fixedActive;                                        // living/telomere, i.e. has genes
                
                if(demoType == 1){                                                                    ////// Remodelling & actuation demo
                                                                                // Fixed base, bone, tendon, muscle, elastic, external actuation
                    if(Pos.z == min.z)                                        EpiGen[11]=fixedActive;   // fixed particle
                    if(Pos.z >= max.z-spacing)                                EpiGen[12]=fixedActive;   // external actuation particle 
                    
                    if(Pos.z >= min.z+5*spacing && Pos.z < min.z+10*spacing)  EpiGen[9] =fixedActive;   // bone
                    if(Pos.z >= min.z+10*spacing && Pos.z < min.z+15*spacing) EpiGen[6] =fixedActive;   // tendon
                    if(Pos.z >= min.z+15*spacing && Pos.z < min.z+20*spacing) EpiGen[7] =fixedActive;   // muscle
                    if(Pos.z >= min.z+20*spacing && Pos.z < min.z+25*spacing) EpiGen[10]=fixedActive;   // elastic tissue
                }else if (demoType == 2){                                                            ////// Diffusion & epigenetics demo
                                                                            // Fixed base, homogeneous particles (initially) 
                    if(Pos.z == min.z) EpiGen[0]=fixedActive;                                           // fixed particle
                    EpiGen[2]=1;                                            // living particle NB set gene behaviour
                }                                                           // => (i) French flag, (ii) polartity, (iii) clock & wave front
                
                p = AddParticleMorphogenesis2 (
                /* Vector3DF* */ &Pos, 
                /* Vector3DF* */ &Vel, 
                /* uint */ Age, 
                /* uint */ Clr, 
                /* uint *_*/ ElastIdxU, 
                /* uint *_*/ ElastIdxF, 
                /* unit * */ Particle_Idx,
                /* uint */ Particle_ID, 
                /* uint */ Mass_Radius, 
                /* uint */ NerveIdx, 
                /* float* */ Conc,
                /* uint* */ EpiGen 
                );
                if(p==-1){
                    if (m_FParams.debug>1){std::cout << "\n SetupAddVolumeMorphogenesis2 exited on p==-1, Pos=("<<Pos.x<<","<<Pos.y<<","<<Pos.z<<"), Particle_ID="<<Particle_ID<<",  EpiGen[0]="<<EpiGen[0]<<" \n " << std::flush ;} 
                    return;
                }
    }
    AddNullPoints ();            // If spare particles remain, fill with null points. NB these can be used to "create" particles.
    if (m_FParams.debug>1)std::cout << "\n SetupAddVolumeMorphogenesis2 finished \n" << std::flush ;
}

/////////////////////////////////////////////////////////////////// 
void FluidSystem::Run (){   // deprecated, rather use: Run(const char * relativePath, int frame, bool debug) 
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
    
//std::cout << "\n\n Chk6 \n"<<std::flush;    
    ComputeDiffusionCUDA();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After ComputeDiffusionCUDA", mbDebug);

//std::cout << "\n\n Chk7 \n"<<std::flush;    
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
    PrefixSumChangesCUDA ( 1 );
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After PrefixSumChangesCUDA", mbDebug);
    
std::cout << "\n\n Chk10 \n"<<std::flush;
    CountingSortChangesCUDA (  );
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After CountingSortChangesCUDA", mbDebug);
    
    
    //  execute particle changes // _should_ be able to run concurrently => no cuCtxSynchronize()
    // => single fn ComputeParticleChangesCUDA ()
std::cout << "\n\n Chk11 \n"<<std::flush;
    ComputeParticleChangesCUDA ();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After ComputeParticleChangesCUDA", mbDebug);

std::cout << "\n\n Chk12 \n"<<std::flush;
    CleanBondsCUDA ();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After CleanBondsCUDA ", mbDebug);
    
std::cout << "\n\n Chk13 \n"<<std::flush;
    TransferPosVelVeval ();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After TransferPosVelVeval ", mbDebug);
    
    AdvanceCUDA ( m_Time, m_DT, m_Param[PSIMSCALE] );
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After AdvanceCUDA", mbDebug);   
    
   // SpecialParticlesCUDA ( m_Time, m_DT, m_Param[PSIMSCALE] );
   // cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After SpecialParticlesCUDA", mbDebug); 
    
    
//TransferFromCUDA ();
    //EmitParticlesCUDA ( m_Time, (int) m_Vec[PEMIT_RATE].x );
    TransferFromCUDA ();	// return for rendering
//std::cout << "\n\n Chk7 \n"<<std::flush;

    AdvanceTime ();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After AdvanceTime", mbDebug);  
//std::cout << " finished \n";
}

void FluidSystem::Run (const char * relativePath, int frame, bool debug, bool gene_activity, bool remodelling ){       // version to save data after each kernel
    m_FParams.frame = frame;                 // used by computeForceCuda( .. Args)
    if (m_FParams.debug>1)std::cout << "\n\n###### FluidSystem::Run (.......) frame = "<<frame<<" #########################################################"<<std::flush;
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "begin Run", mbDebug); 
    if(debug){
        TransferFromCUDA ();
        SavePointsCSV2 (  relativePath, frame );
        std::cout << "\n\nRun(relativePath,frame) Chk1, saved "<< frame <<".csv At start of Run(...) \n"<<std::flush;
    }
    InsertParticlesCUDA ( 0x0, 0x0, 0x0 );
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After InsertParticlesCUDA", mbDebug);
    if(debug){
        TransferFromCUDA ();
        SavePointsCSV2 (  relativePath, frame+1 );
        std::cout << "\n\nRun(relativePath,frame) Chk2, saved "<< frame+1 <<".csv  After InsertParticlesCUDA\n"<<std::flush;
    }
    PrefixSumCellsCUDA ( 1 );
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After PrefixSumCellsCUDA", mbDebug);
    if(debug){
        TransferFromCUDA ();
        SavePointsCSV2 (  relativePath, frame+2 );
        std::cout << "\n\nRun(relativePath,frame) Chk3, saved "<< frame+2 <<".csv  After PrefixSumCellsCUDA\n"<<std::flush;
    }
    CountingSortFullCUDA ( 0x0 );
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After CountingSortFullCUDA", mbDebug);
    if(debug){
        TransferFromCUDA ();
        SavePointsCSV2 (  relativePath, frame+3 );
        std::cout << "\n\nRun(relativePath,frame) Chk4, saved "<< frame+3 <<".csv  After CountingSortFullCUDA\n"<<std::flush;
    }
    
    if(m_FParams.freeze==true){
        InitializeBondsCUDA ();
        cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After InitializeBondsCUDA ", mbDebug);
        if(debug){
            TransferFromCUDA ();
            SavePointsCSV2 (  relativePath, frame+3 );      // NB overwrites previous file.
            std::cout << "\n\nRun(relativePath,frame) Chk4.5, saved "<< frame+3 <<".csv  After InitializeBondsCUDA \n"<<std::flush;
        }
    }
        
    ComputePressureCUDA();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After ComputePressureCUDA", mbDebug);
    if(debug){
        TransferFromCUDA ();
        SavePointsCSV2 (  relativePath, frame+4 );
        std::cout << "\n\nRun(relativePath,frame) Chk5, saved "<< frame+4 <<".csv  After ComputePressureCUDA \n"<<std::flush;
    }
    
    ComputeForceCUDA ();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After ComputeForceCUDA", mbDebug);
    if(debug){
        TransferFromCUDA ();
        SavePointsCSV2 (  relativePath, frame+5 );
        std::cout << "\n\nRun(relativePath,frame) Chk6, saved "<< frame+5 <<".csv  After ComputeForceCUDA \n"<<std::flush;
    }
    // TODO compute nerve activation ? 
    
    // TODO compute muscle action ?
    
    ComputeDiffusionCUDA();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After ComputeDiffusionCUDA", mbDebug);
    if(debug){
        TransferFromCUDA ();
        SavePointsCSV2 (  relativePath, frame+6 );
        std::cout << "\n\nRun(relativePath,frame) Chk7, saved "<< frame+6 <<".csv  After ComputeDiffusionCUDA \n"<<std::flush;
    }
    if(gene_activity){
        ComputeGenesCUDA();     // NB (i)Epigenetic countdown, (ii) GRN gene regulatory network sensitivity to TransciptionFactors (FCONC)
        cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After ComputeGenesCUDA", mbDebug);
        if(debug){
            TransferFromCUDA ();
            SavePointsCSV2 (  relativePath, frame+7 );
            std::cout << "\n\nRun(relativePath,frame) Chk8, saved "<< frame+7 <<".csv  After ComputeGenesCUDA \n"<<std::flush;
        }
        cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After SavePointsCSV2 after ComputeGenesCUDA", mbDebug); // wipes out FEPIGEN
    }
    if(remodelling){
        AssembleFibresCUDA ();
        cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After AssembleFibresCUDA", mbDebug); 
        if(debug){
            TransferFromCUDA ();
            SavePointsCSV2 (  relativePath, frame+8 );
            std::cout << "\n\nRun(relativePath,frame) Chk9.0, saved "<< frame+8 <<".csv  After AssembleFibresCUDA  \n"<<std::flush;
        }
        
        
        ComputeBondChangesCUDA ();
        cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After ComputeBondChangesCUDA", mbDebug); // wipes out FEPIGEN ////////////////////////////////////
        if(debug){
            TransferFromCUDA ();
            SavePointsCSV2 (  relativePath, frame+9 );
            std::cout << "\n\nRun(relativePath,frame) Chk9, saved "<< frame+8 <<".csv  After ComputeBondChangesCUDA  \n"<<std::flush;
        }
        PrefixSumChangesCUDA ( 1 );
        cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After PrefixSumChangesCUDA", mbDebug); // writes mangled (?original?) data to FEPIGEN - not anymore
        if(debug){
            TransferFromCUDA ();
            SavePointsCSV2 (  relativePath, frame+10 );
            std::cout << "\n\nRun(relativePath,frame) Chk10, saved "<< frame+9 <<".csv  After PrefixSumChangesCUDA \n"<<std::flush;
        }
        CountingSortChangesCUDA (  );
        cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After CountingSortChangesCUDA", mbDebug);
        if(debug){
            TransferFromCUDA ();
            SavePointsCSV2 (  relativePath, frame+11 );
            std::cout << "\n\nRun(relativePath,frame) Chk11, saved "<< frame+10 <<".csv  After CountingSortChangesCUDA  \n"<<std::flush;
        }
        ComputeParticleChangesCUDA ();                                     // execute particle changes // _should_ be able to run concurrently => no cuCtxSynchronize()
        cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After ComputeParticleChangesCUDA", mbDebug);
        if(debug){
            TransferFromCUDA ();
            SavePointsCSV2 (  relativePath, frame+12 );
            std::cout << "\n\nRun(relativePath,frame) Chk12, saved "<< frame+11 <<".csv  After  ComputeParticleChangesCUDA.  mMaxPoints="<<mMaxPoints<<"\n"<<std::flush;
        }
        
        //CleanBondsCUDA ();
        //cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After CleanBondsCUDA ", mbDebug);
    }


    TransferPosVelVeval ();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After TransferPosVelVeval ", mbDebug);
    if(debug){
        TransferFromCUDA ();
        SavePointsCSV2 (  relativePath, frame+13 );
        std::cout << "\n\nRun(relativePath,frame) Chk13, saved "<< frame+12 <<".csv  After  TransferPosVelVeval.  mMaxPoints="<<mMaxPoints<<"\n"<<std::flush;
    }
    AdvanceCUDA ( m_Time, m_DT, m_Param[PSIMSCALE] );
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After AdvanceCUDA", mbDebug);
    if(debug){
        TransferFromCUDA ();
        SavePointsCSV2 (  relativePath, frame+14 );
        std::cout << "\n\nRun(relativePath,frame) Chk14, saved "<< frame+13 <<".csv  After  AdvanceCUDA\n"<<std::flush;
    }
/*    SpecialParticlesCUDA ( m_Time, m_DT, m_Param[PSIMSCALE]);
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After SpecialParticlesCUDA", mbDebug);
    if(debug){
        TransferFromCUDA ();
        SavePointsCSV2 (  relativePath, frame+15 );
        std::cout << "\n\nRun(relativePath,frame) Chk15, saved "<< frame+14 <<".csv  After  SpecialParticlesCUDA\n"<<std::flush;
    }
*/
    AdvanceTime ();
/*
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After AdvanceTime", mbDebug); 
    if(debug){
        TransferFromCUDA ();
        SavePointsCSV2 (  relativePath, frame+15 );
    }
*/
/*
//     if(debug){
//         TransferFromCUDA ();
//         SavePointsCSV2 (  relativePath, frame+18 );
//         std::cout << "\n\nRun(relativePath,frame) Chk16, saved "<< frame+6 <<".csv  After AdvanceCUDA \n"<<std::flush;
//     }
    //cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After AdvanceCUDA", mbDebug);    
    //EmitParticlesCUDA ( m_Time, (int) m_Vec[PEMIT_RATE].x );
    //TransferFromCUDA ();	// return for rendering

//     if(debug){
//         TransferFromCUDA ();
//         SavePointsCSV2 (  relativePath, frame+19 );
//         std::cout << "Run(relativePath,frame) finished,  saved "<< frame+7 <<".csv  After AdvanceTime \n";
//     }
*/
}// 0:start, 1:InsertParticles, 2:PrefixSumCellsCUDA, 3:CountingSortFull, 4:ComputePressure, 5:ComputeForce, 6:Advance, 7:AdvanceTime

void FluidSystem::Run2PhysicalSort(){
    InsertParticlesCUDA ( 0x0, 0x0, 0x0 );
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After InsertParticlesCUDA", mbDebug);
    
    PrefixSumCellsCUDA ( 1 );
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After PrefixSumCellsCUDA", mbDebug);
    
    CountingSortFullCUDA ( 0x0 );
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After CountingSortFullCUDA", mbDebug);
}

void FluidSystem::Run2InnerPhysicalLoop(){
    if(m_FParams.freeze==true){
        InitializeBondsCUDA ();
        cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After InitializeBondsCUDA ", mbDebug);
    }
    
    ComputePressureCUDA();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After ComputePressureCUDA", mbDebug);
    
    ComputeForceCUDA ();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After ComputeForceCUDA", mbDebug);
    
    if(launchParams.debug>4){
        TransferFromCUDA ();
        launchParams.file_increment++;
        SavePointsCSV2 (  launchParams.outPath, launchParams.file_num+launchParams.file_increment );
        std::cout << "\n\nRun(relativePath,frame) Chk4, saved "<< launchParams.file_num+3 <<".csv  After CountingSortFullCUDA\n"<<std::flush;
    }
    
    TransferPosVelVeval ();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After TransferPosVelVeval ", mbDebug);
    
    AdvanceCUDA ( m_Time, m_DT, m_Param[PSIMSCALE] );
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After AdvanceCUDA", mbDebug);
    
    SpecialParticlesCUDA ( m_Time, m_DT, m_Param[PSIMSCALE]);
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After SpecialParticlesCUDA", mbDebug);
    
    TransferPosVelVevalFromTemp ();
    AdvanceTime ();
}

void FluidSystem::Run2GeneAction(){//NB gene sorting occurs within Run2PhysicalSort()
    ComputeDiffusionCUDA();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After ComputeDiffusionCUDA", mbDebug);
    
    ComputeGenesCUDA(); // NB (i)Epigenetic countdown, (ii) GRN gene regulatory network sensitivity to TransciptionFactors (FCONC)
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After ComputeGenesCUDA", mbDebug);
}

void FluidSystem::Run2Remodelling(){
    AssembleFibresCUDA ();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After AssembleFibresCUDA", mbDebug); 
    
    ComputeBondChangesCUDA ();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After ComputeBondChangesCUDA", mbDebug); 
    
    PrefixSumChangesCUDA ( 1 );
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After PrefixSumChangesCUDA", mbDebug);
    
    CountingSortChangesCUDA (  );
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After CountingSortChangesCUDA", mbDebug);
    
    ComputeParticleChangesCUDA ();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After ComputeParticleChangesCUDA", mbDebug);
}



void FluidSystem::setFreeze(bool freeze){
    m_FParams.freeze = freeze;
    cuCheck ( cuMemcpyHtoD ( cuFParams,	&m_FParams,		sizeof(FParams) ), "FluidParamCUDA", "cuMemcpyHtoD", "cuFParams", mbDebug);
}


void FluidSystem::Freeze (){
    m_FParams.freeze = true;
    Run();
    m_FParams.freeze = false;
}

void FluidSystem::Freeze (const char * relativePath, int frame, bool debug, bool gene_activity, bool remodelling  ){
    m_FParams.freeze = true;
    Run(relativePath, frame, debug, gene_activity, remodelling );
    m_FParams.freeze = false;
}

void FluidSystem::AdvanceTime () {  // may need to prune unused details from this fn.
    m_Time += m_DT;
}

///////////////////////////////////////////////////
// Ideal grid cell size (gs) = 2 * smoothing radius = 0.02*2 = 0.04
// Ideal domain size = k*gs/d = k*0.02*2/0.005 = k*8 = {8, 16, 24, 32, 40, 48, ..}
//    (k = number of cells, gs = cell size, d = simulation scale)
void FluidSystem::SetupGrid ( Vector3DF min, Vector3DF max, float sim_scale, float cell_size){
    float world_cellsize = cell_size / sim_scale;
    m_GridMin = min;
    m_GridMax = max;
    m_GridSize = m_GridMax;
    m_GridSize -= m_GridMin;
    m_GridRes.x = (int) ceil ( m_GridSize.x / world_cellsize );		// Determine grid resolution
    m_GridRes.y = (int) ceil ( m_GridSize.y / world_cellsize );
    m_GridRes.z = (int) ceil ( m_GridSize.z / world_cellsize );
    m_GridSize.x = m_GridRes.x * cell_size / sim_scale;				// Adjust grid size to multiple of cell size
    m_GridSize.y = m_GridRes.y * cell_size / sim_scale;
    m_GridSize.z = m_GridRes.z * cell_size / sim_scale;
    m_GridDelta = m_GridRes;		// delta = translate from world space to cell #
    m_GridDelta /= m_GridSize;
    m_GridTotal = (int)(m_GridRes.x * m_GridRes.y * m_GridRes.z);

    // Number of cells to search:
    // n = (2r / w) +1,  where n = 1D cell search count, r = search radius, w = world cell width
    m_GridSrch = (int) (floor(2.0f*(m_Param[PSMOOTHRADIUS]/sim_scale) / world_cellsize) + 1.0f);
    if ( m_GridSrch < 2 ) m_GridSrch = 2;
    m_GridAdjCnt = m_GridSrch * m_GridSrch * m_GridSrch ;			// 3D search count = n^3, e.g. 2x2x2=8, 3x3x3=27, 4x4x4=64

    if ( m_GridSrch > 6 ) {
        //if (m_FParams.debug>1)nvprintf ( "ERROR: Neighbor search is n > 6. \n " );
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
/////////////////////////////////////////////////////////////
void FluidSystem::SetupSPH_Kernels (){
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
    m_Param [ PSURFACE_TENSION ] = 0.1f;
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
    m_Param [ PGRAV ] =			1.0f;

    m_Param [ PGROUND_SLOPE ] = 0.0f;
    m_Param [ PFORCE_MIN ] =	0.0f;
    m_Param [ PFORCE_MAX ] =	0.0f;
    m_Param [ PFORCE_FREQ ] =	16.0f;
    m_Vec [ PPLANE_GRAV_DIR ].Set ( 0, -9.8f, 0 );

    // Default sim config
    m_Param [PGRIDSIZE] = m_Param[PSMOOTHRADIUS] * 2;
}

void FluidSystem::SetupExampleParams (uint spacing){
    Vector3DF pos;
    Vector3DF min, max;
    m_Param [ PSPACING ] = spacing;

    switch ( (int) m_Param[PEXAMPLE] ) {

    case 0:	{	// Regression test. N x N x N static grid

        int k = (int) ceil ( pow ( (float) m_Param[PNUM], (float) 1.0f/3.0f ) );
        m_Vec [ PVOLMIN ].Set ( 0, 0, 0 );
        m_Vec [ PVOLMAX ].Set ( 2.0f+(k/2), 2.0f+(k/2), 2.0f+(k/2) );
        m_Vec [ PINITMIN ].Set ( 1.0f, 1.0f, 1.0f );
        m_Vec [ PINITMAX ].Set ( 1.0f+(k/2), 1.0f+(k/2), 1.0f+(k/2) );

        m_Param [ PGRAV ] = 0.0;
        m_Vec [ PPLANE_GRAV_DIR ].Set ( 0.0, 0.0, 0.0 );
        //m_Param [ PSPACING ] = spacing;//0.5;				// Fixed spacing		Dx = x-axis density
        m_Param [ PSMOOTHRADIUS ] =	m_Param [PSPACING];		// Search radius
        //m_Toggle [ PRUN ] = false;				// Do NOT run sim. Neighbors only.
        //m_Param [PDRAWMODE] = 1;				// Point drawing
        //m_Param [PDRAWGRID] = 1;				// Grid drawing
        //m_Param [PDRAWTEXT] = 1;				// Text drawing
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
    case 6:     // Morphogenesis small demo
        m_Param [ PSIMSCALE ] = 1.0f;
        m_Param [ PRADIUS ] = 1.0f;
        m_Param [ PSMOOTHRADIUS ] = 1.0f;
        
        m_Vec [ PVOLMIN ].Set ( 0, 0, 0 );
        m_Vec [ PVOLMAX ].Set ( 10, 20, 20 ); //( 80, 50, 80 );
        m_Vec [ PINITMIN ].Set ( m_Vec [ PVOLMIN ].x,  m_Vec [ PVOLMIN ].y, m_Vec [ PVOLMIN ].z );// will be reset to m_Vec[PBOUNDMIN].
        m_Vec [ PINITMAX ].Set ( 60, 80, 60 );
        
        m_Param [ PGRAV ] = 2.000000f;
        m_Vec [ PPLANE_GRAV_DIR ].Set ( 0, -1, 0 );
        m_Param [ PGROUND_SLOPE ] = 0.1f;
        break;
    case 7:     // From SpecificationFile.txt
        m_Time = launchParams.m_Time;
        m_DT = launchParams.m_DT;
        m_Param [ PGRIDSIZE ] = launchParams.gridsize;
        m_Param [ PSPACING ] = launchParams.spacing;
        m_Param [ PSIMSCALE ] = launchParams.simscale;
        m_Param [ PSMOOTHRADIUS ] = launchParams.smoothradius;
        m_Param [ PVISC ] = launchParams.visc;
        m_Param [ PSURFACE_TENSION ] = launchParams.surface_tension;
        m_Param [ PMASS ] = launchParams.mass;
        m_Param [ PRADIUS ] = launchParams.radius;
        /*m_Param [ PDIST ] = launchParams.dist;*/
        m_Param [ PINTSTIFF ] = launchParams.intstiff;
        m_Param [ PEXTSTIFF ] = launchParams.extstiff;
        m_Param [ PEXTDAMP ] = launchParams.extdamp;
        m_Param [ PACCEL_LIMIT ] = launchParams.accel_limit;
        m_Param [ PVEL_LIMIT ] = launchParams.vel_limit;
        m_Param [ PGRAV ] = launchParams.grav;
        m_Param [ PGROUND_SLOPE ] = launchParams.ground_slope;
        m_Param [ PFORCE_MIN ] = launchParams.force_min;
        m_Param [ PFORCE_MAX ] = launchParams.force_max;
        m_Param [ PFORCE_FREQ ] = launchParams.force_freq;
        
        m_Vec [ PVOLMIN ] = launchParams.volmin;
        m_Vec [ PVOLMAX ] = launchParams.volmax;
        m_Vec [ PINITMIN ] = launchParams.initmin;
        m_Vec [ PINITMAX ] = launchParams.initmax;
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
    m_FGenome.param[0][m_FGenome.elongation_factor]      = 0.02  ;
    m_FGenome.param[0][m_FGenome.strength_threshold]     = 0.1  ;
    m_FGenome.param[0][m_FGenome.strengthening_factor]   = 0.02  ;
    
    m_FGenome.param[0][m_FGenome.max_rest_length]        = 1.0  ;
    m_FGenome.param[0][m_FGenome.min_rest_length]        = 0.3  ;
    m_FGenome.param[0][m_FGenome.max_modulus]            = 0.8  ;
    m_FGenome.param[0][m_FGenome.min_modulus]            = 0.3  ;
    
    m_FGenome.param[0][m_FGenome.elastLim]               = 2  ;
    m_FGenome.param[0][m_FGenome.default_rest_length]    = 0.5  ;
    m_FGenome.param[0][m_FGenome.default_modulus]        = 100000  ;
    m_FGenome.param[0][m_FGenome.default_damping]        = 10  ;
    
    //1=collagen
    m_FGenome.param[1][m_FGenome.elongation_threshold]   = 4.0  ;
    m_FGenome.param[1][m_FGenome.elongation_factor]      = 0.01 ;
    m_FGenome.param[1][m_FGenome.strength_threshold]     = 4.1  ;
    m_FGenome.param[1][m_FGenome.strengthening_factor]   = 0.01 ;
    
    m_FGenome.param[1][m_FGenome.max_rest_length]        = 1.0  ;
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
    
    m_FGenome.param[2][m_FGenome.max_rest_length]        = 1.0  ;
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
    if (m_FParams.debug>0)printf ( "\nSetupSpacing: Density=,%f, Spacing=,%f, PDist=,%f\n", m_Param[PRESTDENSITY], m_Param[PSPACING], m_Param[PDIST] );

    // Particle Boundaries
    m_Vec[PBOUNDMIN] = m_Vec[PVOLMIN];
    m_Vec[PBOUNDMIN] += 2.0*(m_Param[PGRIDSIZE] / m_Param[PSIMSCALE]);
    m_Vec[PBOUNDMAX] = m_Vec[PVOLMAX];
    m_Vec[PBOUNDMAX] -= 2.0*(m_Param[PGRIDSIZE] / m_Param[PSIMSCALE]);
}

void FluidSystem::SetupSimulation(int gpu_mode, int cpu_mode){ // const char * relativePath, int gpu_mode, int cpu_mode
     // Allocate buffers for points
    //std::cout<<"\nSetupSimulation chk1, m_FParams.debug="<<m_FParams.debug<<std::flush;
    m_Param [PNUM] = launchParams.num_particles;                             // NB there is a line of text above the particles, hence -1.
    mMaxPoints = m_Param [PNUM];
    m_Param [PGRIDSIZE] = 2*m_Param[PSMOOTHRADIUS] / m_Param[PGRID_DENSITY];
    //std::cout<<"\nSetupSimulation chk2, m_FParams.debug="<<m_FParams.debug<<std::flush;
    
    SetupSPH_Kernels ();
    SetupSpacing ();
    SetupGrid ( m_Vec[PVOLMIN]/*bottom corner*/, m_Vec[PVOLMAX]/*top corner*/, m_Param[PSIMSCALE], m_Param[PGRIDSIZE]);
    //std::cout<<"\nSetupSimulation chk3, m_FParams.debug="<<m_FParams.debug<<std::flush;
    
    if (gpu_mode != GPU_OFF) {     // create CUDA instance etc.. 
        FluidSetupCUDA ( mMaxPoints, m_GridSrch, *(int3*)& m_GridRes, *(float3*)& m_GridSize, *(float3*)& m_GridDelta, *(float3*)& m_GridMin, *(float3*)& m_GridMax, m_GridTotal, 0 );
        UpdateParams();            //  sends simulation params to device.
        UpdateGenome();            //  sends genome to device.              // NB need to initialize genome from file, or something.
    }
    std::cout<<"\nSetupSimulation chk4, mMaxPoints="<<mMaxPoints<<", gpu_mode="<<gpu_mode<<", cpu_mode="<<cpu_mode<<", m_FParams.debug="<<m_FParams.debug<<std::flush;
    
    AllocateParticles ( mMaxPoints, gpu_mode, cpu_mode );  // allocates only cpu buffer for particles
    std::cout<<"\nSetupSimulation chk5 "<<std::flush;
    
    AllocateGrid(gpu_mode, cpu_mode);
    std::cout<<"\nSetupSimulation chk6 "<<std::flush;
    
}

void FluidSystem::RunSimulation (){
    //std::cout<<"\nRunSimulation chk1 "<<std::flush;
    Init_FCURAND_STATE_CUDA ();
    //std::cout<<"\nRunSimulation chk2 "<<std::flush;
    auto old_begin = std::chrono::steady_clock::now();
    //std::cout<<"\nRunSimulation chk3 "<<std::flush;
    TransferPosVelVeval ();
    //std::cout<<"\nRunSimulation chk4 "<<std::flush;
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After TransferPosVelVeval, before 1st timestep", 1/*mbDebug*/);
    //std::cout<<"\nRunSimulation chk5 "<<std::flush;
    
    setFreeze(true);
    std::cout<<"\nRunSimulation chk6,  launchParams.freeze_steps="<<launchParams.freeze_steps
    <<",  launchParams.file_num="<<launchParams.file_num
    <<",  launchParams.num_files="<<launchParams.num_files
    <<",  launchParams.freeze_steps="<<launchParams.freeze_steps
    <<",  launchParams.save_vtp="<<launchParams.save_vtp
    <<" ."<<std::flush;
    for (int k=0; k<launchParams.freeze_steps; k++){
      std::cout<<"\n\nFreeze()"<<k<<"\n"<<std::flush;
      Run (launchParams.outPath, launchParams.file_num, (launchParams.debug>4), (launchParams.gene_activity=='y'), (launchParams.remodelling=='y') );
      TransferPosVelVeval ();
      if(launchParams.save_csv=='y'||launchParams.save_vtp=='y') TransferFromCUDA ();
      if(launchParams.save_csv=='y') SavePointsCSV2 ( launchParams.outPath, launchParams.file_num+90);
      if(launchParams.save_vtp=='y') SavePointsVTP2 ( launchParams.outPath, launchParams.file_num+90);
      launchParams.file_num+=100;
    }
    setFreeze(false);
    printf("\n\nFreeze finished, starting normal Run ##############################################\n\n");
    
    for ( ; launchParams.file_num<launchParams.num_files; launchParams.file_num+=100 ) {
        for ( int j=0; j<launchParams.steps_per_file; j++ ) {//, bool gene_activity, bool remodelling 
            Run (launchParams.outPath, launchParams.file_num, (launchParams.debug>4), (launchParams.gene_activity=='y'), (launchParams.remodelling=='y') );  // run the simulation  // Run(outPath, file_num) saves file after each kernel,, Run() does not.
        }// 0:start, 1:InsertParticles, 2:PrefixSumCellsCUDA, 3:CountingSortFull, 4:ComputePressure, 5:ComputeForce, 6:Advance, 7:AdvanceTime

        //fluid.SavePoints (i);                         // alternate file formats to write
        // TODO flip mutex
        auto begin = std::chrono::steady_clock::now();
        if(launchParams.save_csv=='y'||launchParams.save_vtp=='y') TransferFromCUDA ();
        if(launchParams.save_csv=='y') SavePointsCSV2 ( launchParams.outPath, launchParams.file_num+90);
        if(launchParams.save_vtp=='y') SavePointsVTP2 ( launchParams.outPath, launchParams.file_num+90);
        cout << "\n File# " << launchParams.file_num << ". " << std::flush;
        
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> time = end - begin;
        std::chrono::duration<double> begin_dbl = begin - old_begin;
        if(launchParams.debug>0) std::cout << "\nLoop duration : "
                    << begin_dbl.count() <<" seconds. Time taken to write files for "
                    << NumPoints() <<" particles : " 
                    << time.count() << " seconds\n" << std::endl;
        old_begin = begin;
    }
    launchParams.file_num++;
    WriteSimParams ( launchParams.outPath ); 
    WriteGenome( launchParams.outPath );
}

void FluidSystem::Run2Simulation(){
    Init_FCURAND_STATE_CUDA ();
    auto old_begin = std::chrono::steady_clock::now();
    TransferPosVelVeval ();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After TransferPosVelVeval, before 1st timestep", 1/*mbDebug*/);
    setFreeze(true);
    for (int k=0; k<launchParams.freeze_steps; k++){
      std::cout<<"\n\nFreeze()"<<k<<"\n"<<std::flush;
      Run (launchParams.outPath, launchParams.file_num, (launchParams.debug>4), (launchParams.gene_activity=='y'), (launchParams.remodelling=='y') );
      TransferPosVelVeval ();
      if(launchParams.save_csv=='y'||launchParams.save_vtp=='y') TransferFromCUDA ();
      if(launchParams.save_csv=='y') SavePointsCSV2 ( launchParams.outPath, launchParams.file_num+90);
      if(launchParams.save_vtp=='y') SavePointsVTP2 ( launchParams.outPath, launchParams.file_num+90);
      launchParams.file_num+=100;
    }
    setFreeze(false);
    printf("\n\nFreeze finished, starting normal Run ##############################################\n\n");
    
    for ( ; launchParams.file_num<launchParams.num_files; launchParams.file_num+=100 ) {
        launchParams.file_increment=0;
        for ( int j=0; j<launchParams.steps_per_file; j++ ) {
            Run2PhysicalSort();
            for (int k=0; k<launchParams.steps_per_InnerPhysicalLoop; k++) Run2InnerPhysicalLoop();
            if(launchParams.gene_activity=='y') Run2GeneAction();
            if(launchParams.remodelling=='y') Run2Remodelling();
        }
        auto begin = std::chrono::steady_clock::now();
        if(launchParams.save_csv=='y'||launchParams.save_vtp=='y') TransferFromCUDA ();
        if(launchParams.save_csv=='y') SavePointsCSV2 ( launchParams.outPath, launchParams.file_num+90);
        if(launchParams.save_vtp=='y') SavePointsVTP2 ( launchParams.outPath, launchParams.file_num+90);
        cout << "\n File# " << launchParams.file_num << ". " << std::flush;
        
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> time = end - begin;
        std::chrono::duration<double> begin_dbl = begin - old_begin;
        if(launchParams.debug>0) std::cout << "\nLoop duration : "
                    << begin_dbl.count() <<" seconds. Time taken to write files for "
                    << NumPoints() <<" particles : " 
                    << time.count() << " seconds\n" << std::endl;
        old_begin = begin;
    }
    launchParams.file_num++;
    WriteSimParams ( launchParams.outPath ); 
    WriteGenome( launchParams.outPath );
    WriteExampleSpecificationFile ( launchParams.outPath );
}


