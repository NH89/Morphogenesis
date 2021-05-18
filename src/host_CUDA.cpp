#include <assert.h>
#include <cuda.h>
#include "cutil_math.h"
#include <unistd.h>
#include <curand_kernel.h> // ../cuda-11.2/targets/x86_64-linux/include/
#include "fluid_system.h"

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

void FluidSystem::TransferFromTempCUDA ( int buf_id, int sz ){
    cuCheck ( cuMemcpyDtoD ( m_Fluid.gpu(buf_id), m_FluidTemp.gpu(buf_id), sz ), "TransferFromTempCUDA", "cuMemcpyDtoD", "m_Fluid", mbDebug);
}

void FluidSystem::FluidSetupCUDA ( int num, int gsrch, int3 res, float3 size, float3 delta, float3 gmin, float3 gmax, int total, int chk ){
    m_FParams.pnum = num;
    m_FParams.maxPoints = num;
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

void FluidSystem::FluidParamCUDA ( float ss, float sr, float pr, float mass, float rest, float3 bmin, float3 bmax, float estiff, float istiff, float visc, float surface_tension, float damp, float fmin, float fmax, float ffreq, float gslope, float gx, float gy, float gz, float al, float vl){
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
    m_FParams.psurface_t = surface_tension;
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
    //m_FParams.pemit = emit;
                                                                            
    m_FParams.pdist = pow ( m_FParams.pmass / m_FParams.prest_dens, 1/3.0f );
                                                                                // Normalization constants.
    m_FParams.poly6kern = 315.0f / (64.0f * 3.141592f * pow( sr, 9.0f) );
    m_FParams.wendlandC2kern = 21 / (16 * 3.141592f );   
    /* My notes from Sympy my notebook. 
    Where Wendland C2 kernel:
    
        wc2 = (1-r*ss/2*sr)**4  * ((2*q) +1)
    
    Normalisation constant = 1/integrate( (wc2*(4*pi*r**2)), (r,0, 2*sr/ss)),  NB *(4*pi*r**2) area of a sphere, & 2=basis of wc2.
    
        =  1/ (288pi - 15552.0πss^2/sr^2 + 77760.0πss^3/sr^3 - 149965.714285714πss^4/sr^4 + 104976.0πss^5/sr^5  )
    
    */
    /* Notes from DualSPHysics Wiki
    // Normalization const = reciprocal of radial integral of (kernel * area of sphere), found using Sympy.      
    // NB using W(r,h)=alpha_D (1-q/2)**4 *(2*q +1), 0<=q<=2, as per DualSPHysics Wiki. Where alpha_D is the normaliation constant.
    // * m_FParams.pmass * m_FParams.psimscale
    */
    m_FParams.spikykern = -45.0f / (3.141592f * pow( sr, 6.0f) );            // spikykern used for force due to pressure.
    m_FParams.lapkern = 45.0f / (3.141592f * pow( sr, 6.0f) );               
    // NB Viscosity uses a different kernel, this is the constant portion of its Laplacian.
    // NB Laplacian is a scalar 2nd order differential, "The divergence of the gradient" 
    // This Laplacian comes from Muller et al 2003, NB The kernel is defined by the properties of  its Laplacian, gradient and value at the basis (outer limit) of the kernel. The Laplacian is the form used in the code. The equation of the kernel in Muller et al seems to be wrong, but this does not matter.
    
/*
    // -32*(1 - r)**3 + 12*(1 - r)**2*(4*r + 1)  // the Laplacian of  WC2 = (1-r)**4 *(1+4*r)
//(15*r**2*(h/r**3 + 2/h**2 - 3*r/h**3)/(2*pi*h**3) + 15*r*(-h/(2*r**2) + 2*r/h**2 - 3*r**2/(2*h**3))/(pi*h**3))/r**2
//(45/pi*h^6)((h^2/12r^3)+(2h/3)-(3r/4))
    
//(r**2*(h/r**3 + 2/h**2 - 3*r/h**3) + 2*r*(-h/(2*r**2) + 2*r/h**2 - 3*r**2/(2*h**3) ) )/r**2
*/
    
    m_FParams.gausskern = 1.0f / pow(3.141592f * 2.0f*sr*sr, 3.0f/2.0f);     // Gaussian not currently used.

    m_FParams.H = m_FParams.psmoothradius / m_FParams.psimscale;
    m_FParams.d2 = m_FParams.psimscale * m_FParams.psimscale;
    m_FParams.rd2 = m_FParams.r2 / m_FParams.d2;
    m_FParams.vterm = m_FParams.lapkern * m_FParams.pvisc;

    // Transfer sim params to device
    cuCheck ( cuMemcpyHtoD ( cuFParams,	&m_FParams,		sizeof(FParams) ), "FluidParamCUDA", "cuMemcpyHtoD", "cuFParams", mbDebug);
}

void FluidSystem::TransferToCUDA (){
if (m_FParams.debug>1) std::cout<<"\nTransferToCUDA ()\n"<<std::flush;
    // Send particle buffers
    cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FPOS),          m_Fluid.bufC(FPOS),         mMaxPoints *sizeof(float) * 3),                     "TransferToCUDA", "cuMemcpyHtoD", "FPOS",           mbDebug);
    cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FVEL),          m_Fluid.bufC(FVEL),         mMaxPoints *sizeof(float)*3 ),                      "TransferToCUDA", "cuMemcpyHtoD", "FVEL",           mbDebug);
    cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FVEVAL),        m_Fluid.bufC(FVEVAL),       mMaxPoints *sizeof(float)*3 ),                      "TransferToCUDA", "cuMemcpyHtoD", "FVELAL",         mbDebug);
    cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FFORCE),        m_Fluid.bufC(FFORCE),       mMaxPoints *sizeof(float)*3 ),                      "TransferToCUDA", "cuMemcpyHtoD", "FFORCE",         mbDebug);
    cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FPRESS),        m_Fluid.bufC(FPRESS),       mMaxPoints *sizeof(float) ),                        "TransferToCUDA", "cuMemcpyHtoD", "FPRESS",         mbDebug);
    cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FDENSITY),      m_Fluid.bufC(FDENSITY),     mMaxPoints *sizeof(float) ),                        "TransferToCUDA", "cuMemcpyHtoD", "FDENSITY",       mbDebug);
    cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FCLR),          m_Fluid.bufC(FCLR),         mMaxPoints *sizeof(uint) ),                         "TransferToCUDA", "cuMemcpyHtoD", "FCLR",           mbDebug);
    // add extra data for morphogenesis
    cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FELASTIDX),     m_Fluid.bufC(FELASTIDX),    mMaxPoints *sizeof(uint[BOND_DATA]) ),              "TransferToCUDA", "cuMemcpyHtoD", "FELASTIDX",      mbDebug);
    cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FPARTICLEIDX),  m_Fluid.bufC(FPARTICLEIDX), mMaxPoints *sizeof(uint[BONDS_PER_PARTICLE *2]) ),  "TransferToCUDA", "cuMemcpyHtoD", "FPARTICLEIDX",   mbDebug);
    cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FPARTICLE_ID),  m_Fluid.bufC(FPARTICLE_ID), mMaxPoints *sizeof(uint) ),                         "TransferToCUDA", "cuMemcpyHtoD", "FPARTICLE_ID",   mbDebug);
    cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FMASS_RADIUS),  m_Fluid.bufC(FMASS_RADIUS), mMaxPoints *sizeof(uint) ),                         "TransferToCUDA", "cuMemcpyHtoD", "FMASS_RADIUS",   mbDebug);
    cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FNERVEIDX),     m_Fluid.bufC(FNERVEIDX),    mMaxPoints *sizeof(uint) ),                         "TransferToCUDA", "cuMemcpyHtoD", "FNERVEIDX",      mbDebug);
    cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FCONC),         m_Fluid.bufC(FCONC),        mMaxPoints *sizeof(float[NUM_TF]) ),                "TransferToCUDA", "cuMemcpyHtoD", "FCONC",          mbDebug);
    cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FEPIGEN),       m_Fluid.bufC(FEPIGEN),      mMaxPoints *sizeof(uint[NUM_GENES]) ),              "TransferToCUDA", "cuMemcpyHtoD", "FEPIGEN",        mbDebug);
if (m_FParams.debug>1) std::cout<<"TransferToCUDA ()  finished\n"<<std::flush;

}
  
void FluidSystem::TransferFromCUDA (){
//std::cout<<"\nTransferFromCUDA () \n"<<std::flush;    
    // Return particle buffers
    cuCheck( cuMemcpyDtoH ( m_Fluid.bufC(FPOS),         m_Fluid.gpu(FPOS),          mMaxPoints *sizeof(float)*3 ),                         "TransferFromCUDA", "cuMemcpyDtoH", "FPOS",         mbDebug);
    cuCheck( cuMemcpyDtoH ( m_Fluid.bufC(FVEL),         m_Fluid.gpu(FVEL),          mMaxPoints *sizeof(float)*3 ),                         "TransferFromCUDA", "cuMemcpyDtoH", "FVEL",         mbDebug);
    cuCheck( cuMemcpyDtoH ( m_Fluid.bufI(FAGE),         m_Fluid.gpu(FAGE),          mMaxPoints *sizeof(uint) ),                            "TransferFromCUDA", "cuMemcpyDtoH", "FAGE",         mbDebug);
    cuCheck( cuMemcpyDtoH ( m_Fluid.bufI(FCLR),         m_Fluid.gpu(FCLR),          mMaxPoints *sizeof(uint) ),                            "TransferFromCUDA", "cuMemcpyDtoH", "FCLR",         mbDebug);
    // add extra data for morphogenesis
    cuCheck( cuMemcpyDtoH ( m_Fluid.bufC(FELASTIDX),	m_Fluid.gpu(FELASTIDX),	    mMaxPoints *sizeof(uint[BOND_DATA]) ),                 "TransferFromCUDA", "cuMemcpyDtoH", "FELASTIDX",    mbDebug); 
    cuCheck( cuMemcpyDtoH ( m_Fluid.bufI(FPARTICLEIDX),	m_Fluid.gpu(FPARTICLEIDX),	mMaxPoints *sizeof(uint[BONDS_PER_PARTICLE *2]) ),     "TransferFromCUDA", "cuMemcpyDtoH", "FPARTICLEIDX", mbDebug);
    cuCheck( cuMemcpyDtoH ( m_Fluid.bufI(FPARTICLE_ID),	m_Fluid.gpu(FPARTICLE_ID),	mMaxPoints *sizeof(uint) ),                            "TransferFromCUDA", "cuMemcpyDtoH", "FPARTICLE_ID", mbDebug);
    cuCheck( cuMemcpyDtoH ( m_Fluid.bufI(FMASS_RADIUS),	m_Fluid.gpu(FMASS_RADIUS),	mMaxPoints *sizeof(uint) ),                            "TransferFromCUDA", "cuMemcpyDtoH", "FMASS_RADIUS", mbDebug);
    cuCheck( cuMemcpyDtoH ( m_Fluid.bufI(FNERVEIDX),	m_Fluid.gpu(FNERVEIDX),	    mMaxPoints *sizeof(uint) ),                            "TransferFromCUDA", "cuMemcpyDtoH", "FNERVEIDX",    mbDebug);
    cuCheck( cuMemcpyDtoH ( m_Fluid.bufF(FCONC),	    m_Fluid.gpu(FCONC),	        mMaxPoints *sizeof(float[NUM_TF]) ),                   "TransferFromCUDA", "cuMemcpyDtoH", "FCONC",        mbDebug);
    cuCheck( cuMemcpyDtoH ( m_Fluid.bufI(FEPIGEN),	    m_Fluid.gpu(FEPIGEN),	    mMaxPoints *sizeof(uint[NUM_GENES]) ),                 "TransferFromCUDA", "cuMemcpyDtoH", "FEPIGEN",      mbDebug);
}   // NB found FEPIGEN needed bufI and mMaxPoints, otherwise produced garbled files.

void FluidSystem::Init_FCURAND_STATE_CUDA (){ // designed to use to bootstrap itself. Set j=0 from host, call kernel repeatedly for 256^n threads, n=0-> to pnum threads.
    unsigned long long  seed=0; // sequence=0, offset=1,
    srand (time(NULL));
    for (int i=0;i<mNumPoints;i++){ // generate seeds
        seed = rand();
        //seed = seed << 32;
        //seed += rand();
        //seed = clock(); 
        m_Fluid.bufI(FCURAND_SEED)[i] = seed;
        //curand_init(seed, sequence, offset, &m_Fluid.bufCuRNDST(FCURAND_STATE)[i]);
        //if (m_FParams.debug>1)printf("\n(2:seed=%llu,(FCURAND_SEED)[i]=%llu, rand()=%u), ",seed, m_Fluid.bufULL(FCURAND_SEED)[i], rand() );
        //if (m_FParams.debug>1) cout<<"\t(seed="<<seed<<",(FCURAND_SEED)[i]="<<m_Fluid.bufI(FCURAND_SEED)[i]<<"), "<<std::flush;
       
    }
    // transfer to cuda
    //cuCheck( cuMemcpyDtoH ( gcell,	m_Fluid.gpu(FGCELL),	mNumPoints *sizeof(uint) ), "InsertParticlesCUDA", "cuMemcpyDtoH", "FGCELL", mbDebug );
    //cuCheck( cuMemcpyDtoH ( m_Fluid.bufCuRNDST(FCURAND_STATE),	m_Fluid.gpu(FCURAND_STATE),	mNumPoints *sizeof(curandState_t) ), 
    //         "Init_FCURAND_STATE_CUDA", "cuMemcpyDtoH", "FCURAND_STATE", mbDebug );
    
    cuCheck( cuMemcpyHtoD (m_Fluid.gpu(FCURAND_SEED), m_Fluid.bufC(FCURAND_SEED), mNumPoints *sizeof(uint) ), 
             "Init_FCURAND_STATE_CUDA", "cuMemcpyDtoH", "FCURAND_SEED", mbDebug );
    
    if (m_FParams.debug>1) std::cout <<"\nInit_FCURAND_STATE_CUDA_2.0\n\n"<<std::flush;
    /*
    int n=0;
    void* args[1] = {&n};
    int numBlocks_=1, numThreads_=1;
    if (m_FParams.debug>1) std::cout <<"\nInit_FCURAND_STATE_CUDA_1.0\t n="<<n<<",  pow(256,n)="<<pow(256,n)<<",  mNumPoints/256="<<mNumPoints/256<<",\t mNumPoints="<<mNumPoints<<", mMaxPoints="<<mMaxPoints<<"  \n"<<std::flush;
    
    do {
        computeNumBlocks ( pow(256,n), m_FParams.threadsPerBlock, numBlocks_, numThreads_);
        
        if (m_FParams.debug>1) std::cout <<"\nInit_FCURAND_STATE_CUDA_2.0\t n="<<n<<",  pow(256,n)="<<pow(256,n)<<",  mNumPoints/256="<<mNumPoints/256<<
        "\t numBlocks_="<<numBlocks_<<", numThreads_="<<numThreads_<<"  \n"<<std::flush;
        
        cuCheck(cuCtxSynchronize(), "Init_FCURAND_STATE_CUDA", "cuCtxSynchronize", "Before m_Func[FUNC_INIT_FCURAND_STATE], in do-while loop", 1);
        
        cuCheck ( cuLaunchKernel ( m_Func[FUNC_INIT_FCURAND_STATE],  numBlocks_, 1, 1, numThreads_, 1, 1, 0, NULL, args, NULL), "Init_FCURAND_STATE_CUDA", "cuLaunch", "FUNC_INIT_FCURAND_STATE", mbDebug);
        
        n++;
    } while (pow(256,n) < mNumPoints/256) ;
    
    if (m_FParams.debug>1) std::cout <<"\nInit_FCURAND_STATE_CUDA_3.0\t n="<<n<<",  pow(256,n)="<<pow(256,n)<<",  mNumPoints/256="<<mNumPoints/256<<
    "\t m_FParams.numBlocks="<<m_FParams.numBlocks<<",  m_FParams.numThreads="<<m_FParams.numThreads<<".  \n"<<std::flush;
        
    */
    void* args[1] = {&mNumPoints};
    
    cuCheck(cuCtxSynchronize(), "Init_FCURAND_STATE_CUDA", "cuCtxSynchronize", "Before m_Func[FUNC_INIT_FCURAND_STATE], after do-while loop", 1);
    
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_INIT_FCURAND_STATE],  m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL), "Init_FCURAND_STATE_CUDA", "cuLaunch", "FUNC_INIT_FCURAND_STATE", mbDebug);

    cuCheck(cuCtxSynchronize(), "Init_FCURAND_STATE_CUDA", "cuCtxSynchronize", "After cuMemcpyDtoH FCURAND_STATE, before 1st timestep", 1);
    if (m_FParams.debug>1) std::cout <<"\nInit_FCURAND_STATE_CUDA_4.0\n"<<std::flush;
    
}

void FluidSystem::InsertParticlesCUDA ( uint* gcell, uint* gndx, uint* gcnt ){   // first zero the counters
    cuCheck ( cuMemsetD8 ( m_Fluid.gpu(FGRIDCNT), 0,	m_GridTotal*sizeof(int) ), "InsertParticlesCUDA", "cuMemsetD8", "FGRIDCNT", mbDebug );
    cuCheck ( cuMemsetD8 ( m_Fluid.gpu(FGRIDOFF), 0,	m_GridTotal*sizeof(int) ), "InsertParticlesCUDA", "cuMemsetD8", "FGRIDOFF", mbDebug );
    
    cuCheck ( cuMemsetD8 ( m_Fluid.gpu(FGRIDCNT_ACTIVE_GENES), 0,	m_GridTotal *sizeof(uint[NUM_GENES]) ), "InsertParticlesCUDA", "cuMemsetD8", "FGRIDCNT", mbDebug );
    cuCheck ( cuMemsetD8 ( m_Fluid.gpu(FGRIDOFF_ACTIVE_GENES), 0,	m_GridTotal *sizeof(uint[NUM_GENES]) ), "InsertParticlesCUDA", "cuMemsetD8", "FGRIDOFF", mbDebug );
    
    // Set long list to sort all particles.
    computeNumBlocks ( m_FParams.pnum, m_FParams.threadsPerBlock, m_FParams.numBlocks, m_FParams.numThreads);				// particles
    // launch kernel "InsertParticles"
    void* args[1] = { &mMaxPoints };  //&mNumPoints
    cuCheck(cuLaunchKernel(m_Func[FUNC_INSERT], m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL),
            "InsertParticlesCUDA", "cuLaunch", "FUNC_INSERT", mbDebug);
    if (m_FParams.debug>1) cout<<"\n########\nCalling InsertParticles kernel: args[1] = {"<<mNumPoints<<"}, mMaxPoints="<<mMaxPoints
        <<"\t m_FParams.numBlocks="<<m_FParams.numBlocks<<", m_FParams.numThreads="<<m_FParams.numThreads<<" \t"<<std::flush;

    // Transfer data back if requested (for validation)
    if (gcell != 0x0) {
        cuCheck( cuMemcpyDtoH ( gcell,	m_Fluid.gpu(FGCELL),	mNumPoints *sizeof(uint) ), "InsertParticlesCUDA", "cuMemcpyDtoH", "FGCELL", mbDebug );
        cuCheck( cuMemcpyDtoH ( gndx,	m_Fluid.gpu(FGNDX),		mNumPoints *sizeof(uint) ), "InsertParticlesCUDA", "cuMemcpyDtoH", "FGNDX", mbDebug);
        cuCheck( cuMemcpyDtoH ( gcnt,	m_Fluid.gpu(FGRIDCNT),	m_GridTotal*sizeof(uint) ), "InsertParticlesCUDA", "cuMemcpyDtoH", "FGRIDCNT", mbDebug);
        cuCtxSynchronize ();
    }
    if(m_debug>4){
        if (m_FParams.debug>1) cout<<"\nSaving (FGCELL) InsertParticlesCUDA: (particleIdx, cell) , mMaxPoints="<<mMaxPoints<<"\t"<<std::flush;
        cuCheck( cuMemcpyDtoH ( m_Fluid.bufI(FGCELL), m_Fluid.gpu(FGCELL),	sizeof(uint[mMaxPoints]) ), "PrefixSumChangesCUDA", "cuMemcpyDtoH", "FGCELL", mbDebug);
        SaveUintArray( m_Fluid.bufI(FGCELL), mMaxPoints, "InsertParticlesCUDA__m_Fluid.bufI(FGCELL).csv" );
    }
}

void FluidSystem::PrefixSumCellsCUDA ( int zero_offsets ){
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
    if ( numElem1 > SCAN_BLOCKSIZE*xlong(SCAN_BLOCKSIZE)*SCAN_BLOCKSIZE) { if (m_FParams.debug>1)printf ( "\nERROR: Number of elements exceeds prefix sum max. Adjust SCAN_BLOCKSIZE.\n" );  }

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
    
    // Loop to PrefixSum the Dense Lists - NB by doing one gene at a time, we reuse the FAUX* arrays & scans.
    // For each gene, input FGRIDCNT_ACTIVE_GENES[gene*m_GridTotal], output FGRIDOFF_ACTIVE_GENES[gene*m_GridTotal]
    CUdeviceptr array0  = m_Fluid.gpu(FGRIDCNT_ACTIVE_GENES);
    CUdeviceptr scan0   = m_Fluid.gpu(FGRIDOFF_ACTIVE_GENES);

    for(int gene=0;gene<NUM_GENES;gene++){
      //if (m_FParams.debug>1) cout<<"\nPrefixSumCellsCUDA()1:gene="<<gene<<"\t"<<std::flush;
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
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_TALLYLISTS], NUM_GENES, 1, 1, NUM_GENES, 1, 1, 0, NULL, argsF, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_TALLYLISTS", mbDebug); //256 threads launched
    cuCheck( cuMemcpyDtoH ( m_Fluid.bufI(FDENSE_LIST_LENGTHS), m_Fluid.gpu(FDENSE_LIST_LENGTHS),	sizeof(uint[NUM_GENES]) ), "PrefixSumCellsCUDA", "cuMemcpyDtoH", "FDENSE_LIST_LENGTHS", mbDebug);
                                                                                    //if active particles for gene > existing buff, then enlarge buff.
    uint * densebuff_len = m_Fluid.bufI(FDENSE_BUF_LENGTHS);                    // and only m_Fluid.bufI(FDENSE_LIST_LENGTHS); copied to host.
    uint * denselist_len = m_Fluid.bufI(FDENSE_LIST_LENGTHS);                   // For each gene allocate intial buffer, 
    
    for(int gene=0;gene<NUM_GENES;gene++){                                          // Note this calculation could be done by a kernel, 
      //if (m_FParams.debug>1) cout<<"\nPrefixSumCellsCUDA()2:gene="<<gene<<", densebuff_len["<<gene<<"]="<<densebuff_len[gene]<<", denselist_len["<<gene<<"]="<<denselist_len[gene]<<" \t"<<std::flush;
        if (denselist_len[gene] > densebuff_len[gene]) {                            // write pointer and size to FDENSE_LISTS and FDENSE_LIST_LENGTHS 
            if (m_FParams.debug>1)printf("\n\nPrefixSumCellsCUDA: enlarging densebuff_len[%u],  m_Fluid.gpuptr(FDENSE_LIST_LENGTHS)[gene]=%llu .\t",gene, m_Fluid.gpuptr(FDENSE_LIST_LENGTHS)[gene] );
            while(denselist_len[gene] >  densebuff_len[gene]) densebuff_len[gene] *=4;                  // m_Fluid.bufI(FDENSE_BUF_LENGTHS)[i]
            AllocateBufferDenseLists( gene, sizeof(uint), m_Fluid.bufI(FDENSE_LIST_LENGTHS)[gene], FDENSE_LISTS );   // NB frees previous buffer &=> clears data
        }
    }
    cuMemcpyHtoD(m_Fluid.gpu(FDENSE_LISTS),         m_Fluid.bufC(FDENSE_LISTS),         NUM_GENES * sizeof(CUdeviceptr)  );  // update pointers to lists on device
    cuMemcpyHtoD(m_Fluid.gpu(FDENSE_BUF_LENGTHS),   m_Fluid.bufC(FDENSE_BUF_LENGTHS),   NUM_GENES * sizeof(CUdeviceptr)  );

    if (m_FParams.debug>1){ 
        std::cout << "\nChk: PrefixSumCellsCUDA 4"<<std::flush;
        for(int gene=0;gene<NUM_GENES;gene++){    std::cout<<"\ngene list_length["<<gene<<"]="<<m_Fluid.bufI(FDENSE_LIST_LENGTHS)[gene]<<"\t"<<std::flush;}
        }
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

    // Loop to PrefixSum the Dense Lists - NB by doing one change_list at a time, we reuse the FAUX* arrays & scans.
    // For each change_list, input FGRIDCNT_ACTIVE_GENES[change_list*m_GridTotal], output FGRIDOFF_ACTIVE_GENES[change_list*m_GridTotal]
    CUdeviceptr array0  = m_Fluid.gpu(FGRIDCNT_CHANGES);
    CUdeviceptr scan0   = m_Fluid.gpu(FGRIDOFF_CHANGES);
    
    if(m_debug>3){
        // debug chk
        cout<<"\nSaving (FGRIDCNT_CHANGES): (bin,#particles) , numElem1="<<numElem1<<"\t"<<std::flush;
        cuCheck( cuMemcpyDtoH ( m_Fluid.bufI(FGRIDCNT_CHANGES), m_Fluid.gpu(FGRIDCNT_CHANGES),	sizeof(uint[numElem1]) ), "PrefixSumChangesCUDA", "cuMemcpyDtoH", "FGRIDCNT_CHANGES", mbDebug); // NUM_CHANGES*
        //### print to a csv file   AND do the same afterwards for FGRIDOFF_CHANGES ###
        SaveUintArray( m_Fluid.bufI(FGRIDCNT_CHANGES), numElem1, "m_Fluid.bufI(FGRIDCNT_CHANGES).csv" );
        //
        cout<<"\nSaving (FGCELL): (particleIdx, cell) , mMaxPoints="<<mMaxPoints<<"\t"<<std::flush;
        cuCheck( cuMemcpyDtoH ( m_Fluid.bufI(FGCELL), m_Fluid.gpu(FGCELL),	sizeof(uint[mMaxPoints]) ), "PrefixSumChangesCUDA", "cuMemcpyDtoH", "FGCELL", mbDebug);
        SaveUintArray( m_Fluid.bufI(FGCELL), mMaxPoints, "m_Fluid.bufI(FGCELL).csv" );
        //
        //   cuCheck( cuMemcpyDtoH ( m_Fluid.bufI(FDENSE_LISTS), m_Fluid.gpu(FDENSE_LISTS),	sizeof(uint[mMaxPoints]) ), "PrefixSumChangesCUDA", "cuMemcpyDtoH", "FDENSE_LISTS", mbDebug);
        //   SaveUintArray( m_Fluid.bufI(FDENSE_LISTS), numElem1, "m_Fluid.bufI(FDENSE_LISTS).csv" );
    }
    cuCtxSynchronize ();

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
                  "PrefixSumChangesCUDA", "cuLaunch", "FUNC_PREFIXSUM", mbDebug);
        void* argsB[5] = { &array2, &scan2, &array3, &numElem2, &zon };             // sum array2. output -> scan2, array3.         i.e. FAUXARRAY1 -> FAUXSCAN1, FAUXARRAY2
        cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXSUM], numElem3, 1, 1, threads, 1, 1, 0, NULL, argsB, NULL ), "PrefixSumChangesCUDA", "cuLaunch", "FUNC_PREFIXSUM", mbDebug);
        if ( numElem3 > 1 ) {
            CUdeviceptr nptr = {0};
            void* argsC[5] = { &array3, &scan3, &nptr, &numElem3, &zon };	        // sum array3. output -> scan3                  i.e. FAUXARRAY2 -> FAUXSCAN2, &nptr
            cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXSUM], 1, 1, 1, threads, 1, 1, 0, NULL, argsC, NULL ), "PrefixSumChangesCUDA", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);
            void* argsD[3] = { &scan2, &scan3, &numElem2 };	                        // merge scan3 into scan2. output -> scan2      i.e. FAUXSCAN2, FAUXSCAN1 -> FAUXSCAN1
            cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXFIXUP], numElem3, 1, 1, threads, 1, 1, 0, NULL, argsD, NULL ), "PrefixSumChangesCUDA", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);
        }
        void* argsE[3] = { &scan1, &scan2, &numElem1 };		                        // merge scan2 into scan1. output -> scan1      i.e. FAUXSCAN1, FGRIDOFF -> FGRIDOFF
        cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXFIXUP], numElem2, 1, 1, threads, 1, 1, 0, NULL, argsE, NULL ), "PrefixSumChangesCUDA", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);
    }
    
    int num_lists = NUM_CHANGES, length = FDENSE_LIST_LENGTHS_CHANGES, fgridcnt = FGRIDCNT_CHANGES, fgridoff = FGRIDOFF_CHANGES;
    void* argsF[4] = {&num_lists, &length,&fgridcnt,&fgridoff};
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_TALLYLISTS], NUM_CHANGES, 1, 1, NUM_CHANGES, 1, 1, 0, NULL, argsF, NULL ), "PrefixSumChangesCUDA", "cuLaunch", "FUNC_TALLYLISTS", mbDebug); //256 threads launched
    cuCheck( cuMemcpyDtoH ( m_Fluid.bufI(FDENSE_LIST_LENGTHS_CHANGES), m_Fluid.gpu(FDENSE_LIST_LENGTHS_CHANGES),	sizeof(uint[NUM_CHANGES]) ), "PrefixSumChangesCUDA", "cuMemcpyDtoH", "FDENSE_LIST_LENGTHS_CHANGES", mbDebug);

                                                                                                                    // If active particles for change_list > existing buff, then enlarge buff.
    for(int change_list=0;change_list<NUM_CHANGES;change_list++){                                                   // Note this calculation could be done by a kernel, 
        uint * densebuff_len = m_Fluid.bufI(FDENSE_BUF_LENGTHS_CHANGES);                                            // and only m_Fluid.bufI(FDENSE_LIST_LENGTHS); copied to host.
        uint * denselist_len = m_Fluid.bufI(FDENSE_LIST_LENGTHS_CHANGES);                                           // For each change_list allocate intial buffer, 
        //if (m_FParams.debug>1)printf("\nPrefixSumChangesCUDA: change_list=%u,  densebuff_len[change_list]=%u, denselist_len[change_list]=%u ,",change_list, densebuff_len[change_list], denselist_len[change_list] );
        if (denselist_len[change_list] > densebuff_len[change_list]) {                                              // write pointer and size to FDENSE_LISTS and FDENSE_LIST_LENGTHS 
            while(denselist_len[change_list] >  densebuff_len[change_list])   densebuff_len[change_list] *=4;       // m_Fluid.bufI(FDENSE_BUF_LENGTHS)[i].  
                                                                                                                    // NB Need 2*densebuff_len[change_list] for particle & bond
            if (m_FParams.debug>1)printf("\nPrefixSumChangesCUDA: ## enlarging buffer## change_list=%u,  densebuff_len[change_list]=%u, denselist_len[change_list]=%u ,",change_list, densebuff_len[change_list], denselist_len[change_list] );
            AllocateBufferDenseLists( change_list, sizeof(uint), 2*densebuff_len[change_list], FDENSE_LISTS_CHANGES );// NB frees previous buffer &=> clears data
        }                                                                                                           // NB buf[2][list_length] holding : particleIdx, bondIdx
    }
    cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FDENSE_BUF_LENGTHS_CHANGES), m_Fluid.bufC(FDENSE_BUF_LENGTHS_CHANGES),	sizeof(uint[NUM_CHANGES]) ), "PrefixSumChangesCUDA", "cuMemcpyHtoD", "FDENSE_BUF_LENGTHS_CHANGES", mbDebug);
    cuMemcpyHtoD(m_Fluid.gpu(FDENSE_LISTS_CHANGES), m_Fluid.bufC(FDENSE_LISTS_CHANGES),  NUM_CHANGES * sizeof(CUdeviceptr)  );                      // update pointers to lists on device
    
    if (m_FParams.debug>1) {
        std::cout << "\nChk: PrefixSumChangesCUDA 4"<<std::flush;
        for(int change_list=0;change_list<NUM_CHANGES;change_list++){
            std::cout<<"\nPrefixSumChangesCUDA: change list_length["<<change_list<<"]="<<m_Fluid.bufI(FDENSE_LIST_LENGTHS_CHANGES)[change_list]<<"\t"<<std::flush;
        }
    }
}

void FluidSystem::CountingSortFullCUDA ( Vector3DF* ppos ){
    if (m_FParams.debug>1) std::cout << "\nCountingSortFullCUDA()1: mMaxPoints="<<mMaxPoints<<", mNumPoints="<<mNumPoints<<",\tmActivePoints="<<mActivePoints<<".\n"<<std::flush;
    // get number of active particles & set short lists for later kernels
    int grid_ScanMax = (m_FParams.gridScanMax.y * m_FParams.gridRes.z + m_FParams.gridScanMax.z) * m_FParams.gridRes.x + m_FParams.gridScanMax.x;
    
    cuCheck( cuMemcpyDtoH ( &mNumPoints,  m_Fluid.gpu(FGRIDOFF)+(m_GridTotal-1/*grid_ScanMax+1*/)*sizeof(int), sizeof(int) ), "CountingSortFullCUDA1", "cuMemcpyDtoH", "FGRIDOFF", mbDebug);
    
    cuCheck( cuMemcpyDtoH ( &mActivePoints,  m_Fluid.gpu(FGRIDOFF)+(grid_ScanMax/*-1*/)*sizeof(int), sizeof(int) ), "CountingSortFullCUDA2", "cuMemcpyDtoH", "FGRIDOFF", mbDebug);
    /*
    int totalPoints = 0;
    cuCheck( cuMemcpyDtoH ( &totalPoints,  m_Fluid.gpu(FGRIDOFF)+(m_GridTotal)*sizeof(int), sizeof(int) ), "CountingSortFullCUDA3", "cuMemcpyDtoH", "FGRIDOFF", mbDebug);
    std::cout<<"\nCountingSortFullCUDA(): totalPoints="<<totalPoints<<std::flush;
    */
    m_FParams.pnumActive = mActivePoints;                                     // TODO eliminate duplication of information & variables between fluid.h and fluid_system.h                               
    cuCheck ( cuMemcpyHtoD ( cuFParams,	&m_FParams, sizeof(FParams) ), "CountingSortFullCUDA3", "cuMemcpyHtoD", "cuFParams", mbDebug); // seems the safest way to update fparam.pnumActive on device.
    
    if (m_FParams.debug>1) std::cout<<"\nCountingSortFullCUDA()2: mMaxPoints="<<mMaxPoints<<" mNumPoints="<<mNumPoints<<",\tmActivePoints="<<mActivePoints<<",  m_GridTotal="<<m_GridTotal<<", grid_ScanMax="<<grid_ScanMax<<"\n"<<std::flush;

    // Transfer particle data to temp buffers
    //  (gpu-to-gpu copy, no sync needed)
    //TransferToTempCUDA ( FPOS,		mMaxPoints *sizeof(Vector3DF) );    // NB if some points have been removed, then the existing list is no longer dense,  
    //TransferToTempCUDA ( FVEL,		mMaxPoints *sizeof(Vector3DF) );    // hence must use mMaxPoints, not mNumPoints
    //TransferToTempCUDA ( FVEVAL,	mMaxPoints *sizeof(Vector3DF) );    // { Could potentially use (old_mNumPoints + mNewPoints) instead of mMaxPoints}
    TransferToTempCUDA ( FFORCE,	mMaxPoints *sizeof(Vector3DF) );    // NB buffers are declared and initialized on mMaxPoints.
    TransferToTempCUDA ( FPRESS,	mMaxPoints *sizeof(float) );
    TransferToTempCUDA ( FDENSITY,	mMaxPoints *sizeof(float) );
    TransferToTempCUDA ( FCLR,		mMaxPoints *sizeof(uint) );
    TransferToTempCUDA ( FAGE,		mMaxPoints *sizeof(uint) );
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

    // debug chk
    //cuCheck( cuMemcpyDtoH ( m_Fluid.bufI(FEPIGEN), m_FluidTemp.gpu(FEPIGEN),	mMaxPoints *sizeof(uint[NUM_GENES]) ), "CountingSortFullCUDA4", "cuMemcpyDtoH", "FGRIDCNT", mbDebug);
    //SaveUintArray_2D( m_Fluid.bufI(FEPIGEN), mMaxPoints, NUM_GENES, "CountingSortFullCUDA__m_FluidTemp.bufI(FEPIGEN)2.csv" );
    
    // reset bonds and forces in fbuf FELASTIDX, FPARTICLEIDX and FFORCE, required to prevent interference between time steps, 
    // because these are not necessarily overwritten by the FUNC_COUNTING_SORT kernel.
    cuCtxSynchronize ();    // needed to prevent colision with previous operations
    
    float max_pos = max(max(m_Vec[PVOLMAX].x, m_Vec[PVOLMAX].y), m_Vec[PVOLMAX].z);
    uint * uint_max_pos = (uint*)&max_pos;
    cuCheck ( cuMemsetD32 ( m_Fluid.gpu(FPOS), *uint_max_pos, mMaxPoints * 3 ),  "CountingSortFullCUDA", "cuMemsetD32", "FELASTIDX",   mbDebug);
    
    //cout<<"\nCountingSortFullCUDA: m_Vec[PVOLMAX]=("<<m_Vec[PVOLMAX].x<<", "<<m_Vec[PVOLMAX].y<<", "<<m_Vec[PVOLMAX].z<<"),  max_pos = "<< max_pos <<std::flush;
    // NB resetting  m_Fluid.gpu(FPOS)  ensures no zombie particles. ?hopefully?
    
    cuCheck ( cuMemsetD32 ( m_Fluid.gpu(FELASTIDX),    UINT_MAX,  mMaxPoints * BOND_DATA              ),  "CountingSortFullCUDA", "cuMemsetD32", "FELASTIDX",    mbDebug);
    cuCheck ( cuMemsetD32 ( m_Fluid.gpu(FPARTICLEIDX), UINT_MAX,  mMaxPoints * BONDS_PER_PARTICLE *2  ),  "CountingSortFullCUDA", "cuMemsetD32", "FPARTICLEIDX", mbDebug);
    cuCheck ( cuMemsetD32 ( m_Fluid.gpu(FPARTICLE_ID), UINT_MAX,  mMaxPoints                          ),  "CountingSortFullCUDA", "cuMemsetD32", "FPARTICLEIDX", mbDebug);
    cuCheck ( cuMemsetD32 ( m_Fluid.gpu(FFORCE),      (uint)0.0,  mMaxPoints * 3 /* ie num elements */),  "CountingSortFullCUDA", "cuMemsetD32", "FFORCE",       mbDebug);
    cuCheck ( cuMemsetD32 ( m_Fluid.gpu(FCONC),             0.0,  mMaxPoints * NUM_TF                 ),  "CountingSortFullCUDA", "cuMemsetD32", "FCONC",        mbDebug);
    cuCheck ( cuMemsetD32 ( m_Fluid.gpu(FEPIGEN),     (uint)0.0,  mMaxPoints * NUM_GENES              ),  "CountingSortFullCUDA", "cuMemsetD32", "FEPIGEN",      mbDebug);
    cuCtxSynchronize ();    // needed to prevent colision with previous operations

    // Reset grid cell IDs
    // cuCheck(cuMemsetD32(m_Fluid.gpu(FGCELL), GRID_UNDEF, numPoints ), "cuMemsetD32(Sort)");
    void* args[1] = { &mMaxPoints };
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_COUNTING_SORT], m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL),
              "CountingSortFullCUDA5", "cuLaunch", "FUNC_COUNTING_SORT", mbDebug );

    // Having sorted the particle data, we can start using a shortened list of particles.
    // NB have to reset to long list at start of time step. 
    computeNumBlocks ( m_FParams.pnumActive, m_FParams.threadsPerBlock, m_FParams.numBlocks, m_FParams.numThreads);				// particles
    
    if (m_FParams.debug>1) std::cout<<"\n CountingSortFullCUDA : FUNC_COUNT_SORT_LISTS\n"<<std::flush;
    // countingSortDenseLists ( int pnum ) // NB launch on bins not particles.
    int blockSize = SCAN_BLOCKSIZE/2 << 1; 
    int numElem1 = m_GridTotal;  
    int numElem2 = 2*  int( numElem1 / blockSize ) + 1;  
    int threads = SCAN_BLOCKSIZE/2;
    cuCtxSynchronize ();
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_COUNT_SORT_LISTS], /*m_FParams.numBlocks*/ numElem2, 1, 1, /*m_FParams.numThreads/2*/ threads , 1, 1, 0, NULL, args, NULL),
              "CountingSortFullCUDA7", "cuLaunch", "FUNC_COUNT_SORT_LISTS", mbDebug );                                   // NB threads/2 required on GTX970m
    cuCtxSynchronize ();
    
    if(m_FParams.debug>3){//debug chk
        std::cout<<"\n### Saving UintArray .csv files."<<std::flush;
        
        cuCheck( cuMemcpyDtoH ( m_Fluid.bufI(FEPIGEN), m_FluidTemp.gpu(FEPIGEN),	mMaxPoints *sizeof(uint[NUM_GENES]) ), "CountingSortFullCUDA8", "cuMemcpyDtoH", "FGRIDCNT", mbDebug);
        SaveUintArray_2D( m_Fluid.bufI(FEPIGEN), mMaxPoints, NUM_GENES, "CountingSortFullCUDA__m_FluidTemp.bufI(FEPIGEN)3.csv" );
        
        cuCheck( cuMemcpyDtoH ( m_Fluid.bufI(FEPIGEN), m_Fluid.gpu(FEPIGEN),	/*mMaxPoints*/mNumPoints *sizeof(uint[NUM_GENES]) ), "PrefixSumChangesCUDA", "cuMemcpyDtoH", "FGRIDCNT", mbDebug);
        SaveUintArray_2D( m_Fluid.bufI(FEPIGEN), mMaxPoints, NUM_GENES, "CountingSortFullCUDA__m_Fluid.bufI(FEPIGEN)3.csv" );
        
        cuCheck( cuMemcpyDtoH ( m_Fluid.bufI(FGRIDCNT), m_Fluid.gpu(FGRIDCNT),	sizeof(uint[m_GridTotal]) ), "CountingSortFullCUDA9", "cuMemcpyDtoH", "FGRIDCNT", mbDebug);
        SaveUintArray( m_Fluid.bufI(FGRIDCNT), m_GridTotal, "CountingSortFullCUDA__m_Fluid.bufI(FGRIDCNT).csv" );
        
        cuCheck( cuMemcpyDtoH ( m_Fluid.bufI(FGRIDOFF), m_Fluid.gpu(FGRIDOFF),	sizeof(uint[m_GridTotal]) ), "CountingSortFullCUDA10", "cuMemcpyDtoH", "FGRIDOFF", mbDebug);
        SaveUintArray( m_Fluid.bufI(FGRIDOFF), m_GridTotal, "CountingSortFullCUDA__m_Fluid.bufI(FGRIDOFF).csv" );
    
       // uint fDenseList2[100000];
       // CUdeviceptr*  _list2pointer = (CUdeviceptr*) &m_Fluid.bufC(FDENSE_LISTS)[2 * sizeof(CUdeviceptr)];
       // cuCheck( cuMemcpyDtoH ( fDenseList2, *_list2pointer,	sizeof(uint[ m_Fluid.bufI(FDENSE_LIST_LENGTHS)[2] ])/*sizeof(uint[2000])*/ ), "CountingSortFullCUDA11", "cuMemcpyDtoH", "FGRIDCNT", mbDebug);
       // SaveUintArray( fDenseList2, m_Fluid.bufI(FDENSE_LIST_LENGTHS)[2], "CountingSortFullCUDA__m_Fluid.bufII(FDENSE_LISTS)[2].csv" );
    }
}

void FluidSystem::CountingSortChangesCUDA ( ){
    if (m_FParams.debug>1) std::cout<<"\n\n#### CountingSortChangesCUDA ( )"<<std::flush;
    /* ////////
    cuCheck( cuMemcpyDtoH ( m_Fluid.bufI(FDENSE_LIST_LENGTHS_CHANGES), m_Fluid.gpu(FDENSE_LIST_LENGTHS_CHANGES),	sizeof(uint[NUM_CHANGES]) ), "PrefixSumCellsCUDA", "cuMemcpyDtoH", "FDENSE_LIST_LENGTHS_CHANGES", mbDebug);
    
                                                                                                                    // If active particles for change_list > existing buff, then enlarge buff.
    for(int change_list=0;change_list<NUM_CHANGES;change_list++){                                                   // Note this calculation could be done by a kernel, 
        uint * densebuff_len = m_Fluid.bufI(FDENSE_BUF_LENGTHS_CHANGES);                                            // and only m_Fluid.bufI(FDENSE_LIST_LENGTHS); copied to host.
        uint * denselist_len = m_Fluid.bufI(FDENSE_LIST_LENGTHS_CHANGES);                                           // For each change_list allocate intial buffer, 
        if (m_FParams.debug>1)printf("\nCountingSortChangesCUDA1: change_list=%u,  densebuff_len[change_list]=%u, denselist_len[change_list]=%u ,",change_list, densebuff_len[change_list], denselist_len[change_list] );
    }
    *//////////
    int blockSize = SCAN_BLOCKSIZE/2 << 1; 
    int numElem1 = m_GridTotal;  
    int numElem2 = 2* int( numElem1 / blockSize ) + 1;  
    int threads = SCAN_BLOCKSIZE/2;
    void* args[1] = { &mActivePoints };
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_COUNTING_SORT_CHANGES], numElem2, 1, 1, threads , 1, 1, 0, NULL, args, NULL),
              "CountingSortChangesCUDA", "cuLaunch", "FUNC_COUNTING_SORT_CHANGES", mbDebug );   
     /////////
    cuCheck(cuCtxSynchronize(), "CountingSortChangesCUDA()", "cuCtxSynchronize", "After FUNC_COUNTING_SORT_CHANGES", mbDebug);
    
    cuCheck( cuMemcpyDtoH ( m_Fluid.bufI(FDENSE_LIST_LENGTHS_CHANGES), m_Fluid.gpu(FDENSE_LIST_LENGTHS_CHANGES),	sizeof(uint[NUM_CHANGES]) ), "PrefixSumCellsCUDA", "cuMemcpyDtoH", "FDENSE_LIST_LENGTHS_CHANGES", mbDebug);
                                                                                                                    // If active particles for change_list > existing buff, then enlarge buff.
    for(int change_list=0;change_list<NUM_CHANGES;change_list++){                                                   // Note this calculation could be done by a kernel, 
        uint * densebuff_len = m_Fluid.bufI(FDENSE_BUF_LENGTHS_CHANGES);                                            // and only m_Fluid.bufI(FDENSE_LIST_LENGTHS); copied to host.
        uint * denselist_len = m_Fluid.bufI(FDENSE_LIST_LENGTHS_CHANGES);                                           // For each change_list allocate intial buffer, 
        if (m_FParams.debug>1)printf("\nCountingSortChangesCUDA2: change_list=%u,  densebuff_len[change_list]=%u, denselist_len[change_list]=%u ,\t\t threads=%u, numElem2=%u,  m_GridTotal=%u \t",
               change_list, densebuff_len[change_list], denselist_len[change_list], threads, numElem2,  m_GridTotal );
        cuCtxSynchronize ();
        if(m_FParams.debug>1){
            uint fDenseList2[1000000] = {UINT_MAX};//TODO make this array size safe!  NB 10* num particles.
            CUdeviceptr*  _list2pointer = (CUdeviceptr*) &m_Fluid.bufC(FDENSE_LISTS_CHANGES)[change_list*sizeof(CUdeviceptr)]; 
                                                                                                                // Get device pointer to FDENSE_LISTS_CHANGES[change_list].
            cuCheck( cuMemcpyDtoH ( fDenseList2, *_list2pointer,	2*sizeof(uint[densebuff_len[change_list]]) ), "PrefixSumChangesCUDA", "cuMemcpyDtoH", "FGRIDCNT", mbDebug);
            char filename[256];
            sprintf(filename, "CountingSortChangesCUDA__m_Fluid.bufII(FDENSE_LISTS_CHANGES)[%u].csv", change_list);
            SaveUintArray_2Columns( fDenseList2, denselist_len[change_list], densebuff_len[change_list], filename );
            ///
            printf("\n\n*_list2pointer=%llu",*_list2pointer);
            
        }
    }
}

void FluidSystem::InitializeBondsCUDA (){
    if (m_FParams.debug>1)cout << "\n\nInitializeBondsCUDA ()\n"<<std::flush;
    uint gene           = 1;                                                            // solid  (has springs)
    uint list_length    = m_Fluid.bufI(FDENSE_LIST_LENGTHS)[gene];
    void* args[3]       = { &m_FParams.pnumActive, &list_length, &gene};                //initialize_bonds (int ActivePoints, uint list_length, uint gene)
    int numBlocks, numThreads;
    computeNumBlocks (list_length, m_FParams.threadsPerBlock, numBlocks, numThreads);
    
    if (m_FParams.debug>1)cout << "\nInitializeBondsCUDA (): list_length="<<list_length<<", m_FParams.threadsPerBlock="<<m_FParams.threadsPerBlock<<", numBlocks="<<numBlocks<<", numThreads="<<numThreads<<" \t args{m_FParams.pnumActive="<<m_FParams.pnumActive<<", list_length="<<list_length<<", gene="<<gene<<"}"<<std::flush;
    
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_INITIALIZE_BONDS],  m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL), "ComputePressureCUDA", "cuLaunch", "FUNC_COMPUTE_PRESS", mbDebug);
}

void FluidSystem::ComputePressureCUDA (){
    void* args[1] = { &mActivePoints };
    //cout<<"\nComputePressureCUDA: mActivePoints="<<mActivePoints<<std::flush;
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_COMPUTE_PRESS],  m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL), "ComputePressureCUDA", "cuLaunch", "FUNC_COMPUTE_PRESS", mbDebug);
}

void FluidSystem::ComputeDiffusionCUDA(){
    //if (m_FParams.debug>1) std::cout << "\n\nRunning ComputeDiffusionCUDA()" << std::endl;
    void* args[1] = { &mActivePoints };
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_COMPUTE_DIFFUSION],  m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL), "ComputeDiffusionCUDA", "cuLaunch", "FUNC_COMPUTE_DIFFUSION", mbDebug);
}

void FluidSystem::ComputeForceCUDA (){
    //if (m_FParams.debug>1)printf("\n\nFluidSystem::ComputeForceCUDA (),  m_FParams.freeze=%s",(m_FParams.freeze==true) ? "true" : "false");
    void* args[3] = { &m_FParams.pnumActive ,  &m_FParams.freeze, &m_FParams.frame};
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_COMPUTE_FORCE],  m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL), "ComputeForceCUDA", "cuLaunch", "FUNC_COMPUTE_FORCE", mbDebug);
}

void FluidSystem::ComputeGenesCUDA (){  // for each gene, call a kernel wih the dese list for that gene
    // NB Must zero ftemp.bufI(FEPIGEN) and ftemp.bufI(FCONC) before calling kernel. ftemp is used to collect changes before FUNC_TALLY_GENE_ACTION.
    cuCheck ( cuMemsetD8 ( m_FluidTemp.gpu(FCONC),   0,	m_FParams.szPnts *sizeof(float[NUM_TF])   ), "ComputeGenesCUDA", "cuMemsetD8", "m_FluidTemp.gpu(FCONC)",   mbDebug );
    cuCheck ( cuMemsetD8 ( m_FluidTemp.gpu(FEPIGEN), 0,	m_FParams.szPnts *sizeof(uint[NUM_GENES]) ), "ComputeGenesCUDA", "cuMemsetD8", "m_FluidTemp.gpu(FEPIGEN)", mbDebug );
    for (int gene=0;gene<NUM_GENES;gene++) {
        uint list_length = m_Fluid.bufI(FDENSE_LIST_LENGTHS)[gene];
        void* args[3] = { &m_FParams.pnumActive, &gene, &list_length };
        int numBlocks, numThreads;
        computeNumBlocks ( list_length , m_FParams.threadsPerBlock, numBlocks, numThreads);
        
        if (m_FParams.debug>1) std::cout<<"\nComputeGenesCUDA (): gene ="<<gene<<", list_length="<<list_length<<", m_FParams.threadsPerBlock="<<m_FParams.threadsPerBlock<<", numBlocks="<<numBlocks<<",  numThreads="<<numThreads<<". args={mNumPoints="<<mNumPoints<<", list_length="<<list_length<<", gene ="<<gene<<"}"<<std::flush;
        
        if( numBlocks>0 && numThreads>0){
            //std::cout<<"\nCalling m_Func[FUNC_COMPUTE_GENE_ACTION], list_length="<<list_length<<", numBlocks="<<numBlocks<<", numThreads="<<numThreads<<"\n"<<std::flush;
            cuCheck ( cuLaunchKernel ( m_Func[FUNC_COMPUTE_GENE_ACTION],  numBlocks, 1, 1, numThreads, 1, 1, 0, NULL, args, NULL), "ComputeGenesCUDA", "cuLaunch", "FUNC_COMPUTE_GENE_ACTION", mbDebug);
        }
    }
    cuCheck(cuCtxSynchronize(), "ComputeGenesCUDA", "cuCtxSynchronize", "After FUNC_COMPUTE_GENE_ACTION & before FUNC_TALLY_GENE_ACTION", mbDebug);
    for (int gene=0;gene<NUM_GENES;gene++) {
        uint list_length = m_Fluid.bufI(FDENSE_LIST_LENGTHS)[gene];
        void* args[3] = { &m_FParams.pnumActive, &gene, &list_length };
        int numBlocks, numThreads;
        computeNumBlocks ( list_length , m_FParams.threadsPerBlock, numBlocks, numThreads);
        
        if( numBlocks>0 && numThreads>0){
            if (m_FParams.debug>1) std::cout<<"\nCalling m_Func[FUNC_TALLY_GENE_ACTION], gene="<<gene<<", list_length="<<list_length<<", numBlocks="<<numBlocks<<", numThreads="<<numThreads<<"\n"<<std::flush;
            cuCheck ( cuLaunchKernel ( m_Func[FUNC_TALLY_GENE_ACTION],  numBlocks, 1, 1, numThreads, 1, 1, 0, NULL, args, NULL), "ComputeGenesCUDA", "cuLaunch", "FUNC_TALLY_GENE_ACTION", mbDebug);
        }
    }
}

void FluidSystem::AssembleFibresCUDA (){  //kernel: void assembleMuscleFibres ( int pnum, uint list, uint list_length )
    if (m_FParams.debug>1)cout << "\n\nAssembleFibresCUDA ()\n"<<std::flush;
    uint gene = 7; // muscle
    uint list_length = m_Fluid.bufI(FDENSE_LIST_LENGTHS)[gene];
    void* args[3] = { &m_FParams.pnumActive, &gene, &list_length };
    int numBlocks, numThreads;
    computeNumBlocks ( list_length , m_FParams.threadsPerBlock, numBlocks, numThreads);
    
    /*
    if( numBlocks>0 && numThreads>0){
        cuCheck ( cuLaunchKernel ( m_Func[FUNC_ASSEMBLE_MUSCLE_FIBRES_OUTGOING],  numBlocks, 1, 1, numThreads, 1, 1, 0, NULL, args, NULL), "ComputeGenesCUDA", "cuLaunch", "FUNC_COMPUTE_GENE_ACTION", mbDebug);
    }
    */
    
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "In AssembleFibresCUDA, after OUTGOING", mbDebug); 
    
    /*
    if( numBlocks>0 && numThreads>0){
        cuCheck ( cuLaunchKernel ( m_Func[FUNC_ASSEMBLE_MUSCLE_FIBRES_INCOMING],  numBlocks, 1, 1, numThreads, 1, 1, 0, NULL, args, NULL), "ComputeGenesCUDA", "cuLaunch", "FUNC_COMPUTE_GENE_ACTION", mbDebug);
    }
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "In AssembleFibresCUDA, after OUTGOING", mbDebug); 
    */
    

    
    
    // Kernels:  call by tissue type using dense lists by gene.
    //assembleMuscleFibres()
    //assembleFasciaFibres ()
    if (m_FParams.debug>1) cout << "\nFinished AssembleFibresCUDA ()\n\n"<<std::flush;
}

void FluidSystem::ComputeBondChangesCUDA (uint steps_per_InnerPhysicalLoop){// Given the action of the genes, compute the changes to particle properties & splitting/combining  NB also "inserts changes" 
//  if (m_FParams.debug>1)printf("\n m_Fluid.gpu(FGRIDOFF_CHANGES)=%llu   ,\t m_Fluid.gpu(FGRIDCNT_CHANGES)=%llu   \n",m_Fluid.gpu(FGRIDOFF_CHANGES) , m_Fluid.gpu(FGRIDCNT_CHANGES)   );
  
    cuCheck ( cuMemsetD8 ( m_Fluid.gpu(FGRIDOFF_CHANGES), 0,	m_GridTotal *sizeof(uint[NUM_CHANGES]) ), "ComputeBondChangesCUDA", "cuMemsetD8", "FGRIDOFF", mbDebug );
                                            //NB list for all living cells. (non senescent) = FEPIGEN[2]
    cuCheck ( cuMemsetD8 ( m_Fluid.gpu(FGRIDCNT_CHANGES), 0,	m_GridTotal *sizeof(uint[NUM_CHANGES]) ), "ComputeBondChangesCUDA", "cuMemsetD8", "FGRIDCNT", mbDebug );
    
    uint list_length = m_Fluid.bufI(FDENSE_LIST_LENGTHS)[2];    // call for dense list of living cells (gene'2'living/telomere (has genes))
    void* args[3] = { &mActivePoints, &list_length, &steps_per_InnerPhysicalLoop};
    int numBlocks, numThreads;
    computeNumBlocks (list_length, m_FParams.threadsPerBlock, numBlocks, numThreads);
    
    //std::cout<<"\n\nComputeBondChangesCUDA (): m_FParams.debug = "<<m_FParams.debug<<", (m_FParams.debug>1)="<<(m_FParams.debug>1)<<"\n"<<std::flush;
    
    if (m_FParams.debug>1) std::cout<<"\n\nComputeBondChangesCUDA (): list_length="<<list_length<<", m_FParams.threadsPerBlock="<<m_FParams.threadsPerBlock<<", numBlocks="<<numBlocks<<",  numThreads="<<numThreads<<". \t\t args={mActivePoints="<<mActivePoints<<", list_length="<<list_length<<"}\n\n"<<std::flush;
    
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_COMPUTE_BOND_CHANGES],  m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL), "computeBondChanges", "cuLaunch", "FUNC_COMPUTE_BOND_CHANGES", mbDebug);
}

void FluidSystem::ComputeParticleChangesCUDA (){// Call each for dense list to execute particle changes. NB Must run concurrently without interfering => no cuCtxSynchronize()
    uint startNewPoints = mActivePoints + 1;
    if (m_FParams.debug>2)printf("\n");
    for (int change_list = 0; change_list<NUM_CHANGES;change_list++){
    //int change_list = 0; // TODO debug, chk one kernel at a time
        uint list_length = m_Fluid.bufI(FDENSE_LIST_LENGTHS_CHANGES)[change_list];  // num blocks and threads by list length
        //uint list_length = m_Fluid.bufI(FDENSE_BUF_LENGTHS_CHANGES)[change_list]; 
        //if (change_list!=0 && change_list!=1)continue; // only test heal() and lengthenTissue() for now.
        
        if (m_FParams.debug>2)printf("\n\nComputeParticleChangesCUDA(): startNewPoints=%u, change_list=%u, list_length=%u, mMaxPoints=%u \t", 
            startNewPoints, change_list, list_length, mMaxPoints); 
    
        if ((change_list >0)&&(startNewPoints + list_length > mMaxPoints)){         // NB heal() does not create new bonds.
            printf("\n\n### Run out of spare particles. startNewPoints=%u, change_list=%u, list_length=%u, mMaxPoints=%u ###\n", 
            startNewPoints, change_list, list_length, mMaxPoints); 
            list_length = mMaxPoints - startNewPoints;
            Exit();
        }//
    
        void* args[5] = {&mActivePoints, &list_length, &change_list, &startNewPoints, &mMaxPoints};
        int numThreads, numBlocks;
        
        //int numThreads = 1;//m_FParams.threadsPerBlock;
        //int numBlocks  = 1;//iDivUp ( list_length, numThreads );
        
        computeNumBlocks (list_length, m_FParams.threadsPerBlock, numBlocks, numThreads);
        
        if (m_FParams.debug>2) std::cout
            <<"\nComputeParticleChangesCUDA ():"
            <<" frame ="                    <<m_FParams.frame
            <<", mActivePoints="            <<mActivePoints
            <<", change_list ="             <<change_list
            <<", list_length="              <<list_length
            <<", m_FParams.threadsPerBlock="<<m_FParams.threadsPerBlock
            <<", numBlocks="                <<numBlocks
            <<", numThreads="               <<numThreads
            <<". args={mActivePoints="      <<mActivePoints
            <<", list_length="              <<list_length
            <<", change_list="              <<change_list
            <<", startNewPoints="           <<startNewPoints
            <<"\t"<<std::flush;
        
        if( (list_length>0) && (numBlocks>0) && (numThreads>0)){
            if (m_FParams.debug>0) std::cout
                <<"\nComputeParticleChangesCUDA ():"
                <<"\tCalling m_Func[FUNC_HEAL+"           <<change_list
                <<"], list_length="                         <<list_length
                <<", numBlocks="                            <<numBlocks
                <<", numThreads="                           <<numThreads
                <<",\t m_FParams.threadsPerBlock="          <<m_FParams.threadsPerBlock
                <<", numBlocks*m_FParams.threadsPerBlock="  <<numBlocks*m_FParams.threadsPerBlock
                <<"\t"<<std::flush;
            
            cuCheck ( cuLaunchKernel ( m_Func[FUNC_HEAL+change_list], numBlocks, 1, 1, numThreads, 1, 1, 0, NULL, args, NULL), 
                  "ComputeParticleChangesCUDA", "cuLaunch", "FUNC_HEAL+change_list", mbDebug);
        }
        cuCheck(cuCtxSynchronize(), "ComputeParticleChangesCUDA", "cuCtxSynchronize", "In ComputeParticleChangesCUDA", mbDebug);
                                                                                // Each thread will pick different new particles from surplus particles.
        if (change_list==2 || change_list==6) startNewPoints+=  list_length;    // Increment by num new particles used by previous kernels. 
        //if (change_list==1 || change_list==5) startNewPoints+=  list_length*3;  // Increment by 3 particles for muscle.    
        /*
    0   #define FUNC_HEAL                       23 //heal
    1   #define FUNC_LENGTHEN_MUSCLE            24 //lengthen_muscle
    2   #define FUNC_LENGTHEN_TISSUE            25 //lengthen_tissue
    3   #define FUNC_SHORTEN_MUSCLE             26 //shorten_muscle
    4   #define FUNC_SHORTEN_TISSUE             27 //shorten_tissue
    
    5   #define FUNC_STRENGTHEN_MUSCLE          28 //strengthen_muscle
    6   #define FUNC_STRENGTHEN_TISSUE          29 //strengthen_tissue
    7   #define FUNC_WEAKEN_MUSCLE              30 //weaken_muscle
    8   #define FUNC_WEAKEN_TISSUE              31 //weaken_tissue
         */
    }
    if (m_FParams.debug>1) std::cout<<"\nFinished ComputeParticleChangesCUDA ()\n"<<std::flush;
}

void FluidSystem::CleanBondsCUDA (){
    void* args[3] = { &m_FParams.pnumActive};
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_CLEAN_BONDS],  m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL), "CleanBondsCUDA", "cuLaunch", "FUNC_CLEAN_BONDS", mbDebug);
}

void FluidSystem::TransferPosVelVeval (){
    TransferToTempCUDA ( FPOS,		mMaxPoints *sizeof(Vector3DF) );    // NB if some points have been removed, then the existing list is no longer dense,  
    TransferToTempCUDA ( FVEL,		mMaxPoints *sizeof(Vector3DF) );    // hence must use mMaxPoints, not mNumPoints
    TransferToTempCUDA ( FVEVAL,	mMaxPoints *sizeof(Vector3DF) );
}

void FluidSystem::TransferPosVelVevalFromTemp (){
    TransferFromTempCUDA ( FPOS,	mMaxPoints *sizeof(Vector3DF) );    // NB if some points have been removed, then the existing list is no longer dense,  
    TransferFromTempCUDA ( FVEL,	mMaxPoints *sizeof(Vector3DF) );    // hence must use mMaxPoints, not mNumPoints
    TransferFromTempCUDA ( FVEVAL,	mMaxPoints *sizeof(Vector3DF) );
}

void FluidSystem::AdvanceCUDA ( float tm, float dt, float ss ){
    void* args[4] = { &tm, &dt, &ss, &m_FParams.pnumActive };
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_ADVANCE],  m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL), "AdvanceCUDA", "cuLaunch", "FUNC_ADVANCE", mbDebug);
    //cout<<"\nAdvanceCUDA: m_FParams.pnumActive="<<m_FParams.pnumActive<<std::flush;
}

void FluidSystem::SpecialParticlesCUDA (float tm, float dt, float ss){   // For interaction.Using dense lists for gene 1 & 0.
    int gene = 12;                                                           // 'externally actuated' particles
    uint list_length = m_Fluid.bufI(FDENSE_LIST_LENGTHS)[gene];
    void* args[5] = {&list_length, &tm, &dt, &ss, &m_FParams.pnumActive};         // void externalActuation (uint list_len,  float time, float dt, float ss, int numPnts )
    int numBlocks, numThreads;
    computeNumBlocks ( list_length , m_FParams.threadsPerBlock, numBlocks, numThreads);
    
    if (m_FParams.debug>1) std::cout<<"\nSpecialParticlesCUDA:EXTERNAL_ACTUATION: list_length="<<list_length<<" , m_FParams.threadsPerBlock="<< m_FParams.threadsPerBlock <<", numBlocks="<< numBlocks <<", numThreads="<< numThreads <<", args{m_FParams.pnum="<< m_FParams.pnum <<",  gene="<< gene <<", list_length="<< list_length <<"  }  \n"<<std::flush;
    
    if( numBlocks>0 && numThreads>0){
        if (m_FParams.debug>1) std::cout<<"\nCalling m_Func[FUNC_EXTERNAL_ACTUATION], list_length="<<list_length<<", numBlocks="<<numBlocks<<", numThreads="<<numThreads<<"\n"<<std::flush;
        cuCheck ( cuLaunchKernel ( m_Func[FUNC_EXTERNAL_ACTUATION],  numBlocks, 1, 1, numThreads, 1, 1, 0, NULL, args, NULL), "SpecialParticlesCUDA", "cuLaunch", "FUNC_EXTERNAL_ACTUATION", mbDebug);
    }
    gene =11;                                                                // 'fixed' particles
    list_length = m_Fluid.bufI(FDENSE_LIST_LENGTHS)[gene];
    args[0] = &list_length;                                                 // void fixedParticles (uint list_len, int numPnts )
    args[1] = &m_FParams.pnum;
    computeNumBlocks ( list_length , m_FParams.threadsPerBlock, numBlocks, numThreads);
    
    if (m_FParams.debug>1) std::cout<<"\nSpecialParticlesCUDA:FIXED: list_length="<<list_length<<" , m_FParams.threadsPerBlock="<< m_FParams.threadsPerBlock <<", numBlocks="<< numBlocks <<", numThreads="<< numThreads <<", args{m_FParams.pnum="<< m_FParams.pnum <<",  gene="<< gene <<", list_length="<< list_length <<"  }  \n"<<std::flush;
    
    if( numBlocks>0 && numThreads>0){
        if (m_FParams.debug>1) std::cout<<"\nCalling m_Func[FUNC_FIXED], list_length="<<list_length<<", numBlocks="<<numBlocks<<", numThreads="<<numThreads<<"\n"<<std::flush;
        cuCheck ( cuLaunchKernel ( m_Func[FUNC_FIXED],  numBlocks, 1, 1, numThreads, 1, 1, 0, NULL, args, NULL), "SpecialParticlesCUDA", "cuLaunch", "FUNC_FIXED", mbDebug);
    }
}

void FluidSystem::EmitParticlesCUDA ( float tm, int cnt ){
    void* args[3] = { &tm, &cnt, &m_FParams.pnum };
    cuCheck ( cuLaunchKernel ( m_Func[FUNC_EMIT],  m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL), "EmitParticlesCUDA", "cuLaunch", "FUNC_EMIT", mbDebug);
}

