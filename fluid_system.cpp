
#include <assert.h>
#include <iostream>//<stdio.h>
#include <cuda.h>	
#include "cutil_math.h"
#include "fluid_system.h"

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
	mPackBuf = 0x0;
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
	char cfn[512];		strcpy ( cfn, func.c_str() );

	if ( m_Func[fid] == (CUfunction) -1 )
		cuCheck ( cuModuleGetFunction ( &m_Func[fid], m_Module, cfn ), "LoadKernel", "cuModuleGetFunction", cfn, mbDebug );	
}

// Must have a CUDA context to initialize
void FluidSystem::Initialize ()     // sets up simulation parameters etc.
{
    std::cout << "FluidSystem::Initialize () \n";
	cuCheck ( cuModuleLoad ( &m_Module, "fluid_system_cuda.ptx" ), "LoadKernel", "cuModuleLoad", "fluid_system_cuda.ptx", mbDebug);  // loads the file "fluid_system_cuda.ptx" as a module with pointer  m_Module.

    std::cout << "Chk1.1 \n";
    
	LoadKernel ( FUNC_INSERT,			"insertParticles" );
	LoadKernel ( FUNC_COUNTING_SORT,	"countingSortFull" );
	LoadKernel ( FUNC_QUERY,			"computeQuery" );
	LoadKernel ( FUNC_COMPUTE_PRESS,	"computePressure" );
	LoadKernel ( FUNC_COMPUTE_FORCE,	"computeForce" );
	LoadKernel ( FUNC_ADVANCE,			"advanceParticles" );
	LoadKernel ( FUNC_EMIT,				"emitParticles" );
	LoadKernel ( FUNC_RANDOMIZE,		"randomInit" );
	LoadKernel ( FUNC_SAMPLE,			"sampleParticles" );
	LoadKernel ( FUNC_FPREFIXSUM,		"prefixSum" );
	LoadKernel ( FUNC_FPREFIXFIXUP,		"prefixFixup" );

    std::cout << "Chk1.2 \n";
    
	size_t len = 0;
	cuCheck ( cuModuleGetGlobal ( &cuFBuf, &len,		m_Module, "fbuf" ),		"LoadKernel", "cuModuleGetGlobal", "cuFBuf", mbDebug);      // Returns a global pointer (cuFBuf) from a module  (m_Module), see line 81.
	cuCheck ( cuModuleGetGlobal ( &cuFTemp, &len,		m_Module, "ftemp" ),	"LoadKernel", "cuModuleGetGlobal", "cuFTemp", mbDebug);     // fbuf, ftemp, fparam are defined in fluid_system_cuda.cu,
	cuCheck ( cuModuleGetGlobal ( &cuFParams, &len,	m_Module, "fparam" ),		"LoadKernel", "cuModuleGetGlobal", "cuFParams", mbDebug);   // based on structs defined in fluid.h 
                                                                                                                                            // NB defined differently in kernel vs cpu code.
                                                                                                                                            // An FBufs struct holds an array of pointers. 
    std::cout << "Chk1.3 \n";
    
	// Clear all buffers
	memset ( &m_Fluid, 0,		sizeof(FBufs) );
	memset ( &m_FluidTemp, 0,	sizeof(FBufs) );
	memset ( &m_FParams, 0,		sizeof(FParams) );
    
    std::cout << "Chk1.4 \n";
    
	//m_Param [ PMODE ]		= RUN_VALIDATE;			// debugging
	m_Param [ PMODE ]		= RUN_GPU_FULL;		
	m_Param [ PEXAMPLE ]	= 2;            //0=Regression test. N x N x N static grid, 1=Tower , 
                                            //2=Wave pool , 3=Small dam break , 4=Dual-Wave pool , 5=Microgravity . See  FluidSystem::SetupExampleParams ().
	m_Param [ PGRID_DENSITY ] = 2.0;
	m_Param [ PNUM ]		= 65536 * 128;

    std::cout << "Chk1.5 \n";
    
	// Allocate the sim parameters
	AllocateBuffer ( FPARAMS,		sizeof(FParams),	0,	1,					 GPU_SINGLE, CPU_OFF );
    std::cout << "Chk1.6 \n";
}

void FluidSystem::Start ( int num )     // #### creates the particles ####
{
    std::cout << "FluidSystem::Start ( "<< num <<" ) \n";
    
	#ifdef TEST_PREFIXSUM
		TestPrefixSum ( 16*1024*1024 );		
		exit(-2);
	#endif

	m_Time = 0;

	ClearNeighborTable ();
	mNumPoints = 0;			// reset count
	
	SetupDefaultParams ();	
	SetupExampleParams ();	
	m_Param[PNUM] = (float) num;	// maximum number of points
	mMaxPoints = num;

	m_Param [PGRIDSIZE] = 2*m_Param[PSMOOTHRADIUS] / m_Param[PGRID_DENSITY];	

	// Setup stuff
	AllocatePackBuf ();
	
	SetupKernels ();
	
	SetupSpacing ();

	SetupGrid ( m_Vec[PVOLMIN], m_Vec[PVOLMAX], m_Param[PSIMSCALE], m_Param[PGRIDSIZE], 1.0f );	// Setup grid
		
	FluidSetupCUDA ( mMaxPoints, m_GridSrch, *(int3*)& m_GridRes, *(float3*)& m_GridSize, *(float3*)& m_GridDelta, *(float3*)& m_GridMin, *(float3*)& m_GridMax, m_GridTotal, 0 );

	UpdateParams();

	// Allocate data
	AllocateParticles( mMaxPoints );

	AllocateGrid();

	// Create the particles (after allocate)
	SetupAddVolume(m_Vec[PINITMIN], m_Vec[PINITMAX], m_Param[PSPACING], 0.1f, (int)m_Param[PNUM]);		// increases mNumPoints

	TransferToCUDA ();		 // Initial transfer
}
/////////////////////////////////////////////////////////////////


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
	// Free fluid buffers
	for (int n=0; n < MAX_BUF; n++ ) {
		if ( m_Fluid.bufC(n) != 0x0 )
			free ( m_Fluid.bufC(n) );
	}

	//cudaExit ();
}

void FluidSystem::AllocateBuffer ( int buf_id, int stride, int cpucnt, int gpucnt, int gpumode, int cpumode )   // mallocs a buffer
{
	if (cpumode == CPU_YES) {
		char* src_buf = m_Fluid.bufC(buf_id);
		char* dest_buf = (char*) malloc(cpucnt*stride);                   //  ####  malloc the buffer   ####
		if (src_buf != 0x0) {
			memcpy(dest_buf, src_buf, cpucnt*stride);
			free(src_buf);
		}
		m_Fluid.setBuf(buf_id, dest_buf);
	}
	if (gpumode == GPU_SINGLE || gpumode == GPU_DUAL )	{
		if (m_Fluid.gpuptr(buf_id) != 0x0) cuCheck(cuMemFree(m_Fluid.gpu(buf_id)), "AllocateBuffer", "cuMemFree", "Fluid.gpu", mbDebug);
		cuCheck( cuMemAlloc(m_Fluid.gpuptr(buf_id), stride*gpucnt), "AllocateBuffer", "cuMemAlloc", "Fluid.gpu", mbDebug);
	}
	if (gpumode == GPU_TEMP || gpumode == GPU_DUAL ) {
		if (m_FluidTemp.gpuptr(buf_id) != 0x0) cuCheck(cuMemFree(m_FluidTemp.gpu(buf_id)), "AllocateBuffer", "cuMemFree", "FluidTemp.gpu", mbDebug);
		cuCheck( cuMemAlloc(m_FluidTemp.gpuptr(buf_id), stride*gpucnt), "AllocateBuffer", "cuMemAlloc", "FluidTemp.gpu", mbDebug);
	}
}

// Allocate particle memory
void FluidSystem::AllocateParticles ( int cnt )                         // calls AllocateBuffer(..) for each buffer.  Called by FluidSystem::Start(..), cnt = mMaxPoints.
{
	AllocateBuffer ( FPOS,		sizeof(Vector3DF),	cnt,	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
	AllocateBuffer ( FCLR,		sizeof(uint),		cnt,	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
	AllocateBuffer ( FVEL,		sizeof(Vector3DF),	cnt,	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
	AllocateBuffer ( FVEVAL,	sizeof(Vector3DF),	cnt,	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
	AllocateBuffer ( FAGE,		sizeof(unsigned short), cnt,m_FParams.szPnts,	GPU_DUAL, CPU_YES );
	AllocateBuffer ( FPRESS,	sizeof(float),		cnt,	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
	AllocateBuffer ( FDENSITY,	sizeof(float),		cnt, 	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
	AllocateBuffer ( FFORCE,	sizeof(Vector3DF),	cnt,	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
	AllocateBuffer ( FCLUSTER,	sizeof(uint),		cnt,	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
	AllocateBuffer ( FGCELL,	sizeof(uint),		cnt,	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
	AllocateBuffer ( FGNDX,		sizeof(uint),		cnt,	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
	AllocateBuffer ( FGNEXT,	sizeof(uint),		cnt,	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
	AllocateBuffer ( FNBRNDX,	sizeof(uint),		cnt,	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
	AllocateBuffer ( FNBRCNT,	sizeof(uint),		cnt,	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
	AllocateBuffer ( FSTATE,	sizeof(uint),		cnt,	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
    // extra buffers for morphogenesis
    AllocateBuffer ( FELASTIDX,	sizeof(uint[BONDS_PER_PARTICLE +1]), cnt,   m_FParams.szPnts,	GPU_DUAL, CPU_YES );
    AllocateBuffer ( FNERVEIDX,	sizeof(uint),		                 cnt,	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
    AllocateBuffer ( FCONC,	    sizeof(uint[NUM_TF]),		         cnt,	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
    AllocateBuffer ( FEPIGEN,	sizeof(uint[NUM_GENES]),	         cnt,	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
    
	
	// Update GPU access pointers
	cuCheck( cuMemcpyHtoD(cuFBuf, &m_Fluid, sizeof(FBufs)),			"AllocateParticles", "cuMemcpyHtoD", "cuFBuf", mbDebug);
	cuCheck( cuMemcpyHtoD(cuFTemp, &m_FluidTemp, sizeof(FBufs)),	"AllocateParticles", "cuMemcpyHtoD", "cuFTemp", mbDebug);
	cuCheck( cuMemcpyHtoD(cuFParams, &m_FParams, sizeof(FParams)),  "AllocateParticles", "cuMemcpyHtoD", "cuFParams", mbDebug);
	cuCheck(cuCtxSynchronize(), "AllocateParticles", "cuCtxSynchronize", "", mbDebug );

	m_Param[PSTAT_PMEM] = 68.0f * 2 * cnt;

	// Allocate auxiliary buffers (prefix sums)
	int blockSize = SCAN_BLOCKSIZE << 1;
	int numElem1 = m_GridTotal;
	int numElem2 = int ( numElem1 / blockSize ) + 1;
	int numElem3 = int ( numElem2 / blockSize ) + 1;

	AllocateBuffer ( FAUXARRAY1,	sizeof(uint),		0,	numElem2, GPU_SINGLE, CPU_OFF );
	AllocateBuffer ( FAUXSCAN1,	    sizeof(uint),		0,	numElem2, GPU_SINGLE, CPU_OFF );
	AllocateBuffer ( FAUXARRAY2,	sizeof(uint),		0,	numElem3, GPU_SINGLE, CPU_OFF );
	AllocateBuffer ( FAUXSCAN2,	    sizeof(uint),		0,	numElem3, GPU_SINGLE, CPU_OFF );
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
	
	mNumPoints++;
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
				clr *= 0.8; clr += 0.2;				
				clr.Clamp (0, 1.0);				
				m_Fluid.bufI(FCLR) [p] = COLORA( clr.x, clr.y, clr.z, 1); 				
				// = COLORA( 0.25, +0.25 + (y-min.y)*.75/dy, 0.25 + (z-min.z)*.75/dz, 1);  // (x-min.x)/dx
			}
		}
	}		
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
    //std::cout << " FluidSystem::Run () \n";
	//case RUN_GPU_FULL:					// Full CUDA pathway, GRID-accelerted GPU, /w deep copy sort		
		InsertParticlesCUDA ( 0x0, 0x0, 0x0 );		
		PrefixSumCellsCUDA ( 0x0, 1 );		
		CountingSortFullCUDA ( 0x0 );
		ComputePressureCUDA();		
		ComputeForceCUDA ();			
		AdvanceCUDA ( m_Time, m_DT, m_Param[PSIMSCALE] );		
		//EmitParticlesCUDA ( m_Time, (int) m_Vec[PEMIT_RATE].x );		
		TransferFromCUDA ();	// return for rendering			
	AdvanceTime ();
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
	
	AL = m_Param[PACCEL_LIMIT];	AL2 = AL*AL;
	SL = m_Param[PVEL_LIMIT];	SL2 = SL*SL;
	
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
			accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
		}		
		diff = radius - ( bmax.y - ppos->y )*ss;
		if (diff > EPSILON) {
			norm.Set ( 0, -1, 0 );
			adj = stiff * diff - damp * (float) norm.Dot ( *pveleval );
			accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
		}		
		
		// X-axis walls
		if ( !m_Toggle[PWRAP_X] ) {
			diff = radius - ( ppos->x - (bmin.x + (sin(m_Time*m_Param[PFORCE_FREQ])+1)*0.5f * m_Param[PFORCE_MIN]) )*ss;	
			//diff = 2 * radius - ( p->pos.x - min.x + (sin(m_Time*10.0)-1) * m_Param[FORCE_XMIN_SIN] )*ss;	
			if (diff > EPSILON ) {
				norm.Set ( 1.0, 0, 0 );
				adj = (m_Param[ PFORCE_MIN ]+1) * stiff * diff - damp * (float) norm.Dot ( *pveleval ) ;
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;					
			}

			diff = radius - ( (bmax.x - (sin(m_Time*m_Param[PFORCE_FREQ])+1)*0.5f* m_Param[PFORCE_MAX]) - ppos->x )*ss;	
			if (diff > EPSILON) {
				norm.Set ( -1, 0, 0 );
				adj = (m_Param[ PFORCE_MAX ]+1) * stiff * diff - damp * (float) norm.Dot ( *pveleval );
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
			}
		}

		// Z-axis walls
		diff = radius - ( ppos->z - bmin.z )*ss;			
		if (diff > EPSILON) {
			norm.Set ( 0, 0, 1 );
			adj = stiff * diff - damp * (float) norm.Dot ( *pveleval );
			accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
		}
		diff = radius - ( bmax.z - ppos->z )*ss;
		if (diff > EPSILON) {
			norm.Set ( 0, 0, -1 );
			adj = stiff * diff - damp * (float) norm.Dot ( *pveleval );
			accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
		}
		

		// Wall barrier
		if ( m_Toggle[PWALL_BARRIER] ) {
			diff = 2 * radius - ( ppos->x - 0 )*ss;					
			if (diff < 2*radius && diff > EPSILON && fabs(ppos->y) < 3 && ppos->z < 10) {
				norm.Set ( 1.0, 0, 0 );
				adj = 2*stiff * diff - damp * (float) norm.Dot ( *pveleval ) ;	
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;					
			}
		}
		
		// Levy barrier
		if ( m_Toggle[PLEVY_BARRIER] ) {
			diff = 2 * radius - ( ppos->x - 0 )*ss;					
			if (diff < 2*radius && diff > EPSILON && fabs(ppos->y) > 5 && ppos->z < 10) {
				norm.Set ( 1.0, 0, 0 );
				adj = 2*stiff * diff - damp * (float) norm.Dot ( *pveleval ) ;	
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;					
			}
		}
		// Drain barrier
		if ( m_Toggle[PDRAIN_BARRIER] ) {
			diff = 2 * radius - ( ppos->z - bmin.z-15 )*ss;
			if (diff < 2*radius && diff > EPSILON && (fabs(ppos->x)>3 || fabs(ppos->y)>3) ) {
				norm.Set ( 0, 0, 1);
				adj = stiff * diff - damp * (float) norm.Dot ( *pveleval );
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
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
			clr.x -= float(1/255.0f);		if ( clr.x < 0.2f ) clr.x = 0.2f;
			clr.y -= float(1/255.0f);		if ( clr.y < 0.2f ) clr.y = 0.2f;
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
	gc.y = c / xz;				c -= gc.y*xz;
	gc.z = c / m_GridRes.x;		c -= gc.z*m_GridRes.x;
	gc.x = c;
	return gc;
}



////////////////////////////////////////////////////////


void FluidSystem::InsertParticles ()
{
	int gs;	
	
	// Reset all grid pointers and neighbor tables to empty
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
					if ( i==j ) { j = m_Fluid.bufI(FGNEXT)[j]; continue; }
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
					if ( i==j ) { j = m_Fluid.bufI(FGNEXT)[j]; continue; }
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
	ftype = 2; fcnt = 3;		// float, 3 channel
	fwrite ( &ftype, sizeof(int), 1, fp );		
	fwrite ( &fcnt,  sizeof(int), 1, fp );	
	fwrite ( m_Fluid.bufC(FPOS),  numpnt*sizeof(Vector3DF), 1, fp );

	// write velocities
	ftype = 2; fcnt = 3;		// float, 3 channel
	fwrite ( &ftype, sizeof(int), 1, fp );		
	fwrite ( &fcnt,  sizeof(int), 1, fp );	
	fwrite ( m_Fluid.bufC(FVEL),  numpnt*sizeof(Vector3DF), 1, fp );

	// write colors
	ftype = 0; fcnt = 4;		// char, 4 channel
	fwrite ( &ftype, sizeof(int), 1, fp );		
	fwrite ( &fcnt,  sizeof(int), 1, fp );	
	fwrite ( m_Fluid.bufC(FCLR),  numpnt*sizeof(unsigned char)*4, 1, fp );

	fclose ( fp );

	fflush ( fp );
}


void FluidSystem::SavePointsCSV ( int frame )
{
	char buf[256];
    frame += 100000;    // ensures numerical and alphabetic order match
	sprintf ( buf, "particles_pos_vel_color%04d.csv", frame );
	FILE* fp = fopen ( buf, "w" );

	int numpnt = NumPoints();
	int numfield = 3;
	int ftype;         // 0=char, 1=int, 2=float, 3=double
	int fcnt;
    
    Vector3DF* Pos;
    Vector3DF* Vel;
    uint* Clr;

    fprintf(fp, "x coord, y coord, z coord, x vel, y vel, z vel,  color \n");

    for(int i=0;i<numpnt;i++){
        Pos = getPos(i);
        Vel = getVel(i);
        Clr = getClr(i);
        fprintf(fp, "%f,%f,%f,%f,%f,%f, %u \n", Pos->x, Pos->y,Pos->z, Vel->x,Vel->y,Vel->z, *Clr); 
    }
	fclose ( fp );
	fflush ( fp );
}

void FluidSystem::SavePoints_asciiPLY ( int frame )
{
	char buf[256];
    frame += 100000;    // ensures numerical and alphabetic order match
	sprintf ( buf, "particles_pos%04d.ply", frame );
	FILE* fp = fopen ( buf, "w" );

	int numpnt = NumPoints();
	int numfield = 3;
	int ftype;         // 0=char, 1=int, 2=float, 3=double
	int fcnt;
    
    Vector3DF* Pos;
    Vector3DF* Vel;
    uint* Clr;

    fprintf(fp, "ply \n format ascii 1.0\n comment particle cloud from Fluids_v4\n element vertex %i\n", numpnt );
    fprintf(fp, "property float x\nproperty float y\nproperty float z\n");
    fprintf(fp, "end_header\n");
    
    for(int i=0;i<numpnt;i++){
        Pos = getPos(i);
        Vel = getVel(i);
        Clr = getClr(i);
        fprintf(fp, "%f %f %f\n", Pos->x, Pos->y,Pos->z); 
    }
	fclose ( fp );
	fflush ( fp );
}




////////////////////////
// adapted from Example 1 of http://web.mit.edu/fwtools_v3.1.0/www/Intro/IntroExamples.html#CreateExample
#include <hdf5/serial/hdf5.h>	//hdf5/serial/
#include <stdio.h>
#include <stdlib.h>

#define DATASETNAME "Vec3DF_Array" 
//#define NX     5                      /* dataset dimensions */
#define NY     3
#define RANK   2

int FluidSystem::WriteParticlesToHDF5File (int filenum)     
{
    std::cout << "WriteParticlesToHDF5File \n" << std::flush;
    hid_t       file, dataset;         /* file and dataset handles */
    hid_t       datatype, dataspace;   /* handles */
    hsize_t     dimsf[2];              /* dataset dimensions */
    herr_t      status;

    const int NX =  NumPoints();                 

    int         i, j;
    float **    data = new float*[NX];  /* allocate data to write */
    for(i=0; i<NX; ++i) 
	data[i] =  new float[NY];          

    if (data == nullptr) {
	std::cout << "Error: memory could not be allocated";
	return -1;
    }
        
    /* Data  and output buffer initialization.  */
    for (j = 0; j < NX; j++) {
	for (i = 0; i < NY; i++)
	    data[j][i] = (float)(i + j);
    }
    
    // edit filename
    char filename[256];
    filenum += 100000;    // ensures numerical and alphabetic order match
	sprintf ( filename, "particles_pos_%04d.h5", filenum );

    /* Create a new file using H5F_ACC_TRUNC access,
     * default file creation properties, and default file
     * access properties. */
    file = H5Fcreate(/*FILE2*/filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    /* Describe the size of the array and create the data space for fixed
     * size dataset. */
    dimsf[0] = NX;
    dimsf[1] = NY;
    dataspace = H5Screate_simple(RANK, dimsf, NULL); 

    /* Define datatype for the data in the file.
     * We will store little endian INT numbers.*/
    datatype = H5Tcopy(H5T_IEEE_F64LE/*H5T_NATIVE_INT*/);
    status = H5Tset_order(datatype, H5T_ORDER_LE);

    /* Create a new dataset within the file using defined dataspace and
     * datatype and default dataset creation properties. */
    dataset = H5Dcreate(file, DATASETNAME, datatype, dataspace,
			H5P_DEFAULT,
			H5P_DEFAULT,
			H5P_DEFAULT);

    /* Write the data to the dataset using default transfer properties. */
    status = H5Dwrite(dataset, H5T_IEEE_F64LE/*H5T_NATIVE_INT*/, H5S_ALL, H5S_ALL,
		      H5P_DEFAULT, m_Fluid.bufV3(FPOS) /*data*/);

    /* Close/release resources. */
    H5Sclose(dataspace);
    H5Tclose(datatype);
    H5Dclose(dataset);
    H5Fclose(file);
 
    return 0;
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
	
		} break;
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
	m_Vec[PBOUNDMIN] = m_Vec[PVOLMIN];		m_Vec[PBOUNDMIN] += 2.0*(m_Param[PGRIDSIZE] / m_Param[PSIMSCALE]);
	m_Vec[PBOUNDMAX] = m_Vec[PVOLMAX];		m_Vec[PBOUNDMAX] -= 2.0*(m_Param[PGRIDSIZE] / m_Param[PSIMSCALE]);
}



int iDivUp (int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}
void computeNumBlocks (int numPnts, int minThreads, int &numBlocks, int &numThreads)
{
    numThreads = min( minThreads, numPnts );
    numBlocks = (numThreads==0) ? 1 : iDivUp ( numPnts, numThreads );
}

/*void FluidSystem::cudaExit ()
{
	cudaDeviceReset();
}*/


void FluidSystem::TransferToTempCUDA ( int buf_id, int sz )
{
	cuCheck ( cuMemcpyDtoD ( m_FluidTemp.gpu(buf_id), m_Fluid.gpu(buf_id), sz ), "TransferToTempCUDA", "cuMemcpyDtoD", "m_FluidTemp", mbDebug);
}


///////////////////////////////////////////////


void FluidSystem::FluidSetupCUDA ( int num, int gsrch, int3 res, float3 size, float3 delta, float3 gmin, float3 gmax, int total, int chk )
{	
	m_FParams.pnum = num;	
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
	// Send particle buffers	
	cuCheck(cuMemcpyHtoD(m_Fluid.gpu(FPOS), m_Fluid.bufC(FPOS),				mNumPoints *sizeof(float) * 3),			"TransferToCUDA", "cuMemcpyHtoD", "FPOS", mbDebug);
	cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FVEL),	m_Fluid.bufC(FVEL),			mNumPoints *sizeof(float)*3 ),	"TransferToCUDA", "cuMemcpyHtoD", "FVEL", mbDebug);
	cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FVEVAL),	m_Fluid.bufC(FVEVAL),	mNumPoints *sizeof(float)*3 ), "TransferToCUDA", "cuMemcpyHtoD", "FVELAL", mbDebug);
	cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FFORCE),	m_Fluid.bufC(FFORCE),	mNumPoints *sizeof(float)*3 ), "TransferToCUDA", "cuMemcpyHtoD", "FFORCE", mbDebug);
	cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FPRESS),	m_Fluid.bufC(FPRESS),	mNumPoints *sizeof(float) ),	"TransferToCUDA", "cuMemcpyHtoD", "FPRESS", mbDebug);
	cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FDENSITY), m_Fluid.bufC(FDENSITY),	mNumPoints *sizeof(float) ),	"TransferToCUDA", "cuMemcpyHtoD", "FDENSITY", mbDebug);
	cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FCLR),	m_Fluid.bufC(FCLR),			mNumPoints *sizeof(uint) ),		"TransferToCUDA", "cuMemcpyHtoD", "FCLR", mbDebug);
}

void FluidSystem::TransferFromCUDA ()
{
	// Return particle buffers	
	cuCheck( cuMemcpyDtoH ( m_Fluid.bufC(FPOS),	m_Fluid.gpu(FPOS),	mNumPoints *sizeof(float)*3 ), "TransferFromCUDA", "cuMemcpyDtoH", "FPOS", mbDebug);
	cuCheck( cuMemcpyDtoH ( m_Fluid.bufC(FVEL),	m_Fluid.gpu(FVEL),	mNumPoints *sizeof(float)*3 ), "TransferFromCUDA", "cuMemcpyDtoH", "FVEL", mbDebug);
	cuCheck( cuMemcpyDtoH ( m_Fluid.bufC(FCLR),	m_Fluid.gpu(FCLR),	mNumPoints *sizeof(uint) ),	"TransferFromCUDA", "cuMemcpyDtoH", "FCLR", mbDebug);
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

	// Reset grid cell IDs
	//cuCheck(cuMemsetD32(m_Fluid.gpu(FGCELL), GRID_UNDEF, numPoints ), "cuMemsetD32(Sort)");

	void* args[1] = { &mNumPoints };
	cuCheck ( cuLaunchKernel ( m_Func[FUNC_COUNTING_SORT], m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL),
				"CountingSortFullCUDA", "cuLaunch", "FUNC_COUNTING_SORT", mbDebug );

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

void FluidSystem::ComputeForceCUDA ()
{
	void* args[1] = { &m_FParams.pnum };
	cuCheck ( cuLaunchKernel ( m_Func[FUNC_COMPUTE_FORCE],  m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL), "ComputeForceCUDA", "cuLaunch", "FUNC_COMPUTE_FORCE", mbDebug);
}

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








