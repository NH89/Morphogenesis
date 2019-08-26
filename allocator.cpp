#include "allocator.h"
#if defined(_WIN32)
#	include <windows.h>
#endif
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda.h>
#include <assert.h>

bool cudaCheck ( CUresult launch_stat, const char* obj, const char* method, const char* apicall, const char* arg, bool bDebug)
{
	CUresult kern_stat = CUDA_SUCCESS;
	
	if ( bDebug ) {
		kern_stat = cuCtxSynchronize();	
	}		
	if ( kern_stat != CUDA_SUCCESS || launch_stat != CUDA_SUCCESS ) {
		const char* launch_statmsg = "";
		const char* kern_statmsg = "";
		cuGetErrorString ( launch_stat, &launch_statmsg );
		cuGetErrorString ( kern_stat, &kern_statmsg);
        std::cout<<"GVDB CUDA ERROR:\n";
        std::cout<<"  Launch status: "<< launch_statmsg <<"\n";
        std::cout<<"  Kernel status: "<< kern_statmsg <<"\n";
        std::cout<<"  Caller: "<<obj<<"::"<<method<<"\n";
        std::cout<<"  Call:   "<<apicall<<"\n";
        std::cout<<"  Args:   "<<arg<<"\n";

		if (bDebug) {
            std::cout<<"  Generating assert so you can examine call stack.\n";
			assert(0);		// debug - trigger break (see call stack)
		} else {
            std::cout<<"Error. Application will exit.\n"; // exit - return 0  
		}		
		return false;
	} 
	return true;
}

void StartCuda ( int devsel, CUcontext ctxsel, CUdevice& dev, CUcontext& ctx, CUstream* strm, bool verbose)
{
	// NOTES:
	// CUDA and OptiX Interop: (from Programming Guide 3.8.0)
	// - CUDA must be initialized using run-time API
	// - CUDA may call driver API, but only after context created with run-time API
	// - Once app created CUDA context, OptiX will latch onto the existing CUDA context owned by run-time API
	// - Alternatively, OptiX can create CUDA context. Then set runtime API to it. (This is how Ocean SDK sample works.)

	int version = 0;
    char name[128];
    
	int cnt = 0;
	CUdevice dev_id;	
	cuInit(0);

	//--- List devices
	cuDeviceGetCount ( &cnt );
	if (cnt == 0) {
        std::cout<<"ERROR: No CUDA devices found.\n";
		dev = (int) NULL; ctx = NULL;
        std::cout<<"Error. Application will exit.\n";
		return;
	}	
	if (verbose) std::cout<<"  Device List:\n";
	for (int n=0; n < cnt; n++ ) {
		cuDeviceGet(&dev_id, n);
		cuDeviceGetName ( name, 128, dev_id);

		int pi;
		cuDeviceGetAttribute ( &pi, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH, dev_id ) ;
		if (verbose) std::cout<<"Max. texture3D width: "<<pi<<"\n";
        
		cuDeviceGetAttribute ( &pi, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT, dev_id ) ;
		if (verbose) std::cout<<"Max. texture3D height: "<<pi<<"\n";
        
		cuDeviceGetAttribute ( &pi, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH, dev_id ) ;
		if (verbose) std::cout<<"Max. texture3D depth: "<<pi<<"\n";
		if (verbose) std::cout<<"   "<<n<<". "<<name<<"\n";
	}

    //--- Create new context with Driver API 
    cudaCheck(cuDeviceGet(&dev, devsel), "(global)", "StartCuda", "cuDeviceGet", "", false );
    cudaCheck(cuCtxCreate(&ctx, CU_CTX_SCHED_AUTO, dev), "(global)", "StartCuda", "cuCtxCreate", "", false );

	cuDeviceGetName(name, 128, dev);
	if (verbose) std::cout<<"   Using Device: "<<(int) dev<<", "<<name<<", Context: "<<(void*) ctx<<"\n";
	
	cuCtxSetCurrent( NULL );
	cuCtxSetCurrent( ctx );
}

Vector3DF cudaGetMemUsage ()
{
	Vector3DF mem;
	size_t free, total;	
	cuMemGetInfo ( &free, &total );
	free /= (1024.0f*1024.0f);		// MB
	total /= (1024.0f*1024.0f);
	mem.x = total - free;	// used
	mem.y = free;
	mem.z = total;
	return mem;
}
