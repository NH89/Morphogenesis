#ifndef DEF_ALLOCATOR
	#define DEF_ALLOCATOR

	#include <vector>
	#include <cuda.h>
	#include "vector.h"

	// Maximum number of GVDB Pool levels
	#define MAX_POOL		10

	// Global CUDA helpers
	#define MIN_RUNTIME_VERSION		4010
	#define MIN_COMPUTE_VERSION		0x20
	extern void StartCuda( int devsel, CUcontext ctxsel, CUdevice& dev, CUcontext& ctx, CUstream* strm, bool verbose );
	extern bool	cudaCheck ( CUresult e, const char* obj, const char* method, const char* apicall, const char* arg, bool bDebug);
	extern Vector3DF cudaGetMemUsage();
    
#endif //DEF_ALLOCATOR
