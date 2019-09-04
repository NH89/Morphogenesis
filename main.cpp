// Fluid System
#include "fluid_system.h"

int main ( int argc, const char** argv ) 
{
	cuInit(0);					// Initialize
	int deviceCount = 0;
	cuDeviceGetCount(&deviceCount);
	if (deviceCount == 0) {printf("There is no device supporting CUDA.\n"); exit (0);}

	CUdevice cuDevice;
	cuDeviceGet(&cuDevice, 0);
	CUcontext cuContext;
	cuCtxCreate(&cuContext, 0, cuDevice);



    FluidSystem fluid;
    fluid.SetDebug ( false );
    int m_numpnts = 150000;   	// ### here is where number of particles is set
    
    fluid.Initialize ();
    fluid.Start ( m_numpnts );
    for(int i=0;i<20;i++){    fluid.Run ();  }

    return 0;
}


