

// Fluid System
#include <stdio.h>
#include <inttypes.h>
#include <errno.h>
#include <string.h>
#include <chrono>
#include "fluid_system.h"
typedef	unsigned int		uint;	

int main ( int argc, const char** argv ) 
{
    char spec_file[256];
    if ( argc != 2 ) {
        printf ( "usage: make_demo2 specification_file.txt \n" );
        return 0;
    } else {
        sprintf ( spec_file, "%s", argv[1] );
        printf ( "specification_file = %s\n", spec_file );
    }
    
    // Initialize
    cuInit ( 0 );
    int deviceCount = 0;
    cuDeviceGetCount ( &deviceCount );
    if ( deviceCount == 0 ) {
        printf ( "There is no device supporting CUDA.\n" );
        exit ( 0 );
    }
    CUdevice cuDevice;
    cuDeviceGet ( &cuDevice, 0 );
    CUcontext cuContext;
    cuCtxCreate ( &cuContext, 0, cuDevice );
    
    FluidSystem fluid;
    fluid.InitializeCuda ();
    std::cout<<"\n\nmake_demo2 chk0,"<<std::flush;
    
    fluid.ReadSpecificationFile ( spec_file );
    std::cout<<"\n\nmake_demo2 chk1, fluid.launchParams.debug="<<fluid.launchParams.debug<<", fluid.launchParams.paramsPath=" <<fluid.launchParams.paramsPath <<std::flush;
    
    fluid.WriteDemoSimParams(
        fluid.launchParams.paramsPath, GPU_DUAL, CPU_YES, fluid.launchParams.num_particles, fluid.launchParams.spacing, fluid.launchParams.x_dim, fluid.launchParams.y_dim, fluid.launchParams.z_dim, fluid.launchParams.demoType, fluid.launchParams.simSpace, fluid.launchParams.debug
    ); /*const char * relativePath*/ 
    std::cout<<"\n\nmake_demo2 chk2 "<<std::flush;
    fluid.TransferToCUDA (); 
    fluid.RunSimulation ();
    std::cout<<"\n\nmake_demo2 chk3 "<<std::flush;
    
    // clean up and exit
    fluid.Exit ();
    
    size_t   free1, free2, total;
    cudaMemGetInfo(&free1, &total);
    printf("\n\nCuda Memory, before cuCtxDestroy(cuContext): free=%lu, total=%lu.\t",free1,total);
    
    CUresult cuResult = cuCtxDestroy ( cuContext ) ;
    if ( cuResult!=0 ) {printf ( "error closing, cuResult = %i \n",cuResult );}
    
    cudaMemGetInfo(&free2, &total);
    printf("\nAfter cuCtxDestroy(cuContext): free=%lu, total=%lu, released=%lu.\n",free2,total,(free2-free1) );
    
    printf ( "\nClosed make_demo2.\n" );
    return 0;
}
