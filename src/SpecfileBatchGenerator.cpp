


// Fluid System
#include <stdio.h>
#include <inttypes.h>
#include <errno.h>
#include <string.h>
#include <chrono>
#include <sys/stat.h>
#include "fluid_system.h"
typedef	unsigned int		uint;	

int main ( int argc, const char** argv ) 
{
    char spec_file[256];
    if ( argc != 2 ) {
        printf ( "usage: SpecfileBatchGenerator specification_file.txt \n" );
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
    std::cout<<"\n\nSpecfileBatchGenerator chk0, spec_file = "<< spec_file << std::flush;
    
    fluid.ReadSpecificationFile ( spec_file );
    std::cout<<"\n\nSpecfileBatchGenerator chk1, fluid.launchParams.debug="<<fluid.launchParams.debug<<", fluid.launchParams.paramsPath=" <<fluid.launchParams.paramsPath <<std::flush;
    
    // Params to loop around:
    // (i)(fluid)m_Param [ PINTSTIFF ] (0.5-8), (ii)m_Param[PVISC] ( - ), (iii)surface_t ( - ), (iv)(elastic) stiffness, (v)damping, (vi)m_Param [ PMASS ], 
    char Specification_file_path[256];
    //sprintf ( Specification_file_path, "%s", spec_file );
    
    for (int i=2; i<=8; i*=2){
        float intstiff = 0.5 * (float)i;                                                                // set parameter
        sprintf ( Specification_file_path, "%s_intstiff%f", spec_file, intstiff);  
        int check = mkdir(Specification_file_path,0777);                                                // make sub dir
        if (check  ){ printf("\nUnable to create directory: %s\n", Specification_file_path);  exit(1);}
        printf("\n%s\t",Specification_file_path);
        fluid.WriteExampleSpecificationFile (Specification_file_path);                                  // write spec file
    }
    
    std::cout<<"\n\nSpecfileBatchGenerator chk2 "<<std::flush;
    
    // clean up and exit
    fluid.Exit ();
    
    size_t   free1, free2, total;
    cudaMemGetInfo(&free1, &total);
    printf("\n\nCuda Memory, before cuCtxDestroy(cuContext): free=%lu, total=%lu.\t",free1,total);
    
    CUresult cuResult = cuCtxDestroy ( cuContext ) ;
    if ( cuResult!=0 ) {printf ( "error closing, cuResult = %i \n",cuResult );}
    
    cudaMemGetInfo(&free2, &total);
    printf("\nAfter cuCtxDestroy(cuContext): free=%lu, total=%lu, released=%lu.\n",free2,total,(free2-free1) );
    
    printf ( "\nClosed SpecfileBatchGenerator.\n" );
    return 0;
}
