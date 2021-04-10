#include <stdio.h>
#include <inttypes.h>
#include <errno.h>
#include <string.h>
#include <chrono>
#include "fluid_system.h"
#include <filesystem>
typedef	unsigned int		uint;	

int main ( int argc, const char** argv ) 
{
    char input_folder[256];
    char output_folder[256];
    if ((argc != 3) && (argc !=2)) {
        printf ( "usage: make_demo2 input_folder output_folder.\
        \nNB input_folder must contain \"SpecificationFile.txt\", output will be wrtitten to \"output_folder/out_data_time/\".\
        \nIf output_folder is not given the value from SpecificationFile.txt will be used.\n" );
        return 0;
    } else {
        sprintf ( input_folder, "%s", argv[1] );
        sprintf ( output_folder, "%s", argv[2] );
        printf ( "input_folder = %s , output_folder = %s\n", input_folder, output_folder );
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
    
    fluid.ReadSpecificationFile ( input_folder );
    std::cout<<"\n\nmake_demo2 chk1, fluid.launchParams.debug="<<fluid.launchParams.debug<<", fluid.launchParams.paramsPath=" <<fluid.launchParams.paramsPath <<std::flush;
    
    for(int i=0; i<256; i++){fluid.launchParams.paramsPath[i] = input_folder[i];}
    for(int i=0; i<256; i++){fluid.launchParams.pointsPath[i] = input_folder[i];}
    for(int i=0; i<256; i++){fluid.launchParams.genomePath[i] = input_folder[i];}
    if(argc==3)for(int i=0; i<256; i++){fluid.launchParams.outPath[i] = output_folder[i];}
    //char mkdir_cmd[256];
    //sprintf(mkdir_cmd,"mkdir -p %s", output_folder);
    //std::system("mkdir_cmd");
    //std::filesystem::create_directory(output_folder);
    
    if(mkdir(output_folder, 0755) == -1) cerr << "\nError :  failed to create output_folder.\n" << strerror(errno) << endl;
    else cout << "output_folder created\n"; // NB 0755 = rwx owner, rx for others.
    
    
    fluid.WriteDemoSimParams(
        fluid.launchParams.paramsPath, GPU_DUAL, CPU_YES, fluid.launchParams.num_particles, fluid.launchParams.spacing, fluid.launchParams.x_dim, fluid.launchParams.y_dim, fluid.launchParams.z_dim, fluid.launchParams.demoType, fluid.launchParams.simSpace, fluid.launchParams.debug
    ); /*const char * relativePath*/ 
    std::cout<<"\n\nmake_demo2 chk2 "<<std::flush;
    fluid.TransferToCUDA (); 
    fluid.Run2Simulation ();
    
    
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