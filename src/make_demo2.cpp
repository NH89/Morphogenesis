#include <stdio.h>
#include <inttypes.h>
#include <errno.h>
#include <string.h>
#include <chrono>
#include <filesystem>

#include "fluid_system.h"

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
																						cout << "\nmake_demo2: chk_1 "<<std::flush;
																						// Initialize
    cuInit ( 0 );
    int deviceCount = 0;
    cuDeviceGetCount ( &deviceCount );
    if ( deviceCount == 0 ) {
        printf ( "There is no device supporting CUDA.\n" );
        exit ( 0 );
    }
																						cout << "\nmake_demo2: chk_2 "<<std::flush;
    CUdevice cuDevice;
    cuDeviceGet ( &cuDevice, 0 );
    CUcontext cuContext;
    																					//cuCtxCreate ( &cuContext, 0, cuDevice );
    CUctxCreateParams ctxCreateParams = {};
    cuCtxCreate(&cuContext, &ctxCreateParams, CU_CTX_SCHED_AUTO, cuDevice);
																						cout <<"\n\n### make_demo2: chk_3 "<<std::flush;
    FluidSystem fluid;																	cout <<"\n\n### make_demo2: chk_4 "<<std::flush;
    fluid.InitializeCuda ();															cout <<"\n\n### make_demo2: chk_5 , input_folder : "<<input_folder<<std::flush;
    fluid.ReadSpecificationFile ( input_folder );										cout <<"\n\n### make_demo2 chk_6,\nfluid.launchParams.debug      = "<<fluid.launchParams.debug
																													<<", \nfluid.launchParams.paramsPath = "<<fluid.launchParams.paramsPath
																													<<", \nfluid.launchParams.genomePath = "<<fluid.launchParams.genomePath
																													<<", \nfluid.launchParams.pointsPath = "<<fluid.launchParams.pointsPath
																													<<", \nfluid.launchParams.spacing    = "<<fluid.launchParams.spacing	<<std::flush;
	for(int i=0; i<256; i++){fluid.launchParams.outPath[i] = output_folder[i];}
    
    if(mkdir(output_folder, 0755) == -1) 								cerr << "\nError :  failed to create output_folder.\n" << strerror(errno) << endl;
    																					else cout << "output_folder created\n"; // NB 0755 = rwx owner, rx for others.

																						cout <<"\n\n### make_demo2 chk7, \nfluid.launchParams.debug      = "<<fluid.launchParams.debug
																													<<", \nfluid.launchParams.paramsPath = "<<fluid.launchParams.paramsPath
																													<<", \nfluid.launchParams.genomePath = "<<fluid.launchParams.genomePath
																													<<", \nfluid.launchParams.pointsPath = "<<fluid.launchParams.pointsPath
																													<<", \nfluid.launchParams.spacing    = "<<fluid.launchParams.spacing	<<std::flush;
	//fluid.save_stdout(std::filesystem::path( output_folder),  std::string( "stdout.txt") );	// seriously slows down execution.
	uint num_particles_start=0;
    if(fluid.launchParams.loadSim=='y'){												cout <<"\n\n### make_demo2 chk8,  (fluid.launchParams.loadSim=='y') : "<<fluid.launchParams.loadSim<<std::flush;
        fluid.ReadSimParams( 	fluid.launchParams.paramsPath);
        fluid.ReadGenome(		fluid.launchParams.genomePath);
        fluid.ReadPointsCSV2(	fluid.launchParams.pointsPath, GPU_DUAL, CPU_YES);	// int gpu_mode, int cpu_mode
        num_particles_start=	fluid.ActivePoints();
        fluid.SavePointsVTP2(	fluid.launchParams.outPath, 1);
        fluid.SavePointsCSV2(	fluid.launchParams.outPath, 1);
    }else{																				cout <<"\n\n### make_demo2 chk8,  (fluid.launchParams.loadSim=='y') : "<<fluid.launchParams.loadSim<<std::flush;
    	fluid.WriteDemoSimParams(           // Generates the simulation from data previously loaded from SpecificationFile.txt .
        	fluid.launchParams.paramsPath,
        	GPU_DUAL,
        	CPU_YES,
        	fluid.launchParams.num_particles,
        	fluid.launchParams.spacing,
        	fluid.launchParams.x_dim,
        	fluid.launchParams.y_dim,
        	fluid.launchParams.z_dim,
        	fluid.launchParams.demoType,
        	fluid.launchParams.simSpace,
        	fluid.launchParams.debug
    	);
        num_particles_start=fluid.ActivePoints();
    	fluid.TransferToCUDA ();
    }

    																					if(fluid.launchParams.save_csv=='y'){
                                                                                            fluid.SavePointsCSV2 ( fluid.launchParams.outPath, fluid.launchParams.file_num+5);
                                                                                            fluid.TransferFromCUDA ();
                                                                                            fluid.cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "Run2Simulation After TransferFromCUDA", fluid.launchParams.debug);
                                                                                            fluid.SavePointsCSV2 ( fluid.launchParams.outPath, fluid.launchParams.file_num+10);
                                                                                        }
    																					cout<<"\n\n### make_demo2 chk9 "<<std::flush;
    fluid.Run3Simulation ();															// ###  Using benchmarking edits, instead of Run2Simulation()

    fluid.WriteResultsCSV(input_folder, output_folder, num_particles_start);			// NB post-slurm script to (i) cat results.csv files, (ii)tar-gzip and ftp folders to recipient.
    
    size_t   free1, free2, total;
    cudaMemGetInfo(&free1, &total);
    																					printf("\n\n### make_demo2 chk10: Cuda Memory, before cuCtxDestroy(cuContext): free=%lu, total=%lu.\t",free1,total);
    CUresult cuResult = cuCtxDestroy ( cuContext ) ;
    if ( cuResult!=0 ) {printf ( "error closing, cuResult = %i \n",cuResult );}
    
    cudaMemGetInfo(&free2, &total);
    																					printf("\n### make_demo2 chk11: After cuCtxDestroy(cuContext): free=%lu, total=%lu, released=%lu.\n",free2,total,(free2-free1) );
    																					printf ( "\nClosed make_demo2.\n" );
                                                                                        std::cerr<<"\n\nClosed make_demo2 \n"<<std::flush;
	//std::fclose(stdout);
    return 0;
}
