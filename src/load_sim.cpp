
#include <stdio.h>
#include <inttypes.h>
#include <errno.h>
#include <string.h>
#include <chrono>
#include "fluid_system.h"

int main ( int argc, const char** argv )
{
    char paramsPath[256];
    char pointsPath[256];
    char genomePath[256];
    char outPath[256];
    uint num_files, steps_per_file, freeze_steps;
    int file_num=0;
    char save_ply, save_csv, save_vtp;
    if ( argc != 9 ) {
        printf ( "usage: load_sim  simulation_data_folder output_folder num_files steps_per_file freeze_steps save_ply(y/n) save_csv(y/n) save_vtp(y/n)\n" );
        return 0;
    } else {
        sprintf ( paramsPath, "%s/SimParams.txt", argv[1] );
        printf ( "simulation parameters file = %s\n", paramsPath );

        sprintf ( pointsPath, "%s/particles_pos_vel_color100001.csv", argv[1] );
        printf ( "simulation points file = %s\n", pointsPath );

        sprintf ( genomePath, "%s/genome.csv", argv[1] );
        printf ( "simulation genome file = %s\n", genomePath );

        sprintf ( outPath, "%s/", argv[2] );
        printf ( "output folder = %s\n", outPath );
        
        num_files = atoi(argv[3]);
        printf ( "num_files = %u\n", num_files );
        
        steps_per_file = atoi(argv[4]);
        printf ( "steps_per_file = %u\n", steps_per_file );
        
        freeze_steps = atoi(argv[5]);
        
        save_ply = *argv[6];
        printf ( "save_ply = %c\n", save_ply );
        
        save_csv = *argv[7];
        printf ( "save_csv = %c\n", save_csv );
        
        save_vtp = *argv[8];
        printf ( "save_vtp = %c\n", save_vtp );
        
    }

    cuInit ( 0 );                                       // Initialize
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
    fluid.SetDebug ( false );
    fluid.InitializeCuda ();

    fluid.ReadSimParams ( paramsPath );
    fluid.ReadGenome ( genomePath );
    // NB currently GPU allocation is by Allocate particles, called by ReadPointsCSV.
    fluid.ReadPointsCSV2 ( pointsPath, GPU_DUAL, CPU_YES );

std::cout <<"\nchk load_sim_1.0\n"<<std::flush;
    auto old_begin = std::chrono::steady_clock::now();
    
    fluid.TransferFromCUDA ();
    fluid.SavePointsCSV2 ( outPath, file_num );
    //fluid.SavePoints_asciiPLY_with_edges ( outPath, file_num );
    file_num++;
    
    fluid.TransferPosVelVeval ();
    cuCheck(cuCtxSynchronize(), "Run", "cuCtxSynchronize", "After TransferPosVelVeval, before 1st timestep", 1/*mbDebug*/);
    
 
std::cout <<"\nchk load_sim_2.0\n"<<std::flush;
    for (int k=0; k<freeze_steps; k++){
        std::cout<<"\n\nFreeze()\n"<<std::flush;
         /*
        fluid.Freeze (outPath, file_num);                   // save csv after each kernel - to investigate bugs
        file_num+=10;
         */
        fluid.Freeze ();       // creates the bonds // fluid.Freeze(outPath, file_num) saves file after each kernel,, fluid.Freeze() does not.
        if(save_csv=='y') fluid.SavePointsCSV2 ( outPath, file_num);
        //if(save_ply=='y') fluid.SavePoints_asciiPLY_with_edges ( outPath, file_num );
        if(save_vtp=='y') fluid.SavePointsVTP2( outPath, file_num);
        file_num+=10;
    }

    printf("\n\nFreeze finished, starting normal Run ##############################################\n\n");
    
    for ( ; file_num<num_files; file_num+=10 ) {
        
        
        for ( int j=0; j<steps_per_file; j++ ) {
            fluid.Run ();                               // run the simulation  // Run(outPath, file_num) saves file after each kernel,, Run() does not.
        }// 0:start, 1:InsertParticles, 2:PrefixSumCellsCUDA, 3:CountingSortFull, 4:ComputePressure, 5:ComputeForce, 6:Advance, 7:AdvanceTime

        //fluid.SavePoints (i);                         // alternate file formats to write
        // TODO flip mutex
        auto begin = std::chrono::steady_clock::now();
        
        if(save_csv=='y') fluid.SavePointsCSV2 ( outPath, file_num);
        //if(save_ply=='y') fluid.SavePoints_asciiPLY_with_edges ( outPath, file_num );
        if(save_vtp=='y') fluid.SavePointsVTP2( outPath, file_num);
        
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> time = end - begin;
        std::chrono::duration<double> begin_dbl = begin - old_begin;
        std::cout << "\nLoop duration : "
                    << begin_dbl.count() <<" seconds. Time taken to write files for "
                    << fluid.NumPoints() <<" particles : " 
                    << time.count() << " seconds\n" << std::endl;
        old_begin = begin;
        
        //fluid.WriteParticlesToHDF5File(i);
        //printf ( "\nsaved file_num=%u, frame number =%i \n",file_num,  file_num*steps_per_file );
    }
    
    file_num++;
    fluid.WriteSimParams ( outPath ); 
    fluid.WriteGenome( outPath );
  //  fluid.SavePointsCSV2 ( outPath, file_num );                 //fluid.SavePointsCSV ( outPath, 1 );
  //  fluid.SavePointsVTP2 ( outPath, file_num );

    fluid.Exit ();                                      // Clean up and close
    CUresult cuResult = cuCtxDestroy ( cuContext ) ;
    if ( cuResult!=0 ) {printf ( "error closing, cuResult = %i \n",cuResult );}
    printf ( "\nClosing load_sim.\n" );
    return 0;
}

// note: Any memory allocated with cuMemAllocManaged should be released with cuMemFree.
