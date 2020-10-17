#include <stdio.h>
#include <inttypes.h>
#include <errno.h>
#include <string.h>
#include <chrono>
#include "fluid_system.h"

void file_writer_thread_function(FluidSystem &fluid, output &output_){   // thread for hiding latency of writing files. NB pass ref to fluid object.            
    //  file writing & mutexes
    while (output_.running){
        std::cout << "\nfile_writer_thread_function(..file_num = "<< output_.file_num <<".),  "<<std::flush;  
        std::unique_lock<std::mutex> lock(output_.condMutex);
        output_.dataReady.wait(lock);
        if (!output_.running) break;
        output_.isDiskReady = false;
        
        if(output_.save_csv=='y') fluid.SavePointsCSV2 ( output_.outPath, output_.file_num);
        if(output_.save_ply=='y') fluid.SavePoints_asciiPLY_with_edges ( output_.outPath, output_.file_num );
        if(output_.save_vtp=='y') fluid.SavePointsVTP2( output_.outPath, output_.file_num);
        //output_.file_num++;//+=10;
        
        output_.isDiskReady = true;
        output_.diskReady.notify_all();
    }
    std::cout << "\nExiting file_writer_thread_function(...),  "<<std::flush;
}

int main ( int argc, const char** argv )
{
    /*
    char paramsPath[256];
    char pointsPath[256];
    char genomePath[256];
    char outPath[256];
    uint num_files, steps_per_file, freeze_steps;
    int file_num=0;
    char save_ply, save_csv, save_vtp;
    */
    output output_;
    
    if ( argc != 9 ) {
        printf ( "usage: load_sim  simulation_data_folder output_folder num_files steps_per_file freeze_steps save_ply(y/n) save_csv(y/n) save_vtp(y/n)\n" );
        return 0;
    } else {
        sprintf ( output_.paramsPath, "%s/SimParams.txt", argv[1] );
        printf ( "simulation parameters file = %s\n", output_.paramsPath );

        sprintf ( output_.pointsPath, "%s/particles_pos_vel_color100001.csv", argv[1] );
        printf ( "simulation points file = %s\n", output_.pointsPath );

        sprintf ( output_.genomePath, "%s/genome.csv", argv[1] );
        printf ( "simulation genome file = %s\n", output_.genomePath );

        sprintf ( output_.outPath, "%s/", argv[2] );
        printf ( "output folder = %s\n", output_.outPath );
        
        output_.num_files = atoi(argv[3]);
        printf ( "num_files = %u\n", output_.num_files );
        
        output_.steps_per_file = atoi(argv[4]);
        printf ( "steps_per_file = %u\n", output_.steps_per_file );
        
        output_.freeze_steps = atoi(argv[5]);
        
        output_.save_ply = *argv[6];
        printf ( "save_ply = %c\n", output_.save_ply );
        
        output_.save_csv = *argv[7];
        printf ( "save_csv = %c\n", output_.save_csv );
        
        output_.save_vtp = *argv[8];
        printf ( "save_vtp = %c\n", output_.save_vtp );
        
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

    fluid.ReadSimParams ( output_.paramsPath );
    fluid.ReadGenome ( output_.genomePath, GPU_DUAL, CPU_YES );
    // NB currently GPU allocation is by Allocate particles, called by ReadPointsCSV.
    fluid.ReadPointsCSV2 ( output_.pointsPath, GPU_DUAL, CPU_YES );

std::cout <<"\nchk load_sim_1.0\n"<<std::flush;
    //fluid.TransferFromCUDA ();
    //fluid.SavePointsCSV2 ( output_.outPath, output_.file_num );
    //fluid.SavePoints_asciiPLY_with_edges ( output_.outPath, output_.file_num );
    //output_.file_num++;
    
    //auto old_begin = std::chrono::steady_clock::now();
                                                                        // thread for hiding latency of writing files
    output_.running = true;
    output_.isDiskReady = true; // disk thread starts ready to receive data
//    std::thread file_writer_thread(&file_writer_thread_function, std::ref(fluid), std::ref(output_) );    
    
 
std::cout <<"\nchk load_sim_2.0\n"<<std::flush;
    for (int k=0; k<output_.freeze_steps; k++){
         /*
        fluid.Freeze (outPath, file_num);                               // save csv after each kernel - to investigate bugs
        file_num+=10;
         */
        fluid.Freeze ();       // creates the bonds // fluid.Freeze(outPath, file_num) saves file after each kernel,, fluid.Freeze() does not.
        output_.file_num++;
        /*
        if(save_csv=='y') fluid.SavePointsCSV2 ( outPath, file_num);
        if(save_ply=='y') fluid.SavePoints_asciiPLY_with_edges ( outPath, file_num );
        if(save_vtp=='y') fluid.SavePointsVTP2( outPath, file_num);
        file_num+=10;
        */
        /*
        std::unique_lock<std::mutex> lock(output_.condMutex);
        if (!output_.isDiskReady){
            std::cout << "\nfile_writer_thread not yet ready, waiting for it to finish" << std::endl;
            output_.diskReady.wait(lock);
            if (!output_.running) break;
        }
        output_.dataReady.notify_all();
        */
        fluid.SavePointsPVTP2( output_.outPath, 0);
    }

    printf("\n\nFreeze finished, starting normal Run ##############################################\n\n");
    
    for ( ; output_.file_num<output_.num_files; output_.file_num++/*+=10*/ ) {
        auto begin = std::chrono::steady_clock::now();
        for ( int j=0; j<output_.steps_per_file; j++ ) {
            fluid.Run ();
            /*
            // run the simulation  // Run(outPath, file_num) saves file after each kernel,, Run() does not.
            // 0:start, 1:InsertParticles, 2:PrefixSumCellsCUDA, 3:CountingSortFull, 4:ComputePressure, 5:ComputeForce, 6:Advance, 7:AdvanceTime
            */
        }
        
/*        
        std::unique_lock<std::mutex> lock(output_.condMutex);                                  // synchronize with file_writer_thread
        if (!output_.isDiskReady){
            std::cout << "\nfile_writer_thread not yet ready, waiting for it to finish" << std::endl;
            output_.diskReady.wait(lock);
            if (!output_.running) break;
        }
        output_.dataReady.notify_all();
*/        
        
        fluid.SavePointsPVTP2( output_.outPath, output_.file_num);
        
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> time = end - begin;
        std::cout << "\nTime taken to run "<< output_.steps_per_file <<" steps, "<< fluid.NumPoints() <<" particles : " << time.count() << " seconds" << std::endl;
        /*
        //fluid.SavePoints (i);                         // alternate file formats to write
        // TODO flip mutex
        
        //begin = std::chrono::steady_clock::now();
        
        //if(output_.save_csv=='y') fluid.SavePointsCSV2 ( output_.outPath, output_.file_num);
        //if(output_.save_ply=='y') fluid.SavePoints_asciiPLY_with_edges ( output_.outPath, output_.file_num );
        //if(output_.save_vtp=='y') fluid.SavePointsVTP2( output_.outPath, output_.file_num);
        
        //end = std::chrono::steady_clock::now();
        //time = end - begin;
        //std::chrono::duration<double> begin_dbl = begin - old_begin;
        //std::cout << "\nLoop duration : "<< begin_dbl.count() <<" seconds. Time taken to write files for "<< fluid.NumPoints() <<" particles : " << time.count() << " seconds" << std::endl;
        //old_begin = begin;
        
        //fluid.WriteParticlesToHDF5File(i);
        //printf ( "\nsaved file_num=%u, frame number =%i \n",file_num,  file_num*steps_per_file );
        */
    }
    //fluid.WriteSimParams ( output_.outPath ); 
    //fluid.WriteGenome( output_.outPath );
    //fluid.SavePointsCSV2 ( outPath, 20*30 );          //fluid.SavePointsCSV ( outPath, 1 );
    //fluid.SavePointsVTP2 ( outPath, 20*30 );
    
//    output_.running = false;                            // file_writer_thread teardown
//    output_.dataReady.notify_all();                     // notify both threads if they're blocked waiting on a condition variable
//    output_.diskReady.notify_all();
//    file_writer_thread.join();                          // wait for threads to finish to prevent errors (graceful shutdown)
    
    fluid.Exit ();                                      // Clean up and close
    CUresult cuResult = cuCtxDestroy ( cuContext ) ;
    if ( cuResult!=0 ) {printf ( "error closing, cuResult = %i \n",cuResult );}
    printf ( "\nClosing load_sim.\n" );
    return 0;
}

// note: Any memory allocated with cuMemAllocManaged should be released with cuMemFree.
