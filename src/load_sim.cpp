
#include <stdio.h>
#include <inttypes.h>
#include <errno.h>
#include <string.h>

#include "fluid_system.h"

int main ( int argc, const char** argv )
{
    char paramsPath[256];
    char pointsPath[256];
    char genomePath[256];
    char outPath[256];
    uint num_files, steps_per_file, freeze_steps;
    int file_num=0;
    char save_ply, save_csv;
    if ( argc != 8 ) {
        printf ( "usage: load_sim  simulation_data_folder output_folder num_files steps_per_file freeze_steps save_ply(y/n) save_csv(y/n)\n" );
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
        printf ( "save_ply = %u\n", save_ply );
        
        save_csv = *argv[7];
        printf ( "save_csv = %u\n", save_csv );
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
    fluid.ReadGenome ( genomePath, GPU_DUAL, CPU_YES );
    // NB currently GPU allocation is by Allocate particles, called by ReadPointsCSV.
    fluid.ReadPointsCSV2 ( pointsPath, GPU_DUAL, CPU_YES );

std::cout <<"\nchk load_sim_1.0\n"<<std::flush;
    fluid.TransferFromCUDA ();
    fluid.SavePointsCSV2 ( outPath, file_num );
    fluid.SavePoints_asciiPLY_with_edges ( outPath, file_num );
    file_num++;
/*
std::cout <<"\nchk load_sim_1.1\n"<<std::flush;
    fluid.InsertParticlesCUDA ( 0x0, 0x0, 0x0 );
    fluid.PrefixSumCellsCUDA ( 0x0, 1 );
    fluid.CountingSortFullCUDA ( 0x0 );
std::cout <<"\nchk load_sim_1.2\n"<<std::flush;   
    fluid.TransferFromCUDA ();
    fluid.SavePointsCSV2 ( outPath, 1 );
    fluid.SavePoints_asciiPLY ( outPath, 1 );
std::cout <<"\nchk load_sim_1.3\n"<<std::flush;   
    fluid.FreezeCUDA();
    fluid.TransferFromCUDA ();
    fluid.SavePointsCSV2 ( outPath, 2 );
    fluid.SavePoints_asciiPLY ( outPath, 2 );
std::cout <<"\nchk load_sim_1.4\n"<<std::flush;   
    fluid.Run (); 
    fluid.SavePointsCSV2 ( outPath, 3 );
    fluid.SavePoints_asciiPLY ( outPath, 3 );
 */   



std::cout <<"\nchk load_sim_2.0\n"<<std::flush;
    

    for (int k=0; k<freeze_steps; k++){
         /*
        fluid.Freeze (outPath, file_num);                   // save csv after each kernel - to investigate bugs
        file_num+=10;
         */
        // /*
        fluid.Freeze ();       // creates the bonds // fluid.Freeze(outPath, file_num) saves file after each kernel,, fluid.Freeze() does not.
//temporary comment out//   if(save_csv=='y') fluid.SavePointsCSV2 ( outPath, file_num);
        if(save_ply=='y') fluid.SavePoints_asciiPLY_with_edges ( outPath, file_num );
     //   printf("\nsaved file_num=%u",file_num);
        file_num+=10;
        
        // */
    }

    printf("\n\nFreeze finished, starting normal Run ##############################################\n\n");
    
    for ( ; file_num<num_files; file_num+=10 ) {
        for ( int j=0; j<steps_per_file; j++ ) {
            fluid.Run ();                               // run the simulation  // Run(outPath, file_num) saves file after each kernel,, Run() does not.
        }// 0:start, 1:InsertParticles, 2:PrefixSumCellsCUDA, 3:CountingSortFull, 4:ComputePressure, 5:ComputeForce, 6:Advance, 7:AdvanceTime

        //fluid.SavePoints (i);                         // alternate file formats to write
        if(save_csv=='y') fluid.SavePointsCSV2 ( outPath, file_num);
        if(save_ply=='y') fluid.SavePoints_asciiPLY_with_edges ( outPath, file_num );
        //fluid.WriteParticlesToHDF5File(i);
        printf ( "\nsaved file_num=%u, frame number =%i \n",file_num,  file_num*steps_per_file );
    }


    /*//fluid.TransferFromCUDA ();	// retrieve outcome
    //int filenum = 0;
    	//fluid.SavePoints(filenum);
    // NB fluid.m_Fluid.bufC(FPOS) returns a char* for fluid.m_Fluid.mcpu[n]

    //fluid.SaveResults ();
    //int NumPoints (){ return mNumPoints; }
    //Vector3DF* getPos(int n){ return &m_Fluid.bufV3(FPOS)[n]; }
    //Vector3DF* getVel(int n){ return &m_Fluid.bufV3(FVEL)[n]; }
    //uint* getClr (int n){ return &m_Fluid.bufI(FCLR)[n]; }

    //write fluid.m_Fluid.mcpu[n] to file. where n=bufferID : FPOS=0, FVEL=1

    // mcpu[MAX_BUF] where MAX_BUF = 25, is an array of buffers.
    // FluidSystem::AllocateParticles ( int cnt=numParticles )
    // calls FluidSystam::AllocateBuffer(...) for each each buffer
    // which calls m_Fluid.setBuf(buf_id, dest_buf);
    // to save the pointer to the allocated buffer to fluid.m_Fluid.mcpu[n]
    // FPOS, has a stride : sizeof(Vector3DF) , and 'numParticles' elements

    // create file

    // write HDF5 data to store fluid.m_Fluid.mcpu[FPOS][numParticles][Vector3DF]
    // use h5ex_t_array.c example

    	//int stride = sizeof(Vector3DF);
    	//fluid.SavePointsCSV (filenum);
    //fluid.WriteFileTest2(m_numpnts);
    //fluid.WriteParticlesToHDF5File(filenum);*/
    
    fluid.WriteSimParams ( outPath ); 
    fluid.WriteGenome( outPath );
    fluid.SavePointsCSV2 ( outPath, 20*30 );                 //fluid.SavePointsCSV ( outPath, 1 );

    fluid.Exit ();                                      // Clean up and close
    CUresult cuResult = cuCtxDestroy ( cuContext ) ;
    if ( cuResult!=0 ) {printf ( "error closing, cuResult = %i \n",cuResult );}
    printf ( "\nClosing load_sim.\n" );
    return 0;
}

// note: Any memory allocated with cuMemAllocManaged should be released with cuMemFree.
