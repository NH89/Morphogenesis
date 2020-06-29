
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
    if ( argc != 3 ) {
        printf ( "usage: fluid_system  simulation_data_folder output_folder\n" );
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
std::cout <<"\nchk A_1.0\n"<<std::flush;
    for ( int i=0; i<10; i++ ) {
        for ( int j=0; j<30; j++ ) {
            fluid.Run ();                               // run the simulation
        }

        //fluid.SavePoints (i);                         // alternate file formats to write
        //fluid.SavePointsCSV (i);
        fluid.SavePoints_asciiPLY ( outPath, i );
        //fluid.WriteParticlesToHDF5File(i);
        printf ( "\t i=%i frame number =%i \n",i, i*20 );
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



    fluid.Exit ();                                      // Clean up and close
    CUresult cuResult = cuCtxDestroy ( cuContext ) ;
    if ( cuResult!=0 ) {
        printf ( "error closing, cuResult = %i \n",cuResult );
    }
    printf ( "\nClosing fluids_v4.\n" );
    return 0;
}

// note: Any memory allocated with cuMemAllocManaged should be released with cuMemFree.
