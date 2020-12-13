
#include <stdio.h>
#include <inttypes.h>
#include <errno.h>
#include <string.h>

#include "fluid_system.h"

int main ( int argc, const char** argv ) 
{
    char paramsPath[256];
    char genomePath[256];
    char pointsPath[256];
    char outPath[256];
	if ( argc != 3 ){
	    printf("usage: check_demo  simulation_data_folder  output_folder\n");
	    return 0;
	}else {
        sprintf ( paramsPath, "%s/SimParams.txt", argv[1] );
        printf("simulation parameters file = %s\n", paramsPath);
        
        sprintf ( genomePath, "%s/genome.csv", argv[1] );
        printf("simulation parameters file = %s\n", genomePath);
        
        sprintf ( pointsPath, "%s/particles_pos_vel_color100001.csv", argv[1] );  // particles_pos_vel_color100001_test_data.csv
        printf("simulation points file = %s\n", pointsPath);
        
        sprintf ( outPath, "%s", argv[2] );
        printf("output_folder = %s\n", outPath);
	}	

    FluidSystem fluid;
    
    // Clear all buffers
    fluid.Initialize();   // where do the buffers for params and genome get allocated when there is no fluid.InitializeCuda (); ?

    fluid.ReadSimParams(paramsPath);
    fluid.ReadGenome(genomePath);
    fluid.ReadPointsCSV2(pointsPath, GPU_OFF, CPU_YES);  //fluid.ReadPointsCSV(pointsPath, GPU_OFF, CPU_YES);
    printf("\nchk1\n");
    fluid.WriteSimParams ( outPath ); 
    printf("\nchk2\n");
    fluid.WriteGenome( outPath );
    printf("\nchk3\n");
    fluid.SavePointsCSV2 ( outPath, 1 );
    printf("\nchk4\n");
    fluid.SavePointsVTP2(outPath, 1 );
    printf("\nchk5\n");
    
    fluid.Exit ();	
    printf("\ncheck_demo finished.\n");
    return 0;
}
