
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
	    printf("usage: fluid_system  simulation_data_folder  output_folder\n");
	    return 0;
	}else {
        sprintf ( paramsPath, "%s/SimParams.txt", argv[1] );
        printf("simulation parameters file = %s\n", paramsPath);
        
        sprintf ( genomePath, "%s/genome.csv", argv[1] );
        printf("simulation parameters file = %s\n", genomePath);
        
        sprintf ( pointsPath, "%s/particles_pos_vel_color100001.csv", argv[1] );
        printf("simulation points file = %s\n", pointsPath);
        
        sprintf ( outPath, "%s", argv[2] );
        printf("output_folder = %s\n", outPath);
	}	

    FluidSystem fluid;
    
    fluid.ReadSimParams(paramsPath);
    fluid.ReadGenome(genomePath, GPU_OFF, CPU_YES);
    fluid.ReadPointsCSV(pointsPath, GPU_OFF, CPU_YES);
    
    fluid.WriteSimParams ( outPath ); 
    fluid.WriteGenome( outPath );
    fluid.SavePointsCSV ( outPath, 1 );
    fluid.SavePoints_asciiPLY ( outPath, 1 );
    
    fluid.Exit ();	
    printf("\nmake_demo.\n");
    return 0;
}
