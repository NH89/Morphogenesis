
// Fluid System
#include <stdio.h>
#include <inttypes.h>
#include <errno.h>
#include <string.h>

#include "fluid_system.h"

typedef	unsigned int		uint;	

int main ( int argc, const char** argv ) 
{
    printf ( "\n\n### Starting Morphogenesis  \"make_demo\" ### \n" );
    uint num_particles, demoType, simSpace;
    float spacing, x_dim, y_dim, z_dim;
    if ( argc != 8 && argc !=1 ) {
        printf ( "usage: make_demo num_particles spacing x_dim y_dim z_dim \n \
        demoType(0:free falling, 1: remodelling & actuation, 2: diffusion & epigenetics.) \n \
        simSpace(0:regression test, 1:tower, 2:wavepool, 3:small dam break, 4:dual-wavepool, 5: microgravity, \n \
            6:Morphogenesis small demo  7:use SpecificationFile.txt  8:parameter sweep default )\n" );
        return 0;
    } else if (argc == 8) {
        printf ( "\nargc == 8");

        num_particles = atoi(argv[1]);
        printf ( "num_particles = %u\n", num_particles );
        
        spacing = atof(argv[2]);
        printf ( "spacing = %f\n", spacing );
        
        x_dim = atof(argv[3]);
        printf ( "x_dim = %f\n", x_dim );
        
        y_dim = atof(argv[4]);
        printf ( "y_dim = %f\n", y_dim );
        
        z_dim = atof(argv[5]);
        printf ( "z_dim = %f\n", z_dim );
        
        demoType = atof(argv[6]);
        printf ( "demoType = %u, (0:free falling, 1: remodelling & actuation, 2: diffusion & epigenetics.)\n", demoType );
        
        simSpace = atof(argv[7]);
        printf ( "simSpace = %u, (0:regression test, 1:tower, 2:wavepool, 3:small dam break, 4:dual-wavepool, 5: microgravity, \n \
            6:Morphogenesis small demo  7:use SpecificationFile.txt  8:parameter sweep default )\n\n", simSpace);
    }  else {
        printf ( "\nargc == 1");

        num_particles = 4000;
        printf ( "num_particles = %u\n", num_particles );
        
        spacing = 1.0;
        printf ( "spacing = %f\n", spacing );
        
        x_dim = 10.0;
        printf ( "x_dim = %f\n", x_dim );
        
        y_dim = 10.0;
        printf ( "y_dim = %f\n", y_dim );
        
        z_dim = 3;
        printf ( "z_dim = %f\n", z_dim );
        
        demoType = 0;
        printf ( "demoType = %u, (0:free falling, 1: remodelling & actuation, 2: diffusion & epigenetics.)\n", demoType );
        
        simSpace = 8;
        printf ( "simSpace = %u, (0:regression test, 1:tower, 2:wavepool, 3:small dam break, 4:dual-wavepool, 5: microgravity, \n \
            6:Morphogenesis small demo  7:use SpecificationFile.txt  8:parameter sweep default )\n\n", simSpace );
    }
    
    uint debug = 2;  // same values as in load_sim and in specification_file.txt .
    FluidSystem fluid;
    fluid.Initialize();
    
    std::cout <<"\nmake_demo : launchParams.read_genome = " << fluid.launchParams.read_genome <<",\t  launchParams.genomePath = "<<  fluid.launchParams.genomePath<<std::endl<<std::flush;

    fluid.WriteDemoSimParams("./demo", GPU_OFF, CPU_YES , num_particles, spacing, x_dim, y_dim, z_dim, demoType, simSpace, debug);/*const char * relativePath*/ 
    
    if(argc !=1){                                           // i.e not relying on defaults in simspace 8
    fluid.launchParams.num_particles    = num_particles;    // Write default values to fluid.launchParams...
    fluid.launchParams.demoType         = demoType;
    fluid.launchParams.simSpace         = 7;                // i.e. use the Specfile.txt generated. 
    fluid.launchParams.x_dim            = x_dim;
    fluid.launchParams.y_dim            = y_dim;
    fluid.launchParams.z_dim            = z_dim;
    
    fluid.launchParams.num_files        = 400;
    fluid.launchParams.steps_per_InnerPhysicalLoop = 3;
    fluid.launchParams.steps_per_file   = 6;
    fluid.launchParams.freeze_steps     = 1;
    fluid.launchParams.debug            = 0;
    fluid.launchParams.file_num         = 0;
    
    fluid.launchParams.save_ply         = 'n';
    fluid.launchParams.save_csv         = 'n';
    fluid.launchParams.save_vtp         = 'y';
    fluid.launchParams.gene_activity    = 'n';
    fluid.launchParams.remodelling      = 'n';

    }
 
    std::string paramsPath("demo");     // Set file paths relative to data/ , where SpecfileBatchGenerator will be run.
    std::string pointsPath("demo/particles_pos_vel_color100001.csv");
    std::string genomePath("demo");
    std::string outPath("out");
	sprintf(fluid.launchParams.paramsPath,"%s", paramsPath.c_str() );
    sprintf(fluid.launchParams.pointsPath,"%s", pointsPath.c_str() );
    sprintf(fluid.launchParams.genomePath,"%s", genomePath.c_str() );
    sprintf(fluid.launchParams.outPath,   "%s", outPath.c_str() );

    printf("\nfluid.launchParams.paramsPath :  %s\n", fluid.launchParams.paramsPath );
	printf("\nfluid.launchParams.pointsPath :  %s\n", fluid.launchParams.pointsPath );
	printf("\nfluid.launchParams.genomePath :  %s\n", fluid.launchParams.genomePath );
	printf("\nfluid.launchParams.outPath    :  %s\n", fluid.launchParams.outPath );

    fluid.WriteExampleSpecificationFile("./demo");

    fluid.SavePointsVTP2( "demo", 1 );
    fluid.SavePointsCSV2( "demo", 1 );

    printf("\nmake_demo finished.\n");
    fluid.Exit_no_CUDA ();	
    return 0;
}
