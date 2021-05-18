
// Fluid System
#include <stdio.h>
#include <inttypes.h>
#include <errno.h>
#include <string.h>

#include "fluid_system.h"

typedef	unsigned int		uint;	

int main ( int argc, const char** argv ) 
{
    uint num_particles, demoType, simSpace;
    float spacing, x_dim, y_dim, z_dim;
    if ( argc != 8 && argc !=1 ) {
        printf ( "usage: make_demo num_particles spacing x_dim y_dim z_dim \n \
        demoType(0:free falling, 1: remodelling & actuation, 2: diffusion & epigenetics.) \n \
        simSpace(0:regression test, 1:tower, 2:wavepool, 3:small dam break, 4:dual-wavepool, 5: microgravity, \n \
            6:Morphogenesis small demo  7:use SpecificationFile.txt  8:parameter sweep default )\n" );
        return 0;
    } else if (argc == 8) {
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
 
    std::string paramsPath("demo/SimParams.txt");     // Set file paths relative to data/ , where SpecfileBatchGenerator will be run.
    std::string pointsPath("demo");
    std::string genomePath("demo/genome.csv");
    std::string outPath("out");
    for(int i=0;i<paramsPath.length();i++)fluid.launchParams.paramsPath[i] = paramsPath[i];
    for(int i=0;i<pointsPath.length();i++)fluid.launchParams.pointsPath[i] = pointsPath[i];
    for(int i=0;i<genomePath.length();i++)fluid.launchParams.genomePath[i] = genomePath[i];
    for(int i=0;i<outPath.length(); i++)  fluid.launchParams.outPath[i]    = outPath[i];
    
    fluid.WriteExampleSpecificationFile("./demo");
    printf("\nmake_demo finished.\n");
    fluid.Exit_no_CUDA ();	
    return 0;
}
