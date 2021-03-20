
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
    if ( argc != 8 ) {
        printf ( "usage: make_demo num_particles spacing x_dim y_dim z_dim demoType(0:free falling, 1: remodelling & actuation, 2: diffusion & epigenetics.)  simSpace(0:regression test, 1:tower, 2:wavepool, 3: small dam break, 4:dual-wavepool, 5: microgravity)\n" );
        return 0;
    } else {
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
        printf ( "simSpace = %u, (0:regression test, 1:tower, 2:wavepool, 3: small dam break, 4:dual-wavepool, 5: microgravity)\n\n", simSpace);
    }    
    uint debug = 2;  // same values as in load_sim and in specification_file.txt .
    FluidSystem fluid;
    fluid.WriteDemoSimParams("./demo", GPU_OFF, CPU_YES , num_particles, spacing, x_dim, y_dim, z_dim, demoType, simSpace, debug);/*const char * relativePath*/ 
    
    fluid.launchParams.num_particles    = num_particles;    // Write default values to fluid.launchParams...
    fluid.launchParams.demoType         = demoType;
    fluid.launchParams.simSpace         = simSpace;
    fluid.launchParams.x_dim            = x_dim;
    fluid.launchParams.y_dim            = y_dim;
    fluid.launchParams.z_dim            = z_dim;
    
    fluid.launchParams.num_files        = 400;
    fluid.launchParams.steps_per_file   = 6;
    fluid.launchParams.freeze_steps     = 1;
    fluid.launchParams.debug            = 2;
    fluid.launchParams.file_num         = 0;
    
    fluid.launchParams.save_ply         = 'n';
    fluid.launchParams.save_csv         = 'n';
    fluid.launchParams.save_vtp         = 'y';
    fluid.launchParams.gene_activity    = 'n';
    fluid.launchParams.remodelling      = 'n';
 
    std::string str1("../demo");                            // Set file paths relative to data/test , where makeDemo2 will be run.
    for (int i = 0; i < str1.length(); i++) {
        fluid.launchParams.paramsPath[i] = str1[i];
        fluid.launchParams.pointsPath[i] = str1[i];
        fluid.launchParams.genomePath[i] = str1[i];
    }
    std::string str2("../out");
    for (int i = 0; i < str2.length(); i++) {
        fluid.launchParams.outPath[i]    = str2[i];
    }
    
    fluid.WriteExampleSpecificationFile("./demo");
    fluid.Exit ();	
    printf("\nmake_demo finished.\n");
    return 0;
}
