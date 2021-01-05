
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
        printf ( "simSpace = %u, (0:regression test, 1:tower, 2:wavepool, 3: small dam break, 4:dual-wavepool, 5: microgravity)\n", simSpace);
        
        printf ( "### still need to edit fluid.WriteDemoSimParams(\"./demo\"); to use the new input ! \n");
    }    
    
    
    
    FluidSystem fluid;
    fluid.WriteDemoSimParams("./demo", num_particles, spacing, x_dim, y_dim, z_dim, demoType, simSpace);/*const char * relativePath*/ 
    fluid.Exit ();	
    printf("\nmake_demo finished.\n");
    return 0;
}
