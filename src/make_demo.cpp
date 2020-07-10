
// Fluid System
#include <stdio.h>
#include <inttypes.h>
#include <errno.h>
#include <string.h>

#include "fluid_system.h"

int main ( int argc, const char** argv ) 
{
    uint num_particles;
    float spacing, x_dim, y_dim, z_dim;
    if ( argc != 6 ) {
        printf ( "usage: make_demo num_particles spacing x_dim y_dim z_dim\n" );
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
        
        printf ( "### still need to edit fluid.WriteDemoSimParams(\"./demo\"); to use the new input ! \n");
    }    
    
    
    
    FluidSystem fluid;
    fluid.WriteDemoSimParams("./demo", num_particles, spacing, x_dim, y_dim, z_dim);/*const char * relativePath*/ 
    fluid.Exit ();	
    printf("\nmake_demo finished.\n");
    return 0;
}
