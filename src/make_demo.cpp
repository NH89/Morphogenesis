
// Fluid System
#include <stdio.h>
#include <inttypes.h>
#include <errno.h>
#include <string.h>

#include "fluid_system.h"

int main ( ) 
{
    FluidSystem fluid;
    fluid.WriteDemoSimParams("./demo");/*const char * relativePath*/ 
    fluid.Exit ();	
    printf("\nmake_demo finished.\n");
    return 0;
}
