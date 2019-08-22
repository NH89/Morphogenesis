// Fluid System
#include "fluid_system.h"

int main ( int argc, const char** argv ) 
{
    FluidSystem fluid;
    fluid.SetDebug ( false );
    int m_numpnts = 150000;   // ### here is where number of particles is set
    
    fluid.Initialize ();
    fluid.Start ( m_numpnts );
    fluid.Run ();

    return 0;
}


