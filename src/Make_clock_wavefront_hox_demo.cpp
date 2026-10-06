#include <stdio.h>
#include <inttypes.h>
#include <errno.h>
#include <string.h>
#include <chrono>
#include <filesystem>

#include "fluid_system.h"

typedef	unsigned int		uint;




int main ( int argc, const char** argv ){
	char folder[256];
	sprintf( folder, "%s", argv[1] );

	uint debug 				= 2;  // same values as in load_sim and in specification_file.txt .
    FluidSystem fluid;
    fluid.Initialize();

	fluid.ReadGenome(		folder);
	fluid.ReadSimParams(	folder);

	fluid.SetupAddVolume_clock_wavefront_hox();
/*
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

	fluid.WriteExampleSpecificationFile("./demo");												// specify path only. File is always called "SpecificationFile.txt"
*/
    fluid.SavePointsVTP2( "demo", 1 );
    fluid.SavePointsCSV2( "demo", 1 );
    fluid.Exit_no_CUDA ();
    return 0;
}

void FluidSystem::SetupAddVolume_clock_wavefront_hox(/*Vector3DF min, Vector3DF max, float spacing, float offs*/ ){  // NB ony used in WriteDemoSimParams() called by make_demo.cpp . Creates a cuboid with all particle values definable.
if (m_FParams.debug>1)std::cout << "\n SetupAddVolumeMorphogenesis2 \t" << std::flush ;

	Vector3DF 	min			= m_Vec[PINITMIN];
	Vector3DF 	max			= m_Vec[PINITMAX];//{5,3,22};									// (x,y,z)
	float		spacing		= spacing;
	float		offs		= 0.1f;

    Vector3DF 	pos;
    float 		dx, dy, dz;
    int 		cntx, cntz, p, c2;
    cntx 	=	(int) ceil( (max.x-min.x-offs) / spacing );
    cntz 	=	(int) ceil( (max.z-min.z-offs) / spacing );
    int cnt =	cntx * cntz;
    min		+=	offs;													// NB by default offs=0.1f, & min=m_Vec[PINITMIN], when called in WriteDemoSimParams(..)
    max		-=	offs;													// m_Vec[PINITMIN] is set in SetupExampleParams()
    dx		=	max.x-min.x;											// m_Vec[PBOUNDMIN] is set in SetupSpacing
    dy		=	max.y-min.y;
    dz		=	max.z-min.z;
    Vector3DF	rnd;
    c2		=	cnt/2;
    Vector3DF 	Pos, Vel;
    uint		Age, Clr, Particle_ID, Mass_Radius, NerveIdx;
    uint		ElastIdxU[BOND_DATA];
    float		ElastIdxF[BOND_DATA];
    uint		Particle_Idx[BONDS_PER_PARTICLE*2];						// FPARTICLE_IDX : other particles with incoming bonds attaching here.
    float		Conc[NUM_TF];
    uint		EpiGen[NUM_GENES]={0};
    Particle_ID = 0;													// NB Particle_ID=0 means "no particle" in ElastIdx.
    Vector3DF 	volV3DF					= max-min;
    int 		num_particles_to_make	= std::min( mMaxPoints, (8 * int(volV3DF.x*volV3DF.y*volV3DF.z)) );		// 27 * //int(volV3DF.x*volV3DF.y*volV3DF.z / spacing*spacing*spacing);

    srand((unsigned int)time(NULL));
    if (m_FParams.debug>1)std::cout<<"\nSetupAddVolumeMorphogenesis2: num_particles_to_make="<<num_particles_to_make<<",   min=("<<min.x<<","<<min.y<<","<<min.z<<"), max=("<<max.x<<","<<max.y<<","<<max.z<<") "<<std::flush;

    for (int i=0; i<num_particles_to_make; i++){
        Pos.x 	=  min.x + (float(rand())/float((RAND_MAX)) * dx) ;
        Pos.y 	=  min.y + (float(rand())/float((RAND_MAX)) * dy) ;
		Pos.z 	=  min.z + (float(rand())/float((RAND_MAX)) * dz) ;

        Particle_ID ++;  												// NB AddParticleMorphogenesis2(...) checks not to exceed max num particles
        Vel.x	= 0; Vel.y=0; Vel.z=0;
        Age 	= 0;
        																// Colour of particles
        Vector3DF clr ( (pos.x-min.x)/dx, 0, (pos.z-min.z)/dz );
		clr		*= 0.8;
		clr		+= 0.2;
		clr.Clamp (0, 1.0);
		Clr		= COLORA( clr.x, clr.y, clr.z, 1);
																		// Modulus & length of elastic bonds
																		// 8bits log modulus + 24bit uid, with fixed length // but for now 16bit modulus and radius

		uint modulus	= uint(m_Param [ PINTSTIFF ]) ;					// m_Param [ PINTSTIFF ] =		1.0f;
		uint length		= uint(1000 * m_Param [ PSMOOTHRADIUS ]);		// m_Param [ PSMOOTHRADIUS ] =	0.015f;	// m // related to spacing, but also max particle range i.e. ....
		uint mod_len	= ( modulus <<16 | length ); 					// NB should mask length to prevent it exceeding 16bits, i.e. 255*255

		for (int bond_idx = 0; bond_idx<BONDS_PER_PARTICLE;bond_idx++){
			for (int j = 0; j< DATA_PER_BOND; j++){

                ElastIdxU[bond_idx*DATA_PER_BOND +j] = UINT_MAX;
				ElastIdxF[bond_idx*DATA_PER_BOND +j] = 0;
			}
                ElastIdxU[bond_idx*DATA_PER_BOND +8] = 0;
		}
                														//NB #define DATA_PER_BOND 6 		//6 : [0]current index, [1]elastic limit, [2]restlength, [3]modulus, [4]damping coeff, [5]particle ID, [6]bond index
		for (int bond_idx_2 = 0; bond_idx_2<BONDS_PER_PARTICLE*2; bond_idx_2++) { Particle_Idx[bond_idx_2] = UINT_MAX; }

                // Mass & radius of particles
                // 4bit mass + 4bit radius + 24bit uid																				// but for now, 16bit mass & radius
                // Note m_params[] is set in "FluidSystem::SetupDefaultParams ()" and "FluidSystem::SetupExampleParams ()"
                // mass = m_Param[PMASS]; 																							// 0.00020543f; // kg
                // radius = m_Param[PRADIUS]; 																						// 0.015f; 		// m

		Mass_Radius =  ( (uint(m_Param[PMASS]*255.0f*255.0f)<<16) | uint(m_Param[PRADIUS]*255.0f*255.0f) ) ; 						// mass=>13, radius=>975	Packs two 16bit int numbers into one 32bit uint.

		// ###   Epigenetics and tissue types
		for (int tf_idx  =0; tf_idx  < NUM_TF;    tf_idx++)   {		Conc[	tf_idx]     = 0 ;}     // morphogen & transcription factor concentrations
		for (int gene_idx=0; gene_idx< NUM_GENES; gene_idx++) {		EpiGen[	gene_idx] 	= 0 ;}     // epigenetic state of each gene in this particle

		uint fixedActive = INT_MAX;                             		// FEPIGEN below INT_MAX will count down to inactivation. Count down is inactivated by adding INT_MAX.
		EpiGen[0] = fixedActive;                                        // active, i.e. not reserve
		EpiGen[1] = fixedActive;                                        // solid, i.e. have elastic bonds
		EpiGen[2] = fixedActive;                                        // living/telomere, i.e. has genes

		// Set up pre-somatic mesoderm NB volume 5x3x22 bins (x,y,z)
		if(  ( (Pos.x >1.0f && Pos.x <2.0f) || (Pos.x >3.0f && Pos.x <4.0f) )   && Pos.y >1.0f  && Pos.y <2.0f   && Pos.z >1.0f  && Pos.z <20.0f ){
			EpiGen[3]		=	INT_MAX;								// Presomatic mesoderm
			if( Pos.z < 2.0f){
				EpiGen[4]	=	INT_MAX;								// Initial activated somite
			}
		}
		/////////////////
		p = AddParticleMorphogenesis2 (
                /* Vector3DF* */	&Pos,
                /* Vector3DF* */	&Vel,
                /* uint */			Age,
                /* uint */			Clr,
                /* uint *_*/		ElastIdxU,
                /* uint *_*/		ElastIdxF,
                /* unit * */		Particle_Idx,
                /* uint */			Particle_ID,
                /* uint */			Mass_Radius,
                /* uint */			NerveIdx,
                /* float* */		Conc,
                /* uint* */			EpiGen
		);
		if(p==-1){
			if (m_FParams.debug>1){std::cout << "\n SetupAddVolumeMorphogenesis2 exited on p==-1, Pos=("<<Pos.x<<","<<Pos.y<<","<<Pos.z<<"), Particle_ID="<<Particle_ID<<",  EpiGen[0]="<<EpiGen[0]<<" \n " << std::flush ;}
			return;
		}
	}
	mActivePoints=mNumPoints;														// Initial active points, used in make_demo2.cpp, by WriteResultsCSV()
	AddNullPoints ();																// If spare particles remain, fill with null points. NB these can be used to "create" particles.
	if (m_FParams.debug>1)std::cout << "\n SetupAddVolumeMorphogenesis2 finished \n" << std::flush ;
}

/*
// void FluidSystem::WriteDemoSimParams ( const char * relativePath, int gpu_mode, int cpu_mode, uint num_particles, float spacing, float x_dim, float y_dim, float z_dim, uint demoType, uint simSpace, uint debug){
//
// 	AllocateBuffer ( FPARAMS, sizeof(FParams), 1,0, GPU_OFF, CPU_YES );
// /_*
// 	m_FParams.debug					= debug;
// 	mNumPoints						=     0;				// reset count
// 	m_Time							=     0.0f;				// Start at T=0
// 	m_DT							=     0.003f;
//
//     m_Param [ PEXAMPLE]				= simSpace;				// simSpace==2 : wave pool example.
//     m_Param [ PGRID_DENSITY]		=     2.0;				// gives gridsize = 2*smoothradius/griddensity = smoothradius.
//     m_Param [ PNUM]					= num_particles;		// 1000000;    //1000 = minimal simulation, 1000000 = large simulation
//
//     m_Param [ PSIMSCALE ]			=     0.005f;			// unit size
//     m_Param [ PVISC ]				=     0.50f;			// pascal-second (Pa.s) = 1 kg m^-1 s^-1  (see wikipedia page on viscosity)
//     m_Param [ PSURFACE_TENSION ]	=     0.1f;
//     m_Param [ PRESTDENSITY ]		=   400.0f;				// kg / m^3
//     m_Param [ PSPACING ]			=     0.0f;				// spacing will be computed automatically from density in most examples (set to 0 for autocompute)
//     m_Param [ PMASS ]				=     0.00020543f;		// kg
//     m_Param [ PRADIUS ]				=     0.015f;			// m
//     m_Param [ PDIST ]				=     0.0059f;			// m
//     m_Param [ PSMOOTHRADIUS ]		=     0.015f;			// m
//     m_Param [ PINTSTIFF ]			=     1.0f;
//     m_Param [ PEXTSTIFF ]			= 50000.0f;
//     m_Param [ PEXTDAMP ]			=   100.0f;
//     m_Param [ PACCEL_LIMIT ]		=   150.0f;				// m / s^2
//     m_Param [ PVEL_LIMIT ]			=     3.0f;				// m / s
//     m_Param [ PGRAV ]				=     1.0f;
//
//     m_Param [ PGROUND_SLOPE ]		=     0.0f;
//     m_Param [ PFORCE_MIN ]			=     0.0f;
//     m_Param [ PFORCE_MAX ]			=     0.0f;
//     m_Param [ PFORCE_FREQ ]			=    16.0f;
//
//     m_Param [ PACTUATION_FACTOR ]	=     0;
//     m_Param [ PACTUATION_PERIOD ]	=     1;
//
//     m_Vec [ PPLANE_GRAV_DIR ].Set ( 0, -9.8f, 0 );
//
//     // Default sim config
//     m_Param [PGRIDSIZE]				= m_Param[PSMOOTHRADIUS] * 2;
//
// 	//////////////    SetupExampleParams(spacing);
// 	m_Time = launchParams.m_Time;
// 	m_DT = launchParams.m_DT;
// 	m_Param [ PGRIDSIZE ] = launchParams.gridsize;
// 	m_Param [ PSPACING ] = launchParams.spacing;
// 	m_Param [ PSIMSCALE ] = launchParams.simscale;
// 	m_Param [ PSMOOTHRADIUS ] = launchParams.smoothradius;
// 	m_Param [ PVISC ] = launchParams.visc;
// 	m_Param [ PSURFACE_TENSION ] = launchParams.surface_tension;
// 	m_Param [ PMASS ] = launchParams.mass;
// 	m_Param [ PRADIUS ] = launchParams.radius;
//
// 	m_Param [ PINTSTIFF ] = launchParams.intstiff;
// 	m_Param [ PEXTSTIFF ] = launchParams.extstiff;
// 	m_Param [ PEXTDAMP ] = launchParams.extdamp;
// 	m_Param [ PACCEL_LIMIT ] = launchParams.accel_limit;
// 	m_Param [ PVEL_LIMIT ] = launchParams.vel_limit;
// 	m_Param [ PGRAV ] = launchParams.grav;
// 	m_Param [ PGROUND_SLOPE ] = launchParams.ground_slope;
// 	m_Param [ PFORCE_MIN ] = launchParams.force_min;
// 	m_Param [ PFORCE_MAX ] = launchParams.force_max;
// 	m_Param [ PFORCE_FREQ ] = launchParams.force_freq;
//
// 	m_Vec [ PVOLMIN ] = launchParams.volmin;
// 	m_Vec [ PVOLMAX ] = launchParams.volmax;
// 	m_Vec [ PINITMIN ] = launchParams.initmin;
// 	m_Vec [ PINITMAX ] = launchParams.initmax;
//
// 	m_Param [ PACTUATION_FACTOR ] = launchParams.actuation_factor;
// 	m_Param [ PACTUATION_PERIOD ] = launchParams.actuation_period;
//
// 	/////////////.....SetupExampleGenome();
//
//     /////////////     SetupSimulation(gpu_mode, cpu_mode);
//
//     m_Vec[PBOUNDMIN].x		= m_Vec[PVOLMIN].x + 2*(m_Param[PGRIDSIZE]/m_Param[PSIMSCALE]);
//     m_Vec[PBOUNDMIN].y		= m_Vec[PVOLMIN].y + 2*(m_Param[PGRIDSIZE]/m_Param[PSIMSCALE]);
//     m_Vec[PBOUNDMIN].z		= m_Vec[PVOLMIN].z + 2*(m_Param[PGRIDSIZE]/m_Param[PSIMSCALE]);
//     m_Vec[PINITMIN].x 		= std::max(m_Vec[PINITMIN].x , m_Vec[PBOUNDMIN].x+1 );
//     m_Vec[PINITMIN].y 		= std::max(m_Vec[PINITMIN].y , m_Vec[PBOUNDMIN].y+1 );
//     m_Vec[PINITMIN].z 		= std::max(m_Vec[PINITMIN].z , m_Vec[PBOUNDMIN].z+1 );
//
//     Vector3DF pinit_max 	= {x_dim,y_dim,z_dim};
//     pinit_max 				+= m_Vec[PINITMIN];
//
//     m_Vec[PBOUNDMAX].x		= m_Vec[PVOLMAX].x - 2*(m_Param[PGRIDSIZE]/m_Param[PSIMSCALE]);
//     m_Vec[PBOUNDMAX].y		= m_Vec[PVOLMAX].y - 2*(m_Param[PGRIDSIZE]/m_Param[PSIMSCALE]);
//     m_Vec[PBOUNDMAX].z		= m_Vec[PVOLMAX].z - 2*(m_Param[PGRIDSIZE]/m_Param[PSIMSCALE]);
//     pinit_max.x 			= std::min(pinit_max.x , m_Vec[PBOUNDMAX].x-1 );
//     pinit_max.y 			= std::min(pinit_max.y , m_Vec[PBOUNDMAX].y-1 );
//     pinit_max.z 			= std::min(pinit_max.z , m_Vec[PBOUNDMAX].z-1 );
// *_/
// 	ReadGenome(			relativePath);
// 	ReadSimParams(		relativePath);
//
//     SetupAddVolumeMorphogenesis2(m_Vec[PINITMIN], pinit_max, spacing, 0.1f, demoType);
//
//     WriteSimParams(		relativePath);
//     WriteGenome(		relativePath);
//     SavePointsCSV2(		relativePath, 1 );
// }
//
// void FluidSystem::WriteExampleSpecificationFile ( const char * relativePath ){ // writes a default version
//     char Specification_file_path[256];
//     sprintf ( Specification_file_path, "%s/SpecificationFile.txt", relativePath );
//     //const char * SimParams_file_path = relativePath;
//     FILE * SpecFile = fopen ( Specification_file_path, "w" );
//     int ret =0;
//
//     ret += std::fprintf ( SpecFile, "num_particles = %u\n ", launchParams.num_particles );
//     ret += std::fprintf ( SpecFile, "demoType = %u\n ", launchParams.demoType );
//     ret += std::fprintf ( SpecFile, "simSpace = %u\n ", launchParams.simSpace );
//     ret += std::fprintf ( SpecFile, "\n");
//
//     ret += std::fprintf ( SpecFile, "m_Time = %f\n ", m_Time );
//     ret += std::fprintf ( SpecFile, "m_DT = %f\n ", m_DT );
//     ret += std::fprintf ( SpecFile, "\n");
//
//     ret += std::fprintf ( SpecFile, "gridsize = %f\n ", m_Param [ PGRIDSIZE ]);
//     ret += std::fprintf ( SpecFile, "spacing = %f\n ", m_Param [ PSPACING ]);
//     ret += std::fprintf ( SpecFile, "\n");
//
//     ret += std::fprintf ( SpecFile, "simscale = %f\n ", m_Param [ PSIMSCALE ]);
//     ret += std::fprintf ( SpecFile, "smooth_radius = %f\n ", m_Param [ PSMOOTHRADIUS ]);
//     ret += std::fprintf ( SpecFile, "\n");
//
//     ret += std::fprintf ( SpecFile, "visc = %f\n ", m_Param [ PVISC ]);
//     ret += std::fprintf ( SpecFile, "surface_t = %f\n ", m_Param [ PSURFACE_TENSION ]);
//     ret += std::fprintf ( SpecFile, "\n");
//
//     ret += std::fprintf ( SpecFile, "mass = %f\n ", m_Param [ PMASS ]);
//     ret += std::fprintf ( SpecFile, "radius = %f\n ", m_Param [ PRADIUS ]);
//     /_*ret += std::fprintf ( SpecFile, "dist = %f\n ", m_Param [ PDIST ]);*_/
//     ret += std::fprintf ( SpecFile, "\n");
//
//     ret += std::fprintf ( SpecFile, "int_stiff = %f\n ", m_Param [ PINTSTIFF ]);
//     ret += std::fprintf ( SpecFile, "ext_stiff = %f\n ", m_Param [ PEXTSTIFF ]);
//     ret += std::fprintf ( SpecFile, "ext_damp = %f\n ", m_Param [ PEXTDAMP ]);
//     ret += std::fprintf ( SpecFile, "\n");
//
//     ret += std::fprintf ( SpecFile, "accel_limit = %f\n ", m_Param [ PACCEL_LIMIT ]);
//     ret += std::fprintf ( SpecFile, "vel_limit = %f\n ", m_Param [ PVEL_LIMIT ]);
//     ret += std::fprintf ( SpecFile, "\n");
//
//     ret += std::fprintf ( SpecFile, "grav = %f\n ", m_Param [ PGRAV ]);
//     ret += std::fprintf ( SpecFile, "slope = %f\n ", m_Param [ PGROUND_SLOPE ]);
//     ret += std::fprintf ( SpecFile, "\n");
//
//     ret += std::fprintf ( SpecFile, "force_min = %f\n ", m_Param [ PFORCE_MIN ]);
//     ret += std::fprintf ( SpecFile, "force_max = %f\n ", m_Param [ PFORCE_MAX ]);
//     ret += std::fprintf ( SpecFile, "force_freq = %f\n ", m_Param [ PFORCE_FREQ ]);
//     ret += std::fprintf ( SpecFile, "\n");
//
//     ret += std::fprintf ( SpecFile, "x_dim = %f\n ", launchParams.x_dim );
//     ret += std::fprintf ( SpecFile, "y_dim = %f\n ", launchParams.y_dim );
//     ret += std::fprintf ( SpecFile, "z_dim = %f\n ", launchParams.z_dim );
//     ret += std::fprintf ( SpecFile, "\n");
//
//     ret += std::fprintf ( SpecFile, "pos_x = %f\n ", launchParams.pos_x );
//     ret += std::fprintf ( SpecFile, "pos_y = %f\n ", launchParams.pos_y );
//     ret += std::fprintf ( SpecFile, "pos_z = %f\n ", launchParams.pos_z );
//     ret += std::fprintf ( SpecFile, "\n");
//
//     ret += std::fprintf ( SpecFile, "volmin_x = %f\n ", m_Vec [ PVOLMIN ].x);
//     ret += std::fprintf ( SpecFile, "volmin_y = %f\n ", m_Vec [ PVOLMIN ].y );
//     ret += std::fprintf ( SpecFile, "volmin_z = %f\n ", m_Vec [ PVOLMIN ].z );
//     ret += std::fprintf ( SpecFile, "\n");
//
//     ret += std::fprintf ( SpecFile, "volmax_x = %f\n ", m_Vec [ PVOLMAX ].x );
//     ret += std::fprintf ( SpecFile, "volmax_y = %f\n ", m_Vec [ PVOLMAX ].y );
//     ret += std::fprintf ( SpecFile, "volmax_z = %f\n ", m_Vec [ PVOLMAX ].z );
//     ret += std::fprintf ( SpecFile, "\n");
//
//     ret += std::fprintf ( SpecFile, "initmin_x = %f\n ", m_Vec [ PINITMIN ].x );
//     ret += std::fprintf ( SpecFile, "initmin_y = %f\n ", m_Vec [ PINITMIN ].y );
//     ret += std::fprintf ( SpecFile, "initmin_z = %f\n ", m_Vec [ PINITMIN ].z );
//     ret += std::fprintf ( SpecFile, "\n");
//
//     ret += std::fprintf ( SpecFile, "initmax_x = %f\n ", m_Vec [ PINITMAX ].x );
//     ret += std::fprintf ( SpecFile, "initmax_y = %f\n ", m_Vec [ PINITMAX ].y );
//     ret += std::fprintf ( SpecFile, "initmax_z = %f\n ", m_Vec [ PINITMAX ].z );
//     ret += std::fprintf ( SpecFile, "\n");
//
//     ret += std::fprintf ( SpecFile, "paramsPath = %s\n ", launchParams.paramsPath );
//     ret += std::fprintf ( SpecFile, "pointsPath = %s\n ", launchParams.pointsPath );
//     ret += std::fprintf ( SpecFile, "genomePath = %s\n ", launchParams.genomePath );
//     ret += std::fprintf ( SpecFile, "outPath = %s\n ", launchParams.outPath );
//     ret += std::fprintf ( SpecFile, "loadSim = %c\n ", launchParams.loadSim );
//     ret += std::fprintf ( SpecFile, "\n");
//
//     ret += std::fprintf ( SpecFile, "num_files = %u\n ", launchParams.num_files );
//     ret += std::fprintf ( SpecFile, "steps_per_InnerPhysicalLoop = %u\n ", launchParams.steps_per_InnerPhysicalLoop );
//     ret += std::fprintf ( SpecFile, "steps_per_file = %u\n ", launchParams.steps_per_file );
//     ret += std::fprintf ( SpecFile, "freeze_steps = %u\n ", launchParams.freeze_steps );
//     ret += std::fprintf ( SpecFile, "motion = %u\n ", launchParams.freeze_steps );
//     ret += std::fprintf ( SpecFile, "\n");
//
//     ret += std::fprintf ( SpecFile, "debug = %u\n ", launchParams.debug );
//     ret += std::fprintf ( SpecFile, "file_num = %u\n ", launchParams.file_num );
//     ret += std::fprintf ( SpecFile, "\n");
//
//     ret += std::fprintf ( SpecFile, "save_ply = %c\n ", launchParams.save_ply );
//     ret += std::fprintf ( SpecFile, "save_csv = %c\n ", launchParams.save_csv );
//     ret += std::fprintf ( SpecFile, "save_vtp = %c\n ", launchParams.save_vtp );
//     ret += std::fprintf ( SpecFile, "\n");
//
//     ret += std::fprintf ( SpecFile, "gene_activity = %c\n ", launchParams.gene_activity );
//     ret += std::fprintf ( SpecFile, "remodelling = %c\n ", launchParams.remodelling );
//     ret += std::fprintf ( SpecFile, "read_genome = %c\n ", launchParams.read_genome );
//     ret += std::fprintf ( SpecFile, "\n");
//
//     ret += std::fprintf ( SpecFile, "actuation_factor = %f\n ", m_Param[PACTUATION_FACTOR] );//  launchParams.actuation_factor );
//     ret += std::fprintf ( SpecFile, "actuation_period = %f\n ", m_Param[PACTUATION_PERIOD] );//  launchParams.actuation_period );
//
//     ret += std::fprintf ( SpecFile, "\n");
//     fflush(SpecFile); fclose(SpecFile);
//     return;
// }
//
// void FluidSystem::SetupDefaultParams (){
//     //  Range = +/- 10.0 * 0.006 (r) =	   0.12			m (= 120 mm = 4.7 inch)
//     //  Container Volume (Vc) =			   0.001728		m^3
//     //  Rest Density (D) =				1000.0			kg / m^3
//     //  Particle Mass (Pm) =			   0.00020543	kg						(mass = vol * density)
//     //  Number of Particles (N) =		4000.0
//     //  Water Mass (M) =				   0.821		kg (= 821 grams)
//     //  Water Volume (V) =				   0.000821     m^3 (= 3.4 cups, .21 gals)
//     //  Smoothing Radius (R) =             0.02			m (= 20 mm = ~3/4 inch)
//     //  Particle Radius (Pr) =			   0.00366		m (= 4 mm  = ~1/8 inch)
//     //  Particle Volume (Pv) =			   2.054e-7		m^3	(= .268 milliliters)
//     //  Rest Distance (Pd) =			   0.0059		m
//     //
//     //  Given: D, Pm, N
//     //    Pv = Pm / D			0.00020543 kg / 1000 kg/m^3        = 2.054e-7 m^3
//     //    Pv = 4/3*pi*Pr^3      cuberoot( 2.054e-7 m^3 * 3/(4pi) ) = 0.00366 m
//     //     M = Pm * N			0.00020543 kg * 4000.0             = 0.821 kg
//     //     V =  M / D           0.821 kg / 1000 kg/m^3             = 0.000821 m^3
//     //     V = Pv * N			2.054e-7 m^3 * 4000                = 0.000821 m^3
//     //    Pd = cuberoot(Pm/D)   cuberoot(0.00020543/1000)          = 0.0059 m
//     //
//     // Ideal grid cell size (gs) = 2 * smoothing radius = 0.02*2 = 0.04
//     // Ideal domain size = k*gs/d = k*0.02*2/0.005 = k*8 = {8, 16, 24, 32, 40, 48, ..}
//     //    (k = number of cells, gs = cell size, d = simulation scale)
//
//     // "The viscosity coefficient is the dynamic viscosity, visc > 0 (units Pa.s),
//     // and to include a reasonable damping contribution, it should be chosen
//     // to be approximately a factor larger than any physical correct viscosity
//     // coefficient that can be looked up in the literature. However, care should
//     // be taken not to exaggerate the viscosity coefficient for fluid materials.
//     // If the contribution of the viscosity force density is too large, the net effect
//     // of the viscosity term will introduce energy into the system, rather than
//     // draining the system from energy as intended."
//     //    Actual visocity of water = 0.001 Pa.s    // viscosity of water at 20 deg C.
//
//     m_Time = 0.0f;							// Start at T=0
//     m_DT = 0.003f;
//
//     m_Param [ PSIMSCALE ] =		0.005f;			// unit size
//     m_Param [ PVISC ] =			0.50f;			// pascal-second (Pa.s) = 1 kg m^-1 s^-1  (see wikipedia page on viscosity)
//     m_Param [ PSURFACE_TENSION ] = 0.1f;
//     m_Param [ PRESTDENSITY ] =	400.0f;			// kg / m^3
//     m_Param [ PSPACING ]	=	0.0f;			// spacing will be computed automatically from density in most examples (set to 0 for autocompute)
//     m_Param [ PMASS ] =			0.00020543f;		// kg
//     m_Param [ PRADIUS ] =		0.015f;			// m
//     m_Param [ PDIST ] =			0.0059f;			// m
//     m_Param [ PSMOOTHRADIUS ] =	0.015f;			// m
//     m_Param [ PINTSTIFF ] =		1.0f;
//     m_Param [ PEXTSTIFF ]     = 50000.0f;
//     m_Param [ PEXTDAMP ]      = 100.0f;
//     m_Param [ PACCEL_LIMIT ]  = 150.0f;			// m / s^2
//     m_Param [ PVEL_LIMIT ]    =   3.0f;			// m / s
//     m_Param [ PGRAV ]         =   1.0f;
//
//     m_Param [ PGROUND_SLOPE ] =  0.0f;
//     m_Param [ PFORCE_MIN ]    =  0.0f;
//     m_Param [ PFORCE_MAX ]    =  0.0f;
//     m_Param [ PFORCE_FREQ ]   = 16.0f;
//     m_Vec [ PPLANE_GRAV_DIR ].Set ( 0, -9.8f, 0 );
//
//     // Default sim config
//     m_Param [PGRIDSIZE]           = m_Param[PSMOOTHRADIUS] * 2;
//
//     m_Param [ PACTUATION_FACTOR ] = 0;
//     m_Param [ PACTUATION_PERIOD ] = 1;
// }
//
// void FluidSystem::SetupExampleParams (uint spacing){
//     Vector3DF pos;
//     Vector3DF min, max;
//     m_Param [ PSPACING ] = spacing;
//
//     //std::cout<<"\nSetupExampleParams()1: m_Param[PEXAMPLE] = "<<m_Param[PEXAMPLE]<<"\n"<<std::flush;
//     //std::cout<<"\nSetupExampleParams()2: launchParams.genomePath = "<<launchParams.genomePath<<"\n"<<std::flush;
//
//     switch ( (int) m_Param[PEXAMPLE] ) {
//
//     case 0:	{	// Regression test. N x N x N static grid
//
//         int k = (int) ceil ( pow ( (float) m_Param[PNUM], (float) 1.0f/3.0f ) );
//         m_Vec [ PVOLMIN ].Set ( 0, 0, 0 );
//         m_Vec [ PVOLMAX ].Set ( 2.0f+(k/2), 2.0f+(k/2), 2.0f+(k/2) );
//         m_Vec [ PINITMIN ].Set ( 1.0f, 1.0f, 1.0f );
//         m_Vec [ PINITMAX ].Set ( 1.0f+(k/2), 1.0f+(k/2), 1.0f+(k/2) );
//
//         m_Param [ PGRAV ] = 0.0;
//         m_Vec [ PPLANE_GRAV_DIR ].Set ( 0.0, 0.0, 0.0 );
//         //m_Param [ PSPACING ] = spacing;//0.5;				// Fixed spacing		Dx = x-axis density
//         m_Param [ PSMOOTHRADIUS ] =	m_Param [PSPACING];		// Search radius
//         //m_Toggle [ PRUN ] = false;				// Do NOT run sim. Neighbors only.
//         //m_Param [PDRAWMODE] = 1;				// Point drawing
//         //m_Param [PDRAWGRID] = 1;				// Grid drawing
//         //m_Param [PDRAWTEXT] = 1;				// Text drawing
//         m_Param [PSIMSCALE ] = 1.0f;
//         launchParams.read_genome = 'y';
//     }
//     break;
//     case 1:		// Tower
//         m_Vec [ PVOLMIN ].Set (   0,   0,   0 );
//         m_Vec [ PVOLMAX ].Set (  256, 128, 256 );
//         m_Vec [ PINITMIN ].Set (  5,   5,  5 );
//         m_Vec [ PINITMAX ].Set ( 256*0.3, 128*0.9, 256*0.3 );
//         break;
//     case 2:		// Wave pool
//         m_Vec [ PVOLMIN ].Set (   0,   0,   0 );
//         m_Vec [ PVOLMAX ].Set (  400, 200, 400 );
//         m_Vec [ PINITMIN ].Set ( 100, 80,  100 );
//         m_Vec [ PINITMAX ].Set ( 300, 190, 300 );
//         m_Param [ PFORCE_MIN ] = 100.0f;
//         m_Param [ PFORCE_FREQ ] = 6.0f;
//         m_Param [ PGROUND_SLOPE ] = 0.10f;
//         break;
//     case 3:		// Small dam break
//         m_Vec [ PVOLMIN ].Set ( -40, 0, -40  );
//         m_Vec [ PVOLMAX ].Set ( 40, 60, 40 );
//         m_Vec [ PINITMIN ].Set ( 0, 8, -35 );
//         m_Vec [ PINITMAX ].Set ( 35, 55, 35 );
//         m_Param [ PFORCE_MIN ] = 0.0f;
//         m_Param [ PFORCE_MAX ] = 0.0f;
//         m_Vec [ PPLANE_GRAV_DIR ].Set ( 0.0f, -9.8f, 0.0f );
//         break;
//     case 4:		// Dual-Wave pool
//         m_Vec [ PVOLMIN ].Set ( -100, 0, -15 );
//         m_Vec [ PVOLMAX ].Set ( 100, 100, 15 );
//         m_Vec [ PINITMIN ].Set ( -80, 8, -10 );
//         m_Vec [ PINITMAX ].Set ( 80, 90, 10 );
//         m_Param [ PFORCE_MIN ] = 20.0;
//         m_Param [ PFORCE_MAX ] = 20.0;
//         m_Vec [ PPLANE_GRAV_DIR ].Set ( 0.0f, -9.8f, 0.0f );
//         break;
//     case 5:		// Microgravity
//         m_Vec [ PVOLMIN ].Set ( -80, 0, -80 );
//         m_Vec [ PVOLMAX ].Set ( 80, 100, 80 );
//         m_Vec [ PINITMIN ].Set ( -60, 40, -60 );
//         m_Vec [ PINITMAX ].Set ( 60, 80, 60 );
//         m_Vec [ PPLANE_GRAV_DIR ].Set ( 0, -1, 0 );
//         m_Param [ PGROUND_SLOPE ] = 0.1f;
//         break;
//     case 6:     // Morphogenesis small demo
//         m_Param [ PSIMSCALE ] = 1.0f;
//         m_Param [ PRADIUS ] = 1.0f;
//         m_Param [ PSMOOTHRADIUS ] = 1.0f;
//         m_Param [ PVISC ] = 0.1f;
//
//         m_Vec [ PVOLMIN ].Set ( 0, 0, 0 );
//         m_Vec [ PVOLMAX ].Set ( 10, 20, 50 ); //( 80, 50, 80 );
//         m_Vec [ PINITMIN ].Set ( m_Vec [ PVOLMIN ].x,  m_Vec [ PVOLMIN ].y, m_Vec [ PVOLMIN ].z );// will be reset to m_Vec[PBOUNDMIN].
//         m_Vec [ PINITMAX ].Set ( 10, 20, 30 );
//
//         m_Param [ PGRAV ] = 2.000000f;
//         m_Vec [ PPLANE_GRAV_DIR ].Set ( 0, -1, 0 );
//         m_Param [ PGROUND_SLOPE ] = 0.1f;
//         break;
//     case 7:     // From SpecificationFile.txt
//         m_Time = launchParams.m_Time;
//         m_DT = launchParams.m_DT;
//         m_Param [ PGRIDSIZE ] = launchParams.gridsize;
//         m_Param [ PSPACING ] = launchParams.spacing;
//         m_Param [ PSIMSCALE ] = launchParams.simscale;
//         m_Param [ PSMOOTHRADIUS ] = launchParams.smoothradius;
//         m_Param [ PVISC ] = launchParams.visc;
//         m_Param [ PSURFACE_TENSION ] = launchParams.surface_tension;
//         m_Param [ PMASS ] = launchParams.mass;
//         m_Param [ PRADIUS ] = launchParams.radius;
//         /_*m_Param [ PDIST ] = launchParams.dist;*_/
//         m_Param [ PINTSTIFF ] = launchParams.intstiff;
//         m_Param [ PEXTSTIFF ] = launchParams.extstiff;
//         m_Param [ PEXTDAMP ] = launchParams.extdamp;
//         m_Param [ PACCEL_LIMIT ] = launchParams.accel_limit;
//         m_Param [ PVEL_LIMIT ] = launchParams.vel_limit;
//         m_Param [ PGRAV ] = launchParams.grav;
//         m_Param [ PGROUND_SLOPE ] = launchParams.ground_slope;
//         m_Param [ PFORCE_MIN ] = launchParams.force_min;
//         m_Param [ PFORCE_MAX ] = launchParams.force_max;
//         m_Param [ PFORCE_FREQ ] = launchParams.force_freq;
//
//         m_Vec [ PVOLMIN ] = launchParams.volmin;
//         m_Vec [ PVOLMAX ] = launchParams.volmax;
//         m_Vec [ PINITMIN ] = launchParams.initmin;
//         m_Vec [ PINITMAX ] = launchParams.initmax;
//
//         m_Param [ PACTUATION_FACTOR ] = launchParams.actuation_factor;
//         m_Param [ PACTUATION_PERIOD ] = launchParams.actuation_period;
//
//         break;
//     case 8:  // default demo for parameter sweeps
//         launchParams.num_particles = 4000;
//         launchParams.demoType = 0;
//         launchParams.simSpace = 7;
//
//         m_Time = 0.000000;
//         m_DT = 0.003000;
//
//         m_Param [ PGRIDSIZE ] = 1.0;
//         m_Param [ PSPACING ] = 1.000000;
//
//         m_Param [ PSIMSCALE ] = 1.0;
//         m_Param [ PSMOOTHRADIUS ] = 1.0;
//
//         m_Param [ PVISC ] = 0.500000;
//         m_Param [ PSURFACE_TENSION ] = 1.000000;
//
//         m_Param [ PMASS ] = 0.00205;
//         m_Param [ PRADIUS ] = 1.0000;
//
//         m_Param [ PINTSTIFF ] = 2.000000;
//         m_Param [ PEXTSTIFF ] = 50000.000000;
//         m_Param [ PEXTDAMP ] = 100.000000;
//
//         m_Param [ PACCEL_LIMIT ] = 150.000000;
//         m_Param [ PVEL_LIMIT ] = 3.000000;
//
//         m_Param [ PGRAV ] = 10.000000;
//         m_Param [ PGROUND_SLOPE ] = 0.100000;
//
//         m_Param [ PFORCE_MIN ] = 0.000000;
//         m_Param [ PFORCE_MAX ] = 0.000000;
//         m_Param [ PFORCE_FREQ ] = 16.000000;
//
//         launchParams.x_dim = 10.000000;
//         launchParams.y_dim = 10.000000;
//         launchParams.z_dim = 3.000000;
//
//         launchParams.pos_x = 0.000000;
//         launchParams.pos_y = 0.000000;
//         launchParams.pos_z = 0.000000;
//
//         m_Vec [ PVOLMIN ].Set ( 0.0, 0.0, 0.0 );
//         m_Vec [ PVOLMAX ].Set ( 10.0, 20.0, 20.0 );
//         m_Vec [ PINITMIN ].Set ( 2.0, 2.0, 2.0 );
//         m_Vec [ PINITMAX ].Set ( 10.0, 20.0, 10.0 );
//
//         launchParams.num_files = 4000;
//         launchParams.steps_per_InnerPhysicalLoop = 3;
//         launchParams.steps_per_file = 6;
//         launchParams.freeze_steps = 1;
//
//         launchParams.debug = 0;
//         launchParams.file_num = 0;
//
//         launchParams.save_ply = 'n';
//         launchParams.save_csv = 'n';
//         launchParams.save_vtp = 'n';
//
//         launchParams.gene_activity = 'n';
//         launchParams.remodelling = 'n';
//         launchParams.read_genome = 'y';
//
//         m_Param [ PACTUATION_FACTOR ] = 0;
//         m_Param [ PACTUATION_PERIOD ] = 1;
//
//         break;
//     }
//     //std::cout<<"\nSetupExampleParams()3: launchParams.genomePath = "<<launchParams.genomePath<<"\n"<<std::flush;
// }
//
// void FluidSystem::SetupExampleGenome()  {   // need to set up a demo genome
//     // Null genome
//     for(int i=0; i< NUM_GENES; i++) m_FGenome.mutability[i] = 0;
//     for(int i=0; i< NUM_GENES; i++) m_FGenome.delay[i] = 1;
//     for(int i=0; i< NUM_GENES; i++) for(int j=0; j< NUM_GENES; j++) m_FGenome.sensitivity[i][j] = j;
//
//     for(int i=0; i< NUM_TF/2; i++)      m_FGenome.tf_diffusability[i]    = 0;           // 1st half of TFs are non-diffusible.
//     for(int i=NUM_TF/2; i< NUM_TF; i++) m_FGenome.tf_diffusability[i]    = 1;
//     for(int i=0; i< NUM_TF; i++)        m_FGenome.tf_breakdown_rate[i]  = 1;
//
//     for(int i=0; i< NUM_GENES; i++) for(int j=0; j< 2*NUM_TF+1; j++) m_FGenome.secrete[i][j]=0;     // 1st zero arrays.
//     for(int i=0; i< NUM_GENES; i++) for(int j=0; j< 2*NUM_TF+1; j++) m_FGenome.activate[i][j]=0;
//
//     m_FGenome.secrete[0][2*NUM_TF] = 2; // gene [0] secretes TF 1 & 3, at rates 1 & 4. // minimal test case.
//     m_FGenome.secrete[0][2*0] = 1;
//     m_FGenome.secrete[0][2*0+1] = 1;
//     m_FGenome.secrete[0][2*1] = 3;
//     m_FGenome.secrete[0][2*1+1] = 4;
//
//     m_FGenome.activate[0][2*NUM_TF] = 1; // gene [0] activates TF 5, at rates 6. // minimal test case.
//     m_FGenome.activate[0][2*0] = 5;
//     m_FGenome.activate[0][2*0+1] = 6;
//
//     //FBondParams *params_ =  &m_FGenome.fbondparams[0];
//     //Particle remodelling, bond defaults & limits  m_FGenome.param[3][12]
//     //0=elastin
//     m_FGenome.param[0][m_FGenome.elongation_threshold]   = 0.5  ;
//     m_FGenome.param[0][m_FGenome.elongation_factor]      = 0.002  ;
//     m_FGenome.param[0][m_FGenome.strength_threshold]     = 1.0  ;
//     m_FGenome.param[0][m_FGenome.strengthening_factor]   = 0.002  ;
//
//     m_FGenome.param[0][m_FGenome.max_rest_length]        = 1.0  ;
//     m_FGenome.param[0][m_FGenome.min_rest_length]        = 0.3  ;
//     m_FGenome.param[0][m_FGenome.max_modulus]            = 100000;
//     m_FGenome.param[0][m_FGenome.min_modulus]            = 10  ;
//
//     m_FGenome.param[0][m_FGenome.elastLim]               = 8  ;
//     m_FGenome.param[0][m_FGenome.default_rest_length]    = 0.5  ;
//     m_FGenome.param[0][m_FGenome.default_modulus]        = 1000;
//     m_FGenome.param[0][m_FGenome.default_damping]        = 10;
//
//     //1=collagen
//     m_FGenome.param[1][m_FGenome.elongation_threshold]   = 4.0  ;
//     m_FGenome.param[1][m_FGenome.elongation_factor]      = 0.01 ;
//     m_FGenome.param[1][m_FGenome.strength_threshold]     = 4.1  ;
//     m_FGenome.param[1][m_FGenome.strengthening_factor]   = 0.01 ;
//
//     m_FGenome.param[1][m_FGenome.max_rest_length]        = 1.0  ;
//     m_FGenome.param[1][m_FGenome.min_rest_length]        = 0.3  ;
//     m_FGenome.param[1][m_FGenome.max_modulus]            = 1000000000;//0.8  ;
//     m_FGenome.param[1][m_FGenome.min_modulus]            = 1000000;//0.3  ;
//
//     m_FGenome.param[1][m_FGenome.elastLim]               = 0.55  ;
//     m_FGenome.param[1][m_FGenome.default_rest_length]    = 0.5  ;
//     m_FGenome.param[1][m_FGenome.default_modulus]        = 10000000  ;
//     m_FGenome.param[1][m_FGenome.default_damping]        = 100  ;
//
//     //2=apatite
//     m_FGenome.param[2][m_FGenome.elongation_threshold]   = 1.0  ;
//     m_FGenome.param[2][m_FGenome.elongation_factor]      = 0.001  ;
//     m_FGenome.param[2][m_FGenome.strength_threshold]     = 1.0  ;
//     m_FGenome.param[2][m_FGenome.strengthening_factor]   = 0.001  ;
//
//     m_FGenome.param[2][m_FGenome.max_rest_length]        = 1.0  ;
//     m_FGenome.param[2][m_FGenome.min_rest_length]        = 0.3  ;
//     m_FGenome.param[2][m_FGenome.max_modulus]            = 1000000000;//0.8  ;
//     m_FGenome.param[2][m_FGenome.min_modulus]            = 1000000;//0.3  ;
//
//     m_FGenome.param[2][m_FGenome.elastLim]               = 0.05  ;
//     m_FGenome.param[2][m_FGenome.default_rest_length]    = 0.5  ;
//     m_FGenome.param[2][m_FGenome.default_modulus]        = 10000000  ;
//     m_FGenome.param[2][m_FGenome.default_damping]        = 1000  ;
//
//     //Bond remodelling  m_FGenome.tanh_param[3][8];                 // lengthening/shortening
//     m_FGenome.tanh_param[m_FGenome.elastin][m_FGenome.l_a] = 0.0;   // y-shift
//     m_FGenome.tanh_param[m_FGenome.elastin][m_FGenome.l_b] = 0.0;   // y-scaling
//     m_FGenome.tanh_param[m_FGenome.elastin][m_FGenome.l_c] = 0.0;   // x-scaling
//     m_FGenome.tanh_param[m_FGenome.elastin][m_FGenome.l_d] = 0.0;   // x-shift
//
//     m_FGenome.tanh_param[m_FGenome.elastin][m_FGenome.s_a] = 1.008; // strengthening/weakening  // mod_mul = 1.008+0.01*np.tanh(10*si2[i+1]-7)
//     m_FGenome.tanh_param[m_FGenome.elastin][m_FGenome.s_b] = 0.01;
//     m_FGenome.tanh_param[m_FGenome.elastin][m_FGenome.s_c] = 10.0;
//     m_FGenome.tanh_param[m_FGenome.elastin][m_FGenome.s_d] = 7.0;
//
//     m_FGenome.tanh_param[m_FGenome.collagen][m_FGenome.l_a] = 0.0;  // lengthening/shortening
//     m_FGenome.tanh_param[m_FGenome.collagen][m_FGenome.l_b] = 0.0;
//     m_FGenome.tanh_param[m_FGenome.collagen][m_FGenome.l_c] = 0.0;
//     m_FGenome.tanh_param[m_FGenome.collagen][m_FGenome.l_d] = 0.0;
//
//     m_FGenome.tanh_param[m_FGenome.collagen][m_FGenome.s_a] = 0.0;  // strengthening/weakening
//     m_FGenome.tanh_param[m_FGenome.collagen][m_FGenome.s_b] = 0.0;
//     m_FGenome.tanh_param[m_FGenome.collagen][m_FGenome.s_c] = 0.0;
//     m_FGenome.tanh_param[m_FGenome.collagen][m_FGenome.s_d] = 0.0;
//
//     m_FGenome.tanh_param[m_FGenome.apatite][m_FGenome.l_a] = 0.0;   // lengthening/shortening
//     m_FGenome.tanh_param[m_FGenome.apatite][m_FGenome.l_b] = 0.0;
//     m_FGenome.tanh_param[m_FGenome.apatite][m_FGenome.l_c] = 0.0;
//     m_FGenome.tanh_param[m_FGenome.apatite][m_FGenome.l_d] = 0.0;
//
//     m_FGenome.tanh_param[m_FGenome.apatite][m_FGenome.s_a] = 0.0;   // strengthening/weakening
//     m_FGenome.tanh_param[m_FGenome.apatite][m_FGenome.s_b] = 0.0;
//     m_FGenome.tanh_param[m_FGenome.apatite][m_FGenome.s_c] = 0.0;
//     m_FGenome.tanh_param[m_FGenome.apatite][m_FGenome.s_d] = 0.0;
//
// }
*/
