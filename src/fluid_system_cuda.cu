//----------------------------------------------------------------------------------
//
// FLUIDS v.3 - SPH Fluid Simulator for CPU and GPU
// Copyright (C) 2012-2013. Rama Hoetzlein, http://fluids3.com
//
// BSD 3-clause:
// Redistribution and use in source and binary forms, with or without modification, 
// are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this 
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this 
//    list of conditions and the following disclaimer in the documentation and/or 
//    other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors may 
//    be used to endorse or promote products derived from this software without specific 
//   prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
// SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT 
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) 
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR 
// TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//----------------------------------------------------------------------------------

#define CUDA_KERNEL
#include "fluid_system_cuda.cuh"

#include "cutil_math.h"			// cutil32.lib
#include <string.h>
#include <assert.h>
#include <curand.h>
#include <curand_kernel.h>

__constant__ FParams		fparam;			// CPU Fluid params
__constant__ FBufs			fbuf;			// GPU Particle buffers (unsorted). An FBufs struct holds an array of pointers. 
__constant__ FBufs			ftemp;			// GPU Particle buffers (sorted)
__constant__ FGenome		fgenome;		// GPU Genome for particle automata behaviour. Also holds morphogen diffusability.
__constant__ uint			gridActive;

#define SCAN_BLOCKSIZE		512

extern "C" __global__ void insertParticles ( int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;

	//-- debugging (pointers should match CUdeviceptrs on host side)
	// printf ( " pos: %012llx, gcell: %012llx, gndx: %012llx, gridcnt: %012llx\n", fbuf.bufC(FPOS), fbuf.bufC(FGCELL), fbuf.bufC(FGNDX), fbuf.bufC(FGRIDCNT) );

	register float3 gridMin =	fparam.gridMin;      // "register" is a compiler 'hint', to keep this variable in thread register
	register float3 gridDelta = fparam.gridDelta;    //  even if other variable have to be moved to slower 'local' memory  
	register int3 gridRes =		fparam.gridRes;      //  in the streaming multiprocessor's cache.
	register int3 gridScan =	fparam.gridScanMax;

	register int		gs;
	register float3		gcf;
	register int3		gc;	

	gcf = (fbuf.bufF3(FPOS)[i] - gridMin) * gridDelta; 
	gc = make_int3( int(gcf.x), int(gcf.y), int(gcf.z) );
	gs = (gc.y * gridRes.z + gc.z)*gridRes.x + gc.x;

	if ( gc.x >= 1 && gc.x <= gridScan.x && gc.y >= 1 && gc.y <= gridScan.y && gc.z >= 1 && gc.z <= gridScan.z ) {
		fbuf.bufI(FGCELL)[i] = gs;											// Grid cell insert.
		fbuf.bufI(FGNDX)[i] = atomicAdd ( &fbuf.bufI(FGRIDCNT)[ gs ], 1 );		// Grid counts.

		//gcf = (-make_float3(poff,poff,poff) + fbuf.bufF3(FPOS)[i] - gridMin) * gridDelta;
		//gc = make_int3( int(gcf.x), int(gcf.y), int(gcf.z) );
		//gs = ( gc.y * gridRes.z + gc.z)*gridRes.x + gc.x;
	} else {
		fbuf.bufI(FGCELL)[i] = GRID_UNDEF;		
	}
}

// Counting Sort - Full (deep copy)
extern "C" __global__ void countingSortFull ( int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;		// particle index				
	if ( i >= pnum ) return;

	// Copy particle from original, unsorted buffer (msortbuf),
	// into sorted memory location on device (mpos/mvel)
	uint icell = ftemp.bufI(FGCELL) [ i ];	

	if ( icell != GRID_UNDEF ) {	  
		// Determine the sort_ndx, location of the particle after sort		
		uint indx =  ftemp.bufI(FGNDX)  [ i ];		
	    int sort_ndx = fbuf.bufI(FGRIDOFF) [ icell ] + indx ;	// global_ndx = grid_cell_offet + particle_offset	
		//printf ( "%d: cell: %d, off: %d, ndx: %d\n", i, icell, fbuf.bufI(FGRIDOFF)[icell], indx );
		float3 zero; zero.x=0;zero.y=0;zero.z=0;
		// Transfer data to sort location
		fbuf.bufI (FGRID) [ sort_ndx ] =	sort_ndx;			// full sort, grid indexing becomes identity		
		fbuf.bufF3(FPOS) [sort_ndx] =		ftemp.bufF3(FPOS) [i];
		fbuf.bufF3(FVEL) [sort_ndx] =		ftemp.bufF3(FVEL) [i];
		fbuf.bufF3(FVEVAL)[sort_ndx] =		ftemp.bufF3(FVEVAL) [i];
		fbuf.bufF3(FFORCE)[sort_ndx] =	zero;// old:	ftemp.bufF3(FFORCE) [i];  
                                            // fbuf.bufF3(FFORCE)[ i ] += force; in contributeForce() requires value setting to 0
		fbuf.bufF (FPRESS)[sort_ndx] =		ftemp.bufF(FPRESS) [i];
		fbuf.bufF (FDENSITY)[sort_ndx] =	ftemp.bufF(FDENSITY) [i];
		fbuf.bufI (FCLR) [sort_ndx] =		ftemp.bufI(FCLR) [i];
		fbuf.bufI (FGCELL) [sort_ndx] =		icell;
		fbuf.bufI (FGNDX) [sort_ndx] =		indx;		
        
        // add extra data for morphogenesis
        for (int a=0;a<BONDS_PER_PARTICLE*2;a++){
            fbuf.bufI (FELASTIDX) [sort_ndx*BONDS_PER_PARTICLE*2 + a] =	ftemp.bufI(FELASTIDX) [i*BONDS_PER_PARTICLE*2 + a]; //sort_ndx= grid_cell_offet + particle_offset   , i=particle index
        }
        fbuf.bufI (FPARTICLE_ID) [sort_ndx] =	ftemp.bufI(FPARTICLE_ID) [i];
        fbuf.bufI (FMASS_RADIUS) [sort_ndx] =	ftemp.bufI(FMASS_RADIUS) [i];
        
        fbuf.bufI (FNERVEIDX) [sort_ndx] =	ftemp.bufI(FNERVEIDX) [i];
        fbuf.bufI (FCONC) [sort_ndx] =		ftemp.bufI(FCONC) [i];
        fbuf.bufI (FEPIGEN) [sort_ndx] =	ftemp.bufI(FEPIGEN) [i];  
	}
} 

extern "C" __device__ float contributePressure ( int i, float3 p, int cell )  
// pressure due to particles in 'cell'. NB for each particle there are 27 cells in which interacting particles might be.
{			
	if ( fbuf.bufI(FGRIDCNT)[cell] == 0 ) return 0.0;                                  // If the cell is empty, skip it.

	float3 dist;
	float dsq, c, sum = 0.0;
	register float d2 = fparam.psimscale * fparam.psimscale;
	register float r2 = fparam.r2 / d2;
	
	int clast = fbuf.bufI(FGRIDOFF)[cell] + fbuf.bufI(FGRIDCNT)[cell];     // off set of this cell in the list of particles,  PLUS  the count of particles in this cell.

	for ( int cndx = fbuf.bufI(FGRIDOFF)[cell]; cndx < clast; cndx++ ) {   // For particles in this cell.
		int pndx = fbuf.bufI(FGRID) [cndx];                                       // index of this particle
		dist = p - fbuf.bufF3(FPOS) [pndx];                                       // float3 distance between this particle, and the particle for which the loop has been called.
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);                    // scalar distance squared
		if ( dsq < r2 && dsq > 0.0) {                                             // IF in-range && not the same particle. 
			c = (r2 - dsq)*d2;                                                           //(NB this means all unused particles can be stored at one point)
			sum += c * c * c;				
		} 
	}
	
	return sum;                                                     // NB a scalar value for pressure contribution, at the current particle, due to particles in this cell.
}
			
extern "C" __global__ void computePressure ( int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                // particle index
	if ( i >= pnum ) return;

	// Get search cell
	int nadj = (1*fparam.gridRes.z + 1)*fparam.gridRes.x + 1;
	uint gc = fbuf.bufI(FGCELL) [i];                                       // get grid cell of the current particle.
	if ( gc == GRID_UNDEF ) return;                                        // IF particle not in the simulation
	gc -= nadj;

	// Sum Pressures
	float3 pos = fbuf.bufF3(FPOS) [i];
	float sum = 0.0;
	for (int c=0; c < fparam.gridAdjCnt; c++) {                                    
		sum += contributePressure ( i, pos, gc + fparam.gridAdj[c] );
	}
	__syncthreads();
		
	// Compute Density & Pressure
	sum = sum * fparam.pmass * fparam.poly6kern;
	if ( sum == 0.0 ) sum = 1.0;
	fbuf.bufF(FPRESS)  [ i ] = ( sum - fparam.prest_dens ) * fparam.pintstiff;
	fbuf.bufF(FDENSITY)[ i ] = 1.0f / sum;
}

extern "C" __device__ float3 contributeForce ( int i, float3 ipos, float3 iveleval, float ipress, float idens, int cell, uint particleID,
                                               uint elastIdx[BONDS_PER_PARTICLE * 2], uint bond[BONDS_PER_PARTICLE][2], 
                                               uchar intact[BONDS_PER_PARTICLE]
                                               /*uint particleID[BONDS_PER_PARTICLE], uint  BondType[BONDS_PER_PARTICLE]*/)
{			
	if ( fbuf.bufI(FGRIDCNT)[cell] == 0 ) return make_float3(0,0,0);               // If the cell is empty, skip it.

	float dsq, c, pterm;	
	float3 dist, eterm, force = make_float3(0,0,0);
	int j;
    

	int clast = fbuf.bufI(FGRIDOFF)[cell] + fbuf.bufI(FGRIDCNT)[cell];

	for ( int cndx = fbuf.bufI(FGRIDOFF)[cell]; cndx < clast; cndx++ ) {          // For particles in this cell.
		j = fbuf.bufI(FGRID)[ cndx ];				                                     
		dist = ( ipos - fbuf.bufF3(FPOS)[ j ] );                                  // dist in cm (Rama's comment)
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);                    // scalar distance squared
		if ( dsq < fparam.rd2 && dsq > 0) {                                       // IF in-range && not the same particle
			dsq = sqrt(dsq * fparam.d2);                                          // sqrt(dist^2 * sim_scale^2))
			c = ( fparam.psmoothradius - dsq ); 
			pterm = fparam.psimscale * -0.5f * c * fparam.spikykern * ( ipress + fbuf.bufF(FPRESS)[ j ] ) / dsq;			
            
            //  elastic force due to bonds
            uint rest_len = REST_LENGTH * fparam.pradius;
            for(int a=1; a < BONDS_PER_PARTICLE ; a++){ // for this other particle, check list of bonds for particle of this thread
                if(particleID == elastIdx[a]){      // bond present
                    float abs_dist = sqrt(dsq);
                    if( abs_dist < bond[a][1] ){    // bond intact
                        eterm = dist * ( (rest_len - abs_dist) * bond[a][0] / abs_dist ); // exerts force
                        intact[a]+=1;                // mark bond intact n.b. this catches double counting as well as out of range bonds.
                    }
                    break;                          // bond processed => break out of for loop 
                }
            }            
            
			force += (eterm + pterm * dist + fparam.vterm * ( fbuf.bufF3(FVEVAL)[ j ] - iveleval )) * c * idens * (fbuf.bufF(FDENSITY)[ j ] ); 
            /*
            // force due to pressure gradient PLUS viscosity.
            //   pterm(-ve scalar) * distance(float3) => repulsion
            //   viscosity * (vel(float3)[j] - vel(float3)[i]) * c * density[i] * desity[j]  => drag
               
                //uint b = fbuf.bufI(FELASTIDX)[i+a]   ;//[i][a];  
//                 if(fbuf.bufI(FELASTIDX)[j] == b){
//                     float abs_dist = sqrt(dsq);
//                     eterm = dist * ( (rest_len - abs_dist) * YoungsModulus / abs_dist );  // elastic_force  // NB need to split abs_force by xyz of 
//                     fbuf.bufI(FELASTIDX)[j]  |= TWO_POW_31;                       // bitwise inclusive OR, mark bond as intact, NB this is a uint,
//                 }
            */
            /*
            // fbuf.bufI(FELASTIDX)[0]   is 24bits particle ID => 2^24 = 16,777,216 particles, and 8 bits bond type => 2^8 = 256 bond types.
            //                              look up bond type in texture memory => modulus. Use fixed bond length.
            // fbuf.bufI(FELASTIDX)[0]   is 24 bits particle ID of bonded particle, and 8 bits of other data. Includes 'broken bond' flag.
            
            
            //  for(int a=1; a < BONDS_PER_PARTICLE ; a++)
            //  { 
            //      unit b = m_Fluid.gpu(FELASTIDX)[i][a];     
            //      if(m_Fluid.gpu(FELASTIDX)[j][0] == b)
            //      {
            //          contribute_elastic_force()
            //          m_Fluid.gpu(FELASTIDX)[i][a]  *= (-1);     //mark bond as intact  ? is this a uint !?
            //      }
            //
            //  if(  m_Fluid.gpu(FELASTIDX)[0] == 
            */
        }
	}
	return force;
}


extern "C" __global__ void computeForce ( int pnum)
{			
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	          // particle index				
	if ( i >= pnum ) return;

	// Get search cell	
	uint gc = fbuf.bufI(FGCELL)[ i ];
	if ( gc == GRID_UNDEF ) return;						              // particle out-of-range
	gc -= (1*fparam.gridRes.z + 1)*fparam.gridRes.x + 1;

	// Sum Pressures	
	register float3 force;                                            // request to compiler to store in a register for speed.
	force = make_float3(0,0,0);		
    
    uint elastIdx[BONDS_PER_PARTICLE];
    uint bond[BONDS_PER_PARTICLE][2];
    for (int a=0;a<BONDS_PER_PARTICLE;a++){                             // copy FELASTIDX to thread memory for particle i.
        elastIdx[a] = fbuf.bufI(FELASTIDX)[i*BONDS_PER_PARTICLE*2 + a*2];// particle IDs   i*BONDS_PER_PARTICLE*2 + a  //  [i][a*2]
        uint temp = fbuf.bufI(FELASTIDX)[i*BONDS_PER_PARTICLE*2 + a*2 +1]; //[i][a*2]
        bond[a][0] = temp & TWO_POW_24_MINUS_1;     // modulus          // '&' bitwise AND is bit masking.  
        bond[a][1] = (temp >> 24);                  // elastic limit    // '>>' Bit shift can deliver high bits to bottom
    }
    uchar intact[BONDS_PER_PARTICLE];
    for (int a=0;a<BONDS_PER_PARTICLE;a++){ intact[a]=0;}               // ledger for chcking bonds are intact.
    
    uint particleID = fbuf.bufI(FPARTICLE_ID)[i];   // ID of this particle
    
/*    
//    // get ElastIdx[4], particle IDs and young's modulus. // NB need yield strain and other info too.
//     uint BondType[BONDS_PER_PARTICLE];                                               // BondType[0] can store flag bit for broken bonds. 
//     uint particleID[BONDS_PER_PARTICLE];
// 
//     for (int i=0;i<BONDS_PER_PARTICLE;i++){
//         particleID[i] = fbuf.bufI(FELASTIDX)[i]  & TWO_POW_24 ;     // '&' bitwise AND is bit masking.  '>>' Bit shift can deliver high bits to bottom.
//         BondType[i] = fbuf.bufI(FELASTIDX)[i];
//         BondType[i] >> 24;                               // Young's modulus = pow(2,(BondType/6.4)) => 0-256 log spread from gel to diamond.
//     }
*/    
    
	for (int c=0; c < fparam.gridAdjCnt; c++) {
		force += contributeForce ( i, fbuf.bufF3(FPOS)[ i ], fbuf.bufF3(FVEVAL)[ i ], fbuf.bufF(FPRESS)[ i ], fbuf.bufF(FDENSITY)[ i ], gc + fparam.gridAdj[c], particleID, elastIdx, bond , intact); /*particleID, BondType*/
	}
	fbuf.bufF3(FFORCE)[ i ] += force;  //  += req for elastic bonds. NB need to reset to zero in  CountingSortFull(..)
	
	for (int a=0;a<BONDS_PER_PARTICLE;a++){                            // remove broken bonds
        if(intact[a]!=1){fbuf.bufII(FELASTIDX)[i*BONDS_PER_PARTICLE*2 + a*2 +1]=0;} // nb particle_ID = 0 must be "NO_PARTICLE"  // [i][a*2]
    }
}


extern "C" __global__ void randomInit ( int seed, int numPnts )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= numPnts ) return;

	// Initialize particle random generator	
	curandState_t* st = (curandState_t*) (fbuf.bufC(FSTATE) + i*sizeof(curandState_t));
	curand_init ( seed + i, 0, 0, st );		
}

#define CURANDMAX		2147483647

extern "C" __global__ void emitParticles ( float frame, int emit, int numPnts )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= emit ) return;

	curandState_t* st = (curandState_t*) (fbuf.bufC(FSTATE) + i*sizeof(curandState_t));
	uint v = curand( st);
	uint j = v & (numPnts-1);
	float3 bmin = make_float3(-170,10,-20);
	float3 bmax = make_float3(-190,60, 20);

	float3 pos = make_float3(0,0,0);	
	pos.x = float( v & 0xFF ) / 256.0;
	pos.y = float((v>>8) & 0xFF ) / 256.0;
	pos.z = float((v>>16) & 0xFF ) / 256.0;
	pos = bmin + pos*(bmax-bmin);	
	
	fbuf.bufF3(FPOS)[j] = pos;
	fbuf.bufF3(FVEVAL)[j] = make_float3(0,0,0);
	fbuf.bufF3(FVEL)[j] = make_float3(5,-2,0);
	fbuf.bufF3(FFORCE)[j] = make_float3(0,0,0);	
	
}

__device__ uint getGridCell ( float3 pos, uint3& gc )
{	
	gc.x = (int)( (pos.x - fparam.gridMin.x) * fparam.gridDelta.x);			// Cell in which particle is located
	gc.y = (int)( (pos.y - fparam.gridMin.y) * fparam.gridDelta.y);
	gc.z = (int)( (pos.z - fparam.gridMin.z) * fparam.gridDelta.z);		
	return (int) ( (gc.y*fparam.gridRes.z + gc.z)*fparam.gridRes.x + gc.x);	
}

extern "C" __global__ void sampleParticles ( float* brick, uint3 res, float3 bmin, float3 bmax, int numPnts, float scalar )
{
	float3 dist;
	float dsq;
	int j, cell;	
	register float r2 = fparam.r2;
	register float h2 = 2.0*r2 / 8.0;		// 8.0=smoothing. higher values are sharper

	uint3 i = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + threadIdx;
	if ( i.x >= res.x || i.y >= res.y || i.z >= res.z ) return;
	
	float3 p = bmin + make_float3(float(i.x)/res.x, float(i.y)/res.y, float(i.z)/res.z) * (bmax-bmin);
	//float3 v = make_float3(0,0,0);
	float v = 0.0;

	// Get search cell
	int nadj = (1*fparam.gridRes.z + 1)*fparam.gridRes.x + 1;
	uint3 gc;
	uint gs = getGridCell ( p, gc );
	if ( gc.x < 1 || gc.x > fparam.gridRes.x-fparam.gridSrch || gc.y < 1 || gc.y > fparam.gridRes.y-fparam.gridSrch || gc.z < 1 || gc.z > fparam.gridRes.z-fparam.gridSrch ) {
		brick[ (i.y*int(res.z) + i.z)*int(res.x) + i.x ] = 0.0;
		return;
	}

	gs -= nadj;	

	for (int c=0; c < fparam.gridAdjCnt; c++) {
		cell = gs + fparam.gridAdj[c];		
		if ( fbuf.bufI(FGRIDCNT)[cell] != 0 ) {				
			for ( int cndx = fbuf.bufI(FGRIDOFF)[cell]; cndx < fbuf.bufI(FGRIDOFF)[cell] + fbuf.bufI(FGRIDCNT)[cell]; cndx++ ) {
				j = fbuf.bufI(FGRID)[cndx];
				dist = p - fbuf.bufF3(FPOS)[ j ];
				dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
				if ( dsq < fparam.rd2 && dsq > 0 ) {
					dsq = sqrt(dsq * fparam.d2);					
					//v += fbuf.mvel[j] * (fparam.gausskern * exp ( -(dsq*dsq)/h2 ) / fbuf.mdensity[ j ]);
					v += fparam.gausskern * exp ( -(dsq*dsq)/h2 );
				}
			}
		}
	}
	__syncthreads();

	brick[ (i.z*int(res.y) + i.y)*int(res.x) + i.x ] = v * scalar;
	//brick[ (i.z*int(res.y) + i.y)*int(res.x) + i.x ] = length(v) * scalar;
}

extern "C" __global__ void computeQuery ( int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;

	// Get search cell
	int nadj = (1*fparam.gridRes.z + 1)*fparam.gridRes.x + 1;
	uint gc = fbuf.bufI(FGCELL) [i];
	if ( gc == GRID_UNDEF ) return;						// particle out-of-range
	gc -= nadj;

	// Sum Pressures
	float sum = 0.0;
	for (int c=0; c < fparam.gridAdjCnt; c++) {
		sum += 1.0;
	}
	__syncthreads();
	
}

		
extern "C" __global__ void advanceParticles ( float time, float dt, float ss, int numPnts )
{		
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= numPnts ) return;
	
	if ( fbuf.bufI(FGCELL)[i] == GRID_UNDEF ) {
		fbuf.bufF3(FPOS)[i] = make_float3(-1000,-1000,-1000);
		fbuf.bufF3(FVEL)[i] = make_float3(0,0,0);
		return;
	}
			
	// Get particle vars
	register float3 accel, norm;
	register float diff, adj, speed;
	register float3 pos = fbuf.bufF3(FPOS)[i];
	register float3 veval = fbuf.bufF3(FVEVAL)[i];

	// Leapfrog integration						
	accel = fbuf.bufF3(FFORCE)[i];
	accel *= fparam.pmass;	
		
	// Boundaries
	// Y-axis
	
	diff = fparam.pradius - (pos.y - (fparam.pboundmin.y + (pos.x-fparam.pboundmin.x)*fparam.pground_slope )) * ss;
	if ( diff > EPSILON ) {
		norm = make_float3( -fparam.pground_slope, 1.0 - fparam.pground_slope, 0);
		adj = fparam.pextstiff * diff - fparam.pdamp * dot(norm, veval );
		norm *= adj; accel += norm;
	}

	diff = fparam.pradius - ( fparam.pboundmax.y - pos.y )*ss;
	if ( diff > EPSILON ) {
		norm = make_float3(0, -1, 0);
		adj = fparam.pextstiff * diff - fparam.pdamp * dot(norm, veval );
		norm *= adj; accel += norm;
	}

	// X-axis
	diff = fparam.pradius - (pos.x - (fparam.pboundmin.x + (sin(time*fparam.pforce_freq)+1)*0.5 * fparam.pforce_min))*ss;
	if ( diff > EPSILON ) {
		norm = make_float3( 1, 0, 0);
		adj = (fparam.pforce_min+1) * fparam.pextstiff * diff - fparam.pdamp * dot(norm, veval );
		norm *= adj; accel += norm;
	}
	diff = fparam.pradius - ( (fparam.pboundmax.x - (sin(time*fparam.pforce_freq)+1)*0.5*fparam.pforce_max) - pos.x)*ss;
	if ( diff > EPSILON ) {
		norm = make_float3(-1, 0, 0);
		adj = (fparam.pforce_max+1) * fparam.pextstiff * diff - fparam.pdamp * dot(norm, veval );
		norm *= adj; accel += norm;
	}

	// Z-axis
	diff = fparam.pradius - (pos.z - fparam.pboundmin.z ) * ss;
	if ( diff > EPSILON ) {
		norm = make_float3( 0, 0, 1 );
		adj = fparam.pextstiff * diff - fparam.pdamp * dot(norm, veval );
		norm *= adj; accel += norm;
	}
	diff = fparam.pradius - ( fparam.pboundmax.z - pos.z )*ss;
	if ( diff > EPSILON ) {
		norm = make_float3( 0, 0, -1 );
		adj = fparam.pextstiff * diff - fparam.pdamp * dot(norm, veval );
		norm *= adj; accel += norm;
	}
		
	// Gravity
	accel += fparam.pgravity;

	// Accel Limit
	speed = accel.x*accel.x + accel.y*accel.y + accel.z*accel.z;
	if ( speed > fparam.AL2 ) {
		accel *= fparam.AL / sqrt(speed);
	}

	// Velocity Limit
	float3 vel = fbuf.bufF3(FVEL)[i];
	speed = vel.x*vel.x + vel.y*vel.y + vel.z*vel.z;
	if ( speed > fparam.VL2 ) {
		speed = fparam.VL2;
		vel *= fparam.VL / sqrt(speed);
	}

	// Ocean colors
	/*uint clr = fbuf.bufI(FCLR)[i];
	if ( speed > fparam.VL2*0.2) {
		adj = fparam.VL2*0.2;		
		clr += ((  clr & 0xFF) < 0xFD ) ? +0x00000002 : 0;		// decrement R by one
		clr += (( (clr>>8) & 0xFF) < 0xFD ) ? +0x00000200 : 0;	// decrement G by one
		clr += (( (clr>>16) & 0xFF) < 0xFD ) ? +0x00020000 : 0;	// decrement G by one
		fbuf.bufI(FCLR)[i] = clr;
	}
	if ( speed < 0.03 ) {		
		int v = int(speed/.01)+1;
		clr += ((  clr & 0xFF) > 0x80 ) ? -0x00000001 * v : 0;		// decrement R by one
		clr += (( (clr>>8) & 0xFF) > 0x80 ) ? -0x00000100 * v : 0;	// decrement G by one
		fbuf.bufI(FCLR)[i] = clr;
	}*/
	
	//-- surface particle density 
	//fbuf.mclr[i] = fbuf.mclr[i] & 0x00FFFFFF;
	//if ( fbuf.mdensity[i] > 0.0014 ) fbuf.mclr[i] += 0xAA000000;

	// Leap-frog Integration
	float3 vnext = accel*dt + vel;					// v(t+1/2) = v(t-1/2) + a(t) dt		
	fbuf.bufF3(FVEVAL)[i] = (vel + vnext) * 0.5;	// v(t+1) = [v(t-1/2) + v(t+1/2)] * 0.5			
	fbuf.bufF3(FVEL)[i] = vnext;
	fbuf.bufF3(FPOS)[i] += vnext * (dt/ss);			// p(t+1) = p(t) + v(t+1/2) dt		
}


extern "C" __global__ void prefixFixup(uint *input, uint *aux, int len)     // merge *aux into *input  
{
	unsigned int t = threadIdx.x;
	unsigned int start = t + 2 * blockIdx.x * SCAN_BLOCKSIZE;
	if (start < len)					input[start] += aux[blockIdx.x];      
	if (start + SCAN_BLOCKSIZE < len)   input[start + SCAN_BLOCKSIZE] += aux[blockIdx.x];
}

extern "C" __global__ void prefixSum(uint* input, uint* output, uint* aux, int len, int zeroff) // sum *input, write to *output
{
	__shared__ uint scan_array[SCAN_BLOCKSIZE << 1];
	unsigned int t1 = threadIdx.x + 2 * blockIdx.x * SCAN_BLOCKSIZE;
	unsigned int t2 = t1 + SCAN_BLOCKSIZE;

	// Pre-load into shared memory
	scan_array[threadIdx.x] = (t1<len) ? input[t1] : 0.0f;
	scan_array[threadIdx.x + SCAN_BLOCKSIZE] = (t2<len) ? input[t2] : 0.0f;
	__syncthreads();

	// Reduction
	int stride;
	for (stride = 1; stride <= SCAN_BLOCKSIZE; stride <<= 1) {
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index < 2 * SCAN_BLOCKSIZE)
			scan_array[index] += scan_array[index - stride];
		__syncthreads();
	}

	// Post reduction
	for (stride = SCAN_BLOCKSIZE >> 1; stride > 0; stride >>= 1) {
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index + stride < 2 * SCAN_BLOCKSIZE)
			scan_array[index + stride] += scan_array[index];
		__syncthreads();
	}
	__syncthreads();

	// Output values & aux
	if (t1 + zeroff < len)	output[t1 + zeroff] = scan_array[threadIdx.x];
	if (t2 + zeroff < len)	output[t2 + zeroff] = (threadIdx.x == SCAN_BLOCKSIZE - 1 && zeroff) ? 0 : scan_array[threadIdx.x + SCAN_BLOCKSIZE];
	if (threadIdx.x == 0) {
		if (zeroff) output[0] = 0;
		if (aux) aux[blockIdx.x] = scan_array[2 * SCAN_BLOCKSIZE - 1];
	}
}

