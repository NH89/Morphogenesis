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
#include <cfloat>
#include <cstdint>
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
//#define FLT_MIN  0.000000001                // set here as 2^(-30)
//#define UINT_MAX 65535

extern "C" __global__ void insertParticles ( int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index
	if ( i >= pnum ) return;
for (int a=0;a<BONDS_PER_PARTICLE;a++){                                          // The list of bonds from other particles 
            uint j = fbuf.bufI(FPARTICLEIDX) [i*BONDS_PER_PARTICLE*2 + a];       // NB j is valid only in ftemp.*
            uint k = ftemp.bufI(FPARTICLEIDX) [i*BONDS_PER_PARTICLE*2 + a];
//if(i<34)printf("\nAA(i=%i,a=%i,j=%u,k=%u)\t",i,a,j,k);
}
	//-- debugging (pointers should match CUdeviceptrs on host side)
	// printf ( " pos: %012llx, gcell: %012llx, gndx: %012llx, gridcnt: %012llx\n", fbuf.bufC(FPOS), fbuf.bufC(FGCELL), fbuf.bufC(FGNDX), fbuf.bufC(FGRIDCNT) );

	register float3 gridMin =	fparam.gridMin;                                  // "register" is a compiler 'hint', to keep this variable in thread register
	register float3 gridDelta = fparam.gridDelta;                                //  even if other variable have to be moved to slower 'local' memory  
	register int3 gridRes =		fparam.gridRes;                                  //  in the streaming multiprocessor's cache.
	register int3 gridScan =	fparam.gridScanMax;

	register int		gs;
	register float3		gcf;
	register int3		gc;	

	gcf = (fbuf.bufF3(FPOS)[i] - gridMin) * gridDelta; 
	gc = make_int3( int(gcf.x), int(gcf.y), int(gcf.z) );
	gs = (gc.y * gridRes.z + gc.z)*gridRes.x + gc.x;

	if ( gc.x >= 1 && gc.x <= gridScan.x && gc.y >= 1 && gc.y <= gridScan.y && gc.z >= 1 && gc.z <= gridScan.z ) {
		fbuf.bufI(FGCELL)[i] = gs;											     // Grid cell insert.
		fbuf.bufI(FGNDX)[i] = atomicAdd ( &fbuf.bufI(FGRIDCNT)[ gs ], 1 );		 // Grid counts.
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
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;            // particle index
	if ( i >= pnum ) return;

	// Copy particle from original, unsorted buffer (msortbuf),
	// into sorted memory location on device (mpos/mvel)
	uint icell = ftemp.bufI(FGCELL) [ i ];                             // icell is bin into which i is sorted in fbuf.*

	if ( icell != GRID_UNDEF ) {	  
		// Determine the sort_ndx, location of the particle after sort		
        uint indx =  ftemp.bufI(FGNDX)  [ i ];                         // indx is off set within new cell
        int sort_ndx = fbuf.bufI(FGRIDOFF) [ icell ] + indx ;          // global_ndx = grid_cell_offet + particle_offset	
		float3 zero; zero.x=0;zero.y=0;zero.z=0;
		// Transfer data to sort location
		fbuf.bufI (FGRID) [ sort_ndx ] =	sort_ndx;                  // full sort, grid indexing becomes identity		
		fbuf.bufF3(FPOS) [sort_ndx] =		ftemp.bufF3(FPOS) [i];
		fbuf.bufF3(FVEL) [sort_ndx] =		ftemp.bufF3(FVEL) [i];
		fbuf.bufF3(FVEVAL)[sort_ndx] =		ftemp.bufF3(FVEVAL) [i];
		fbuf.bufF3(FFORCE)[sort_ndx] =      zero;                      // fbuf.bufF3(FFORCE)[ i ] += force; in contributeForce() requires value setting to 0 // old:	ftemp.bufF3(FFORCE) [i];  
		fbuf.bufF (FPRESS)[sort_ndx] =		ftemp.bufF(FPRESS) [i];
		fbuf.bufF (FDENSITY)[sort_ndx] =	ftemp.bufF(FDENSITY) [i];
		fbuf.bufI (FCLR) [sort_ndx] =		ftemp.bufI(FCLR) [i];
		fbuf.bufI (FGCELL) [sort_ndx] =		icell;
		fbuf.bufI (FGNDX) [sort_ndx] =		indx;
        float3 pos = ftemp.bufF3(FPOS) [i];
        // add extra data for morphogenesis
        // track the sort index of the other particle
        
        for (int a=0;a<BONDS_PER_PARTICLE;a++){         // [0]current index, [1]elastic limit, [2]restlength, [3]modulus, [4]damping coeff, [5]particle ID, [6]bond index 
            uint j = ftemp.bufI(FELASTIDX) [i*BOND_DATA + a*DATA_PER_BOND];             // NB j is valid only in ftemp.*
            uint j_sort_ndx = UINT_MAX;
            uint jcell = GRID_UNDEF;
            if (j<pnum){
                jcell = ftemp.bufI(FGCELL) [ j ];                                       // jcell is bin into which j is sorted in fbuf.*
                uint jndx = UINT_MAX;
                if ( jcell != GRID_UNDEF ) {                                            // avoid out of bounds array reads
                    jndx =  ftemp.bufI(FGNDX)  [ j ];      
                    if((fbuf.bufI(FGRIDOFF) [ jcell ] + jndx) <pnum){
                        j_sort_ndx = fbuf.bufI(FGRIDOFF) [ jcell ] + jndx ;             // new location in the list of the other particle
                    }
                }                                                                       // set modulus and length to zero if ( jcell != GRID_UNDEF ) 
            }
            fbuf.bufI (FELASTIDX) [sort_ndx*BOND_DATA + a*DATA_PER_BOND]  = j_sort_ndx; // NB if (j<pnum) j_sort_ndx = UINT_MAX; preserves non-bonds
            for (int b=1;b<DATA_PER_BOND;b++){                                          // copy [1]elastic limit, [2]restlength, [3]modulus, [4]damping coeff, iff unbroken
                fbuf.bufI (FELASTIDX) [sort_ndx*BOND_DATA + a*DATA_PER_BOND +b] = ftemp.bufI (FELASTIDX) [i*BOND_DATA + a*DATA_PER_BOND + b]  * ( jcell != GRID_UNDEF ) ; 
            }                                                                           // old: copy the modulus & length
        }
        
        for (int a=0;a<BONDS_PER_PARTICLE;a++){                                         // The list of bonds from other particles 
            uint k = ftemp.bufI(FPARTICLEIDX) [i*BONDS_PER_PARTICLE*2 + a*2];           // NB j is valid only in ftemp.*
            uint b = ftemp.bufI(FPARTICLEIDX) [i*BONDS_PER_PARTICLE*2 + a*2 +1];
            uint ksort_ndx = UINT_MAX; 
            uint kndx, kcell;
            if (k<pnum){                                                                //(k>=pnum) => bond broken // crashes when j=0 (as set in demo), after run().
                kcell = ftemp.bufI(FGCELL) [ k ];                                       // jcell is bin into which j is sorted in fbuf.*
                if ( kcell != GRID_UNDEF ) {
                    kndx =  ftemp.bufI(FGNDX)  [ k ];  
                    ksort_ndx = fbuf.bufI(FGRIDOFF) [ kcell ] + kndx ;            
                }
            }
            fbuf.bufI(FPARTICLEIDX) [sort_ndx*BONDS_PER_PARTICLE*2 + a*2] =  ksort_ndx; // ftemp.bufI(FPARTICLEIDX) [i*BONDS_PER_PARTICLE + a]
            fbuf.bufI(FPARTICLEIDX) [sort_ndx*BONDS_PER_PARTICLE*2 + a*2 +1] =  b;
            ftemp.bufI(FPARTICLEIDX) [i*BONDS_PER_PARTICLE*2 + a*2] = UINT_MAX;         // set ftemp copy for use as a lock when inserting new bonds in ComputeForce(..)
        }
        fbuf.bufI (FPARTICLE_ID) [sort_ndx] =	ftemp.bufI(FPARTICLE_ID) [i];
        fbuf.bufI (FMASS_RADIUS) [sort_ndx] =	ftemp.bufI(FMASS_RADIUS) [i];
        fbuf.bufI (FNERVEIDX)    [sort_ndx] =	ftemp.bufI(FNERVEIDX) [i];
        
        for (int a=0;a<NUM_TF;a++){fbuf.bufF (FCONC)   [sort_ndx * NUM_TF + a]      =	ftemp.bufF(FCONC) [i * NUM_TF + a]    ;}
        for (int a=0;a<NUM_TF;a++){fbuf.bufI (FEPIGEN) [sort_ndx * NUM_GENES + a]   =	ftemp.bufI(FEPIGEN) [i * NUM_GENES + a];}
	}
} 

extern "C" __device__ float contributePressure ( int i, float3 p, int cell )  
// pressure due to particles in 'cell'. NB for each particle there are 27 cells in which interacting particles might be.
{			
	if ( fbuf.bufI(FGRIDCNT)[cell] == 0 ) return 0.0;                       // If the cell is empty, skip it.

	float3 dist;
	float dsq, c, sum = 0.0;
	register float d2 = fparam.psimscale * fparam.psimscale;
	register float r2 = fparam.r2 / d2;
	
	int clast = fbuf.bufI(FGRIDOFF)[cell] + fbuf.bufI(FGRIDCNT)[cell];      // off set of this cell in the list of particles,  PLUS  the count of particles in this cell.

	for ( int cndx = fbuf.bufI(FGRIDOFF)[cell]; cndx < clast; cndx++ ) {    // For particles in this cell.
		int pndx = fbuf.bufI(FGRID) [cndx];                                       // index of this particle
		dist = p - fbuf.bufF3(FPOS) [pndx];                                       // float3 distance between this particle, and the particle for which the loop has been called.
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);                    // scalar distance squared
		if ( dsq < r2 && dsq > 0.0) {                                             // IF in-range && not the same particle.
			c = (r2 - dsq)*d2;                                                    //(NB this means all unused particles can be stored at one point)
			sum += c * c * c;
		}
	}
	return sum;                                                             // NB a scalar value for pressure contribution, at the current particle, due to particles in this cell.
}
			
extern "C" __global__ void computePressure ( int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                 // particle index
	if ( i >= pnum ) return;

	// Get search cell
	int nadj = (1*fparam.gridRes.z + 1)*fparam.gridRes.x + 1;
	uint gc = fbuf.bufI(FGCELL) [i];                                        // get grid cell of the current particle.
	if ( gc == GRID_UNDEF ) return;                                         // IF particle not in the simulation
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

extern "C" __device__ float contributeDiffusion(int i, float3 p, int cell){
    // if the cell is empty, skip it
    if (fbuf.bufI(FGRIDCNT)[cell] == 0) return 0.0f;

    float3 dist;
    float dsq, c, sum = 0.0;
    register float d2 = fparam.psimscale * fparam.psimscale;
    register float r2 = fparam.r2 / d2;

    // process will be something like:
    // - look at neighbours around me, add their chemicals to this particle, and subtract some from myself as well
    // - return that

    // USE FCONC - should be float

    // add to neighbours, subtract from myself

    return 1.0f;
}


extern "C" __global__ void computeDiffusion(int pnum){
    // get particle index
    uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    // if the particle is outside the simulation, quit processing
    if (i >= pnum) return;

    // Get search cell
    // TODO - what does this block do?
    int nadj = (1 * fparam.gridRes.z + 1) * fparam.gridRes.x + 1;
    uint gc = fbuf.bufI(FGCELL) [i];
    if (gc == GRID_UNDEF) return;
    gc -= nadj;

    // Sum diffusion? (or in this case subtract it?)
    __syncthreads();

    // Compute diffusion?
}

extern "C" __device__ float3 contributeForce ( int i, float3 ipos, float3 iveleval, float ipress, float idens, int cell, uint _bondsToFill, uint _bonds[BONDS_PER_PARTICLE][2], float _bond_dsq[BONDS_PER_PARTICLE], bool freeze)
{			
	if ( fbuf.bufI(FGRIDCNT)[cell] == 0 ) return make_float3(0,0,0);                // If the cell is empty, skip it.
	float dsq, sdist, c, pterm;
	float3 dist = make_float3(0,0,0), eterm  = make_float3(0,0,0), force = make_float3(0,0,0);
	uint j;
	int clast = fbuf.bufI(FGRIDOFF)[cell] + fbuf.bufI(FGRIDCNT)[cell];              // index of last particle in this cell
    for ( int cndx = fbuf.bufI(FGRIDOFF)[cell]; cndx < clast; cndx++ ) {            // For particles in this cell.
		j = fbuf.bufI(FGRID)[ cndx ];
		dist = ( ipos - fbuf.bufF3(FPOS)[ j ] );                                    // dist in cm (Rama's comment)
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);                      // scalar distance squared
		if ( dsq < fparam.rd2 && dsq > 0) {                                         // IF in-range && not the same particle
            sdist = sqrt(dsq * fparam.d2);                                          // smoothing distance = sqrt(dist^2 * sim_scale^2))
			c = ( fparam.psmoothradius - sdist ); 
			pterm = fparam.psimscale * -0.5f * c * fparam.spikykern * ( ipress + fbuf.bufF(FPRESS)[ j ] ) / sdist;                       // pressure term
			force += ( pterm * dist + fparam.vterm * ( fbuf.bufF3(FVEVAL)[ j ] - iveleval )) * c * idens * (fbuf.bufF(FDENSITY)[ j ] );  // fluid force
            if (_bondsToFill >0 && dist.x+dist.y+dist.z > 0.0 && freeze==true){                             // collect particles, in the x+ve hemisphere, for potential bond formation 
                bool known = false;
                uint bond_index = UINT_MAX;

                for (int a=0; a<BONDS_PER_PARTICLE; a++){                                                   // chk if known, i.e. already bonded 
                    if (fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + a*2] == i        ) known = true;   // particle 'j' has a bond to particle 'i'
                    if (fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + a*2] == UINT_MAX ) bond_index = a; // patricle 'j' has an empty bond 'a' : picks last empty bond
                    //if (_bonds[a][0] == j )known = true;                                                  // particle 'i' already has a bond to particle 'j'  // not req, _bonds starts empty && only touch 'j' once
                }
                if (known == false && bond_index<UINT_MAX){       
                    //int bond_direction = 1*(dist.x-dist.y+dist.z>0.0) + 2*(dist.x+dist.y-dist.z>0.0);       // booleans divide bond space into quadrants of x>0.
                    float approx_zero = 0.02*fparam.rd2;
                    int bond_direction = ((dist.x+dist.y+dist.z)>0) * (1*(dist.x*dist.x>approx_zero) + 2*(dist.y*dist.y>approx_zero) + 4*(dist.z*dist.z>approx_zero)) -1; // booleans select +ve quadrant x,y,z axes and their planar diagonals
                    printf("\ni=%u, bond_direction=%i, dist=(%f,%f,%f), dsq=%f, approx_zero=%f", i, bond_direction, dist.x, dist.y, dist.z, dsq, approx_zero);
                    if(0<=bond_direction && bond_direction<BONDS_PER_PARTICLE && dsq<_bond_dsq[bond_direction]){ //if new candidate bond is shorter, for this quadrant. 
                                                                                                                //lacks a candidate bond _bonds[bond_direction][1]==0
                        _bonds[bond_direction][0] = j;                                                      // index of other particle
                        _bonds[bond_direction][1] = bond_index;                                             // FPARTICLEIDX vacancy index of other particle
                        _bond_dsq[bond_direction] = dsq;                                                    // scalar distance squared 
                    }
                }
            }                                                                                               // end of collect potential bonds
        }                                                                                                   // end of: IF in-range && not the same particle
    }                                                                                                       // end of loop round particles in this cell
    return force;                                                                                           // return fluid force && list of potential bonds fron this cell
}

extern "C" __global__ void computeForce ( int pnum, bool freeze, uint frame)
{			
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                         // particle index
	if ( i >= pnum ) return;
	uint gc = fbuf.bufI(FGCELL)[ i ];                                               // Get search cell	
	if ( gc == GRID_UNDEF ) return;                                                 // particle out-of-range

	gc -= (1*fparam.gridRes.z + 1)*fparam.gridRes.x + 1;
	register float3 force, eterm, dist;                                             // request to compiler to store in a register for speed.
	force = make_float3(0,0,0);    eterm = make_float3(0,0,0);     dist  = make_float3(0,0,0);
    float dsq, abs_dist;                                                            // elastic force // new version computes here using particle index rather than ID.
    uint bondsToFill = 0;
    uint bonds[BONDS_PER_PARTICLE][2];                                               // [0] = index of other particle, [1] = bond_index
    float bond_dsq[BONDS_PER_PARTICLE];                                             // length of bond, for potential new bonds
    for (int a=0; a<BONDS_PER_PARTICLE;a++) {
        bonds[a][0]= UINT_MAX;
        bonds[a][1]= UINT_MAX;
        bond_dsq[a]= fparam.rd2;                                                    // NB if ( dsq < fparam.rd2 && dsq > 0) is the cut off for fluid interaction range
    } 
    if (freeze==true){                                                              // If we are going to make new bonds, first check for broken incomming bonds //////////////////
        for (int a=0; a<BONDS_PER_PARTICLE;a++){                                    // loop round this particle's list of _incomming_ bonds /////
            bool intact = false;
            uint k = fbuf.bufI(FPARTICLEIDX)[i*BONDS_PER_PARTICLE*2 + a*2];
            uint b = fbuf.bufI(FPARTICLEIDX)[i*BONDS_PER_PARTICLE*2 + a*2 +1];      // chk bond intact. nb short circuit evaluation of if conditions.
            // k is a particle, bond_idx is in range, AND k's reciprocal record matches i's record of the bond
            if(k<pnum && b<BONDS_PER_PARTICLE && i==fbuf.bufI(FELASTIDX)[k*BOND_DATA + b*DATA_PER_BOND] && a==fbuf.bufI(FELASTIDX)[k*BOND_DATA + b*DATA_PER_BOND +6] )intact=true;   
            if(i==k)intact=false;
            //if(intact==true)printf("\ncomputeForce: incomming bond intact  i=%u, k=%u, a=%u, b=%u",i,k,a,b);
            if(intact==false){                                                      // remove broken/missing _incomming_ bonds
                fbuf.bufI(FPARTICLEIDX)[i*BONDS_PER_PARTICLE*2 + a*2] = UINT_MAX;   // particle
                fbuf.bufI(FPARTICLEIDX)[i*BONDS_PER_PARTICLE*2 + a*2 +1] = UINT_MAX;// bond index
            }
        }
        
        for (int a=0; a<BONDS_PER_PARTICLE;a++){                                    // loop round this particle's list of _outgoing_ bonds /////
            bool intact = false;
            uint j = fbuf.bufI(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND];
            uint bond_idx = fbuf.bufI(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND + 6];  // chk bond intact nb short circuit evaluation of if conditions.
            // j is a particle, bond_idx is in range, AND j's reciprocal record matches i's record of the bond
            if(j<pnum && bond_idx<BONDS_PER_PARTICLE && i==fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + bond_idx*2] && a==fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + bond_idx*2 +1])intact=true; 
            if(i==j)fbuf.bufI(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND]=false;
            if(intact==false){                                                      // remove missing _outgoing_ bonds 
                fbuf.bufI(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND]=UINT_MAX;         // [0]current index, [1]elastic limit, [2]restlength, [3]modulus, [4]damping_coeff, [5]particle ID, [6]bond index 
                fbuf.bufF(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND+1]=0.0;
                fbuf.bufF(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND+2]=1.0;
                fbuf.bufF(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND+3]=0.0;
                fbuf.bufF(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND+4]=0.0;
                fbuf.bufI(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND+5]=UINT_MAX;
                fbuf.bufI(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND+6]=UINT_MAX;
            }
        }
    }
    float3  pvel = {fbuf.bufF3(FVEVAL)[ i ].x,  fbuf.bufF3(FVEVAL)[ i ].y,  fbuf.bufF3(FVEVAL)[ i ].z}; // copy i's FEVAL to thread memory
    for (int a=0;a<BONDS_PER_PARTICLE;a++){                                         // compute elastic force due to bonds /////////////////////////////////////////////////////////
        uint bond = i*BOND_DATA + a*DATA_PER_BOND;                                  // bond's index within i's FELASTIDX 
        uint j                      = fbuf.bufI(FELASTIDX)[bond];                   // particle IDs   i*BOND_DATA + a
        if(j<pnum){                                                                 // copy FELASTIDX to thread memory for particle i.
            float elastic_limit     = fbuf.bufF(FELASTIDX)[bond + 1];               // [0]current index, [1]elastic limit, [2]restlength, [3]modulus, [4]damping_coeff, [5]particle ID, [6]bond index 
            float restlength        = fbuf.bufF(FELASTIDX)[bond + 2];               // NB fbuf.bufF() for floats, fbuf.bufI for uints.
            float modulus           = fbuf.bufF(FELASTIDX)[bond + 3];
            float damping_coeff     = fbuf.bufF(FELASTIDX)[bond + 4];
            uint  other_particle_ID = fbuf.bufI(FELASTIDX)[bond + 5];
            uint  bondIndex         = fbuf.bufI(FELASTIDX)[bond + 6];
            
            float3 j_pos = make_float3(fbuf.bufF3(FPOS)[ j ].x,  fbuf.bufF3(FPOS)[ j ].y,  fbuf.bufF3(FPOS)[ j ].z); // copy j's FPOS to thread memory
        
            dist = ( fbuf.bufF3(FPOS)[ i ] - j_pos  );                              // dist in cm (Rama's comment)  /*fbuf.bufF3(FPOS)[ j ]*/
            dsq  = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);                 // scalar distance squared
            abs_dist = sqrt(dsq) + FLT_MIN;                                         // FLT_MIN adds minimum +ve float, to prevent division by abs_dist=zero
            float3 rel_vel = fbuf.bufF3(FVEVAL)[ j ] - pvel;                        // add optimal damping:  -l*v , were v is relative velocity, and l= 2*sqrt(m*k)  where k is the spring stiffness.
                                                                                    // eterm = (bool within elastic limit) * (spring force + damping)
                                                                                     
            eterm = ((float)(abs_dist < elastic_limit)) * ( ((dist/abs_dist) * modulus * (abs_dist-restlength)/restlength) - damping_coeff*rel_vel); // Elastic force due to bond ####
            force -= eterm;                                                         // elastic force towards other particle, if (rest_len -abs_dist) is -ve
            atomicAdd( &fbuf.bufF3(FFORCE)[ j ].x, eterm.x);                        // NB Must send equal and opposite force to the other particle
            atomicAdd( &fbuf.bufF3(FFORCE)[ j ].y, eterm.y);
            atomicAdd( &fbuf.bufF3(FFORCE)[ j ].z, eterm.z);                        // temporary hack, ? better to write a float3 attomicAdd using atomicCAS  #########

            if (abs_dist >= elastic_limit){                                         // If (out going bond broken)
                fbuf.bufI(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +1]=0;           // remove broken bond by setting elastic limit to zero.
                fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +3]=0;           // set modulus to zero
                
                uint bondIndex_ = fbuf.bufI(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +6];
                fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + bondIndex_] = UINT_MAX ;
                fbuf.bufI(FELASTIDX)[bond] = UINT_MAX;
                printf("\n#### Set to broken, i=%i, j=%i, b=%i, fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + b]=UINT_MAX\t####",i,j,bondIndex_);
                bondsToFill++;
            }
        }
        __syncthreads();    // when is this needed ? ############
    }   
//printf("\tComputeForce: i=%u, bondsToFill=%u", i, bondsToFill);  // was always zero . why ?
	bondsToFill=BONDS_PER_PARTICLE; // remove and use result from loop above ? ############
    for (int c=0; c < fparam.gridAdjCnt; c++) {                                 // Call contributeForce(..) for fluid forces AND potential new bonds /////////////////////////
        float3 fluid_force = make_float3(0,0,0);
        fluid_force = contributeForce ( i, fbuf.bufF3(FPOS)[ i ], fbuf.bufF3(FVEVAL)[ i ], fbuf.bufF(FPRESS)[ i ], fbuf.bufF(FDENSITY)[ i ], gc + fparam.gridAdj[c], bondsToFill, bonds ,bond_dsq, freeze); 
        if (freeze==true) fluid_force *=0.1;                                        // slow fluid movement while forming bonds
        force += fluid_force;
    }
    
    //printf("\ni=%u, bond_dsq=(%f,%f,%f,%f,%f,%f),",i,bond_dsq[0],bond_dsq[1],bond_dsq[2],bond_dsq[3],bond_dsq[4],bond_dsq[5]);

	__syncthreads();   // when is this needed ? ############
    atomicAdd(&fbuf.bufF3(FFORCE)[ i ].x, force.x);                                 // atomicAdd req due to other particles contributing forces via incomming bonds. 
    atomicAdd(&fbuf.bufF3(FFORCE)[ i ].y, force.y);                                 // NB need to reset FFORCE to zero in  CountingSortFull(..)
    atomicAdd(&fbuf.bufF3(FFORCE)[ i ].z, force.z);                                 // temporary hack, ? better to write a float3 attomicAdd using atomicCAS ?  ########

    // Add new bonds /////////////////////////////////////////////////////////////////////////////
    int a = BONDS_PER_PARTICLE * (int)(freeze!=true);                               // if (freeze!=true) skip for loop, else a=0
    for (; a< BONDS_PER_PARTICLE; a++){
        int otherParticleBondIndex = BONDS_PER_PARTICLE*2*bonds[a][0] + 2*a /*bonds[a][1]*/; // fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex]
        
        if((uint)bonds[a][0]==i) printf("\n (uint)bonds[a][0]==i, i=%u a=%u",i,a);  // float bonds[BONDS_PER_PARTICLE][3];  [0] = index of other particle, [1] = dsq, [2] = bond_index
                                                                                    // If outgoing bond empty && proposed bond for this quadrant is valid
        if( fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +1] == 0.0  &&  bonds[a][0] < pnum  && bonds[a][0]!=i  && bond_dsq[a]<3 ){  // ie dsq < 3D diagonal of cube ##### hack #####
                                                                                    // NB "bonds[b][0] = UINT_MAX" is used to indicate no candidate bond found
                                                                                    //    (FELASTIDX) [1]elastic limit = 0.0 isused to indicate out going bond is empty
            printf("\nBond making loop i=%u, a=%i, bonds[a][1]=%u, bond_dsq[a]=%f",i,a,bonds[a][1],bond_dsq[a]);
            
            do {} while( atomicCAS(&ftemp.bufI(FPARTICLEIDX)[otherParticleBondIndex], UINT_MAX, 0) );               // lock ///////////////// ###### //  if (not locked) write zero to 'ftemp' to lock.
            if (fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex]==UINT_MAX)  fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex] = i;                     //  if (bond is unoccupied) write to 'fbuf' to assign this bond
            ftemp.bufI(FPARTICLEIDX)[otherParticleBondIndex] = UINT_MAX;                                            // release lock ///////// ######

            
            if (fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex] == i){                                              // if (this bond is assigned) write bond data
                fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex +1] = a;                                             // write i's outgoing bond_index to j's incoming bonds
                uint i_ID = fbuf.bufI(FPARTICLE_ID)[i];                                                             // retrieve permenant particle IDs for 'i' and 'j'
                uint j_ID = fbuf.bufI(FPARTICLE_ID)[bonds[a][0]];
                float bond_length = sqrt(bond_dsq[a]);
                float modulus = 100000;       // 100 000 000                                                                    // 1000000 = min for soft matter integrity // 
                fbuf.bufI(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND]    = bonds[a][0];                               // [0]current index,
                fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +1] = 2 * bond_length ;                          // [1]elastic limit  = 2x restlength i.e. %100 strain
                fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +2] = 0.5*bond_length;                               // [2]restlength = initial length  
                fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +3] = modulus;                                   // [3]modulus
                fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +4] = 2*sqrt(fparam.pmass*modulus);              // [4]damping_coeff = optimal for mass-spring pair.
                fbuf.bufI(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +5] = j_ID;                                      // [5]save particle ID of the other particle NB for debugging
                fbuf.bufI(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +6] = bonds[a][1];                               // [6]bond index at the other particle 'j's incoming bonds
                printf("\nNew Bond a=%u, i=%u, j=%u, bonds[a][1]=%u, fromPID=%u, toPID=%u,, fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex]=%u, otherParticleBondIndex=%u",
                       a,i,bonds[a][0],bonds[a][1],i_ID,j_ID, fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex], otherParticleBondIndex);
            }            
        }// end if 
        __syncthreads();    // NB applies to all threads _if_ the for loop runs, i.e. if(freeze==true)
    }                                                                               // end loop around FELASTIDX bonds
}                                                                                   // end computeForce (..)

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
		fbuf.bufF3(FPOS)[i] = make_float3(fparam.pboundmin.x,fparam.pboundmin.y,fparam.pboundmin.z-2*fparam.gridRes.z);
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
//    printf(" accel+gravity=%f,%f,%f  gravity=%f,%f,%f\t",accel.x,accel.y,accel.z,fparam.pgravity.x,fparam.pgravity.y,fparam.pgravity.z);

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

