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
for (int a=0;a<BONDS_PER_PARTICLE;a++){                                                                                          // The list of bonds from other particles 
            uint j = fbuf.bufI(FPARTICLEIDX) [i*BONDS_PER_PARTICLE*2 + a];                                                                // NB j is valid only in ftemp.*
            uint k = ftemp.bufI(FPARTICLEIDX) [i*BONDS_PER_PARTICLE*2 + a];
//if(i<34)printf("\nAA(i=%i,a=%i,j=%u,k=%u)\t",i,a,j,k);
}
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
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;            // particle index
	if ( i >= pnum ) return;
/*
for (int a=0;a<BONDS_PER_PARTICLE;a++){                                                                                          // The list of bonds from other particles 
            uint k = ftemp.bufI(FPARTICLEIDX) [i*BONDS_PER_PARTICLE + a];                                                                // NB j is valid only in ftemp.*
            uint ksort_ndx = UINT_MAX; 
            uint kndx, kcell;
//if(i<34)printf("\nAB(i=%i,a=%i,k=%u)",i,a,k);
}
*/
	// Copy particle from original, unsorted buffer (msortbuf),
	// into sorted memory location on device (mpos/mvel)
	uint icell = ftemp.bufI(FGCELL) [ i ];                             // icell is bin into which i is sorted in fbuf.*

	if ( icell != GRID_UNDEF ) {	  
		// Determine the sort_ndx, location of the particle after sort		
        uint indx =  ftemp.bufI(FGNDX)  [ i ];                         // indx is off set within new cell
        int sort_ndx = fbuf.bufI(FGRIDOFF) [ icell ] + indx ;          // global_ndx = grid_cell_offet + particle_offset	
//if (i<34)printf ( "\nAC i=%d: icell: %d, off: %d, ndx: %d\t", i, icell, fbuf.bufI(FGRIDOFF)[icell], indx );
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
 /*        
if (i<34)printf("\nAD Before countingSortFull(): particle i=%u, pos=(%f,%f,%f) \t(fbuf.bufI(FELASTIDX)[0][0]=%u, fbuf.bufF(FELASTIDX)[0][1]=%f),\t(fbuf.bufI(FELASTIDX)[1][0]=%u, fbuf.bufF(FELASTIDX)[1][1]=%f),\t(fbuf.bufI(FELASTIDX)[2][0]=%u, fbuf.bufF(FELASTIDX)[2][1]=%f),\t(fbuf.bufF(FELASTIDX)[3][0]=%u, fbuf.bufI(FELASTIDX)[3][1]=%f),\t",i,pos.x,pos.y,pos.z,
       ftemp.bufI(FELASTIDX)[i*BOND_DATA + 0*DATA_PER_BOND],  ftemp.bufF(FELASTIDX)[i*BOND_DATA + 0*DATA_PER_BOND +1], 
       ftemp.bufI(FELASTIDX)[i*BOND_DATA + 1*DATA_PER_BOND],  ftemp.bufF(FELASTIDX)[i*BOND_DATA + 1*DATA_PER_BOND +1],
       ftemp.bufI(FELASTIDX)[i*BOND_DATA + 2*DATA_PER_BOND],  ftemp.bufF(FELASTIDX)[i*BOND_DATA + 2*DATA_PER_BOND +1],
       ftemp.bufI(FELASTIDX)[i*BOND_DATA + 3*DATA_PER_BOND],  ftemp.bufF(FELASTIDX)[i*BOND_DATA + 3*DATA_PER_BOND +1]
      ); 
 */
        // add extra data for morphogenesis
/*        
//        for (int a=0;a<BONDS_PER_PARTICLE*2;a++){
//            fbuf.bufI (FELASTIDX) [sort_ndx*BONDS_PER_PARTICLE*2 + a] =	ftemp.bufI(FELASTIDX) [i*BONDS_PER_PARTICLE*2 + a]; //sort_ndx= grid_cell_offet + particle_offset   , i=particle index
//        }
*/
        // track the sort index of the other particle
        
        for (int a=0;a<BONDS_PER_PARTICLE;a++){         // [0]current index, [1]elastic limit, [2]restlength, [3]modulus, [4]damping coeff, [5]particle ID, [6]bond index 
            uint j = ftemp.bufI(FELASTIDX) [i*BOND_DATA + a*DATA_PER_BOND];                                                             // NB j is valid only in ftemp.*
            uint j_sort_ndx = UINT_MAX;
            uint jcell = GRID_UNDEF;
            if (j<pnum){
                jcell = ftemp.bufI(FGCELL) [ j ];                                                                                       // jcell is bin into which j is sorted in fbuf.*
//if (i<34)printf("\nAE (i=%u, j=%u, jcell=%u ), \t",i,j,jcell);
                uint jndx = UINT_MAX;
                if ( jcell != GRID_UNDEF ) {                                                                                            // avoid out of bounds array reads
                    jndx =  ftemp.bufI(FGNDX)  [ j ];      
                    // jndx is off set within new cell  
//if (i<34)printf("\nAF (i=%u, j=%u, jcell=%u != GRID_UNDEF, jndx =%u, fbuf.bufI(FGRIDOFF)[jcell]=%u, sum =%u ), \t",i,j,jcell,jndx,fbuf.bufI(FGRIDOFF)[jcell],(fbuf.bufI(FGRIDOFF)[jcell]+jndx) );
                    if((fbuf.bufI(FGRIDOFF) [ jcell ] + jndx) <pnum){
                        j_sort_ndx = fbuf.bufI(FGRIDOFF) [ jcell ] + jndx ;                    // new location in the list of the other particle
                    }
                }                                                                                                                       // set modulus and length to zero if ( jcell != GRID_UNDEF ) 
//if (i<34)printf("\nAG (i=%u, j=%u, jcell=%u != GRID_UNDEF), jndx=%u \t",i,j,jcell,jndx);            

            //fbuf.bufI (FELASTIDX) [sort_ndx*BOND_DATA + a*DATA_PER_BOND +2] = ftemp.bufI (FELASTIDX) [i*BOND_DATA + a*DATA_PER_BOND +DATA_PER_BOND -1]; // copy the partcle ID of the other particle
//if (i<34)printf("\nAH (i=%u, j=%u, jcell=%u != GRID_UNDEF), \t",i,j,jcell);
            }
            fbuf.bufI (FELASTIDX) [sort_ndx*BOND_DATA + a*DATA_PER_BOND]  = j_sort_ndx;                                                 // NB if (j<pnum) j_sort_ndx = UINT_MAX; preserves non-bonds
            for (int b=1;b<DATA_PER_BOND;b++){                                                                    // copy [1]elastic limit, [2]restlength, [3]modulus, [4]damping coeff, iff unbroken
                fbuf.bufI (FELASTIDX) [sort_ndx*BOND_DATA + a*DATA_PER_BOND +b] = ftemp.bufI (FELASTIDX) [i*BOND_DATA + a*DATA_PER_BOND + b]  * ( jcell != GRID_UNDEF ) ; 
            }                                                                                                                           // old: copy the modulus & length
        }

        
        for (int a=0;a<BONDS_PER_PARTICLE;a++){                                                                                         // The list of bonds from other particles 
            uint k = ftemp.bufI(FPARTICLEIDX) [i*BONDS_PER_PARTICLE*2 + a];                                                               // NB j is valid only in ftemp.*
            uint ksort_ndx = UINT_MAX; 
            uint kndx, kcell;
//if(i<34)printf("\nA(i=%i,a=%i,k=%u)",i,a,k);
            if (k>0 && k<pnum){// k=pnum-1;                                         //(k>=pnum) => bond broken // crashes when j=0 (as set in demo), after run().
                kcell = ftemp.bufI(FGCELL) [ k ];                              // jcell is bin into which j is sorted in fbuf.*
//if(i<34)printf("\nB(i=%u,a=%u,k=%u,kcell=%i)",i,a,k,kcell);          
                if ( kcell != GRID_UNDEF ) {
                    kndx =  ftemp.bufI(FGNDX)  [ k ];  
                    ksort_ndx = fbuf.bufI(FGRIDOFF) [ kcell ] + kndx ;            
//if(i<34)printf("\nC(i=%u,a=%u,k=%i,kcell=%i,kndx=%i,ksort_ndx=%i)",i,a,k,kcell, kndx, ksort_ndx);           //, ,jcell=%i ,jcell   //  jndx=%i   ,ftemp.bufI(FGNDX)[ j ]
                }
            }
//if(i<34)printf("\nD(i=%u,a=%u, ksort_ndx=%u),",i,a,ksort_ndx);
            fbuf.bufI(FPARTICLEIDX) [sort_ndx*BONDS_PER_PARTICLE*2 + a] =  ksort_ndx; // ftemp.bufI(FPARTICLEIDX) [i*BONDS_PER_PARTICLE + a]
            ftemp.bufI(FPARTICLEIDX) [i*BONDS_PER_PARTICLE*2 + a] = UINT_MAX;         // set ftemp copy for use as a lock when inserting new bonds in ComputeForce(..)
        }
        fbuf.bufI (FPARTICLE_ID) [sort_ndx] =	ftemp.bufI(FPARTICLE_ID) [i];
        fbuf.bufI (FMASS_RADIUS) [sort_ndx] =	ftemp.bufI(FMASS_RADIUS) [i];
        fbuf.bufI (FNERVEIDX)    [sort_ndx] =	ftemp.bufI(FNERVEIDX) [i];
        
        for (int a=0;a<NUM_TF;a++){fbuf.bufI (FCONC) [sort_ndx * NUM_TF + a] =		ftemp.bufI(FCONC) [i * NUM_TF + a];}
        
        for (int a=0;a<NUM_TF;a++){fbuf.bufI (FEPIGEN) [sort_ndx * NUM_GENES + a] =	ftemp.bufI(FEPIGEN) [i * NUM_GENES + a];}
	}
//if(i<4)printf("\nend countingSortFull, i=%i\n",i);
} 



/*
extern "C" __global__ void writeParticleIndex ( int pnum )      // needed for.ply edges  
                                                                // or perhaps there is a better way. built into bond detection
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;		// particle index				
	if ( i >= pnum ) return;
    
    uint particle_ID = fbuf.bufI (FPARTICLE_ID) [i];
    fbuf.bufI (FPARTICLE_IDX) [particle_ID] = i;
}
*/


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

extern "C" __device__ float3 contributeForce ( int i, float3 ipos, float3 iveleval, float ipress, float idens, int cell, uint _bondsToFill, float _bonds[BONDS_PER_PARTICLE][3], bool freeze)
{			
	if ( fbuf.bufI(FGRIDCNT)[cell] == 0 ) return make_float3(0,0,0);                // If the cell is empty, skip it.
	float dsq, sdist, c, pterm;
	float3 dist = make_float3(0,0,0), eterm  = make_float3(0,0,0), force = make_float3(0,0,0);
	int j;
	int clast = fbuf.bufI(FGRIDOFF)[cell] + fbuf.bufI(FGRIDCNT)[cell];              // index of last particle in this cell
/*
//if (i<1)printf("\n##T: i=%u,fbuf.bufI(FELASTIDX)[i*BOND_DATA ]=%u",i,fbuf.bufI(FELASTIDX)[i*BOND_DATA]);
*/
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
/*                
//if (i<4)printf("\nA: _bondsToFill=%u, dist.x=%f, freeze=%i",_bondsToFill,dist.x,freeze);
*/
                bool known = false;
                int bond_index = -1;

                for (int a=0; a<BONDS_PER_PARTICLE; a++){                                                   // chk if known, i.e. already bonded 
/*
//if (i<4)printf("\nB: fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE + a]=%u, i=%u, a=%u, fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE + a+1]=%u, _bonds[a][0]=%f , j=%u",fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE + a], i, a, fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE + a+1], _bonds[a][0], j);
*/
                    if (fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + a*2] == i        ) known = true;   // particle 'j' has a bond to particle 'i'
                    if (fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + a*2] == UINT_MAX ) bond_index = a; // patricle 'j' has an empty bond 'a'
                    //if (_bonds[a][0] == j )known = true;                                                  // particle 'i' already has a bond to particle 'j'  // not req, _bonds starts empty && only touch 'j' once
                }
/*
//if (i<4)printf("\nC: known =%i, bond_index=%i",known, bond_index);
*/
                if (known == false && bond_index !=-1){                             // [0]current index, [1]elastic limit, [2]restlength, [3]modulus, [4]damping_coeff, [5]particle ID, [6]bond index
                    int bond_direction = 1*(dist.x-dist.y+dist.z>0.0) + 2*(dist.x+dist.y-dist.z>0.0);       // booleans divide bond into quadrants of x>0.
/*
//if (i<4)printf("\nD: bond_direction=%i, dsq=%f, _bonds[bond_direction][1]=%f",bond_direction,dsq,_bonds[bond_direction][1]);
*/
                    if(_bonds[bond_direction][1]==0 || dsq<_bonds[bond_direction][1]){                      // if lacks a candidate bond OR new candidate bond is shorter, for this quadrant.
                        _bonds[bond_direction][0] = j;                                                      // index of other particle
                        _bonds[bond_direction][1] = dsq;                                                    // scalar distance squared 
                        _bonds[bond_direction][2] = bond_index;                                             // FPARTICLEIDX vacancy index of other particle
/*
//if (i<4)printf("\nE: potential bond i=%u j=%u,dsq=%f, bond_index=%i",i,j,dsq,bond_index);
*/
                    }
                }
            }                                                                                               // end of collect potential bonds
        }                                                                                                   // end of: IF in-range && not the same particle
        __syncthreads();    // when is this needed ? ############
/*
//if (i<1)printf("\n##U: i=%u,fbuf.bufI(FELASTIDX)[i*BOND_DATA ]=%u",i,fbuf.bufI(FELASTIDX)[i*BOND_DATA]);
*/
    }                                                                                                       // end of loop round particles in this cell
/*
//float absforce = sqrt(force.x*force.x+force.y*force.y+force.z*force.z);
//printf("cForce: particle=%u, cell=%i, force=(%f,%f,%f), absforce=%f\n",i,cell,force.x,force.y,force.z,absforce);
*/
    return force;                                                                                           // return fluid force && list of potential bonds fron this cell
}

extern "C" __global__ void computeForce ( int pnum, bool freeze)
{			
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                         // particle index
	if ( i >= pnum ) return;
	uint gc = fbuf.bufI(FGCELL)[ i ];                                               // Get search cell	
	if ( gc == GRID_UNDEF ) return;                                                 // particle out-of-range
/*	
if (i<4)printf("A: Before computeForce(): particle i=%u,\t(fbuf.bufI(FELASTIDX)[0][0]=%u, fbuf.bufI(FELASTIDX)[0][1]=%u),\t(fbuf.bufI(FELASTIDX)[1][0]=%u, fbuf.bufI(FELASTIDX)[1][1]=%u),\t(fbuf.bufI(FELASTIDX)[2][0]=%u, fbuf.bufI(FELASTIDX)[2][1]=%u),\t(fbuf.bufI(FELASTIDX)[3][0]=%u, fbuf.bufI(FELASTIDX)[3][1]=%u), freeze=%s\n",i,
       fbuf.bufI(FELASTIDX)[i*BOND_DATA + 0*DATA_PER_BOND],  fbuf.bufI(FELASTIDX)[i*BOND_DATA + 0*DATA_PER_BOND +1], 
       fbuf.bufI(FELASTIDX)[i*BOND_DATA + 1*DATA_PER_BOND],  fbuf.bufI(FELASTIDX)[i*BOND_DATA + 1*DATA_PER_BOND +1],
       fbuf.bufI(FELASTIDX)[i*BOND_DATA + 2*DATA_PER_BOND],  fbuf.bufI(FELASTIDX)[i*BOND_DATA + 2*DATA_PER_BOND +1],
       fbuf.bufI(FELASTIDX)[i*BOND_DATA + 3*DATA_PER_BOND],  fbuf.bufI(FELASTIDX)[i*BOND_DATA + 3*DATA_PER_BOND +1],
        (freeze==true) ? "true" : "false"
      );  
//if (i<1)printf("\n##V: i=%u,fbuf.bufI(FELASTIDX)[i*BOND_DATA ]=%u",i,fbuf.bufI(FELASTIDX)[i*BOND_DATA]);


//for (int a=0;a<BONDS_PER_PARTICLE;a++)if(i<16)printf("\nXA: i=%u, a=%u, k=%u",i,a,fbuf.bufI(FPARTICLEIDX)[i*BONDS_PER_PARTICLE+a]);
*/
	gc -= (1*fparam.gridRes.z + 1)*fparam.gridRes.x + 1;
	register float3 force, eterm, dist;                                             // request to compiler to store in a register for speed.
	force = make_float3(0,0,0);    eterm = make_float3(0,0,0);     dist  = make_float3(0,0,0);
    float dsq, abs_dist;                                                            // elastic force // new version computes here using particle index rather than ID. 
    uint bondsToFill = 0;
    float bonds[BONDS_PER_PARTICLE][3];                                             // [0] = index of other particle, [1] = dsq, [2] = bond_index
    float bond_dsq = fparam.rd2;                                                    // NB if ( dsq < fparam.rd2 && dsq > 0) is the cut off for fluid interaction range
    for (int a=0; a<BONDS_PER_PARTICLE;a++) {bonds[a][0]=-1; bonds[a][1]=bond_dsq;} // to hold particle and length of bond, for potential new bonds

    if (freeze==true){                                                              // If we are going to make new bonds, first check for broken incomming bonds //////////////////
/*
            if(i<16)printf("\nOuter loop start: i=%u, k=(%u,%u,%u,%u) ",i
            ,fbuf.bufI(FPARTICLEIDX)[i*BONDS_PER_PARTICLE],fbuf.bufI(FPARTICLEIDX)[i*BONDS_PER_PARTICLE+1],fbuf.bufI(FPARTICLEIDX)[i*BONDS_PER_PARTICLE+2],fbuf.bufI(FPARTICLEIDX)[i*BONDS_PER_PARTICLE+3] );
*/ 
        for (int a=0; a<BONDS_PER_PARTICLE;a++){                                    // loop round this particle's list of _incomming_ bonds /////
            bool intact = false;
            uint k = fbuf.bufI(FPARTICLEIDX)[i*BONDS_PER_PARTICLE*2 + a*2];
/*
              if((i<16)&&(k<pnum))printf("\nInner loop start: i=%u, a=%u, k=%u j=(%u,%u,%u,%u)",i, a, k, 
                fbuf.bufI(FELASTIDX)[k*BOND_DATA], fbuf.bufI(FELASTIDX)[k*BOND_DATA+DATA_PER_BOND], fbuf.bufI(FELASTIDX)[k*BOND_DATA+2*DATA_PER_BOND], fbuf.bufI(FELASTIDX)[k*BOND_DATA+3*DATA_PER_BOND]
                );
*/ 
            uint b = fbuf.bufI(FPARTICLEIDX)[i*BONDS_PER_PARTICLE*2 + a*2 +1];      // chk bond intact. nb short circuit evaluation of if conditions.
            // k is a particle, bond_idx is in range, AND k's reciprocal record matches i's record of the bond
            if(k<pnum && b<BONDS_PER_PARTICLE && i==fbuf.bufI(FELASTIDX)[k*BOND_DATA + b*DATA_PER_BOND] && a==fbuf.bufI(FELASTIDX)[k*BOND_DATA + b*DATA_PER_BOND +6] )intact=true;   
/*            
            for(int b=0;b<BONDS_PER_PARTICLE;b++){
                if(k<pnum){
                    if(i== fbuf.bufI(FELASTIDX)[k*BOND_DATA+b*DATA_PER_BOND])intact=true;

//                     * printf("\nTest: i=%u, a=%u, b=%u, k=%u, j=%u, bond=%s, intact=%s",i,a,b,k,fbuf.bufI(FELASTIDX)[k*BOND_DATA+b*DATA_PER_BOND],     
//                        (i== fbuf.bufI(FELASTIDX)[k*BOND_DATA+b*DATA_PER_BOND]) ? "true" : "false",
//                        intact ? "true" : "false"
//                    ); 
                }
            }
*/
            if(i==k)intact=false;
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
/*
                //for (int b=0; b<BONDS_PER_PARTICLE; b++){  if (i==fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + b])intact=true;  }
*/
            if(i==j)fbuf.bufI(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND]=false;
            if(intact==false){                                                      // remove missing _outgoing_ bonds 
                fbuf.bufI(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND]=UINT_MAX;         // [0]current index, [1]elastic limit, [2]restlength, [3]modulus, [4]damping_coeff, [5]particle ID, [6]bond index 
                fbuf.bufF(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND+1]=0.0;
                fbuf.bufF(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND+2]=1.0;
                fbuf.bufF(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND+3]=0.0;
                fbuf.bufF(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND+4]=0.0;
                fbuf.bufI(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND+5]=UINT_MAX;
                fbuf.bufF(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND+6]=UINT_MAX;
            }
        }
    }
/*
            for (int b=0; b<BONDS_PER_PARTICLE; b++){                               // loop round the other particle's list of outgoing bonds
                uint bond = k*BOND_DATA + b*DATA_PER_BOND;
if(i<16 && k < pnum)printf("\nInner loop: i=%u, a=%u, b=%u, bond=%u, fbuf.bufI(FELASTIDX)[bond]=%u, match=%s",i,a,b,bond,fbuf.bufI(FELASTIDX)[bond],(i==fbuf.bufI(FELASTIDX)[bond]) ? "true" : "false" );                
                if(bond < pnum && i==fbuf.bufI(FELASTIDX)[bond]) intact = true;     // NB must chk it is not already outside of pnum.
            }
            if (intact == false ) fbuf.bufI(FPARTICLEIDX)[i*BONDS_PER_PARTICLE + a] = UINT_MAX; // set to "broken"
//if(i<16)printf("\nOuter loop end: i=%u, a=%u, incoming bond intact = %s,",i,a, (intact==true) ? "true" : "false" );
        }

//if (i<4)printf("{{B: particle i=%u, bondsToFill=%u, force=(%f,%f,%f)}}\n",i,bondsToFill,force.x,force.y,force.z);
//if (i<1)printf("\n##W: i=%u,fbuf.bufI(FELASTIDX)[i*BOND_DATA ]=%u",i,fbuf.bufI(FELASTIDX)[i*BOND_DATA]);
  
    //for (int a=0;a<BONDS_PER_PARTICLE;a++){ if(i<16)printf("\nXB: i=%u, a=%u, k=%u",i,a,fbuf.bufI(FPARTICLEIDX)[i*BONDS_PER_PARTICLE+a]);}
*/
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
            if(i<8)printf("\neterm=(%f,%f,%f)",eterm.x,eterm.y,eterm.z);
/*
            //if(freeze!=true)eterm=((float)(abs_dist < elastic_limit)) * ( ((dist/abs_dist) * (abs_dist-restlength)/restlength )  / *  - damping_coeff*rel_vel  * / );   
*/
            if(restlength!=abs_dist)printf("\ni=%u,j=%u,i_ID=%u,j_ID=%u,bond_j_ID=%u,bondIndex=%u ,restlength=%f,abs_dist=%f, (restlength-abs_dist)/restlength=%f",
                i,j,fbuf.bufI(FPARTICLE_ID)[i],fbuf.bufI(FPARTICLE_ID)[j],other_particle_ID,bondIndex,restlength,abs_dist, (restlength-abs_dist)/restlength );  
/*
,j_pos=(%f,%f,%f) ,j_pos.x,j_pos.y,j_pos.z
///
            float3 vel_i = {fbuf.bufF3(FVEL)[ i ].x,fbuf.bufF3(FVEL)[ i ].y,fbuf.bufF3(FVEL)[ i ].z };
            float3 vel_j = {fbuf.bufF3(FVEL)[ j ].x,fbuf.bufF3(FVEL)[ j ].y,fbuf.bufF3(FVEL)[ j ].z };
            float3 pos_i = {fbuf.bufF3(FPOS)[ i ].x,fbuf.bufF3(FPOS)[ i ].y,fbuf.bufF3(FPOS)[ i ].z };
            float3 pos_j = {fbuf.bufF3(FPOS)[ j ].x,fbuf.bufF3(FPOS)[ j ].y,fbuf.bufF3(FPOS)[ j ].z };


// if ( (i<32 || j<32) &&  (eterm.x!=0 || eterm.y!=0 || eterm.z!=0) &&(modulus>0.0)&&(abs_dist>0.0) )
///
printf("\npart_i=%u, part_j=%u, vel_i=(%f,%f,%f), vel_j(%f,%f,%f), pos_i=(%f,%f,%f), pos_j=(%f,%f,%f), dist=(%f,%f,%f), abs_dist=%f, restlength=%f, ## i=%u, ,a=%u,j=%u, eterm=(%f,%f,%f), (abs_dist < elastic_limit)=%f, (dist/abs_dist)=(%f,%f,%f), modulus=%f, (abs_dist-restlength)=%f, (abs_dist-restlength)/restlength)=%f , - damping_coeff*rel_vel= - (%f,%f,%f)   ",
     fbuf.bufI(FPARTICLE_ID)[i], fbuf.bufI(FPARTICLE_ID)[j],  vel_i.x, vel_i.y,  vel_i.z,  vel_j.x, vel_j.y, vel_j.z,      pos_i.x, pos_i.y, pos_i.z,    pos_j.x, pos_j.y, pos_j.z,        dist.x, dist.y, dist.z, abs_dist,  restlength,
      i,a,j,eterm.x, eterm.y, eterm.z, ((float)(abs_dist < elastic_limit)), (dist/abs_dist).x, (dist/abs_dist).y, (dist/abs_dist).z, modulus, (abs_dist-restlength), ((abs_dist-restlength)/restlength), (damping_coeff*rel_vel).x, (damping_coeff*rel_vel).y,(damping_coeff*rel_vel).z  );
///
if ((j<16)&&(modulus>0.0)&&(abs_dist>0.0))printf("\npart_i=%u, part_j=%u, i=%u, ,a=%u,j=%u, eterm=(%f,%f,%f), (abs_dist < elastic_limit)=%f, (dist/abs_dist)=(%f,%f,%f), modulus=%f, (abs_dist-restlength)=%f, (abs_dist-restlength)/restlength)=%f , - damping_coeff*rel_vel= - (%f,%f,%f)   ",
    fbuf.bufI(FPARTICLE_ID)[i], fbuf.bufI(FPARTICLE_ID)[j], i,a,j,eterm.x, eterm.y, eterm.z, ((float)(abs_dist < elastic_limit)), (dist/abs_dist).x, (dist/abs_dist).y, (dist/abs_dist).z, modulus, (abs_dist-restlength), ((abs_dist-restlength)/restlength), (damping_coeff*rel_vel).x, (damping_coeff*rel_vel).y, (damping_coeff*rel_vel).z  );

                                                                                    // elastic force NB the direction & magnitude of the x,y,z components of the force.
       
float abs_eterm = sqrt(eterm.x*eterm.x+eterm.y*eterm.y+eterm.z*eterm.z);
float3 damping_force = damping_coeff*rel_vel;
if (elastic_limit>0.0) printf("\nCF: particle i=%u, j=%u, a=%i, (abs_dist < elastic_limit)=%f, dsq=%f,  elastic_limit=%f, rest_len=%f, abs_dist=%f,  modulus=%f, dist=(%f,%f,%f), eterm=(%f,%f,%f), abs_eterm=%f, damping_coeff=%f, rel_vel=(%f,%f,%f), damping_force=(%f,%f,%f)\t",
                    i, j, a,  (float)(abs_dist<elastic_limit),  dsq,  elastic_limit,  restlength, abs_dist,  modulus, dist.x,dist.y,dist.z, eterm.x,eterm.y,eterm.z, abs_eterm, damping_coeff, rel_vel.x, rel_vel.y, rel_vel.z, damping_force.x, damping_force.y, damping_force.z  );
*/
            force -= eterm;                                                         // elastic force towards other particle, if (rest_len -abs_dist) is -ve
            atomicAdd( &fbuf.bufF3(FFORCE)[ j ].x, eterm.x);                        // NB Must send equal and opposite force to the other particle
            atomicAdd( &fbuf.bufF3(FFORCE)[ j ].y, eterm.y);
            atomicAdd( &fbuf.bufF3(FFORCE)[ j ].z, eterm.z);                        // temporary hack, ? better to write a float3 attomicAdd using atomicCAS  #########

            if (abs_dist >= elastic_limit){                                         // If (out going bond broken)
                fbuf.bufI(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +1]=0;           // remove broken bond by setting elastic limit to zero.
                fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +3]=0;           // set modulus to zero
                
                uint bondIndex_ = fbuf.bufI(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +6];
                fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + bondIndex_] = UINT_MAX ;
                printf("\n#### Set to broken, i=%i, j=%i, b=%i, fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + b]=UINT_MAX\t####",i,j,bondIndex_);
/*
                for (int b=0; b<BONDS_PER_PARTICLE ; b++){                          // set broken bond UNINT_MAX
                    if ( fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE + b] == i){
                        fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE + b] = UINT_MAX ;
                        printf("\n#### Set to broken, i=%i, j=%i, b=%i, fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE + b]=UINT_MAX\t####",i,j,b / * ,fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE + b] * /);
                    }
                }
*/
                bondsToFill++;
            }
        }
        __syncthreads();    // when is this needed ? ############
    }   
/*    
//if (i<1)printf("\n##X: i=%u,fbuf.bufI(FELASTIDX)[i*BOND_DATA ]=%u",i,fbuf.bufI(FELASTIDX)[i*BOND_DATA]);

//printf("D: {particle i=%u, bondsToFill=%u, own bonds eforce=(%f,%f,%f), freeze=%s}\n",i,bondsToFill,force.x,force.y,force.z, (freeze==true) ? "true" : "false");

for (int a=0;a<BONDS_PER_PARTICLE;a++){  
printf("\nXC: i=%u, k=%u",i,fbuf.bufI(FPARTICLEIDX)[i*BONDS_PER_PARTICLE+a]);
}
*/
	bondsToFill=4; // remove and use result from loop above ? ############
/*
    //if(freeze==true){
*/
        for (int c=0; c < fparam.gridAdjCnt; c++) {                                 // Call contributeForce(..) for fluid forces AND potential new bonds /////////////////////////
            float3 fluid_force = make_float3(0,0,0);
            fluid_force = contributeForce ( i, fbuf.bufF3(FPOS)[ i ], fbuf.bufF3(FVEVAL)[ i ], fbuf.bufF(FPRESS)[ i ], fbuf.bufF(FDENSITY)[ i ], gc + fparam.gridAdj[c], bondsToFill, bonds , freeze); 
/*
            //if(i<8)printf("\nfluid_force=(%f,%f,%f)",fluid_force.x,fluid_force.y,fluid_force.z);
*/
            if(freeze==true) force += fluid_force/10;   // hack temporary reduction of fluid force for debuging elasticity ######### 
        }
/*
    //}
*/
	__syncthreads();   // when is this needed ? ############
    atomicAdd(&fbuf.bufF3(FFORCE)[ i ].x, force.x);                                 // atomicAdd req due to other particles contributing forces via incomming bonds. 
    atomicAdd(&fbuf.bufF3(FFORCE)[ i ].y, force.y);                                 // NB need to reset FFORCE to zero in  CountingSortFull(..)
    atomicAdd(&fbuf.bufF3(FFORCE)[ i ].z, force.z);                                 // temporary hack, ? better to write a float3 attomicAdd using atomicCAS ?  ########
/*
 * for (int a=0;a<BONDS_PER_PARTICLE;a++){  
printf("\nXD: i=%u, k=%u",i,fbuf.bufI(FPARTICLEIDX)[i*BONDS_PER_PARTICLE+a]);
}
*/      
                                                                                    // Add new bonds /////////////////////////////////////////////////////////////////////////////
    int a = BONDS_PER_PARTICLE * (int)(freeze!=true);                               // if (freeze==true) skip for loop, else a=0
/*    
//if (i<4)printf("\n##Y: i=%u,fbuf.bufI(FELASTIDX)[i*BOND_DATA ]=%u, freeze=%s, a=%i",i,fbuf.bufI(FELASTIDX)[i*BOND_DATA], (freeze==true) ? "true" : "false", a);   
*/
    for (; a< BONDS_PER_PARTICLE; a++){
        int otherParticleBondIndex = (int)bonds[a][0] * BONDS_PER_PARTICLE*2 + a*2; // fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex]
/*        
//if (i<34)printf("\n##YZ:  (uint)bonds[a][0] =%u",(uint)bonds[a][0] );  
*/        
        if((uint)bonds[a][0]==i) printf("\n (uint)bonds[a][0]==i, i=%u a=%u",i,a);  // float bonds[BONDS_PER_PARTICLE][3];  [0] = index of other particle, [1] = dsq, [2] = bond_index
                                                                                    // If outgoing bond empty && proposed bond for this quadrant is valid
        if( fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +1] == 0.0  &&  -1 < (uint)bonds[a][0] < pnum  && (uint)bonds[a][0]!=i){  
                                                                                    // NB "bonds[b][0] = -1" is used to indicate no bond found
/*
            // ftemp.bufI(FPARTICLEIDX) [i*BONDS_PER_PARTICLE + a]       // use as lock 
            //do {} while(atomicCAS(&lock,0,1));
            //...
            //__threadfence(); // wait for writes to finish// 
            //free locklock = 0;}}
            
            //result = atomicCAS(&fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex], UINT_MAX, index);                  // atomicCompareAndSwap to prevent race condition   
*/
                                                                                                                    // using 'ftemp' as a lock for 'fbuf'
            do {} while( atomicCAS(&ftemp.bufI(FPARTICLEIDX)[otherParticleBondIndex], UINT_MAX, 0) );               // lock ///////////////// ###### //  if (not locked) write zero to 'ftemp' to lock.
            if (fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex]==UINT_MAX)  fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex] = i;                     //  if (bond is unoccupied) write to 'fbuf' to assign this bond
            ftemp.bufI(FPARTICLEIDX)[otherParticleBondIndex] = UINT_MAX;                                            // release lock ///////// ######
/*
//if(i<34)printf("\n##YZZ: fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex] =%u, index=%u", fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex], index);   
*/
            if (fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex] == i){                                              // if (this bond is assigned) write bond data
                fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex +1] = a;                                            // write i's outgoing bond_index to j's incoming bonds
                uint i_ID = fbuf.bufI(FPARTICLE_ID)[i];                                                             // retrieve permenant particle IDs for 'i' and 'j'
                uint j_ID = fbuf.bufI(FPARTICLE_ID)[(uint)bonds[a][0]];
                float modulus = 10000000;       // 100 000 000                                                                    // 1000000 = min for soft matter integrity // 
                fbuf.bufI(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND]    = (uint)bonds[a][0];                         // [0]current index,
                fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +1] = 2 * sqrt(bonds[a][1]) ;                    // [1]elastic limit  = 2x restlength i.e. %100 strain
                fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +2] = sqrt(bonds[a][1]);                         // [2]restlength = initial length  
                fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +3] = modulus;                                   // [3]modulus
                fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +4] = 2*sqrt(fparam.pmass*modulus);              // [4]damping_coeff = optimal for mass-spring pair.
                fbuf.bufI(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +5] = j_ID;                                      // [5]save particle ID of the other particle NB for debugging
                fbuf.bufI(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +6] = bonds[a][2];                               // [6]bond index at the other particle 'j's incoming bonds
                printf("\nNew Bond i=%u, j=%u, fromPID=%u, toPID=%u",i,(uint)bonds[a][0],i_ID,j_ID);
            }
        }
        __syncthreads();    // when is this needed ? ############
/*
        if (result == UINT_MAX){                                                                                    // if (atomicCAS succeeded)
        	fbuf.bufI(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND]    = (int)bonds[a][0];                              // [0]current index,
                fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +1] = 2 * sqrt(bonds[a][1]) ;                    // [1]elastic limit  = 2x restlength i.e. %100 strain
                fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +2] = sqrt(bonds[a][1]);                         // [2]restlength = initial length  
                float modulus = 10000000; // 1000000 = min for soft matter integrity // 
                fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +3] = modulus;                                   // [3]modulus
                fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +4] = 2*sqrt(fparam.pmass*modulus);              // [4]damping_coeff = optimal for mass-spring pair.
                fbuf.bufI(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +5] = fbuf.bufI(FPARTICLE_ID)[(int)bonds[a][0]]; // [5]save particle ID of the other particle NB for debugging
        }
//if (i<34)printf("\n##Z: i=%u, a=%u, bonds[a][0]=%u, otherParticleBondIndex(i.e. new k)=%u, index=%i",i,a,(uint)bonds[a][0],otherParticleBondIndex,index); // bonds[a][0]=other particle index
//if (i<4)printf("\n##Z: i=%u,fbuf.bufI(FELASTIDX)[i*BOND_DATA ]=%u",i,fbuf.bufI(FELASTIDX)[i*BOND_DATA]);
*/
    }
/*   
//if (i<34)printf("\n##ZZ: i=%u, fbuf.bufI(FELASTIDX)[i*BOND_DATA ]=%u",i,fbuf.bufI(FELASTIDX)[i*BOND_DATA]);
// for (int a=0;a<BONDS_PER_PARTICLE;a++){  
// printf("\nXE: i=%u, k=%u",i,fbuf.bufI(FPARTICLEIDX)[i+a]);
// }

//printf("E: [particle i=%u, pnum=%i, force=(%f,%f,%f), bondsToFill=%u, (bonds[0][0]=%f, bonds[0][1]=%f), (bonds[1][0]=%f, bonds[1][1]=%f), (bonds[2][0]=%f, bonds[2][1]=%f), (bonds[3][0]=%f, bonds[3][1]=%f)],\n",i,pnum,force.x,force.y,force.z,bondsToFill,bonds[0][0],bonds[0][1],bonds[1][0],bonds[1][1],bonds[2][0],bonds[2][1],bonds[3][0],bonds[3][1] );



if(freeze==true){

//printf("\nF: i=%u j=%f,dsq=%f,bond_index=%f,\t\tbonds[1][0,1,2]=%f,%f,%f,\t\tbonds[2][0,1,2]=%f,%f,%f,\t\tbonds[3][0,1,2]=%f,%f,%f",i,
//    bonds[0][0],bonds[0][1],bonds[0][2],  bonds[1][0],bonds[1][1],bonds[1][2],  bonds[2][0],bonds[2][1],bonds[2][2],  bonds[3][0],bonds[3][1],bonds[3][2] );

for (int a=0; a< BONDS_PER_PARTICLE; a++){                 // make new bonds // [0]current index, [1]elastic limit, [2]restlength, [3]modulus, [4]damping_coeff, [5]particle ID, [6]bond index 
if (i<32)printf("\nPrePreCAS: Inserting bond: i=%i bonds[%i][0]=%f, fbuf.bufF(FELASTIDX)[.. +1]=%f\t",
    i,a,bonds[a][0],fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +1]);                            // fbuf.bufF(FELASTIDX)[.. +1] = elastic limit, 0=> broken
            if( fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +1] == 0.0  &&  bonds[a][0] >=0.0 )      // NB "bonds[b][0] = -1" is used to indicate no bond found
            {   
                int index = i;
                //int otherParticleIndx = bonds[a][0];
                //int otherParticleBondIndex = bonds[a][2];
                int otherParticleBondIndex = (int)bonds[a][0] * BONDS_PER_PARTICLE +  a ;// bonds[a][2];  //
if(i<32)printf("\nPreCAS: i=%u,index=%i,bonds[a][0]=%f,a=%i,otherParticleBondIndex=%i, fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex]=%u ",
    i,index,bonds[a][0],a,otherParticleBondIndex, fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex]);   

                uint result = atomicCAS(&fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex], UINT_MAX, index); // atomicCompareAndSwap to prevent race condition
                
if(i<32)printf("\nPostCAS: fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex]=%u, result =%u",fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex], result);                
                if (result == UINT_MAX){                                                                    // if (atomicCAS succeeded)
                    fbuf.bufI(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND]    = (int)bonds[a][0];                   // [0]current index,
                    fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +1] = 2 * sqrt(bonds[a][1]) ;        // [1]elastic limit  = 2x restlength i.e. %100 strain
                    fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +2] = sqrt(bonds[a][1]);             // [2]restlength = initial length  
                    float modulus = 10000000; // 1000000 = min for soft matter integrity // 
                    fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +3] = modulus;                       // [3]modulus
                    fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +4] = 2*sqrt(fparam.pmass*modulus);  // [4]damping_coeff = optimal for mass-spring pair.
                    fbuf.bufI(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +5] = fbuf.bufI(FPARTICLE_ID)[(int)bonds[a][0]];  // [5]save particle ID of the other particle NB for debugging
if(i<32)printf("\nPostPostCAS: result=%u Inserting bond: i=%i bonds[%i][0]=%i, %i, partID=%i,\t restlength=%f\t",
       result,i,a,(int)bonds[a][0], fbuf.bufI(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND], 
       fbuf.bufI(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +5], fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +2]  ); 
                }
            }
if(i<32)printf("particle i=%u, a=%u,\t((FELASTIDX)[0][0]=%u, (FELASTIDX)[0][1]=%f),\t\t\t(FELASTIDX)[1][0]=%u, (FELASTIDX)[1][1]=%f),\t\t\t((FELASTIDX)[2][0]=%u, (FELASTIDX)[2][1]=%f),\t\t\t((FELASTIDX)[3][0]=%u, (FELASTIDX)[3][1]=%f),\n",i,a,
       fbuf.bufI(FELASTIDX)[i*BOND_DATA + 0*DATA_PER_BOND],  fbuf.bufF(FELASTIDX)[i*BOND_DATA + 0*DATA_PER_BOND +1], 
       fbuf.bufI(FELASTIDX)[i*BOND_DATA + 1*DATA_PER_BOND],  fbuf.bufF(FELASTIDX)[i*BOND_DATA + 1*DATA_PER_BOND +1],
       fbuf.bufI(FELASTIDX)[i*BOND_DATA + 2*DATA_PER_BOND],  fbuf.bufF(FELASTIDX)[i*BOND_DATA + 2*DATA_PER_BOND +1],
       fbuf.bufI(FELASTIDX)[i*BOND_DATA + 3*DATA_PER_BOND],  fbuf.bufF(FELASTIDX)[i*BOND_DATA + 3*DATA_PER_BOND +1]
      );    
        }// end for (a=0 ...)
    }  // end if (freeze===true)
    
*/
}
/*
extern "C" __device__ void contributeFreeze ( int i, float3 ipos, float3 iveleval, float ipress, float idens, int cell, 
                                    uint  elastIdx[BONDS_PER_PARTICLE * 2],     uint  new_bond_list[BONDS_PER_PARTICLE],
                                    float new_bond_dsq[BONDS_PER_PARTICLE],     uint  bonds_to_fill )
{
    if ( fbuf.bufI(FGRIDCNT)[cell] == 0 ) return;                                   // If the cell is empty, skip it.
	float dsq;	
	float3 dist;
	int j;
	int clast = fbuf.bufI(FGRIDOFF)[cell] + fbuf.bufI(FGRIDCNT)[cell];

	for ( int cndx = fbuf.bufI(FGRIDOFF)[cell]; cndx < clast; cndx++ ) {            // For particles in this cell.
		j = fbuf.bufI(FGRID)[ cndx ];                                               // j = index of other particle
        bool known = false;
        for(int a=0; a < BONDS_PER_PARTICLE ; a++){                                 // for this other particle, check list of bonds for particle of this thread
            known = known || (j == elastIdx[a]);
        }
		dist = ( ipos - fbuf.bufF3(FPOS)[ j ] );                                    // dist in cm (Rama's comment)
                                                                                    // only consider particles where dist.y>0 (also eliminates bonding to self). NB also only call cells >= in y.
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);                      // scalar distance squared
        known = known ||(dsq > fparam.rd2)||(dist.y<=0);                            // if particle out of range, skip it.
        
		for(int k=0; k< BONDS_PER_PARTICLE ; k++){                                  // fills first 'bonds_to_fill' of uint new_bond_list[4][2] with closest particles, 
                                                                                    // for which dist.x>=0 && not allready bonds.
            if ((dsq < new_bond_dsq[k])&&(!known)){                                 // If closer, or new bond not yet filled.
                new_bond_list[k] = j;                                               // NB need to 
                new_bond_dsq[k] = dsq;                                              // (i) initialize new_bond_list[4][2] to zero, (ii) launch only for 18 bins, 
                known = true;                                                       // (iii) transfer only new_bond_list[bonds_to_fill][2] to unfilled bonds
            }
        }
    }
}


extern "C" __global__ void freeze ( int pnum)                                       // creates elastic bonds 
{
    uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	                        // particle index
	if ( i >= pnum ) return;
    uint particleID = fbuf.bufI(FPARTICLE_ID)[i];                                   // ID of this particle
	uint gc = fbuf.bufI(FGCELL)[ i ];                                               // Get search cell
	if ( (gc == GRID_UNDEF) || (particleID==0) ) return;                            // particle out-of-range
	gc -= (1*fparam.gridRes.z + 1)*fparam.gridRes.x + 1;
    uint elastIdx[BONDS_PER_PARTICLE];
    uint bonds_to_fill = 0;
    
    for (int a=0;a<BONDS_PER_PARTICLE;a++){ 
        elastIdx[a] = fbuf.bufI(FELASTIDX)[i*BONDS_PER_PARTICLE*2 + a*2]; 
        if (fbuf.bufI(FELASTIDX)[i*BONDS_PER_PARTICLE*2 + a*2]==0) bonds_to_fill++; // copy FELASTIDX to thread memory for particle i.
    }   
    if (bonds_to_fill>0){
        uint new_bond_list[BONDS_PER_PARTICLE];
        float new_bond_dsq[BONDS_PER_PARTICLE];
        for (int a=0;a<BONDS_PER_PARTICLE;a++){ 
            new_bond_list[a]=0; 
            new_bond_dsq[a]=fparam.rd2;
        }   
        for (int c=0; c < fparam.gridAdjCnt; c++) {                                 // fetch list of nearest unbonded particles > in x than this particle.
            contributeFreeze ( i, fbuf.bufF3(FPOS)[ i ], fbuf.bufF3(FVEVAL)[ i ], fbuf.bufF(FPRESS)[ i ], fbuf.bufF(FDENSITY)[ i ], gc + fparam.gridAdj[c], elastIdx, new_bond_list, new_bond_dsq, bonds_to_fill);
        }
        for (int b=0, c=0;b<BONDS_PER_PARTICLE; b++){                               // insert new bonds
            if (elastIdx[b]==0  && c<bonds_to_fill){
                fbuf.bufI(FELASTIDX)[i*BONDS_PER_PARTICLE*2 + b*2] = new_bond_list[c];
                c++;
                if (fbuf.bufI(FELASTIDX)[i*BONDS_PER_PARTICLE*2 + b*2 +1] == 0) fbuf.bufI(FELASTIDX)[i*BONDS_PER_PARTICLE*2 + b*2 +1] = 65552; // nb MakeDemo sets 65551
            }
        }
    }
}
*/
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
//if(accel.x!=0.0||accel.y!=0.0||accel.z!=0.0)printf("\navanceParticles: particle=%u, net_force=(%f,%f,%f)",i,accel.x,accel.y,accel.z);   
	accel *= fparam.pmass;	
//    printf(" accel=%f,%f,%f\t",accel.x,accel.y,accel.z);
		
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
/*
	float3 vel_old = {fbuf.bufF3(FVEL)[i].x,fbuf.bufF3(FVEL)[i].y,fbuf.bufF3(FVEL)[i].z};
	if(accel.x!=0.0||accel.y!=float(-9.8)||accel.z!=0.0)printf("\navanceParticles end: particle=%u, accel=(%f,%f,%f), vel_old+(%f,%f,%f)",i,accel.x,accel.y,accel.z, vel_old.x,vel_old.y,vel_old.z);  
*/	
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

