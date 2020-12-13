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
__constant__ FBondParams    fbondparams;    // GPU copy of remodelling parameters. 
__constant__ uint			gridActive;

#define SCAN_BLOCKSIZE		512
//#define FLT_MIN  0.000000001              // set here as 2^(-30)
//#define UINT_MAX 65535

extern "C" __global__ void insertParticles ( int pnum )                                         // decides which bin each particle belongs in.
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index
	if ( i >= pnum ) return;

	//-- debugging (pointers should match CUdeviceptrs on host side)
	// printf ( " pos: %012llx, gcell: %012llx, gndx: %012llx, gridcnt: %012llx\n", fbuf.bufC(FPOS), fbuf.bufC(FGCELL), fbuf.bufC(FGNDX), fbuf.bufC(FGRIDCNT) );
    if (i==0)printf("\ninsertParticles(): pnum=%u\n",pnum);

	register float3 gridMin =	fparam.gridMin;                                  // "register" is a compiler 'hint', to keep this variable in thread register
	register float3 gridDelta = fparam.gridDelta;                                //  even if other variable have to be moved to slower 'local' memory  
	register int3 gridRes =		fparam.gridRes;                                  //  in the streaming multiprocessor's cache.
	register int3 gridScan =	fparam.gridScanMax;
    register int gridTot =      fparam.gridTotal;

	register int		gs;
	register float3		gcf;
	register int3		gc;	

	gcf = (fbuf.bufF3(FPOS)[i] - gridMin) * gridDelta;                           // finds bin as a float3
	gc = make_int3( int(gcf.x), int(gcf.y), int(gcf.z) );                        // crops to an int3
	gs = (gc.y * gridRes.z + gc.z)*gridRes.x + gc.x;                             // linearizes to an int for a 1D array of bins

	if ( gc.x >= 1 && gc.x <= gridScan.x && gc.y >= 1 && gc.y <= gridScan.y && gc.z >= 1 && gc.z <= gridScan.z ) {
		fbuf.bufI(FGCELL)[i] = gs;											     // Grid cell insert.
		fbuf.bufI(FGNDX)[i] = atomicAdd ( &fbuf.bufI(FGRIDCNT)[ gs ], 1 );       // Grid counts.         //  ## counts particles in this bin.
                                                                                                         //  ## add counters for dense lists. ##############
        // for each gene, if active, then atomicAdd bin count for gene
        for(int gene=0; gene<NUM_GENES; gene++){ // NB data ordered FEPIGEN[gene][particle] AND +ve int values -> active genes.
            //if(i==0)printf("\n");
            if ( (int)fbuf.bufI(FEPIGEN) [i + gene*pnum] ){ 
                atomicAdd ( &fbuf.bufI(FGRIDCNT_ACTIVE_GENES)[gene*gridTot  + gs ], 1 );
                //if(i<10)printf("\n,i=%u, gene=%u, gs=%u, fbuf.bufI(FGRIDCNT_ACTIVE_GENES)[ gene*gridTot  + gs ]=%u",
                //    i, gene, gs, fbuf.bufI(FGRIDCNT_ACTIVE_GENES)[ gene*gridTot  + gs ]);
            }
            // could use a small array of uints to store gene activity as bits. This would reduce the reads, but require bitshift and mask to read. 
            //if(i==0)printf("\nfbuf.bufI(FEPIGEN) [i*NUM_GENES + gene]=%u  gene=%u  i=%u,",fbuf.bufI(FEPIGEN)[i*NUM_GENES + gene], gene ,i  );
        }
        //if(i==0)printf("\n");
	} else {
		fbuf.bufI(FGCELL)[i] = GRID_UNDEF;
	}
}

extern "C" __global__ void prefixFixup(uint *input, uint *aux, int len)                         // merge *aux into *input  
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

extern "C" __global__ void tally_denselist_lengths(int num_lists, int fdense_list_lengths, int fgridcnt, int fgridoff )
{
    uint list = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                                  // which dense list is being tallied.
	if ( list >= num_lists ) return;
    register int gridTot =      fparam.gridTotal;
    fbuf.bufI(fdense_list_lengths)[list] = fbuf.bufI(fgridcnt)[(list+1)*gridTot -1] + fbuf.bufI(fgridoff)[(list+1)*gridTot -1];
}

extern "C" __global__ void countingSortFull ( int pnum )                                        // Counting Sort - Full (deep copy)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;            // particle index
	if ( i >= pnum ) return;
    if (i==0)printf("\ncountingSortFull(): pnum=%u\n",pnum);
	// Copy particle from original, unsorted buffer (msortbuf),
	// into sorted memory location on device (mpos/mvel)
	uint icell = ftemp.bufI(FGCELL) [ i ];                             // icell is bin into which i is sorted in fbuf.*

	if ( icell != GRID_UNDEF ) {	  
		// Determine the sort_ndx, location of the particle after sort		
        uint indx =  ftemp.bufI(FGNDX)  [ i ];                         // indx is off set within new cell
        int sort_ndx = fbuf.bufI(FGRIDOFF) [ icell ] + indx ;          // global_ndx = grid_cell_offet + particle_offset	
		float3 zero; zero.x=0;zero.y=0;zero.z=0;
        
        // Make dense lists for (i) available genes (ii) active genes (iii) diffusion particles (iv) active/reserve particles. ######################
        // NB req new FGNDX & FGRIDOFF for each of (i-iv).
        // Write (1) list of current array lengths, (2) arrays containing  [sort_ndx] of relevant particles.
        // In use kernels read the array to access correct particle.
        // If there is data only used by such kernels, then it should be stored in a dense array.  
        
		// Transfer data to sort location
		fbuf.bufI (FGRID) [ sort_ndx ] =	sort_ndx;                  // full sort, grid indexing becomes identity		
		fbuf.bufF3(FPOS) [sort_ndx] =		ftemp.bufF3(FPOS) [i];
		fbuf.bufF3(FVEL) [sort_ndx] =		ftemp.bufF3(FVEL) [i];
      if(i==0){
          printf("\ncountingSortFull :");
          printf("\nsort_ndx=%i",sort_ndx);
          printf("\nfbuf.bufF3(FVEL)[sort_ndx].x=%f",fbuf.bufF3(FVEL)[sort_ndx].x);
      }
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
            if (k<pnum){                                                                // (k>=pnum) => bond broken // crashes when j=0 (as set in demo), after run().
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
        for (int a=0;a<NUM_GENES;a++){fbuf.bufI (FEPIGEN) [sort_ndx + pnum*a]   =	ftemp.bufI(FEPIGEN) [i + pnum*a];}
	}
} 

extern "C" __global__ void countingSortDenseLists ( int pnum )
{
    unsigned int bin = threadIdx.x + blockIdx.x * SCAN_BLOCKSIZE/2;
    register int gridTot =      fparam.gridTotal;
	if ( bin >= gridTot ) return;                                    // for each bin, for each particle, for each gene, 
                                                                     // if gene active, then write to dense list 
    uint count = fbuf.bufI (FGRIDCNT)[bin];
    if (count==0) return;                                            // return here means that if all bins in this threadblock are empty,
                                                                     // then this multiprocessor is free for the next threadblock.
                                                                     // NB Faster still would be a list of occupied bins.
    uint grdoffset = fbuf.bufI (FGRIDOFF)[bin];
    uint gene_counter[NUM_GENES]={0};
    
    register uint* lists[NUM_GENES];
    for (int gene=0; gene<NUM_GENES;gene++) lists[gene]=fbuf.bufII(FDENSE_LISTS)[gene]; // This element entry is a pointer
    
    register uint* offsets[NUM_GENES];
    for (int gene=0; gene<NUM_GENES;gene++) offsets[gene]=&fbuf.bufI(FGRIDOFF_ACTIVE_GENES)[gene * gridTot];   // The address of this element
    
    if (grdoffset+count > pnum){    printf("\n\n!!Overflow: (grdoffset+count > pnum), bin=%u \n",bin);     return;}
    
    for(int particle=grdoffset; particle<grdoffset+count; particle++){
        for(int gene=0; gene<NUM_GENES; gene++){
            if(  (int)fbuf.bufI(FEPIGEN) [particle + pnum*gene] ) {                      // if (this gene is active in this particle)
                lists[gene][ offsets[gene][bin] + gene_counter[gene] ]=particle;
                gene_counter[gene]++;
                if( gene_counter[gene]>fbuf.bufI(FGRIDCNT_ACTIVE_GENES)[gene*gridTot +bin] )   
                    printf("\n\n overflow: particle=%u, gene=%u, bin=%u, fbuf.bufI (FGRIDCNT_ACTIVE_GENES)[bin]=%u \n",
                           particle, gene, bin, fbuf.bufI (FGRIDCNT_ACTIVE_GENES)[bin]);
            }
        }
    }
}

extern "C" __global__ void countingSortChanges ( int pnum )
{
    unsigned int bin = threadIdx.x + blockIdx.x * SCAN_BLOCKSIZE/2;
    register int gridTot =      fparam.gridTotal;
	if ( bin >= gridTot ) return;                                    // for each bin, for each particle, for each change_list, 
                                                                     // if change_list active, then write to dense list 
    uint count = fbuf.bufI (FGRIDCNT_CHANGES)[bin];
    if (count==0) return;                                            // return here means that if all bins in this threadblock are empty,
                                                                     // then this multiprocessor is free for the next threadblock.
    uint grdoffset = fbuf.bufI (FGRIDOFF)[bin];
    uint change_list_counter[NUM_CHANGES]={0};
    
    register uint* lists[NUM_CHANGES];
    for (int change_list=0; change_list<NUM_CHANGES;change_list++) lists[change_list]=fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list];           // This element entry is a pointer
    
    register uint list_length[NUM_CHANGES];
    for (int change_list=0; change_list<NUM_CHANGES;change_list++) list_length[change_list]=fbuf.bufI(FDENSE_LIST_LENGTHS_CHANGES)[change_list];
    
    register uint* offsets[NUM_CHANGES];
    for (int change_list=0; change_list<NUM_CHANGES;change_list++) offsets[change_list]=&fbuf.bufI(FGRIDOFF_CHANGES)[change_list * gridTot];   // The address of this element
    
    if (grdoffset+count > pnum){    printf("\n\n!!Overflow: (grdoffset+count > pnum), bin=%u \n",bin);     return;}
    
    for(int particle=grdoffset; particle<grdoffset+count; particle++){                                                                         // loop through particles in bin
        for(int bond=0; bond<BONDS_PER_PARTICLE; bond++){                                                                                      // loop through bonds on particle
            uint change = fbuf.bufI(FELASTIDX) [particle*BOND_DATA + bond*DATA_PER_BOND + 8];                                                  // binary change indicator per bond.
            if(change) {
                for (uint change_type=1, change_list=0; change_list<NUM_CHANGES; change_type*=2, change_list++){                               // loop through change indicator  
                    if(change & change_type){                                                                                                  // bit mask to ID change type due to this bond
                        lists[change_list][ (offsets[change_list][bin] + change_list_counter[change_list]) ]=particle;                         // write particleIdx to change list
                        lists[change_list][ (offsets[change_list][bin] + change_list_counter[change_list] + list_length[change_list]) ]=bond;  // write bondIdx to change list
                        change_list_counter[change_list]++;
                    }
                }
            }
        }
    }
}

extern "C" __device__ float contributePressure ( int i, float3 p, int cell )  
// pressure due to particles in 'cell'. NB for each particle there are 27 cells in which interacting particles might be.
{			
	if ( fbuf.bufI(FGRIDCNT)[cell] == 0 ) return 0.0;                       // If the cell is empty, skip it.

	float3 dist;
	float dsq, c, sum = 0.0;
	register float d2 = fparam.psimscale * fparam.psimscale; // max length in simulation space
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
	//printf("computePressure, particle %u, sum=%.32f\n", i, sum);
		
	// Compute Density & Pressure
	sum = sum * fparam.pmass * fparam.poly6kern;
	if ( sum == 0.0 ) sum = 1.0;
	fbuf.bufF(FPRESS)  [ i ] = ( sum - fparam.prest_dens ) * fparam.pintstiff;
	fbuf.bufF(FDENSITY)[ i ] = 1.0f / sum;
}

extern "C" __global__ void computeGeneAction ( int pnum, int gene, uint list_len )  //NB here pnum is for the dense list
{
    uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                                         // particle index
    if ( i >= list_len ) return;
    uint particle_index = fbuf.bufII(FDENSE_LISTS)[gene][i];
    if (particle_index >= pnum){
        printf("\ncomputeGeneAction: (particle_index >= pnum),  gene=%u, i=%u, list_len=%u, particle_index=%u, pnum=%u .\t",
            gene, i, list_len, particle_index, pnum);
    }    
    int delay = (int)fbuf.bufI(FEPIGEN)[gene*pnum + particle_index];                                // Change in _epigenetic_ activation of this particle
    if (delay < INT_MAX){                                                                           // (FEPIGEN==INT_MAX) => active & not counting down.
        fbuf.bufI(FEPIGEN)[gene*pnum + particle_index]--;                                           // (FEPIGEN<1) => inactivated @ insertParticles(..)
        if (delay==1  &&  gene<NUM_GENES && fbuf.bufI(FEPIGEN)[ (gene+1)*pnum + particle_index ] )  // If next gene is active, start count down to inactivate it.
            fbuf.bufI(FEPIGEN)[(gene+1)*pnum + particle_index] = fgenome.delay[gene+1] ;            // Start countdown to silence next gene.
    }                                                                                               // (fgenome.delay[gene+1]==INT_MAX) => barrier to spreading inactivation.
    uint sensitivity[NUM_GENES];                                                                    // TF sensitivities : broadcast to threads
    #pragma unroll                                                                                  // speed up by eliminating loop logic.
    for(int j=0;j<NUM_GENES;j++) sensitivity[j]= fgenome.sensitivity[gene][j];                      // for each gene, its sensitivity to each TF or morphogen
    if(i==list_len-1)printf("\ncomputeGeneAction Chk : gene=%u, i=%u, list_len=%u, particle_index=%u, pnum=%u ,  sensitivity[15]=%u.\t",
            gene, i, list_len, particle_index, pnum, sensitivity[15]);                              // debug chk 
    float activity=0;                                                                               // compute current activity of gene
    #pragma unroll
    for (int tf=0;tf<NUM_TF;tf++){                                                                  // read FCONC
        if(sensitivity[tf]!=0){                                                                     // skip reading unused fconc[]
            activity +=  sensitivity[tf] * fbuf.bufI(FCONC)[particle_index + pnum*tf];
        }                                                           
    }
    // Compute actions                                             // Non-difusible TFs inc instructions to particle modification kernel wrt behaviour (cell type). 
    int numTF =  fgenome.secrete[gene][2*NUM_TF];                  // (i) secrete sparse list of TFs  => atomicAdd(ftemp...) to allow async gene kernels.
    for (int j=0;j<numTF;j++){
        int tf = fgenome.secrete[gene][j*2];
        int secretion_rate = fgenome.secrete[gene][j*2 + 1];
        atomicAdd( &ftemp.bufI(FCONC)[particle_index*NUM_TF +tf], secretion_rate*activity);
    }
    int numLRNA = fgenome.activate[gene][2*NUM_GENES];             // (ii) secrete spare list long RNA => activate other genes.  NB threshold.
    for (int j=0;j<numLRNA;j++){
        int other_gene = fgenome.activate[gene][j*2];
        int threshold = fgenome.activate[gene][j*2 + 1];
        if(threshold<activity)
        atomicAdd( &fbuf.bufI(FEPIGEN)[other_gene*pnum + particle_index], 1);   // what should be the initial state of other_gene when activated ?
    }
}

extern "C" __global__ void computeNerveActivation ( int pnum ) //TODO computeNerveActivation    // initially simple sparse random connections + STDP, later neurogenesis
{                                                                 // NB (i) sensors concetrated in hands & feet (ii)stimuls from womb wall 
    
}

extern "C" __global__ void computeMuscleContraction ( int pnum ) //TODO computeMuscleContraction  // read attached nerve, compute force  
{
    
}

extern "C" __global__ void computeBondChanges ( int pnum, uint list_length )// Given the action of the genes, compute the changes to particle properties & splitting/combining 
{                                                                           // Also "inserts changes" 
    uint particle_index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                             // particle index
    if ( particle_index >= list_length ) return; // pnum should be length of list.
    // ## TODO convert i to particle index _iff_ not called for all particles : use a special dense list for "living tissue", made at same time as gene lists
    uint i = fbuf.bufII(FDENSE_LISTS)[2][particle_index]; // call for dense list of living cells (gene'2'living/telomere (has genes))
    if ( i >= pnum ) return; 
    
    float * fbufFCONC = &fbuf.bufF(FCONC)[i*NUM_TF];
    float * ftempFCONC = &ftemp.bufF(FCONC)[i*NUM_TF];
    float * fbufFEPIGEN = &fbuf.bufF(FEPIGEN)[i*NUM_TF];
    float * ftempFEPIGEN = &ftemp.bufF(FEPIGEN)[i*NUM_TF];
    
    for(int j=0; j<NUM_TF;j++)      fbufFCONC[j] += ftempFCONC[j];              // list of transcription factor conc for this particle
    for(int j=0; j<NUM_GENES;j++) fbufFEPIGEN[j] += ftempFEPIGEN[j];            // list of epigenetic activations for this particle
    /*
    // read FCONC, FNERVEIDX, , FPRESS, FDENSITY

    // define FELASTIDX  14  // [0]current index, [1]elastic limit, [2]restlength, [3]modulus, [4]damping coeff, [5]particle ID, [6]bond index [7]stress integrator
    
    //(i)modify  FCONC, FNERVEIDX, FPRESS, FDENSITY, FMASS_RADIUS,  FELASTIDX [1]elastic limit, [2]restlength, [3]modulus, [4]damping coeff
    
    ///// NB this is the new "freeze", responsible for bond formation.
    // Read FCONC[0] "growth", FELASTIDX[all], FEPIGEN[2-10 & 14,15] 
    
    // Equation to modify spring parameters :   NB limit to modulus -> duplicate particles & fibres
    //(i) principal fibre (tendons)                 i.e. spring[0]                      
    //(ii) 1st 3 fibres (fibrocyte, elastocyte)     i.e. spring[0-2]
    //(iii) all fibres (cartilage & bone)           i.e. spring[0-BONDS_PER_PARTICLE]
    */
    uint* bond_uint_ptr = &fbuf.bufI(FELASTIDX)[i*BOND_DATA];//fbuf.bufI(FELASTIDX)[i*BOND_DATA + bond*DATA_PER_BOND +  ] ;
    float*bond_flt_ptr  = &fbuf.bufF(FELASTIDX)[i*BOND_DATA]; //FELASTIDX [0]current index, [1]elastic limit, [2]restlength, [3]modulus, [4]damping coeff, [5]particle ID, [6]bond index [7]stress integrator,  [8]change-type binary indicator
    register int gridTot = fparam.gridTotal;
    /*
    // hold as a texture or similar. // part of fparams ? 
    // 3 materials for bonds - elastin, collagen, apatite - depend on (i)tissue type (ii) additional bonds 
    
    // -- special genes, for simulation efficiency
    // 0 active particles
    // 1 solid  (has springs)
    // 2 living/telomere (has genes)
    
    // -- behavioural celltypes
    // 3 fat
    // 4 nerves 
    // 5 fibrocyte
    // 6 tendon
    // 7 muscle
    // 8 cartilage
    // 9 bone
    // 10 elastic_lig
    */
    uint bond_type[BONDS_PER_PARTICLE] = {0}; //  0=elastin, 1=collagen, 2=apatite
    // calculate material type for bond
    if (fbufFEPIGEN[9]/*bone*/) for (int bond=0; bond<BONDS_PER_PARTICLE; bond++) bond_type[bond] = 2;
    else if (fbufFEPIGEN[6]/*tendon*/||fbufFEPIGEN[7]/*muscle*/||fbufFEPIGEN[10]/*elast lig*/) {bond_type[0] = 1; bond_type[3] = 1;}
    else if (fbufFEPIGEN[6]/*cartilage*/)for (int bond=0; bond<BONDS_PER_PARTICLE; bond++) bond_type[bond] = 1;
    
    for (uint bond=0; bond<BONDS_PER_PARTICLE;bond++, bond_uint_ptr+=DATA_PER_BOND, bond_flt_ptr+=DATA_PER_BOND ){              
        float stress = bond_flt_ptr[7];
        
        //FBondParams *params_ =  &fgenome.fbondparams[bond_type[bond]];    //fluid_system_cuda.cu(466): warning host member "FBondParams::param" cannot be directly read in a device function
        bond_flt_ptr[2]/*rest length*/ +=  (stress - fgenome.param[bond_type[bond]][fgenome.elongation_threshold])   * fgenome.elongation_factor;
        bond_flt_ptr[2]/*rest length*/ +=  (stress - fgenome.param[bond_type[bond]][fgenome.elongation_threshold])   * fgenome.strengthening_factor;
        // "insert changes"
  //      uint * fbufFGRIDCNT_CHANGES = fbuf.bufI(FGRIDCNT_CHANGES);
  //      int m = 1 + (fbufFEPIGEN[7]>0/*muscle*/||fbufFEPIGEN[10]>0);                                // i.e. if (fbufFEPIGEN[7]>0/*muscle*/) m=2 else m=1;
                                                                                                    // NB two different lists for each change, for (muscle & elastic ligg  vs other tissues)
  //      if (bond_uint_ptr[0]/*other particle*/==UINT_MAX/*bond broken*/ && (bond < 3 || fbufFEPIGEN[8] ||  fbufFEPIGEN[9]/*cartilage OR bone*/)  ){
  //          //TODO what happens when bond broken vs never existed ?  NB information about direction of broken bond.
  //          atomicAdd ( &fbufFGRIDCNT_CHANGES[ 0*gridTot  + fbuf.bufI(FGCELL)[i] ], 1 );            //add to heal list 
  //          bond_uint_ptr[8]+=1;                                                                    // FELASTIDX [8]change-type binary indicator NB accumulates all changes for this bond
  //      }else                                                                                       // prevents clash with heal.
        /*{
            if (bond_flt_ptr[2]>fgenome.param[bond_type[bond]][fgenome.max_rest_length]) {  
                atomicAdd ( &fbufFGRIDCNT_CHANGES[ m*gridTot  + fbuf.bufI(FGCELL)[i] ], 1 );        //add to elongate list , store particleIdx & bond 
                bond_uint_ptr[8]+=2*m;
            }
            if (bond_flt_ptr[2]<fgenome.param[bond_type[bond]][fgenome.min_rest_length]) {  
                atomicAdd ( &fbufFGRIDCNT_CHANGES[ (2+m)*gridTot  + fbuf.bufI(FGCELL)[i] ], 1 );    //add to shorten list
                bond_uint_ptr[8]+=8*m;
            }
            if (bond_flt_ptr[3]>fgenome.param[bond_type[bond]][fgenome.max_modulus])     {  
                atomicAdd ( &fbufFGRIDCNT_CHANGES[ (4+m)*gridTot  + fbuf.bufI(FGCELL)[i] ], 1 );    //add to strengthen list 
                bond_uint_ptr[8]+=32*m;
            }
            if (bond_flt_ptr[3]<fgenome.param[bond_type[bond]][fgenome.min_modulus])     {  
                atomicAdd ( &fbufFGRIDCNT_CHANGES[ (6+m)*gridTot  + fbuf.bufI(FGCELL)[i] ], 1 );    //add to weaken list
                bond_uint_ptr[8]+=64*m;
            }
        }*/
        // bond_uint_ptr[8]+=2^n; is ELASTIDX for binary change indicator per bond. 
    }
}

//////   Particle modification kernels called together. Must make sure that they cannot clash. NB atomic operations. 
extern "C" __device__ void addParticle (uint parent_Idx, uint &new_particle_Idx)                    // Template for stregthening & lengthening kernels
{   
    int particle_Idx = atomicAdd(&fparam.pnumActive, 1);  // NOT safe to use fbuf.bufI(FGRIDOFF)[fparam.gridTotal] as active particle count!
    if (particle_Idx >= 0  &&  particle_Idx < fparam.pnum) {
        new_particle_Idx                            = particle_Idx;
        fbuf.bufF3(FVEVAL)[new_particle_Idx]        = fbuf.bufF3(FVEVAL)[parent_Idx]; // NB could use average with next row. Prob not needed, because old bond is stretched.
        fbuf.bufF3(FVEL)[new_particle_Idx]          = fbuf.bufF3(FVEL)[parent_Idx];
        fbuf.bufF3(FFORCE)[new_particle_Idx]        = fbuf.bufF3(FFORCE)[parent_Idx];
        fbuf.bufI(FMASS_RADIUS)[new_particle_Idx]   = fbuf.bufI(FMASS_RADIUS)[parent_Idx];
        fbuf.bufI(FAGE)[new_particle_Idx]           = 0;
        fbuf.bufI(FCLR)[new_particle_Idx]           = fbuf.bufI(FCLR)[parent_Idx];
        fbuf.bufI(FNERVEIDX)[new_particle_Idx]      = fbuf.bufI(FNERVEIDX)[parent_Idx];
        for (int tf=0;tf<NUM_TF;tf++)                   fbuf.bufF(FCONC)[new_particle_Idx*NUM_TF+tf]          = fbuf.bufF(FCONC)[parent_Idx*NUM_TF+tf];
        for (int gene=0;gene<NUM_GENES;gene++)          fbuf.bufF(FEPIGEN)[new_particle_Idx*NUM_GENES+gene]   = fbuf.bufF(FEPIGEN)[parent_Idx*NUM_GENES+gene];
        // TODO should FEPIGEN be float, int or uint?
    }
}

extern "C" __device__ void removeParticle (uint particle_Idx)                                                       // Template for weakening & shortening kernels
{   //  active particle count : done automatically be insert_particles(..)
    //  sets values to null particle, => will be sorted to reserve section of particle list in next time step.
    fbuf.bufF3(FPOS)[particle_Idx]      = fparam.pboundmax;
    fbuf.bufF3(FVEVAL)[particle_Idx]    = make_float3(0,0,0);
    fbuf.bufF3(FVEL)[particle_Idx]      = make_float3(0,0,0);
    fbuf.bufF3(FFORCE)[particle_Idx]    = make_float3(0,0,0);
    for (int incomingBondIdx=0; incomingBondIdx<BONDS_PER_PARTICLE; incomingBondIdx++){                             // Remove reciprocal data for incoming bonds
        uint jIdx       = fbuf.bufI(FPARTICLEIDX)[particle_Idx*BONDS_PER_PARTICLE*2 + incomingBondIdx*2];
        uint bondIdx    = fbuf.bufI(FPARTICLEIDX)[particle_Idx*BONDS_PER_PARTICLE*2 + incomingBondIdx*2 +1];
        if(jIdx!=UINT_MAX){
            uint *ptr_elastidx =  &fbuf.bufI(FELASTIDX)[jIdx*BOND_DATA + bondIdx*DATA_PER_BOND];
            for (int j=0;j<DATA_PER_BOND;j++)  ptr_elastidx[j] = UINT_MAX;
        fbuf.bufI(FPARTICLEIDX)[particle_Idx*BONDS_PER_PARTICLE*2 + incomingBondIdx*2]      = UINT_MAX;
        fbuf.bufI(FPARTICLEIDX)[particle_Idx*BONDS_PER_PARTICLE*2 + incomingBondIdx*2 +1]   = UINT_MAX;
        }
    }
    for (int outgoingBondIdx=0; outgoingBondIdx<BONDS_PER_PARTICLE; outgoingBondIdx++){                             // Remove reciprocal data for outgoing bonds
        uint jIdx       = fbuf.bufI(FELASTIDX)[particle_Idx*DATA_PER_BOND + outgoingBondIdx*BONDS_PER_PARTICLE +0];
        uint bondIdx    = fbuf.bufI(FELASTIDX)[particle_Idx*DATA_PER_BOND + outgoingBondIdx*BONDS_PER_PARTICLE +6];
        if(jIdx!=UINT_MAX){
            fbuf.bufI(FPARTICLEIDX)[jIdx*BONDS_PER_PARTICLE*2 + bondIdx*2]      = UINT_MAX;
            fbuf.bufI(FPARTICLEIDX)[jIdx*BONDS_PER_PARTICLE*2 + bondIdx*2 +1]   = UINT_MAX;
        }
    }
    uint *ptr_elastidx =  &fbuf.bufI(FELASTIDX)[particle_Idx*BOND_DATA];                                            // Null FELASTIDX
    for (int j=0;j<BOND_DATA;j++)  ptr_elastidx[j] = UINT_MAX;
    
    float *ptr_epigen = &fbuf.bufF(FEPIGEN)[particle_Idx*NUM_GENES];                                                // Zero FEPIGEN
    for (int gene=0;gene<NUM_GENES;gene++)  ptr_epigen[gene]=0;
    
    float *ptr_tf = &fbuf.bufF(FCONC)[particle_Idx*NUM_TF];                                                         // Zero FCONC
    for (int tf=0;tf<NUM_TF;tf++) ptr_tf[tf]=0;
}

extern "C" __device__ void find_potential_bonds (int i, float3 ipos, int cell, uint _bonds[BONDS_PER_PARTICLE][2], float _bond_dsq[BONDS_PER_PARTICLE], float max_len_sq)
{			
	if ( fbuf.bufI(FGRIDCNT)[cell] == 0 ) return;                // If the cell is empty, skip it.
	float dsq, sdist, c;
	float3 dist = make_float3(0,0,0), eterm  = make_float3(0,0,0), force = make_float3(0,0,0);
	uint j;
	int clast = fbuf.bufI(FGRIDOFF)[cell] + fbuf.bufI(FGRIDCNT)[cell];              // index of last particle in this cell
    for ( int cndx = fbuf.bufI(FGRIDOFF)[cell]; cndx < clast; cndx++ ) {            // For particles in this cell.
		j = fbuf.bufI(FGRID)[ cndx ];
		dist = ( ipos - fbuf.bufF3(FPOS)[ j ] );                                    // dist in cm (Rama's comment)
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);                      // scalar distance squared
		if ( dsq < max_len_sq && dsq > 0) {                                         // IF in-range && not the same particle
            sdist = sqrt(dsq * fparam.d2);                                          // smoothing distance = sqrt(dist^2 * sim_scale^2))
			c = ( fparam.psmoothradius - sdist ); 
            bool known = false;
            uint bond_index = UINT_MAX;
            for (int a=0; a<BONDS_PER_PARTICLE; a++){                                                   // chk if known, i.e. already bonded 
                    if (fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + a*2] == i        ) known = true;   // particle 'j' has a bond to particle 'i'
                    if (fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + a*2] == UINT_MAX ) bond_index = a; // patricle 'j' has an empty bond 'a' : picks last empty bond
                    if (_bonds[a][0] == j )known = true; // needed?                                                 // particle 'i' already has a bond to particle 'j'  
                                                                                                            // not req, _bonds starts empty && only touch 'j' once
            }
            if (known == false && bond_index<UINT_MAX){       
                    //int bond_direction = 1*(dist.x-dist.y+dist.z>0.0) + 2*(dist.x+dist.y-dist.z>0.0);       // booleans divide bond space into quadrants of x>0.
                    float approx_zero = 0.02*fparam.rd2;
                    int bond_direction = ((dist.x+dist.y+dist.z)>0) * (1*(dist.x*dist.x>approx_zero) + 2*(dist.y*dist.y>approx_zero) + 4*(dist.z*dist.z>approx_zero)) -1; // booleans select +ve quadrant x,y,z axes and their planar diagonals
                    //printf("\ni=%u, bond_direction=%i, dist=(%f,%f,%f), dsq=%f, approx_zero=%f", i, bond_direction, dist.x, dist.y, dist.z, dsq, approx_zero);
                    if(0<=bond_direction && bond_direction<BONDS_PER_PARTICLE && dsq<_bond_dsq[bond_direction]){ //if new candidate bond is shorter, for this quadrant. 
                                                                                                                //lacks a candidate bond _bonds[bond_direction][1]==0
                        _bonds[bond_direction][0] = j;                                                      // index of other particle
                        _bonds[bond_direction][1] = bond_index;                                             // FPARTICLEIDX vacancy index of other particle
                        _bond_dsq[bond_direction] = dsq;                                                    // scalar distance squared 
                    }
                
            }                                                                                               // end of collect potential bonds
        }                                                                                                   // end of: IF in-range && not the same particle
    }                                                                                                       // end of loop round particles in this cell
}


extern "C" __device__ void find_potential_bond (int i, float3 ipos, uint _thisParticleBonds[BONDS_PER_PARTICLE], float3 tpos, int cell, uint &_otherParticleIdx, uint &_otherParticleBondIdx, float &_bond_dsq, float max_len_sq)       // used when just one bond, near a target location "tpos" is sought.
{			
	if ( fbuf.bufI(FGRIDCNT)[cell] == 0 ) return;                                                           // If the cell is empty, skip it.
	float dsq;
	float3 dist = make_float3(0,0,0);
	uint j;
	int clast = fbuf.bufI(FGRIDOFF)[cell] + fbuf.bufI(FGRIDCNT)[cell];                                      // index of last particle in this cell
  //printf("\nclast=%i, ",clast);
    for ( int cndx = fbuf.bufI(FGRIDOFF)[cell]; cndx < clast; cndx++ ) {                                    // For particles in this cell.
		j = fbuf.bufI(FGRID)[ cndx ];
      //printf("\nj=%u, ",j);
		dist = ( ipos - fbuf.bufF3(FPOS)[ j ] );                                                            // dist in cm (Rama's comment)
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);                                              // scalar distance squared
		if ( dsq < max_len_sq && dsq > 0) {                                                                 // IF in-range && not the same particle
            bool known      = false;
            uint bond_index = UINT_MAX;
            for (int a=0; a<BONDS_PER_PARTICLE; a++){                                                       // chk if known, i.e. already bonded 
                if (fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + a*2] == i        ) known = true;       // particle 'j' has a bond to particle 'i'
                if (fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + a*2] == UINT_MAX ) bond_index = a;     // particle 'j' has an empty bond 'a' : picks last empty bond
                if (_thisParticleBonds[a] == j )known = true;                                               // particle 'i' already has a bond to particle 'j'  
            }
            if (known == false && bond_index<UINT_MAX){
                dist = ( tpos - fbuf.bufF3(FPOS)[ j ] );                                                    // dist to target location
                dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);                                      // scalar distance squared
                if(dsq<_bond_dsq){                                                                          // if new candidate bond is shorter, for this quadrant. 
                    _otherParticleIdx = j;                                                                  // index of other particle
                    _otherParticleBondIdx = bond_index;                                                     // FPARTICLEIDX vacancy index of other particle
                    _bond_dsq = dsq;                                                                        // scalar distance squared 
                }
            }                                                                                               // end of collect potential bonds
        }                                                                                                   // end of: IF in-range && not the same particle
    }                                                                                                       // end of loop round particles in this cell
}


extern "C" __device__ void makeBond (uint thisParticleIdx, uint otherParticleIdx, uint bondIdx, uint otherParticleBondIdx, uint bondType /* elastin, collagen, apatite */){
    //FBondParams *params_ =  &fgenome.fbondparams[bondType];
    uint*   uint_ptr = &fbuf.bufI(FELASTIDX)[thisParticleIdx*BOND_DATA + 3*DATA_PER_BOND];
    float* float_ptr = &fbuf.bufF(FELASTIDX)[thisParticleIdx*BOND_DATA + 3*DATA_PER_BOND];
    uint_ptr [0]    = otherParticleIdx;                                                                     //[0]current index, 
    float_ptr[1]    = fgenome.param[bondType][fgenome.elastLim];                                            //[1]elastic limit, 
    float_ptr[2]    = fgenome.param[bondType][fgenome.default_rest_length];                                 //[2]restlength, 
    float_ptr[3]    = fgenome.param[bondType][fgenome.default_modulus];                                     //[3]modulus, 
    float_ptr[4]    = fgenome.param[bondType][fgenome.default_damping];                                     //[4]damping coeff, 
    uint_ptr [5]    = fbuf.bufI(FPARTICLE_ID)[otherParticleIdx];                                            //[5]particle ID,   
    uint_ptr [6]    = otherParticleBondIdx;                                                                 //[6]bond index 
    float_ptr[7]    = 0;                                                                                    //[7]stress integrator 
    uint_ptr [8]    = 0;                                                                                    //[8]change-type 
                                                                                                            // Connect new particle incoming bonds
    fbuf.bufI(FPARTICLEIDX)[otherParticleIdx*2*BONDS_PER_PARTICLE + 1*2]       = thisParticleIdx;           // particle Idx
    fbuf.bufI(FPARTICLEIDX)[otherParticleIdx*2*BONDS_PER_PARTICLE + 1*2 +1]    = bondIdx;                   // bond Idx 
}


extern "C" __device__ int atomicMakeBond(uint thisParticleIndx,  uint otherParticleIdx, uint bondIdx, uint otherParticleBondIndex, uint bond_type){
    int _otherParticleBondIndex = BONDS_PER_PARTICLE*2*otherParticleIdx + 2*bondIdx;
    do {} while( atomicCAS(&ftemp.bufI(FPARTICLEIDX)[_otherParticleBondIndex], UINT_MAX, 0) );                                               // lock ////////// ###### //  if (not locked) write zero to 'ftemp' to lock.
    if (fbuf.bufI(FPARTICLEIDX)[_otherParticleBondIndex]==UINT_MAX)  fbuf.bufI(FPARTICLEIDX)[_otherParticleBondIndex] = thisParticleIndx;    //  if (bond is unoccupied) write to 'fbuf' to assign this bond
    ftemp.bufI(FPARTICLEIDX)[_otherParticleBondIndex] = UINT_MAX;                                                                            // release lock // ######
    if (fbuf.bufI(FPARTICLEIDX)[_otherParticleBondIndex] == thisParticleIndx){                                                               // if (this bond is assigned) write bond data
        makeBond ( thisParticleIndx, otherParticleIdx /*candidate_target_pIDx*/, bondIdx, otherParticleBondIndex, bond_type);
        return 0;
    }else return 1;
}


extern "C" __device__ void find_closest_particle_per_axis(uint particle, float3 pos, uint neighbours[6]){
    // identify which bin to search  NB particle is new => not yet inserted into a cell
	register float3 gridMin   =	fparam.gridMin;                 // "register" is a compiler 'hint', to keep this variable in thread register
	register float3 gridDelta = fparam.gridDelta;               //  even if other variable have to be moved to slower 'local' memory  
	register int3   gridRes   =	fparam.gridRes;                 //  in the streaming multiprocessor's cache.
    int		gs;
	float3	gcf;
	int3	gc;
	gcf = (pos - gridMin) * gridDelta;                          // finds bin as a float3
	gc  = make_int3( int(gcf.x), int(gcf.y), int(gcf.z) );      // crops to an int3
	gs  = (gc.y * gridRes.z + gc.z)*gridRes.x + gc.x;           // linearizes to an int for a 1D array of bins
	
	float neighbours_dsq[6] = {FLT_MAX};
	
	for (int c=0; c < fparam.gridAdjCnt; c++) {
        uint cell = gs + fparam.gridAdj[c];
        if ( fbuf.bufI(FGRIDCNT)[cell] == 0 ) continue;                                                         // If the cell is empty, skip it.
        float dsq;
        float3 dist = make_float3(0,0,0);
        uint j;
        int clast = fbuf.bufI(FGRIDOFF)[cell] + fbuf.bufI(FGRIDCNT)[cell];                                      // index of last particle in this cell
        
        for ( int cndx = fbuf.bufI(FGRIDOFF)[cell]; cndx < clast; cndx++ ) {                                    // For particles in this cell.
            j = fbuf.bufI(FGRID)[ cndx ];
            if (j==particle)continue;
            dist = ( pos - fbuf.bufF3(FPOS)[ j ] );                                                             // dist in cm (Rama's comment)                                   
            float distxsq=dist.x*dist.x, distysq=dist.y*dist.y, distzsq=dist.z*dist.z;
            dsq = distxsq + distysq + distzsq;                                                                  // scalar distance squared
            int axis =  1*(distxsq>distysq && distxsq>distzsq) + 2*(distysq>=distxsq && distysq>distzsq) +3*(distzsq>=distxsq && distzsq>=distysq);
            if ((axis==1 && dist.x) || (axis==2 && dist.y) || (axis==3 && dist.z)) axis +=2; else axis--;       // sort by longest axis +/-ve 
            
            if ( dsq < neighbours_dsq[axis]) {                                                                  // IF in-range && not the same particle
                neighbours_dsq[axis] = dsq;
                neighbours[axis] = j;
            }                                                                                                   // end of: IF in-range && not the same particle
        }                                                                                                       // end of loop round particles in this cell
    }
}


extern "C" __device__ void find_bonds_to_redistribute(uint new_particle_Idx, float3 newParticlePos, uint neighbours[6], uint neighboursBondIdx[6], uint neighbours2[6]){
    float neighbours_dsq[6] = {FLT_MAX};
    for (int neighbour=0; neighbour<6;neighbour++){
        for (int bond =0; bond<BONDS_PER_PARTICLE; bond++){
            uint otherParticle = fbuf.bufI(FELASTIDX)[neighbours[neighbour]*BOND_DATA + bond*DATA_PER_BOND];
            int chk =0;
            for (; chk<6; chk++) if (otherParticle==neighbours[chk] || otherParticle==neighbours2[chk]) chk =7;    // not one of neighbours[6] or neighbours2[6]
            if (chk==7) continue;
            float3 dist = fbuf.bufF3(FPOS)[otherParticle] - newParticlePos ;
            float dsq = dist.x*dist.x+dist.y*dist.y+dist.z*dist.z;
            if (dsq < neighbours_dsq[neighbour]){
                neighbours_dsq[neighbour] = dsq;
                neighbours2[neighbour] = otherParticle;
                neighboursBondIdx[neighbour] = bond;
            }
        }
    }
}

extern "C" __device__ void makeBondIndxMap( uint parentParticleIndx, int bondInxMap[6]){// A tractable way to approximately map the rotation of the bonds wrt the world frame.
    uint bond0otherPartlicleIdx = fbuf.bufI(FELASTIDX)[parentParticleIndx*BOND_DATA];
    uint bond1otherPartlicleIdx = fbuf.bufI(FELASTIDX)[parentParticleIndx*BOND_DATA+DATA_PER_BOND];
    uint bond2otherPartlicleIdx = fbuf.bufI(FELASTIDX)[parentParticleIndx*BOND_DATA+2*DATA_PER_BOND];
    float3 pos      = fbuf.bufF3(FPOS)[parentParticleIndx]; 
    float3 bond0    = fbuf.bufF3(FPOS)[bond0otherPartlicleIdx] - pos;
    float3 bond1    = fbuf.bufF3(FPOS)[bond1otherPartlicleIdx] - pos;
    float3 bond2    = fbuf.bufF3(FPOS)[bond2otherPartlicleIdx] - pos;
    // int axis =  1*(distxsq<distysq && distxsq<distzsq) + 2*(distysq<=distxsq && distysq<distzsq) +3*(distzsq<=distxsq && distzsq<=distysq);
    // if ((axis==1 && dist.x) || (axis==2 && dist.y) || (axis==3 && dist.z)) axis +=2; else axis--;       // sort by longest axis +/-ve 
    float distxsq=bond0.x*bond0.x, distysq=bond0.y*bond0.y, distzsq=bond0.z*bond0.z;
    float dsq = distxsq + distysq + distzsq;         
    int axis0 =  1*(distxsq>distysq && distxsq>distzsq) + 2*(distysq>=distxsq && distysq>distzsq) +3*(distzsq>=distxsq && distzsq>=distysq);
    
    distxsq=bond1.x*bond1.x*(axis0!=1), distysq=bond1.y*bond1.y*(axis0!=2), distzsq=bond1.z*bond1.z*(axis0!=3);
    int axis1 =  1*(distxsq>distysq && distxsq>distzsq) + 2*(distysq>=distxsq && distysq>distzsq) +3*(distzsq>=distxsq && distzsq>=distysq);
    
    int axis2 = 1*((axis0!=1)&&(axis1!=1)) + 2*((axis0!=2)&&(axis1!=2)) + 3*((axis0!=3)&&(axis1!=3));
    
    if ((axis0==1 && bond0.x) || (axis0==2 && bond0.y) || (axis0==3 && bond0.z)){
        bondInxMap[0] = axis0 +2;
        bondInxMap[3] = axis0 -1;
    }else{
        bondInxMap[0] = axis0 -1;
        bondInxMap[3] = axis0 +2;
    }
    
    if ((axis1==1 && bond1.x) || (axis1==2 && bond1.y) || (axis1==3 && bond1.z)){
        bondInxMap[1] = axis1 +2;
        bondInxMap[4] = axis1 -1;
    }else{
        bondInxMap[1] = axis1 -1;
        bondInxMap[4] = axis1 +2;
    }
    
    if ((axis2==1 && bond2.x) || (axis2==2 && bond2.y) || (axis2==3 && bond2.z)){
        bondInxMap[2] = axis0 +2;
        bondInxMap[5] = axis0 -1;
    }else{
        bondInxMap[2] = axis0 -1;
        bondInxMap[5] = axis0 +2;
    }
}


extern "C" __device__ void redistribute_bonds(uint new_particle_Idx, float3 newParticlePos, uint neighbours[6], uint neighboursBondIdx[6], uint neighbours2[6]){
    // for particle removal, given list of bonds ... 
    // for each bond 
    
    
}


extern "C" __device__ int insertNewParticle(float3 newParticlePos, uint parentParticleIndx, uint bondIdx, uint otherParticleBondIndex, uint bond_type[BONDS_PER_PARTICLE]){
                                                                                                                // Inserts particle at newParticlePos AND redistributes bonds with neighbours.
    uint new_particle_Idx;
    addParticle( parentParticleIndx, new_particle_Idx);
    fbuf.bufF3(FPOS)[new_particle_Idx] = newParticlePos;
    uint neighbours[6]= {UINT_MAX}, neighboursBondIdx[6]= {UINT_MAX}, neighbours2[6]= {UINT_MAX};
    find_closest_particle_per_axis(new_particle_Idx, newParticlePos, neighbours);
    find_bonds_to_redistribute(new_particle_Idx, newParticlePos, neighbours, neighboursBondIdx, neighbours2);
    int ret1, ret2, ret3=0;
    int bondInxMap[6]={UINT_MAX};                                                                                                           // map parent particle orientation
    makeBondIndxMap( parentParticleIndx, bondInxMap);
    // ? how to insert the bond being lengthened or strengthened ? // should occur implicitly due to orientation & placement wrt parent particle.
    
    for (int bond=0; bond<6; bond++){
        if (neighboursBondIdx[bondInxMap[bond]]<BONDS_PER_PARTICLE){                                                                        // suitable bond to redistribute was found)
            ret1 = atomicMakeBond(neighbours[bondInxMap[bond]],  new_particle_Idx, neighboursBondIdx[bondInxMap[bond]], bond, bond_type[bond]);   // new outging bond 
            if (ret1 == 0){ 
                ret2= atomicMakeBond(new_particle_Idx,  
                                     neighbours2[bondInxMap[bond]], 
                                     bond, 
                                     fbuf.bufI(FELASTIDX)[neighbours[bondInxMap[bond]]*BOND_DATA + neighboursBondIdx[bondInxMap[bond]]*DATA_PER_BOND + 6], 
                                     bond_type[bond]);
                if (ret2 !=0)                                                                                                               // if (success) makeBonds; else reset any change.
                    atomicMakeBond(neighbours[bondInxMap[bond]],  
                                   neighbours2[bondInxMap[bond]], 
                                   neighboursBondIdx[bondInxMap[bond]], 
                                   fbuf.bufI(FELASTIDX)[neighbours[bondInxMap[bond]]*BOND_DATA + neighboursBondIdx[bondInxMap[bond]]*DATA_PER_BOND + 6], 
                                   bond_type[bond]);                                                                                              
            }
            if (ret1 || ret2) ret3++;
        }else ret3++;
    }
    return ret3;                                                                                                //NB causes incoming bonds to fluid particles -> non-adherent surface.
}


extern "C" __global__ void heal ( int pnum, uint list_length, int change_list) {                                     //TODO heal ( int pnum ) 
    uint particle_index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                                            // particle index
  //printf("\nheal(), particle_index=%u,\t",particle_index);  
    if ( particle_index >= list_length ) return;                                                                    // pnum should be length of list.
    uint i = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index];                                         // call for dense list of broken bonds
  //printf("\nheal(), i=%u,\t",i);  
    if ( i >= pnum ) return;// TODO replace pnum with active particles where appropriate
    uint bondIdx = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index+list_length]; 
  //printf("\nheal(), bondIdx=%u,\t",bondIdx);   
    // FELASTIDX //# currently [0]current index, [1]elastic limit, [2]restlength, [3]modulus, [4]damping coeff, [5]particle ID, [6]bond index [7]stress integrator [8]change-type binary indicator
    uint*  pointerUint  = &fbuf.bufI(FELASTIDX)[i*BOND_DATA+bondIdx*DATA_PER_BOND];
    float* pointerFloat = &fbuf.bufF(FELASTIDX)[i*BOND_DATA+bondIdx*DATA_PER_BOND];
    // A bond is broken and needs to be remade. (NB this is micro-healing. Macro-healing requires inflamatory processes and remodelling)
    uint   otherParticleIdx         = pointerUint[0];
  //printf("\nheal(), otherParticleIdx=%u,\t",otherParticleIdx); 
    if (otherParticleIdx>BONDS_PER_PARTICLE)return;//otherParticleIdx = i;        // if there is no record, look for a particle nearby. Need a different kernel.
  //printf("\nheal(), otherParticleIdx=%u,\t",otherParticleIdx);
    uint   otherParticleBondIndex   = pointerUint[6];
  //printf("\nheal(), otherParticleBondIndex=%u,\t",otherParticleBondIndex);
    float3 Pos                      = fbuf.bufF3(FPOS)[i];
    float3 otherParticlePos         = fbuf.bufF3(FPOS)[otherParticleIdx];
    float3 oldBondLengthF3          = otherParticlePos - Pos;
    float  oldBondLength            = sqrt(oldBondLengthF3.x*oldBondLengthF3.x + oldBondLengthF3.y*oldBondLengthF3.y + oldBondLengthF3.z*oldBondLengthF3.z);
    float3 dirVec                   = otherParticlePos/(oldBondLength+0.02*fparam.rd2);     // float approx_zero = 0.02*fparam.rd2; Prevents division by zero.
  //printf("\nheal(), dirVec=%f,%f,%f,\t,%f",dirVec.x,dirVec.y,dirVec.z,(oldBondLength+0.000001));
    
    // Determine bond type from binary change-type indicator
    float * fbufFEPIGEN = &fbuf.bufF(FEPIGEN)[i*NUM_TF];
    uint bond_type[BONDS_PER_PARTICLE] = {0};                   //  0=elastin, 1=collagen, 2=apatite
    
    // Calculate material type for bond
    if (fbufFEPIGEN[9]/*bone*/) for (int bond=0; bond<BONDS_PER_PARTICLE; bond++) bond_type[bond] = 2;
    else if (fbufFEPIGEN[6]/*tendon*/||fbufFEPIGEN[7]/*muscle*/||fbufFEPIGEN[10]/*elast lig*/) {bond_type[0] = 1; bond_type[3] = 1;}
    else if (fbufFEPIGEN[6]/*cartilage*/)for (int bond=0; bond<BONDS_PER_PARTICLE; bond++) bond_type[bond] = 1;
    
    // Get default bond params
    //FBondParams *params_  =  &fgenome.fbondparams[bond_type[bondIdx]];
    float max_rest_length = fgenome.param[bond_type[bondIdx]][fgenome.max_rest_length];
    float min_rest_length = fgenome.param[bond_type[bondIdx]][fgenome.min_rest_length];
    float elastLim        = fgenome.param[bond_type[bondIdx]][fgenome.elastLim];
    float default_length  = fgenome.param[bond_type[bondIdx]][fgenome.default_rest_length];
    float default_modulus = fgenome.param[bond_type[bondIdx]][fgenome.default_modulus];
    float default_damping = fgenome.param[bond_type[bondIdx]][fgenome.default_damping];
    
 //printf("\nheal(), max_rest_length=%f,\t",max_rest_length);
    // Find new attachment for this particle                    // NB Don't find new attachment for other particle: leave it open for outging bonds to find.
    float3 target_location = Pos + dirVec * default_length;     // target_location = Pos + dirVec * default_length 
    // identify which bin to search
	register float3 gridMin   =	fparam.gridMin;                 // "register" is a compiler 'hint', to keep this variable in thread register
	register float3 gridDelta = fparam.gridDelta;               //  even if other variable have to be moved to slower 'local' memory  
	register int3   gridRes   =	fparam.gridRes;                 //  in the streaming multiprocessor's cache.
    int		gs;
	float3	gcf;
	int3	gc;
	gcf = (target_location - gridMin) * gridDelta;               // finds bin as a float3
	gc  = make_int3( int(gcf.x), int(gcf.y), int(gcf.z) );       // crops to an int3
	gs  = (gc.y * gridRes.z + gc.z)*gridRes.x + gc.x;            // linearizes to an int for a 1D array of bins
  //printf("\ngcf=(%f,%f,%f), gc=(%i,%i,%i), gs=%i,\t",gcf.x,gcf.y,gcf.z, gc.x,gc.y,gc.z, gs);                                                               // TODO chk target_location is inside sim vol & gs valid.
    uint candidate_target_pIDx      = UINT_MAX;                  // particleIDx of candidate_target
    uint candidate_target_bondIdx   = 0;
    float current_dsq = max_rest_length;
    
    float3 dist, dist2;
	float dsq, dsq2;
    int nadj = (1*fparam.gridRes.z + 1)*fparam.gridRes.x + 1;
    gs -= nadj;                                                  // gs is much too large ! must be < fparam.gridTotal 
    
    uint thisParticleBonds[BONDS_PER_PARTICLE];
    uint * thisParticleBond_ptr;
    thisParticleBond_ptr = &fbuf.bufI(FELASTIDX)[i*BONDS_PER_PARTICLE];
    
    for (int bond =0; bond<BONDS_PER_PARTICLE; bond++)      thisParticleBonds[bond] = thisParticleBond_ptr[bond * DATA_PER_BOND]; // list of existing bonds, idx of other particle
    find_potential_bond (i, Pos, thisParticleBonds, target_location, gs, candidate_target_pIDx, candidate_target_bondIdx, dsq, max_rest_length*max_rest_length); // seaches one cell
    
  //printf("\nheal(), atomicMakeBond: i=%u, gs=%i, m_GridTotal=%i ,\t",i, gs, fparam.gridTotal );  // candidate_target_pIDx comes back invalid as a uint!
  /*candidate_target_pIDx=%u, bondIdx=%u, candidate_target_bondIdx=%u, bond_type[bondIdx]=%u*/  //,candidate_target_pIDx, bondIdx, candidate_target_bondIdx, bond_type[bondIdx]
    atomicMakeBond(i, candidate_target_pIDx, bondIdx, candidate_target_bondIdx, bond_type[bondIdx]);
    
}


extern "C" __global__ void lengthen_muscle ( int pnum, int list_length, int change_list) { //lengthen_muscle ( int pnum ) //NB elastic tissues (yellow ligments) are non-innervated muscle 
    // TODO consider divergently and convergently branching cases of lengthen_muscle ( int pnum )
    uint particle_index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                             // particle index
    if ( particle_index >= list_length ) return; // pnum should be length of list.
    uint i = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index]; // call for dense list of lengthen_muscle
    if ( i >= pnum ) return; 
    uint bondIdx = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index+list_length]; // bondIdx, NB FDENSE_LISTS_CHANGES [2][list_length] 
    
    uint bondIdx_reciprocal = fbuf.bufI(FELASTIDX)[i*BOND_DATA+bondIdx*DATA_PER_BOND+6]; //[6]bond index // i.e. the incoming bondindx on next_particle_Idx
    // NB (bondIdx_reciprocal > bondIdx)  => convergent branching, (bondIdx_reciprocal < bondIdx) => divergent branching , (0==bondIdx_reciprocal == bondIdx)=> nonbranching, else => error
    
    // Need to insert 3 new particles for a new turn of the helix. 
    // NB muscle lengthening may be induced by (i) stretching the collagen chain, (ii) ? 
    // Q what happens when side bonds are stretched?
    // Q what happens when the chain branches (i) ahead, (ii) behind, (iii) on the adjacent particles nthe helix ?
    
    // simple case 
    // find two particles ahead (bondIdx[0])=> array of 3 particle indicies. 
    uint first_row_particle_Idx[3];
    first_row_particle_Idx[0]=i;
    first_row_particle_Idx[1]=fbuf.bufI(FELASTIDX)[first_row_particle_Idx[0]*BOND_DATA];// i.e bond[0]
    first_row_particle_Idx[2]=fbuf.bufI(FELASTIDX)[first_row_particle_Idx[1]*BOND_DATA];
    
    uint next_row_particle_Idx[3];
    float3 new_particle_pos[3];
    uint new_particle_Idx[3];
    
    // Determine bond type from binary change-type indicator
    float * fbufFEPIGEN = &fbuf.bufF(FEPIGEN)[i*NUM_TF];
    uint bond_type[BONDS_PER_PARTICLE] = {0};                   //  0=elastin, 1=collagen, 2=apatite
    
    // Calculate material type for bond
    if (fbufFEPIGEN[9]/*bone*/) for (int bond=0; bond<BONDS_PER_PARTICLE; bond++) bond_type[bond] = 2;
    else if (fbufFEPIGEN[6]/*tendon*/||fbufFEPIGEN[7]/*muscle*/||fbufFEPIGEN[10]/*elast lig*/) {bond_type[0] = 1; bond_type[3] = 1;}
    else if (fbufFEPIGEN[6]/*cartilage*/)for (int bond=0; bond<BONDS_PER_PARTICLE; bond++) bond_type[bond] = 1;
    
    // Get default bond params elastin
    //FBondParams *params_  =  &fgenome.fbondparams[fgenome.elastin];
    float e_elastLim        = fgenome.param[bond_type[bondIdx]][fgenome.elastLim];// TODO is bondIdx correct here for the type of bond required ?
    float e_default_length  = fgenome.param[bond_type[bondIdx]][fgenome.default_rest_length];
    
    for (int column=0;column<3;column++){                                                                               // for each particle : //find next row of helix (bondIdx[1]) => uint next_row_particle_Idx[3]
        next_row_particle_Idx[column] = fbuf.bufI(FELASTIDX)[first_row_particle_Idx[column]*BOND_DATA + DATA_PER_BOND]; // i.e bond[1]  //create new particle at mid point of bond => uint new_particle_Idx[3] 
        addParticle(first_row_particle_Idx[column], new_particle_Idx[column]);                                          // addParticle(uint parent_Idx, float3 new_particle_pos)
        fbuf.bufF3(FPOS)[new_particle_Idx[column]] = fbuf.bufF3(FPOS)[first_row_particle_Idx[column]] + (fbuf.bufF3(FPOS)[first_row_particle_Idx[column]] - fbuf.bufF3(FPOS)[next_row_particle_Idx[column]])/2;
        makeBond ( new_particle_Idx[column],        next_row_particle_Idx[column],  1, 1, fgenome.elastin );            //connect new particle incoming & outgoing contractile bonds  bondIdx[1]
        makeBond ( first_row_particle_Idx[column],  new_particle_Idx[column],       1, 1, fgenome.elastin );
    }
    // Connect new particle helical bonds  bondIdx[0] 
    uint helical_Idx[5] = {first_row_particle_Idx[2], new_particle_Idx[0], new_particle_Idx[1], new_particle_Idx[2], next_row_particle_Idx[0]};
    for (int column=0;column<4;column++) makeBond ( helical_Idx[column], helical_Idx[column+1], 0, 0, fgenome.collagen);// Connect new particle outgoing helical bonds  bondIdx[0]
    
    // connect lateral elastin bonds : using dirVec away fom the helix orthogonal to bond[0&1]
    float3 dir1 = fbuf.bufF3(FPOS)[new_particle_Idx[0]]  - fbuf.bufF3(FPOS)[i];                                         // find linear axis of helix from 1st particle's elastin fibre i.e. POS dif to 1st new particle
    float3 dir2, dir3, target_pos, pos;
    float  dsq;
    uint   thisParticleIdx, otherParticleIdx, otherParticleBondIdx;
    uint   thisParticleBonds[BONDS_PER_PARTICLE];
    uint * thisParticleBond_ptr;
    
    for (int column=0;column<3;column++){                                                                               // for each new particle : search for available incoming elastin bond nearest to the target point 
        dsq                     = e_elastLim;
        thisParticleIdx         = new_particle_Idx[column];
        thisParticleBond_ptr    = &fbuf.bufI(FELASTIDX)[thisParticleIdx*BONDS_PER_PARTICLE];
        for (int bond =0; bond<BONDS_PER_PARTICLE; bond++)      thisParticleBonds[bond] = thisParticleBond_ptr[bond * DATA_PER_BOND]; // list of existing bonds, idx of other particle
        otherParticleIdx        = UINT_MAX;
        otherParticleBondIdx    = UINT_MAX;
        pos                     = fbuf.bufF3(FPOS)[thisParticleIdx];
        dir2                    = fbuf.bufF3(FPOS)[helical_Idx[column+2]]  - fbuf.bufF3(FPOS)[helical_Idx[column+1]];   // find tangent to helix from collagen fibre, i.e. POS dif to next particle in helical_Idx
        dir3                    = cross(dir1,dir2);
        target_pos              = pos + dir3*e_default_length/sqrt(dir3.x*dir3.x+dir3.y*dir3.y+dir3.z*dir3.z);          // define target point
        uint gc                 = fbuf.bufI(FGCELL)[ thisParticleIdx ];                                                 // Get search cell	NB new particle not yet inserted in correct cell.
        if ( gc == GRID_UNDEF ) return;                                                                                 // particle out-of-range
        gc -= (1*fparam.gridRes.z + 1)*fparam.gridRes.x + 1;
        for (int c=0; c < fparam.gridAdjCnt; c++)   find_potential_bond ( thisParticleIdx, pos, thisParticleBonds, target_pos, gc + fparam.gridAdj[c], otherParticleIdx, otherParticleBondIdx, dsq, e_elastLim*e_elastLim);
        if (otherParticleIdx < pnum && otherParticleBondIdx < BONDS_PER_PARTICLE)   atomicMakeBond ( thisParticleIdx, otherParticleIdx, bondIdx, otherParticleBondIdx, fgenome.elastin);
    }
}


extern "C" __global__ void lengthen_tissue ( int pnum, int list_length, int change_list) { //TODO lengthen_tissue ( int pnum )  // add particle in bond
    uint particle_index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                             // particle index
    if ( particle_index >= list_length ) return; // pnum should be length of list.
    uint i = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index]; // call for dense list of lengthen_tissue
    if ( i >= pnum ) return; 
    uint bondIdx = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index+list_length]; 
    
    // Insert 1 particle on axis of strecthed bond & share existing/new lateral bonds
    // It would help to know which bond. => where to add new particle
    
    uint next_particle_Idx = fbuf.bufI(FELASTIDX)[i*BOND_DATA+bondIdx*DATA_PER_BOND];
    uint bondIdx_reciprocal = fbuf.bufI(FELASTIDX)[i*BOND_DATA+bondIdx*DATA_PER_BOND+6]; //[6]bond index // i.e. the incoming bondindx on next_particle_Idx
    
    //create new particle at mid point of bond => uint new_particle_Idx[3] 
    uint new_particle_Idx;
    addParticle(i, new_particle_Idx);
    //fbuf.bufF3(FPOS)[new_particle_Idx]          = fbuf.bufF3(FPOS)[i] + (fbuf.bufF3(FPOS)[i] - fbuf.bufF3(FPOS)[next_particle_Idx])/2;
    float3 newParticlePos  = fbuf.bufF3(FPOS)[i] + (fbuf.bufF3(FPOS)[i] - fbuf.bufF3(FPOS)[next_particle_Idx])/2;
    
    // Determine bond type from binary change-type indicator
    float * fbufFEPIGEN = &fbuf.bufF(FEPIGEN)[i*NUM_TF];
    uint bond_type[BONDS_PER_PARTICLE] = {0};                          //  0=elastin, 1=collagen, 2=apatite
    
    // Calculate material type for bond
    if (fbufFEPIGEN[9]/*bone*/) for (int bond=0; bond<BONDS_PER_PARTICLE; bond++) bond_type[bond] = 2;
    else if (fbufFEPIGEN[6]/*tendon*/||fbufFEPIGEN[7]/*muscle*/||fbufFEPIGEN[10]/*elast lig*/) {bond_type[0] = 1; bond_type[3] = 1;}
    else if (fbufFEPIGEN[6]/*cartilage*/)for (int bond=0; bond<BONDS_PER_PARTICLE; bond++) bond_type[bond] = 1;
    
    insertNewParticle(newParticlePos, i, bondIdx, bondIdx_reciprocal,  bond_type);
    /*
    
    //makeBond (uint thisParticleIdx, uint otherParticleIdx, uint bondIdx, uint otherParticleBondIdx, uint bondType /_* elastin, collagen, apatite *_/)
    makeBond ( i, new_particle_Idx, bondIdx, bondIdx, bond_type[bondIdx] );
    makeBond ( new_particle_Idx, next_particle_Idx, bondIdx, bondIdx_reciprocal, bond_type[bondIdx] );  // NB these _might_not_ req atomic because New Particle is not in the main list yet
    
    //// NB bonds in neighbourhood will mostly be already taken.
    //// Need to dedistribue bonds via the new particle.
    // This may in fact replace the step above as well.
    
    // fn "insert particle" will redistribute bonds as well.
    
    */
    /*
    // Get default bond params
    FBondParams *params_  = &fgenome.fbondparams[bond_type[bondIdx]];
    float max_rest_length = params_[0].param[params_->max_rest_length];
    float min_rest_length = params_[0].param[params_->min_rest_length];
    float elastLim        = params_[0].param[params_->elastLim];
    float default_length  = params_[0].param[params_->default_rest_length];
    float default_modulus = params_[0].param[params_->default_modulus];
    float default_damping = params_[0].param[params_->default_damping];
    
    /////////////////
    uint*   uint_ptr_parent = &fbuf.bufI(FELASTIDX)[i*BOND_DATA + bondIdx*DATA_PER_BOND + 0];
    float* float_ptr_parent = &fbuf.bufF(FELASTIDX)[i*BOND_DATA + bondIdx*DATA_PER_BOND + 0];
    
    uint*   uint_ptr_new = &fbuf.bufI(FELASTIDX)[new_particle_Idx*BOND_DATA + bondIdx*DATA_PER_BOND + 0];
    float* float_ptr_new = &fbuf.bufF(FELASTIDX)[new_particle_Idx*BOND_DATA + bondIdx*DATA_PER_BOND + 0];
        
    // 1st connect replacement bonds 
    
    // (i) bond with parent particle
    uint_ptr_parent [0] = new_particle_Idx;                            //[0]current index, 
    float_ptr_parent[1] = elastLim;                                    //[1]elastic limit, 
    float_ptr_parent[2] = default_length;                              //[2]restlength, 
    float_ptr_parent[3] = default_modulus;                             //[3]modulus, 
    float_ptr_parent[4] = default_damping;                             //[4]damping coeff, 
    uint_ptr_parent [5] = fbuf.bufI(FPARTICLE_ID)[new_particle_Idx];   //[5]particle ID,   
    uint_ptr_parent [6] = bondIdx;                                     //[6]bond index
    float_ptr_parent[7] = 0;                                           //[7]stress integrator 
    uint_ptr_parent [8] = 0;                                           //[8]change-type 
    
    // (ii) bond with next particle
    uint_ptr_new [0]    = next_particle_Idx;                           //[0]current index, 
    float_ptr_new[1]    = elastLim;                                    //[1]elastic limit, 
    float_ptr_new[2]    = default_length;                              //[2]restlength, 
    float_ptr_new[3]    = default_modulus;                             //[3]modulus, 
    float_ptr_new[4]    = default_damping;                             //[4]damping coeff, 
    uint_ptr_new [5]    = fbuf.bufI(FPARTICLE_ID)[next_particle_Idx];  //[5]particle ID,   
    uint_ptr_new [6]    = bondIdx_reciprocal;                          //[6]bond index
    float_ptr_new[7]    = 0;                                           //[7]stress integrator 
    uint_ptr_new [8]    = 0;                                           //[8]change-type 
    
    // (iii) reciprocal records
    fbuf.bufI(FPARTICLEIDX)[new_particle_Idx*2*BONDS_PER_PARTICLE + 1*2]       = i;                        // particle Idx
    fbuf.bufI(FPARTICLEIDX)[new_particle_Idx*2*BONDS_PER_PARTICLE + 1*2 +1]    = bondIdx;                  // bond Idx
    
    fbuf.bufI(FPARTICLEIDX)[next_particle_Idx*2*BONDS_PER_PARTICLE + 1*2]      = new_particle_Idx;         // particle Idx
    fbuf.bufI(FPARTICLEIDX)[next_particle_Idx*2*BONDS_PER_PARTICLE + 1*2 +1]   = bondIdx;                  // bond Idx
    
    */
    /*
    // 2nd redistribute bonds from parent particles ?  
    
    // 3rd fill in other bonds NB (i) bond angles, (ii) tissue types : bone & cartilage => (nearly) full bonding & 
    // NB structural celltypes by active FEPIGEN:
    // helical      : // 7 muscle  // 10 elastic_lig
    // linear       : // 6 tendon 
    // bilinear     : // 5 fibrocyte 
    // homogeneous  : // default (mesenchyme/loose ct) //3 fat : elastin
                      // 8 cartilage                           : collagen
                      // 9 bone                                : apatite
    
    // exclude bond index ...
    
    // freeze bonds // Octrant lists 
         
    uint    bonds[BONDS_PER_PARTICLE][2];
    float   bond_dsq[BONDS_PER_PARTICLE];
    uint gc = fbuf.bufI(FGCELL)[ i ];                                               // Get search cell	NB new particle not yet inserted in correct cell.
	if ( gc == GRID_UNDEF ) return;                                                 // particle out-of-range
	gc -= (1*fparam.gridRes.z + 1)*fparam.gridRes.x + 1;
    // NB must select correct "float max_rest_length = params_[0].param[params_->max_rest_length];"
    
    for (int c=0; c < fparam.gridAdjCnt; c++) find_potential_bonds ( i, fbuf.bufF3(FPOS)[ i ], gc + fparam.gridAdj[c], bonds, bond_dsq, max_rest_length*max_rest_length); 
                                            //find_potential_bonds (int i, float3 ipos, int cell, uint _bonds[BONDS_PER_PARTICLE][2], float _bond_dsq[BONDS_PER_PARTICLE], float max_len_sq);
        // Add new bonds /////////////////////////////////////////////////////////////////////////////
    for (int a =0; a< BONDS_PER_PARTICLE; a++){
        int otherParticleBondIndex = BONDS_PER_PARTICLE*2*bonds[a][0] + 2*a /_*bonds[a][1]*_/; // fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex]
        
        if((uint)bonds[a][0]==i) printf("\n (uint)bonds[a][0]==i, i=%u a=%u",i,a);  // float bonds[BONDS_PER_PARTICLE][3];  [0] = index of other particle, [1] = dsq, [2] = bond_index
                                                                                    // If outgoing bond empty && proposed bond for this quadrant is valid
        if( fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +1] == 0.0  &&  bonds[a][0] < pnum  && bonds[a][0]!=i  && bond_dsq[a]<3 ){  // ie dsq < 3D diagonal of cube ##### hack #####
                                                                                    // NB "bonds[b][0] = UINT_MAX" is used to indicate no candidate bond found
                                                                                    //    (FELASTIDX) [1]elastic limit = 0.0 isused to indicate out going bond is empty
            //printf("\nBond making loop i=%u, a=%i, bonds[a][1]=%u, bond_dsq[a]=%f",i,a,bonds[a][1],bond_dsq[a]);
            
            
            do {} while( atomicCAS(&ftemp.bufI(FPARTICLEIDX)[otherParticleBondIndex], UINT_MAX, 0) );               // lock ////////// ###### //  if (not locked) write zero to 'ftemp' to lock.
            if (fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex]==UINT_MAX)  fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex] = i;    //  if (bond is unoccupied) write to 'fbuf' to assign this bond
            ftemp.bufI(FPARTICLEIDX)[otherParticleBondIndex] = UINT_MAX;                                            // release lock // ######

            if (fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex] == i){                                              // if (this bond is assigned) write bond data
                fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex +1] = a;                                             // write i's outgoing bond_index to j's incoming bonds
                uint i_ID = fbuf.bufI(FPARTICLE_ID)[i];                                                             // retrieve permenant particle IDs for 'i' and 'j'
                uint j_ID = fbuf.bufI(FPARTICLE_ID)[bonds[a][0]];                                                   // uint bonds[BONDS_PER_PARTICLE][2];[0]=index of other particle,[1]=bond_index
                float bond_length = sqrt(bond_dsq[a]);
                float modulus = 100000;       // 100 000 000                                                        // 1000000 = min for soft matter integrity // 
                fbuf.bufI(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND]    = bonds[a][0];                               // [0]current index,
                fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +1] = 2 * bond_length ;                          // [1]elastic limit  = 2x restlength i.e. %100 strain
                fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +2] = 0.5*bond_length;                               // [2]restlength = initial length  
                fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +3] = modulus;                                   // [3]modulus
                fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +4] = 2*sqrt(fparam.pmass*modulus);              // [4]damping_coeff = optimal for mass-spring pair.
                fbuf.bufI(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +5] = j_ID;                                      // [5]save particle ID of the other particle NB for debugging
                fbuf.bufI(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +6] = bonds[a][1];                               // [6]bond index at the other particle 'j's incoming bonds
                //printf("\nNew Bond a=%u, i=%u, j=%u, bonds[a][1]=%u, fromPID=%u, toPID=%u,, fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex]=%u, otherParticleBondIndex=%u",
                //       a,i,bonds[a][0],bonds[a][1],i_ID,j_ID, fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex], otherParticleBondIndex);
            }            
        }// end if 
        __syncthreads();    // NB applies to all threads _if_ the for loop runs, i.e. if(freeze==true)
    }           // TODO make this work with incoming & outgoing bonds, NB preserve existing bonds                    // end loop around FELASTIDX bonds

    */
}


extern "C" __global__ void shorten_muscle ( int pnum, int list_length, int change_list) { //TODO shorten_muscle ( int pnum )  // remove particle in chain & update contractile bonds
    uint particle_index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                             // particle index
    if ( particle_index >= list_length ) return; // pnum should be length of list.
    uint i = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index]; // call for dense list of shorten_muscle
    if ( i >= pnum ) return; 
    uint bondIdx = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index+list_length]; 
    // Need to remove 3 particles, and close the gap.
    
    
    
}
extern "C" __global__ void shorten_tissue ( int pnum, int list_length, int change_list) { //TODO shorten_tissue ( int pnum )  // remove particle and connect bonds along their axis
    uint particle_index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                             // particle index
    if ( particle_index >= list_length ) return; // pnum should be length of list.
    uint i = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index]; // call for dense list of shorten_tissue
    if ( i >= pnum ) return; 
    uint bondIdx = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index+list_length]; 
    // Need to remove 1 particle and close the gap
    // It would help to know which bond. => how to close the gap
    
    
}
extern "C" __global__ void strengthen_muscle ( int pnum, int list_length, int change_list) { //TODO strengthen_muscle ( int pnum )  // NB Y branching etc
    uint particle_index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                             // particle index
    if ( particle_index >= list_length ) return; // pnum should be length of list.
    uint i = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index]; // call for dense list of strengthen_muscle
    if ( i >= pnum ) return; 
    uint bondIdx = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index+list_length]; 
    // Need to doulble up the helix i.e. add particles and contractile bonds in parallel.
    // Q Induced by ?
    // Q double up How ? 
    // NB difference between a helix and a zig-zag is only that the contractile bonds reach 2 particles ahead.
    
    
    
}
extern "C" __global__ void strengthen_tissue ( int pnum, int list_length, int change_list) { //TODO strengthen_tissue ( int pnum )  // add particle and bonds in parallel AND reduce original bon's modulus
    uint particle_index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                             // particle index
    if ( particle_index >= list_length ) return; // pnum should be length of list.
    uint i = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index]; // call for dense list of strengthen_tissue
    if ( i >= pnum ) return; 
    uint bondIdx = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index+list_length]; 
    // Need to double up articles and bonds in parallel wrt the affected bond
    // It would help to know which bond. => where to place the new particle i.e. orthogonal to the bond NB place where there is space in the plane.
    
    
}
extern "C" __global__ void weaken_muscle ( int pnum, int list_length, int change_list) { //TODO weaken_muscle ( int pnum )  // NB Y branching etc
    uint particle_index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                             // particle index
    if ( particle_index >= list_length ) return; // pnum should be length of list.
    uint i = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index]; // call for dense list of weaken_muscle
    if ( i >= pnum ) return; 
    uint bondIdx = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index+list_length]; 
    // Need to remove a row of particles in parallel - i.e. form/propagate a branch 
    // 
    
    
}
extern "C" __global__ void weaken_tissue ( int pnum, int list_length, int change_list) { //TODO weaken_tissue ( int pnum )  // remove particle & transfer bonds laterally  
    uint particle_index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                             // particle index
    if ( particle_index >= list_length ) return; // pnum should be length of list.
    uint i = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index]; // call for dense list of weaken_tissue
    if ( i >= pnum ) return; 
    uint bondIdx = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index+list_length]; 
    // Need to remove a particle and close the gap by transfering load laterally
    // It would help to know which bond. => how to close the gap
    
    
}

// NB aim to set particles to their 'correct' bond pattern for their tissue type(s)
// What happens if different bonds cause a particle to be repeatedly created and deleted ? When/how could this happen ? 



    
    
/*   
//     #pragma unroll
//     for(int j=0;j<BONDS_PER_PARTICLE;j++)
//         modulus = (j==0 &&(FEPIGEN[6]/_*tendon*_/||FEPIGEN[7]/_*muscle*_/||FEPIGEN[10]/_*elast_lig*_/) 
//                     ||(j<3&&FEPIGEN[5]/_*fibrocyte*_/)
//                     ||FEPIGEN[8]/_*cartilage*_/)
//                 *collagen_mod   
//                 + FEPIGEN[9]*bone_mod
//                 + elastin_mod; 
//         
//         FELASTIDX[j*DATA_PER_BOND +3]/_*modulus*_/ = FELASTIDX[j*DATA_PER_BOND +3] +  modulus * (FELASTIDX[j*DATA_PER_BOND+7]/_*stress_integrator*_/ - stregthThreshold);  
//         
//         if(FELASTIDX[3]>modulus) FCONC[0] = duplicate  // i.e. need to grow more tissue to dissipate the stress
//     }
//     
//     for(int j=0;j<BONDS_PER_PARTICLE;j++){
//         Adjust rest_length according to stress_integrator
//         
//         if too short, mark particle to remove
//         if too long, du[licate particle along axis of stretch
//     }
//     
//     
//     if (FEPIGEN[6]/_*tendon*_/||FEPIGEN[7]/_*muscle*_/||FEPIGEN[10]/_*elast_lig*_/){
//         NB FELASTIDX[0*DATA_PER_BOND +3] i.e. bond[0] is the principal collaen fibre
//         In muscle & elast lig, bond[1] reaches ahead 2 steps in the chain 
//         In muscle bond[1] will contract iff motor nerve is firing
//         Elast lig has no motor nerve & does not contract
//         
//         moduli & lengths are fixed. 
//         Strength => delete/duplicate particle in parallel.
//         Length => delete/duplicate along bond[0] axis
//         
//         if((FEPIGEN[6]/_*tendon*_/){                              // danger that every particle in chain will elongate. ? elongate at myotendinous jxn?
//             if ( integrator[0] > elongate_threshold ){
//                 new_particle = atomicAdd( &particles_inuse  ); //e.g. atomicAdd ( &fbuf.bufI(FGRIDCNT)[ gs ], 1 ); 
//             }
//             if ( integrator[0] >   
//         }
//     }
//     if (FEPIGEN[8]/_*cartilage*_/||FEPIGEN[9]/_*bone*_/){
//     }
*/   
    
/*    
    // Equation to modify mass/radius:
    // (i)   mass & radius  * TF[0] 
    
    // (ii)  if (radius > 1.36) split : make two new particles, position orthogonal to two strongest bonds 
    
    // (iii) if (radius < 1) combine : find another particle to combine with. closest particle with same cell type(s)
    
    
    
    // Equation to break/make springs, due to:
    // (i) particle split/combine
    // (ii) spring breakage
    // (iii) particle migration  // modify rest length -> stress, break if too short
    // For each unused bond:
    //      Find new attachment in desired direction, at desired distance, and of desired cell type
      
    
    // genes: (expect 64 to get from zygote-> hand)
    
    // -- special genes, for simulation efficiency
    // 0 active particles
    // 1 solid  (has springs)
    // 2 living/telomere (has genes)
    
    // -- behavioural celltypes
    // 3 fat
    // 4 nerves 
    // 5 fibrocyte
    // 6 tendon
    // 7 muscle
    // 8 cartilage
    // 9 bone
    // 10 elastic_lig
    
    
    // -- regulatory genes
    // 11 pluripotency
    // 12 Trophoblast
    // 13 Amniotic Trophoblast
    // 14 Epiblast/ectoderm
    // 15 Hypoblast/endoderm
    // 16 Primitive node
    // 17 Primitive streak
    // 18 Mesoderm
    // 19 Notocord
    // 20 Pre-somitic mesoderm → ‘clock’ 
    // 21 Dorsal ectoderm
    // 22 Somite
    // 23 Hox 1
    // 24 Hox 4
    // 25 Hox 5
    // 26 Hox 6
    // 27 ZPA – zone of polarizing activity
    // 28 AER – apical epidermal ridge
    // 29
    // 30
    // 31
    
    
    // Non-diffusible TFs 
    // 0 growth - treat as int, INT_MIN => die
    // 1 # not used: spring stress integrator_1 slow addition, fast decay, prolonged stress indicator // rather use per spring integrator
    // 2 # not used: spring stress integrator_2 fast addition, slow decay, cyclical peak indicator    // rather use current stress?  
    // 3 clock_1
    // 4 clock_2
    // 5 
    // 6
    // 7
    
    // matrix stiffness 
    // hyaline & apatite => all bonds triangulated and stiff
    // fribrocyte => 2 stiff bonds
    // tenocyte & muscle => only one stiff bond
    // elastin => medium stiffness & long yield strain
    
                                        
    //(ii)make/break bonds - run on dense list of particles, requires (i)search of neighbours, (ii)spare bonds on both 
    
    
    
    //(iii)add/remove particles from/to reserve list. NB ideally don't process particles in reserve list. 
    
    
    
    // Remodelling rules
    // Different cell types respond by secreting or resorbing their characteristic materials. 
    // The general pattern of response is 
    // (i) prolonged tension causes lengthening,                    i.e. slow addition, fast decay integrator_1.
    // (ii) cyclical loading causes strengthening,                  i.e. fast addition, slow decay integrator_2.
    // (iii) low peak strain causes shortening of fibrous tissue,  (low value on integrator_1)
    // (iv) low peak stress causes weakening.                      (low value on integrator_2)
    
    // peak stress & strain can differ for muscle, not for other tissues.
*/    
    
 

/*
    // what is current epigenetic state 
    // which genes should run. 
    
    ////////////////////////////////
    
    // For a given cell (i.e. particle), there is a list of active genes – bit mask on an "active genes" uint. This could be a uint array if >32 genes are required.
    // Genes may contain bit masks for activating other genes.
    // These are equivalent to "Long non-coding RNA" transcription activators →  which help to form the "promotor initiator complex"
    // (As opposed to general silencer/supressor/enhancer binding transcription factors.) NB silencing is permanent.
    // This leads to dense lists for particles on which to execute each gene.
    // For each gene : active/inactive & silenced/not_yet
    
    // Efficient tracking of particles for active gene lists:
    // Copy bond tracking, 
    // Use "update genes" flag to make dense list of changes at particle sorting time.
    // Run update genes kernel on "changes" list -> add to 
    
    // Making dense lists: 
    // # should hold particles in the same order as the general list
    // # should require only processing of (i) existing dense list PLUS (ii) changes list
    
    // Have separate "reinitiate dense lists" kernel - to check / limit error propagation (rarely run).
    // See section on Optimisation below.
    
    
    ///////////////////////////////
    
    // Points from "New Biological Morphogenetic Methods..."
    // 2.1) Mutation of Mutability - this is genome modification, outside of the simulation. 
    //      However it constrains how genes can be implemented.
    //
    // 2.2.1) Epigenetic Cell Lines—Morphogenetic and Histological Identities
    //
    //  Epigenetic variables: 
    //      (i) (float)current_activation, - 'phosphorylation'
    //      (ii) (bool)available/silenced, - 'methylation' => epigenetic type.
    //      (iii) (uint) spread count down - of silencing - e.g. Hox Genes stop spreading -> epigenetic type
    //                                          NB requires sequence of genes on chromosome.
    //      (iv) (bool) stop spreading
    //      (v)  (bool) "Not yet activated"
    //      => bool, bool, bool, uint, float. Could bitshift the uint to get the three bools.
    //
    //  Genetic parameters: 
    //      (i) Mutability, - NB probability of mutation at all, not magnitude of change.
    //                      - types: (a) cis-regulatory (I)degree of sensitivity, (II)to what - morphogen/stress
    //                               (b) change of gene action (not req for morphogenesis, i.e. protien change)
    //                               (c) genome architecture - (I) duplication/relocation of gene, 
    //                                                         (II)repartition of chromosomes
    //
    //      (ii) Delay/insulator, barrier to spread of inactivation.
    //
    //      (iii) Cis-regulatory sensitivities (morphogens, stress & strain cycles) altering current activation.
    //
    //      (iv)  Cell actions - secrete morphogens, move, adhere, divide, secrete/resorb material 
    //                          - dependent on current activation.
    //                          - Q: how to implement ?
    //                                  - move - NB risk of adding energy... 
    //                                  - adhere - make/break springs
    //                                  - divide - add particle from reserve.
    //                                  - secrete/resorb - change particle mass, viscosity, fluid stiffness, 
    //                                                   - spring length/stiffness -> anisotropy
    //
    // 2.2.2) Local Anatomical Coordinates—From Morphogen Gradients
    //                         - implemented via secretion, diffusion, & breakdown of transcription factors
    //                         - need genetic codes for:
    //                                  (I&II) symmetry breaking - establish orthogonality of poles & layers 
    //                                          (blastulation, primitive streak, gastrulation)
    //                                          blastomere->cyst->embryonic disk->primitive streak....
    //                                  
    //                                  (III) clock & wave front -> Hox genes
    //                                  (IV) gap and pair
    //                                  (V) tissue layer co-growth
    //                                  (VI) limb bud location
    //                                  (VII) limb growth & segmentation
    //                                  (VIII) digital ray lateral & lognitudinal segmentation
    //                                  (IX) reuse of synovial joint 'module'
    //                                  (X) location, migration & connection of of muscle-tendon
    //                                  (XI) location & connection of ligaments - articular, retinacular, dermal
    //                                  (XII) dermal specialization - palmar pads, nails, claws, hooves
    //                                  (XIII) nervous system connection & construction 
    //                          - NB epi-genetic branching tree, local coords, repeated patterns 
    //                                                                  -> reuse of cell types & modules 
    //
    // 2.2.3) Remodeling - implemented through cell actions 2.2.1(iv), regulated through 2.2.1(iii).
    // 
    //
    ///////////////////////////////////////////////////////////////////////////
    
    // Required genes: NB #define NUM_GENES  16.  NB tractability + evolvability
    //
    // Basic actions ://////  4+ ... but are these functions of celltype ?
    //
    // Add/remove particles - function of mass/radius
    //
    // Add remove mass/radius
    //
    // Increase/decrease matrix modulus & viscosity
    //
    // Incr/decr spring length, stiffness & damping - which springs? -> anisotropy & adhesion
    // 
    // 
    
    // Tissue modification rules 
    //
    // (i) prolonged tension causes lengthening, 
    // (ii) cyclical loading causes strengthening, 
    // (iii) low peak strain causes shortening of fibrous tissue, 
    // (iv) low peak stress causes weakening.
    // 
    // ? rolling integrators for spring stress & stress^3 -> mean vs peak
    // NB for muscles stress & strain are independent. => need to track strain for lengthening, stress for strengthening
    
    // Bone growth is regulated by morphogen diffusion at the growth plates and articular cartilage.
    // Bone itself is shaped by 
    //      * passive deformation of the bone primordia in the formation of the joints
    //      * active remodeling in response to forces to form the ridges and protrusions where major muscles and tendons attach
    //
    // 
    
    //
    // Cell type genes: 9+2
    //
    // bone, cartilage, tendon, muscle, ligament/fascia, loose tissue, dermis, epidermis, horn, 
    //
    // myotendinous junction, enthesis - poss double-expression
    //
    // 
    //
    // General anatomical modules:///////
    // 
    // articulation/synovial joint
    // 
    // bone primordia
    // 
    // endo/meso/ectoderm
    //
    // secretory
    //
    //
    // 
    // Epigenetic labels:
    // 
    // Hox genes - axial zoning - (6 used, 13x4 available)
    //
    // Limb bud fore/hind  (poss due to Hox combination)
    // 
    // Other homeobox genes - autopod, zeugoopod, stylopod, scapula/pelvis
    //
    // digit rays, carpus, metacarpus, phalanges, plalanx number, nail/claw/hoof
    
    
    
 */   
    
    // Data structures (fluid.h)
    //
    /*
        struct FGenome{   
                        // ## currently using fixed size genome for efficiency. 
                                            // NB Particle data size depends on genome size.
            uint mutability[NUM_GENES];
            uint delay[NUM_GENES];
            uint sensitivity[NUM_GENES][NUM_GENES]; // for each gene, its sensitivity to each TF or morphogen
            uint difusability[NUM_GENES][2];// for each gene, the diffusion and breakdown rates of its TF.
            //uint *function[NUM_GENES];    
                        // Hard code a case-switch that calls each gene's function iff the gene is active.
        };                                  // NB gene functions need to be in fluid_system_cuda.cu
    */
    
    /*
        struct FBufs {  // holds an array of pointers,
            ..
            char*				mcpu[ MAX_BUF ];
            ..
        }
    */
    
    /*
            #define FFORCE		3       //# 3DF        force 
            #define FPRESS		4       //# F      32  pressure
            #define FDENSITY	5       //# F          density

            #define FELASTIDX   14      //# currently [0]current index, [1]elastic limit, [2]restlength, 
                                                    [3]modulus, [4]damping coeff, [5]particle ID, [6]bond index 
            #define FPARTICLEIDX 29     //# uint[BONDS_PER_PARTICLE *2]  
                                                    list of other particles' bonds connecting to this particle AND their indices 
            #define FPARTICLE_ID 30     //# uint original pnum,
            #define FMASS_RADIUS 31     //# uint holding modulus 16bit and limit 16bit.  
            #define FNERVEIDX   15      //# uint
            #define FCONC       16      //# float[NUM_TF]        NUM_TF = num transcription factors & morphogens
            #define FEPIGEN     17      //# uint[NUM_GENES]
    */
/*    ///////////////////////////////////////////////////////////////////////////////
    
    
    // Code Optimisation:
    // Making dense lists
    //           part of particle sorting, build particle index arrays for 
    //                  (i) available genes (ii) active genes (iii) diffusion particles (iv) active/reserve particles. 
    //                   NB sequence of kernels called bu fluid_system::run()
    //                   InsertParticlesCUDA - sort particles into bins
    //                   PrefixSumCellsCUDA - count particles in bins - need to count (i&ii) above. (NB FUNC_FPREFIXSUM & FUNC_FPREFIXFIXUP)
    //                   CountingSortFullCUDA - build arrays - need (NB TransferToTempCUDA(..) for each fbuf array )
    
    // NB bitonic merge sort _may_ be useful to sort particles in active gene lists wrt to their location in the main particle list.
    // This is sorting particles in gene bins, on their particle bin FGNDX.
    // _Alternatively_, could loop on particles in bin and write each to gene bin. Max time : max num particles/bin. 
    
    // could write to a list of particles per gene, then run dense blocks for each gene.
    // NB all particles in block execute identical code.
    // Build active gene arrays during sorting
    // NB sequence : InsertParticles, PrefixSumCells, CountingSortFullCUDA  

    // NB generally in C/C++/Cuda types <32bit have to be converted to 32bit for processing. 
    // => use 32bit int/float, except where there is explicit support.
    
    // ############ convert to FP16 - NB Minimum spec: SM 5.3, so on Tesla-P100 (SM 6.0), NOT on GTX970m (SM 5.2)
    // NB P100 has GP100 with FP16, but the GTX 10xx series have GP104 with INT8 instead.
    // see https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html#arithmetic-primitives
    // see https://github.com/tpn/cuda-samples/tree/master/v8.0/0_Simple/fp16ScalarProduct
    // 
    
    // ############ convert to BFLOAT16 - only available on RTX cores,  i.e. not Tesla-P100
    // see  https://mc.ai/fp64-fp32-fp16-bfloat16-tf32-and-other-members-of-the-zoo/ 
    // https://docs.nvidia.com/cuda/cuda-math-api/modules.html
    // also https://medium.com/@prrama/an-introduction-to-writing-fp16-code-for-nvidias-gpus-da8ac000c17f
    // NB this also applies to much of the data in Morphogenesis.
    ///////////////////////////////////////////////////////////////////////////////
    
    
    
    // Epigenetic data uint[NUM_GENES], per gene :  1st bit -> available/silenced
    //                                              2nd portion -> spread/stop
    //                                              3rd portion -> current activation
    
    // Genome data (once for all particles):    uint sensitivity[NUM_GENES][NUM_GENES]; // cis-regulatory sensitivity to each TF, register for automata kernel 
    //
    //                                          uint difusability[NUM_GENES][2];        // diffusion rates of FCONC, register for diffusion kernel
    //                                          
    //                                          uint delay[NUM_GENES];                  // sets intial spread/stop
    //                                          uint mutability[NUM_GENES];             // used only for mutation
    
    // Epigenetics kernel
    // for each (available) gene: read epigenetic data, compute spread of silencing,  
    
    // Gene execution kernels
    // for each (active) gene:
    // for each particle in dense list of particles where that gene is active:
    //                          read current epigenetic activation, read FCONC, FNERVEIDX, FELASTIDX strain, FPRESS, FDENSITY
    //                          compute current activity of gene, & change in epigenetic activation
    //                          run gene function
    //                              (i)modify  FCONC, FNERVEIDX, FPRESS, FDENSITY, FMASS_RADIUS,
    //                                         FELASTIDX [1]elastic limit, [2]restlength, [3]modulus, [4]damping coeff
    //                                      
    //                              (ii)make/break bonds    - requires (i)search of neighbours, (ii)spare bonds on both 
    //                                                      - run on dense list of particles
    //
    //                              (iii)add/remove particles from/to reserve list. NB ideally don't process particles in reserve list. 
    
    
    // NB difference between sparse representation of gene sensitivities for efficient simulation
    // vs dense representation for mutation.
*/  

    // New plan:
    // 1) make _densely_packed_lists_ 
    //      (i) for each gene
    //      NB needs mods to prefixSum to count for bins of each gene.
    //      Run only on: existing list + changes list. NB most particles run only a few genes at a time.
    // 
    //      i.e. for each particle in list atomic_add to bin count
    //      Then for each bin in main list, write particle index to dense list of each active gene. 
    //      NB running this kernel on bins avoids the need to sort or atomic add.
    //
    //      (ii) Likewise make dense list(s) for epigenetics.
    //      (iii) also for elastic vs fluid,  diffusion vs non-diffusion, reserve vs in use.
    //      NB cuda malloc for list sizes, when enlargement req.
    //
    // 2) call epigenetics kernel for dense list. 
    //          - Used for Hox genes in somites, and probably limb segments.
    // 
    // 3) call gene kernel for each dense list
    //      NB each 'gene' is equivalent to a biological gene cluster under common cis regulatory control.
    //      Each gene has a list of operations :
    //      (i) packed sparse list of sensitivities to TFs.
    //      (ii) gene action(s) ?
    //          - modify params for property update kernel
    //          - activate other genes
    //          - silence self
    //          - secrete a few TFs
    //          - move cell i.e. change which particles it is bonded to.
    //      (iii) nervous interactions
    //          - send sensory data - to nervous system
    //          - contract muscle - on nerve stimulus, NB temporary change of stiffness & rest ln.
    //           
    // 4) call property update kernel for all particles 
    //      reads parameters for each particle to modify properties - tissue type: ectoderm, mesoderm, cartilage, bone, muscle, tendon, fascia, fat, horn.
    //      avoids multiple calculations & edits.
    //          - mass/radius, divide/combine/delete
    //          - bonds - form/break/length/stiffness/elastic limit
    //          - fluid - stiffness & viscosity
    //
    // Automatic genotype Optimisation: (In lieu of full differentiability).  
    // ? Record gradients ? wrt to what ? parameters of genome - but select which. 
    // Record snap shots. Replay to find _when_ change most affected desired result. 
    // i.e. gradient of result wrt time.
    // Replay from snap shot - which genes are active? 
    // Find gradients of result wrt to genes
    // For most significant genes, What are they sensitive to? -> gradient of gene wrt factor
    // Options (1) adjust sensitivity, (2) increase the source of the stimulus
    // 
    
 
 
 
//////////////////////////////////////////////////////////

    // Diffusion kernel: (i) read & use uint difusability[NUM_GENES][2];    // for each gene, the diffusion and breakdown rates of its TF.
    //                   (ii) non-difusability of morphogens outside body, yet we may want fluid & womb fluid-elastic simulation
    //                   (iii) non-difusibility of internal transcription factors
    //                   (iv) breakdown rate of morphogens and transcription factors, ?
//! constant diffusion rate (as a percentage, 0.0 to 1.0) of chemical exchanged per step. change this in future!
#define DIFFUSE_RATE 10.0   // - replace with: FGenome->difusability[NUM_GENES][2] (above)

//! loops over all the chemicals in the given particle and exchanges chemicals
extern "C" __device__ void contributeDiffusion(uint i, float3 p, int cell, const float currentConc[NUM_TF], float newConc[NUM_TF], uint diffusability[NUM_TF]){  
    // if the cell is empty, skip it
    if (fbuf.bufI(FGRIDCNT)[cell] == 0) return;

    // this is all standard setup stuff, borrowed from contributePressure()
    register float d2 = fparam.psimscale * fparam.psimscale; // (particle simulation scale), not PSI
    register float r2 = fparam.r2 / d2;

    // offset of particle in particle list, and number of particles in cell?
    int clast = fbuf.bufI(FGRIDOFF)[cell] + fbuf.bufI(FGRIDCNT)[cell];

    // iterate over particles in cell
    for (int cndx = fbuf.bufI(FGRIDOFF)[cell]; cndx < clast; cndx++) {
        int pndx = fbuf.bufI(FGRID)[cndx];

        // distance between this particle and considered particle (scalar distance squared, to save time I presume)
        float3 dist = p - fbuf.bufF3(FPOS) [pndx];
        float dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);

        // if the particle is in range, and not ourselves
        if (dsq < r2 && dsq > 0.0) {
            // distance falloff, diffusion rate scalar
            float c = (r2 - dsq) * d2;

            // chemical loop
            // for each chemical in this neighbour particle, exchange an amount relative to the diffusion rate with us
            // get the j'th chemical from this particle and exchange
            #pragma unroll
            for (int j = 0; j < NUM_TF; j++) 
                if(diffusability[j]) newConc[j] += diffusability[j] * c * (fbuf.bufF(FCONC)[pndx * NUM_TF + j] - currentConc[j]);
        }
    }
    // method:
    // add to ourselves 1% of what they have, and give away 1% of what we have
    // compute calls contribute once per bin
    // contribute will loop over particles for the bin
    // therefore it needs to loop over each 16 chemical per particle
    // space -> bin -> particle -> chemical in particle

    // this function returns nothing because all arguments are passed to it
    return;
}

//! main function to handle calculating diffusion, visits bins, then particles, then chemicals in particles
extern "C" __global__ void computeDiffusion(int pnum){
    // get particle index
    uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    // if the particle is outside the simulation, quit processing
    if (i >= pnum) return;

    // collect current concentration: copy from global memory to thread memory
    float newConc[NUM_TF] = {0};
    float currentConc[NUM_TF] = {0};
    // TODO maybe memcpy?
    for (int j = 0; j < NUM_TF; j++){
        currentConc[j] = fbuf.bufF(FCONC)[i * NUM_TF + j];
    }
    uint diffusability[NUM_TF];
    for (int j = 0; j < NUM_TF; j++) diffusability[j]=fgenome.tf_diffusability[j];

    // Get search cell
    int nadj = (1 * fparam.gridRes.z + 1) * fparam.gridRes.x + 1;
    uint gc = fbuf.bufI(FGCELL) [i];
    if (gc == GRID_UNDEF) return;
    gc -= nadj;

    // Now we work to exchange diffusion, by adding our neighbours chemicals and subtracting our own chemicals
    float3 pos = fbuf.bufF3(FPOS) [i];
    // bin loop: visit the bins
    for (int c = 0; c < fparam.gridAdjCnt; c++) {
        contributeDiffusion(i, pos, gc + fparam.gridAdj[c], currentConc, newConc, diffusability);
    }
    __syncthreads();

    // for this particular particle, loop over chemicals and write to global memory
    // TODO could also be memcpy
    for (int j = 0; j < NUM_TF; j++){
        fbuf.bufF(FCONC)[i * NUM_TF + j] = fgenome.tf_breakdown_rate[j] * (currentConc[j] + newConc[j]);
    }
}

extern "C" __device__ float3 contributeForce ( int i, float3 ipos, float3 iveleval, float ipress, float idens, int cell, uint _bondsToFill, uint _bonds[BONDS_PER_PARTICLE][2], float _bond_dsq[BONDS_PER_PARTICLE], bool freeze)
{			
	if ( fbuf.bufI(FGRIDCNT)[cell] == 0 ) return make_float3(0,0,0);                                        // If the cell is empty, skip it.
	float  dsq, sdist, c, pterm;
	float3 dist     = make_float3(0,0,0),      eterm = make_float3(0,0,0),    force = make_float3(0,0,0);
	uint   j;
	int    clast    = fbuf.bufI(FGRIDOFF)[cell] + fbuf.bufI(FGRIDCNT)[cell];                                // index of last particle in this cell
    for (int cndx = fbuf.bufI(FGRIDOFF)[cell]; cndx < clast; cndx++ ) {                                     // For particles in this cell.
		j           = fbuf.bufI(FGRID)[ cndx ];
		dist        = ( ipos - fbuf.bufF3(FPOS)[ j ] );                                                     // dist in cm (Rama's comment)
		dsq         = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);                                      // scalar distance squared
		if ( dsq < fparam.rd2 && dsq > 0) {                                                                 // IF in-range && not the same particle
            sdist   = sqrt(dsq * fparam.d2);                                                                // smoothing distance = sqrt(dist^2 * sim_scale^2))
			c       = ( fparam.psmoothradius - sdist ); 
			pterm   = fparam.psimscale * -0.5f * c * fparam.spikykern * ( ipress + fbuf.bufF(FPRESS)[ j ] ) / sdist;                       // pressure term
			force   += ( pterm * dist + fparam.vterm * ( fbuf.bufF3(FVEVAL)[ j ] - iveleval )) * c * idens * (fbuf.bufF(FDENSITY)[ j ] );  // fluid force
            
            if (freeze==true && _bondsToFill >0 && dist.x+dist.y+dist.z > 0.0){                             // collect particles, in the x+ve hemisphere, for potential bond formation 
                bool known      = false;
                uint bond_index = UINT_MAX;

                for (int a=0; a<BONDS_PER_PARTICLE; a++){                                                   // chk if known, i.e. already bonded 
                    if (fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + a*2] == i        ) known = true;   // particle 'j' has a bond to particle 'i'
                    if (fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + a*2] == UINT_MAX ) bond_index = a; // patricle 'j' has an empty bond 'a' : picks last empty bond
                    //if (_bonds[a][0] == j )known = true;                                                  // particle 'i' already has a bond to particle 'j'  // not req, _bonds starts empty && only touch 'j' once
                }
                if (known == false && bond_index<UINT_MAX){       
                    //int bond_direction = 1*(dist.x-dist.y+dist.z>0.0) + 2*(dist.x+dist.y-dist.z>0.0);     // booleans divide bond space into quadrants of x>0.
                    float approx_zero   = 0.02*fparam.rd2;
                    int bond_direction  = ((dist.x+dist.y+dist.z)>0) * (1*(dist.x*dist.x>approx_zero) + 2*(dist.y*dist.y>approx_zero) + 4*(dist.z*dist.z>approx_zero)) -1; // booleans select +ve quadrant x,y,z axes and their planar diagonals
                    //printf("\ni=%u, bond_direction=%i, dist=(%f,%f,%f), dsq=%f, approx_zero=%f", i, bond_direction, dist.x, dist.y, dist.z, dsq, approx_zero);
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
    uint bonds[BONDS_PER_PARTICLE][2];                                              // [0] = index of other particle, [1] = bond_index
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
            if(k<pnum && b<BONDS_PER_PARTICLE && i==fbuf.bufI(FELASTIDX)[k*BOND_DATA + b*DATA_PER_BOND] && a==fbuf.bufI(FELASTIDX)[k*BOND_DATA + b*DATA_PER_BOND +6] && a==fbuf.bufI(FELASTIDX)[k*BOND_DATA + b*DATA_PER_BOND +1]>0)intact=true;   
            if(i==k)intact=false;
            //if(intact==true)printf("\ncomputeForce: incomming bond intact  i=%u, k=%u, a=%u, b=%u",i,k,a,b);
            if(intact==false){                                                      // remove broken/missing _incomming_ bonds
                //fbuf.bufI(FPARTICLEIDX)[i*BONDS_PER_PARTICLE*2 + a*2] = UINT_MAX;   // particle NB retain bond direction info
                fbuf.bufI(FPARTICLEIDX)[i*BONDS_PER_PARTICLE*2 + a*2 +1] = UINT_MAX;// bond index
            }
        }
        
        for (int a=0; a<BONDS_PER_PARTICLE;a++){                                    // loop round this particle's list of _outgoing_ bonds /////
            bool intact = false;
            uint j = fbuf.bufI(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND];
            uint bond_idx = fbuf.bufI(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND + 6];  // chk bond intact nb short circuit evaluation of if conditions.
            // j is a particle, bond_idx is in range, AND j's reciprocal record matches i's record of the bond
            if(j<pnum && bond_idx<BONDS_PER_PARTICLE && i==fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + bond_idx*2] && a==fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + bond_idx*2 +1])intact=true; 
            if(i==j){
                fbuf.bufI(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND]   =UINT_MAX;
                fbuf.bufI(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND+1] =0;          // bond to self not allowed, 
            }
            if(intact==false){                                                      // remove missing _outgoing_ bonds // ## may not want to lose all this data.
                //fbuf.bufI(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND]=UINT_MAX;         // [0]current index, [1]elastic limit, [2]restlength, [3]modulus, [4]damping_coeff, [5]particle ID, [6]bond index 
                fbuf.bufF(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND+1] =0.0;
                //fbuf.bufF(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND+2]=1.0;
                //fbuf.bufF(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND+3]=0.0;
                //fbuf.bufF(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND+4]=0.0;
                //fbuf.bufI(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND+5]=UINT_MAX;
                //fbuf.bufI(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND+6]=UINT_MAX;
            }
        }
    }
    float3  pvel = {fbuf.bufF3(FVEVAL)[ i ].x,  fbuf.bufF3(FVEVAL)[ i ].y,  fbuf.bufF3(FVEVAL)[ i ].z}; // copy i's FEVAL to thread memory
    for (int a=0;a<BONDS_PER_PARTICLE;a++){                                         // compute elastic force due to bonds /////////////////////////////////////////////////////////
        uint bond                   = i*BOND_DATA + a*DATA_PER_BOND;                // bond's index within i's FELASTIDX 
        uint j                      = fbuf.bufI(FELASTIDX)[bond];                   // particle IDs   i*BOND_DATA + a
        float restlength        = fbuf.bufF(FELASTIDX)[bond + 2];               // NB fbuf.bufF() for floats, fbuf.bufI for uints.
        if(j<pnum && restlength>0){                                                                 // copy FELASTIDX to thread memory for particle i.
            float elastic_limit     = fbuf.bufF(FELASTIDX)[bond + 1];               // [0]current index, [1]elastic limit, [2]restlength, [3]modulus, [4]damping_coeff, [5]particle ID, [6]bond index 
            
            float modulus           = fbuf.bufF(FELASTIDX)[bond + 3];
            float damping_coeff     = fbuf.bufF(FELASTIDX)[bond + 4];
            uint  other_particle_ID = fbuf.bufI(FELASTIDX)[bond + 5];
            uint  bondIndex         = fbuf.bufI(FELASTIDX)[bond + 6];
            
            float3 j_pos = make_float3(fbuf.bufF3(FPOS)[ j ].x,  fbuf.bufF3(FPOS)[ j ].y,  fbuf.bufF3(FPOS)[ j ].z); // copy j's FPOS to thread memory
        
            dist            = ( fbuf.bufF3(FPOS)[ i ] - j_pos  );                   // dist in cm (Rama's comment)  /*fbuf.bufF3(FPOS)[ j ]*/
            dsq             = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);      // scalar distance squared
            abs_dist        = sqrt(dsq) + FLT_MIN;                                  // FLT_MIN adds minimum +ve float, to prevent division by abs_dist=zero
            float3 rel_vel  = fbuf.bufF3(FVEVAL)[ j ] - pvel;                       // add optimal damping:  -l*v , were v is relative velocity, and l= 2*sqrt(m*k)  
                                                                                    // where k is the spring stiffness.
                                                                                    // eterm = (bool within elastic limit) * (spring force + damping)
            float spring_strain = /* modulus * */ (abs_dist-restlength)/restlength; // NB this is now a strain accumulator, because stress is too large a number > FLT_MAX
            #define DECAY_FACTOR 0.99                                                                                   // could be a gene.
            fbuf.bufF(FELASTIDX)[bond + 7] = (fbuf.bufI(FELASTIDX)[bond + 7] + spring_strain) * DECAY_FACTOR;           // spring strain integrator
          if(i==0)printf("\ncomputeForce(): restlength=%f, modulus=%f , abs_dist=%f , spring_strain=%f , fbuf.bufI(FELASTIDX)[bond + 7]=%f  ",restlength , modulus , abs_dist , spring_strain , fbuf.bufF(FELASTIDX)[bond + 7]  );  
            
            eterm = ((float)(abs_dist < elastic_limit)) * ( ((dist/abs_dist) * spring_strain * modulus) - damping_coeff*rel_vel); // Elastic force due to bond ####
            force -= eterm;                                                         // elastic force towards other particle, if (rest_len -abs_dist) is -ve
            atomicAdd( &fbuf.bufF3(FFORCE)[ j ].x, eterm.x);                        // NB Must send equal and opposite force to the other particle
            atomicAdd( &fbuf.bufF3(FFORCE)[ j ].y, eterm.y);
            atomicAdd( &fbuf.bufF3(FFORCE)[ j ].z, eterm.z);                        // temporary hack, ? better to write a float3 attomicAdd using atomicCAS  #########

            if (abs_dist >= elastic_limit){                                         // If (out going bond broken)
                fbuf.bufI(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +1]=0;           // remove broken bond by setting elastic limit to zero.
                //fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +3]=0;           // set modulus to zero
                
                uint bondIndex_ = fbuf.bufI(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +6];
                fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + bondIndex_+1] = UINT_MAX ;// set the reciprocal bond index to UINT_MAX, but leave the old particle ID for bond direction.
                //fbuf.bufI(FELASTIDX)[bond] = UINT_MAX;
                //printf("\n#### Set to broken, i=%i, j=%i, b=%i, fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + b]=UINT_MAX\t####",i,j,bondIndex_);
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
            //printf("\nBond making loop i=%u, a=%i, bonds[a][1]=%u, bond_dsq[a]=%f",i,a,bonds[a][1],bond_dsq[a]);
            
            do {} while( atomicCAS(&ftemp.bufI(FPARTICLEIDX)[otherParticleBondIndex], UINT_MAX, 0) );               // lock ////////// ###### //  if (not locked) write zero to 'ftemp' to lock.
            if (fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex]==UINT_MAX)  fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex] = i;    //  if (bond is unoccupied) write to 'fbuf' to assign this bond
            ftemp.bufI(FPARTICLEIDX)[otherParticleBondIndex] = UINT_MAX;                                            // release lock // ######

            if (fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex] == i){                                              // if (this bond is assigned) write bond data
                fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex +1] = a;                                             // write i's outgoing bond_index to j's incoming bonds
                uint i_ID = fbuf.bufI(FPARTICLE_ID)[i];                                                             // retrieve permenant particle IDs for 'i' and 'j'
                uint j_ID = fbuf.bufI(FPARTICLE_ID)[bonds[a][0]];                                                   // uint bonds[BONDS_PER_PARTICLE][2];[0]=index of other particle,[1]=bond_index
                float bond_length = sqrt(bond_dsq[a]);
                float modulus = 100000;       // 100 000 000                                                        // 1000000 = min for soft matter integrity // 
                fbuf.bufI(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND]    = bonds[a][0];                               // [0]current index,
                fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +1] = 2 * bond_length ;                          // [1]elastic limit  = 2x restlength i.e. %100 strain
                fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +2] = 0.5*bond_length;                               // [2]restlength = initial length  
                fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +3] = modulus;                                   // [3]modulus
                fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +4] = 2*sqrt(fparam.pmass*modulus);              // [4]damping_coeff = optimal for mass-spring pair.
                fbuf.bufI(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +5] = j_ID;                                      // [5]save particle ID of the other particle NB for debugging
                fbuf.bufI(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +6] = bonds[a][1];                               // [6]bond index at the other particle 'j's incoming bonds
                //printf("\nNew Bond a=%u, i=%u, j=%u, bonds[a][1]=%u, fromPID=%u, toPID=%u,, fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex]=%u, otherParticleBondIndex=%u",
                //       a,i,bonds[a][0],bonds[a][1],i_ID,j_ID, fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex], otherParticleBondIndex);
            }            
        }// end if 
        __syncthreads();    // NB applies to all threads _if_ the for loop runs, i.e. if(freeze==true)
    }                                                                               // end loop around FELASTIDX bonds
}                                                                                   // end computeForce (..)

extern "C" __global__ void randomInit ( int seed, int numPnts )                                                                 // NB not currently used
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= numPnts ) return;

	// Initialize particle random generator	
	curandState_t* st = (curandState_t*) (fbuf.bufC(FSTATE) + i*sizeof(curandState_t));
	curand_init ( seed + i, 0, 0, st );		
}

#define CURANDMAX		2147483647

extern "C" __global__ void emitParticles ( float frame, int emit, int numPnts )                                                 // NB not currently used, may be a useful template
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

__device__ uint getGridCell ( float3 pos, uint3& gc )                                                                           // NB not currently used
{	
	gc.x = (int)( (pos.x - fparam.gridMin.x) * fparam.gridDelta.x);			// Cell in which particle is located
	gc.y = (int)( (pos.y - fparam.gridMin.y) * fparam.gridDelta.y);
	gc.z = (int)( (pos.z - fparam.gridMin.z) * fparam.gridDelta.z);		
	return (int) ( (gc.y*fparam.gridRes.z + gc.z)*fparam.gridRes.x + gc.x);	
}

extern "C" __global__ void sampleParticles ( float* brick, uint3 res, float3 bmin, float3 bmax, int numPnts, float scalar )     // NB not currently used
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

extern "C" __global__ void computeQuery ( int pnum )                                                                            // NB not currently used
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
	
	// Shield for particle store at fparam.pboundmax . ? does this exist implicitly due to the other boundaries ? 
	float3 dist = fparam.pboundmax - pos;
	diff = 2*fparam.pradius - (dist.x + dist.y + dist.z) * ss;              // use Manhatan norm for speed & 2*pradius for safety
	if ( diff > EPSILON ) {
        norm = make_float3( 1, 1, 1 );                                      // NB planar norm for speed, not spherical
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
	
	// Leap-frog Integration
	float3 vnext = accel*dt + vel;					// v(t+1/2) = v(t-1/2) + a(t) dt		
	fbuf.bufF3(FVEVAL)[i] = (vel + vnext) * 0.5;	// v(t+1) = [v(t-1/2) + v(t+1/2)] * 0.5			
	fbuf.bufF3(FVEL)[i] = vnext;
	fbuf.bufF3(FPOS)[i] += vnext * (dt/ss);			// p(t+1) = p(t) + v(t+1/2) dt		
    
    if (i==0 ){
        printf("\n\nadvanceParticles()2:");
        printf("\nvel.x==%f",vel.x);
        printf("\naccel.x==%f",accel.x);
        printf("\ndt==%f",dt);
        printf("\nvnext.x==%f",vnext.x);
        printf("\nss==%f",ss);
    }
}


