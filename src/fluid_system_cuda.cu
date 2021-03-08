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
//__constant__ FBondParams    fbondparams;    // GPU copy of remodelling parameters. 
__constant__ uint			gridActive;

#define SCAN_BLOCKSIZE		512
//#define FLT_MIN  0.000000001              // set here as 2^(-30)
//#define UINT_MAX 65535

//if(fparam.debug>2) => device printf

extern "C" __global__ void insertParticles ( int pnum )                                         // decides which bin each particle belongs in.
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index
	if ( i >= pnum ) return;

	//-- debugging (pointers should match CUdeviceptrs on host side)
	// printf ( " pos: %012llx, gcell: %012llx, gndx: %012llx, gridcnt: %012llx\n", fbuf.bufC(FPOS), fbuf.bufC(FGCELL), fbuf.bufC(FGNDX), fbuf.bufC(FGRIDCNT) );
  //  if (fparam.debug>2 && i==0)printf("\ninsertParticles(): pnum=%u\n",pnum);

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
	
if(fparam.debug>2 && i==pnum-1) printf("\ninsertParticles()1: gridTot=%i,  i=%u: gc.x=%i, gc.y=%i, gc.z=%i, gs=%i \t gridScan.x=%i, gridScan.y=%i, gridScan.z=%i, gridTot=%u,\t gridDelta=(%f,%f,%f) gridMin=(%f,%f,%f) gridRes=(%i,%i,%i)", 
    gridTot, i, gc.x, gc.y, gc.z, gs,  gridScan.x, gridScan.y, gridScan.z, gridTot, gridDelta.x, gridDelta.y, gridDelta.z,  gridMin.x, gridMin.y, gridMin.z, gridRes.x, gridRes.y, gridRes.z );

	if ( gc.x >= 1 && gc.x <= gridScan.x && gc.y >= 1 && gc.y <= gridScan.y && gc.z >= 1 && gc.z <= gridScan.z ) {
		fbuf.bufI(FGCELL)[i] = gs;											     // Grid cell insert.
		fbuf.bufI(FGNDX)[i] = atomicAdd ( &fbuf.bufI(FGRIDCNT)[ gs ], 1 );       // Grid counts.         //  ## counts particles in this bin.
                                                                                                         //  ## add counters for dense lists. ##############
        // for each gene, if active, then atomicAdd bin count for gene
        for(int gene=0; gene<NUM_GENES; gene++){ // NB data ordered FEPIGEN[gene][particle] AND +ve int values -> active genes.
            //if(fparam.debug>2 && i==0)printf("\n");
            if (fbuf.bufI(FEPIGEN) [i + gene*fparam.maxPoints] >0 ){  // "if((int)fbuf.bufI(FEPIGEN)" may clash with INT_MAX
                atomicAdd ( &fbuf.bufI(FGRIDCNT_ACTIVE_GENES)[gene*gridTot  + gs ], 1 );
                //if(fparam.debug>2 && (gene==6||gene==9) /*i<10*/) printf("\ninsertParticles()2: i=,%u, gene=,%u, gs=,%u, fbuf.bufI(FGRIDCNT_ACTIVE_GENES)[ gene*gridTot  + gs ]=,%u",
                //    i, gene, gs, fbuf.bufI(FGRIDCNT_ACTIVE_GENES)[ gene*gridTot  + gs ]);
            }
            // could use a small array of uints to store gene activity as bits. This would reduce the reads, but require bitshift and mask to read. 
            //if(fparam.debug>2 && i==0)printf("\ninsertParticles()3: fbuf.bufI(FEPIGEN) [i*NUM_GENES + gene]=%u  gene=%u  i=%u,",fbuf.bufI(FEPIGEN)[gene*pnum + i/* i*NUM_GENES + gene*/], gene ,i  );
        }
        //if(fparam.debug>2 && i==0)printf("\n");
	} else {
		fbuf.bufI(FGCELL)[i] = GRID_UNDEF;  // gridTot;//    // m_GridTotal  
		//if(i>pnum-10)fbuf.bufI(FGNDX)[i] = atomicAdd ( &fbuf.bufI(FGRIDCNT)[ gridTot-1 ], 1 );  // NB limit on the number of atomic operations on one variable.
        //if(fparam.debug>2)printf("\ninsertParticles()4: i=%i GRID_UNDEF, gc.x=%i, gc.y=%i, gc.z=%i,  ",i, gc.x, gc.y, gc.z);
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
    
    //if(fparam.debug>2)printf("\ntally_denselist_lengths: gridTot=%u, fbuf.bufI(%i)[%i] = %u, &fdense_list_lengths)[list]=%p \t",
    //       gridTot, fdense_list_lengths, list, fbuf.bufI(fdense_list_lengths)[list], &fbuf.bufI(fdense_list_lengths)[list] );
}

extern "C" __global__ void countingSortFull ( int pnum )                                // Counting Sort - Full (deep copy)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                             // particle index
	if ( i >= pnum ) return;
    if (fparam.debug>2 && i==0)printf("\ncountingSortFull(): pnum=%u\n",pnum);
	// Copy particle from original, unsorted buffer (msortbuf),
	// into sorted memory location on device (mpos/mvel)
	uint icell = ftemp.bufI(FGCELL) [ i ];                                              // icell is bin into which i is sorted in fbuf.*

	//if ( icell == GRID_UNDEF ) printf("\nicell == GRID_UNDEF, i=,%u,",i);   
	
	if ( icell != GRID_UNDEF ) {	                                                    // This line would eliminate out of range particles from the model, inc. NULL particles.
		// Determine the sort_ndx, location of the particle after sort		
        uint indx =  ftemp.bufI(FGNDX)  [ i ];                                          // indx is off set within new cell
        int sort_ndx = fbuf.bufI(FGRIDOFF) [ icell ] + indx ;                           // global_ndx = grid_cell_offet + particle_offset	
		float3 zero; zero.x=0;zero.y=0;zero.z=0;
        
        // Make dense lists for (i) available genes (ii) active genes (iii) diffusion particles (iv) active/reserve particles. ######################
        // NB req new FGNDX & FGRIDOFF for each of (i-iv).
        // Write (1) list of current array lengths, (2) arrays containing  [sort_ndx] of relevant particles.
        // In use kernels read the array to access correct particle.
        // If there is data only used by such kernels, then it should be stored in a dense array.  
        
		// Transfer data to sort location
		fbuf.bufI (FGRID)   [sort_ndx] =	sort_ndx;                                   // full sort, grid indexing becomes identity		
		fbuf.bufF3(FPOS)    [sort_ndx] =	ftemp.bufF3(FPOS)    [i];
		fbuf.bufF3(FVEL)    [sort_ndx] =	ftemp.bufF3(FVEL)    [i];
		fbuf.bufF3(FVEVAL)  [sort_ndx] =	ftemp.bufF3(FVEVAL)  [i];
		fbuf.bufF3(FFORCE)  [sort_ndx] =    zero;                                       // fbuf.bufF3(FFORCE)[ i ] += force; in contributeForce() requires value setting to 0 
		fbuf.bufF (FPRESS)  [sort_ndx] =	ftemp.bufF(FPRESS)   [i];
		fbuf.bufF (FDENSITY)[sort_ndx] =	ftemp.bufF(FDENSITY) [i];
        fbuf.bufI (FAGE)    [sort_ndx] =	ftemp.bufI(FAGE)     [i];
		fbuf.bufI (FCLR)    [sort_ndx] =	ftemp.bufI(FCLR)     [i];
		fbuf.bufI (FGCELL)  [sort_ndx] =	icell;
		fbuf.bufI (FGNDX)   [sort_ndx] =	indx;
        float3 pos = ftemp.bufF3(FPOS) [i];
        // add extra data for morphogenesis
        // track the sort index of the other particle
        for (int a=0;a<BONDS_PER_PARTICLE;a++){
            // FELASTIDX: [0]current index, [1]elastic limit, [2]restlength, [3]modulus, [4]damping coeff, [5]particle ID, [6]bond index, [7]stress integrator, [8]change-type binary indicator
            uint j = ftemp.bufI(FELASTIDX) [i*BOND_DATA + a*DATA_PER_BOND];             // NB i,j are valid only in ftemp.*
            uint j_sort_ndx = UINT_MAX;
            uint jcell = GRID_UNDEF;
   
            if (j<pnum){
                jcell       = ftemp.bufI(FGCELL) [ j ];                                 // jcell is bin into which j is sorted in fbuf.*
                uint jndx   = UINT_MAX;
                if ( jcell != GRID_UNDEF ) {                                            // avoid out of bounds array reads
                    jndx    =  ftemp.bufI(FGNDX)  [ j ];      
                    if((fbuf.bufI(FGRIDOFF) [ jcell ] + jndx) <pnum){
                        j_sort_ndx = fbuf.bufI(FGRIDOFF) [ jcell ] + jndx ;             // new location in the list of the other particle
                    }
                }                                                                       // set modulus and length to zero if ( jcell != GRID_UNDEF ) // No longer done.
            }
            fbuf.bufI (FELASTIDX) [sort_ndx*BOND_DATA + a*DATA_PER_BOND]  = j_sort_ndx; // NB if (j>=pnum) j_sort_ndx = UINT_MAX; preserves non-bonds
            for (int b=1;b<5/*DATA_PER_BOND*/;b++){                                     // copy [1]elastic limit, [2]restlength, [3]modulus, [4]damping coeff, etc // no longer (iff unbroken)
                fbuf.bufF (FELASTIDX) [sort_ndx*BOND_DATA + a*DATA_PER_BOND +b] = ftemp.bufF (FELASTIDX) [i*BOND_DATA + a*DATA_PER_BOND + b]; // uints
            }                                                                           // old: copy the modulus & length
            fbuf.bufI (FELASTIDX) [sort_ndx*BOND_DATA + a*DATA_PER_BOND +5] = ftemp.bufI (FELASTIDX) [i*BOND_DATA + a*DATA_PER_BOND + 5];   //[5]partID, uint
            fbuf.bufI (FELASTIDX) [sort_ndx*BOND_DATA + a*DATA_PER_BOND +6] = ftemp.bufI (FELASTIDX) [i*BOND_DATA + a*DATA_PER_BOND + 6];   //[6]bond index, uint
            fbuf.bufF (FELASTIDX) [sort_ndx*BOND_DATA + a*DATA_PER_BOND +7] = ftemp.bufF (FELASTIDX) [i*BOND_DATA + a*DATA_PER_BOND + 7];   //[7]stress integrator, float
            fbuf.bufI (FELASTIDX) [sort_ndx*BOND_DATA + a*DATA_PER_BOND +8] = ftemp.bufI (FELASTIDX) [i*BOND_DATA + a*DATA_PER_BOND + 8];   //[8]change-type, uint
        }
        for (int a=0;a<BONDS_PER_PARTICLE;a++){                                         // The list of bonds from other particles 
            uint k = ftemp.bufI(FPARTICLEIDX) [i*BONDS_PER_PARTICLE*2 + a*2];           // NB i,j are valid only in ftemp.*
            uint b = ftemp.bufI(FPARTICLEIDX) [i*BONDS_PER_PARTICLE*2 + a*2 +1];
            uint kndx, kcell, ksort_ndx = UINT_MAX; 
            if (k<pnum){                                                                // (k>=pnum) => bond broken // crashes when j=0 (as set in demo), after run().
                kcell         = ftemp.bufI(FGCELL) [ k ];                               // jcell is bin into which j is sorted in fbuf.*
                if ( kcell   != GRID_UNDEF ) {
                    kndx      = ftemp.bufI(FGNDX)  [ k ];  
                    ksort_ndx = fbuf.bufI(FGRIDOFF)[ kcell ] + kndx ;            
                }
            }
            fbuf.bufI (FPARTICLEIDX) [sort_ndx*BONDS_PER_PARTICLE*2 + a*2]      =  ksort_ndx; // ftemp.bufI(FPARTICLEIDX) [i*BONDS_PER_PARTICLE + a]
            fbuf.bufI (FPARTICLEIDX) [sort_ndx*BONDS_PER_PARTICLE*2 + a*2 +1]   =  b;
            ftemp.bufI(FPARTICLEIDX) [i       *BONDS_PER_PARTICLE*2 + a*2]      = UINT_MAX;   // set ftemp copy for use as a lock when inserting new bonds in ComputeForce(..)
        }
        //if (fparam.debug>2)printf("\n(sort_ndx=%u, i=%u)", sort_ndx, i);
        
        fbuf.bufI (FPARTICLE_ID) [sort_ndx] =	ftemp.bufI(FPARTICLE_ID) [i];
        fbuf.bufI (FMASS_RADIUS) [sort_ndx] =	ftemp.bufI(FMASS_RADIUS) [i];
        fbuf.bufI (FNERVEIDX)    [sort_ndx] =	ftemp.bufI(FNERVEIDX)    [i];
        
        uint* fbuf_epigen  = &fbuf.bufI(FEPIGEN)[sort_ndx];
        uint* ftemp_epigen = &ftemp.bufI(FEPIGEN)[i];
        for (int a=0;a<NUM_GENES;a++)  fbuf_epigen[pnum*a]  = ftemp_epigen[pnum*a];  // NB launched with pnum=mMaxPoints=fparam.maxPoints
        
        float* fbuf_conc  = &fbuf.bufF(FCONC)[sort_ndx * NUM_TF];
        float* ftemp_conc = &ftemp.bufF(FCONC)[i * NUM_TF];
        for (int a=0;a<NUM_TF;a++)     fbuf_conc[a] = ftemp_conc[a]; 
            //fbuf.bufF (FCONC)[sort_ndx * NUM_TF + a] = ftemp.bufF(FCONC)[i * NUM_TF + a];
            //__syncwarp();
	}
}

/*
extern "C" __global__ void countingSortEPIGEN ( int pnum )    
{
    uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                             // particle index
	if ( i >= pnum ) return;
    if (fparam.debug>2 && i==0)printf("\ncountingSortFull(): pnum=%u\n",pnum);
	// Copy particle from original, unsorted buffer (msortbuf),
	// into sorted memory location on device (mpos/mvel)
	uint icell = ftemp.bufI(FGCELL) [ i ];                                              // icell is bin into which i is sorted in fbuf.*

	//if ( icell == GRID_UNDEF ) printf("\nicell == GRID_UNDEF, i=,%u,",i);   
	
	if ( icell != GRID_UNDEF ) {	                                                    // This line would eliminate out of range particles from the model, inc. NULL particles.
		// Determine the sort_ndx, location of the particle after sort		
        uint indx =  ftemp.bufI(FGNDX)  [ i ];                                          // indx is off set within new cell
        int sort_ndx = fbuf.bufI(FGRIDOFF) [ icell ] + indx ;                           // global_ndx = grid_cell_offet + particle_offset	

        uint* fbuf_epigen  = &fbuf.bufI(FEPIGEN)[sort_ndx];
        uint* ftemp_epigen = &ftemp.bufI(FEPIGEN)[i];
        
        for (int a=0;a<NUM_GENES;a++)   {
            fbuf_epigen[pnum*a]  = ftemp_epigen[pnum*a];
            if(sort_ndx>pnum) printf("\ncountingSortEPIGEN: sort_ndx=,%u, i=,%u, pnum=,%u, a=,%u, ftemp_epigen[pnum*a]=,%u,  ",
                   sort_ndx, i, pnum, a, ftemp_epigen[pnum*a]
                  );
        }
    }
}
*/


extern "C" __global__ void countingSortDenseLists ( int pnum )
{
    unsigned int bin = threadIdx.x + blockIdx.x * SCAN_BLOCKSIZE/2;
    register int gridTot =      fparam.gridTotal;
    if (fparam.debug>2 && bin==0) printf("\n\n######countingSortDenseLists###### bin==0  gridTot=%u, fbuf.bufI (FGRIDOFF)[bin]=%u \n",gridTot, fbuf.bufI (FGRIDOFF)[0]);
	if ( bin >= gridTot ) return;                                    // for each bin, for each particle, for each gene, 
                                                                     // if gene active, then write to dense list 
    uint count = fbuf.bufI (FGRIDCNT)[bin];
    //if (fparam.debug>2 && bin%10000==0)printf("|");
    if (count==0) return;                                            // return here means that IFF all bins in this threadblock are empty,
    /*
    //if (fparam.debug>2 && count>0&&bin%10000==0)printf("\n\ncountingSortDenseLists: (count>100) bin=%u\n\n",bin);                           // then this multiprocessor is free for the next threadblock.
    //if (fparam.debug>2 && bin%100==0)printf("!count=%u,",count);                                                                 // NB Faster still would be a list of occupied bins.
    //if (fparam.debug>2 && count>27)printf("\ncount=%u,bin=%u\t",count,bin);
    
    uint grdoff_ =0;
    if(bin>0)grdoff_ =fbuf.bufI (FGRIDOFF)[bin-1];
    */
    uint grdoffset = fbuf.bufI (FGRIDOFF)[bin];
    uint gene_counter[NUM_GENES]={0};
    /*
    int step = grdoff_-grdoffset;
    if (fparam.debug>2 && bin>0 && step>27)  printf("\nbin=%u, gridoff step = %u, grdoff_=%u,  grdoffset=%u \t",bin, step, grdoff_, grdoffset );
    if (fparam.debug>2 && grdoffset>2200 && grdoffset<22100) printf("\ngrdoffset=%u  ",grdoffset);
    */
    register uint* lists[NUM_GENES];
    for (int gene=0; gene<NUM_GENES;gene++) lists[gene]=fbuf.bufII(FDENSE_LISTS)[gene]; // This element entry is a pointer
    
    register uint* offsets[NUM_GENES];
    for (int gene=0; gene<NUM_GENES;gene++) offsets[gene]=&fbuf.bufI(FGRIDOFF_ACTIVE_GENES)[gene * gridTot];   // The address of this element
    
    if (grdoffset+count > pnum){    printf("\n\n!!Overflow: (grdoffset+count > pnum), bin=%u \n",bin);     return;}
    
    for(uint particle=grdoffset; particle<grdoffset+count; particle++){
        if (fparam.debug>2 && particle>=22000 && particle<20030) printf("\nparticle==%u, ",particle);
        for(int gene=0; gene<NUM_GENES; gene++){
            /*
            if (gene==2 && particle%100==0) printf("\n offsets[gene][bin] + gene_counter[gene] =%u , particle=%u , fbuf.bufI(FEPIGEN) [particle + pnum*gene]=%u\t", 
                offsets[gene][bin] + gene_counter[gene] , particle, fbuf.bufI(FEPIGEN) [particle + pnum*gene]);
            */
            if(  /*(int)*/fbuf.bufI(FEPIGEN) [particle + pnum*gene] >0 ) {    // NB launched with pnum=mMaxPoints=fparam.maxPoints      // if (this gene is active in this particle)
                lists[gene][ offsets[gene][bin] + gene_counter[gene] ]=particle;
                gene_counter[gene]++;
                //if (fparam.debug>2 )printf("*");
                /*
                 * if (gene>2/_*particle<10&&gene==2*_/)printf("\ncountingSortDenseLists()1:  particle=,%u, gene=,%u, bin=,%u, grdoffset=,%u, count=,%u, address=,%p, \t offsets[gene][bin]=,%u, gene_counter[gene]=,%u, fbuf.bufI(FEPIGEN) [particle + pnum*gene]=%u ",
                    particle, gene, bin, grdoffset, count,
                    &lists[gene][ offsets[gene][bin] + gene_counter[gene] ],
                    offsets[gene][bin],
                    gene_counter[gene],
                    fbuf.bufI(FEPIGEN) [particle + pnum*gene]//UINT_MAX//
                                                   );
                */
                if (fparam.debug>2 && gene_counter[gene]>fbuf.bufI(FGRIDCNT_ACTIVE_GENES)[gene*gridTot +bin] )   
                    printf("\n Overflow: particle=,%u, ID=,%u, gene=,%u, bin=,%u, gene_counter[gene]=,%u, fbuf.bufI (FGRIDCNT_ACTIVE_GENES)[gene*gridTot +bin]=,%u \t\t",
                           particle, fbuf.bufI(FPARTICLE_ID)[particle], gene, bin, gene_counter[gene], fbuf.bufI (FGRIDCNT_ACTIVE_GENES)[gene*gridTot +bin]);
                    /*
                    //else printf("\n Non-overflow: particle=%u, ID=%u, gene=%u, bin=%u, gene_counter[gene]=%u, fbuf.bufI (FGRIDCNT_ACTIVE_GENES)[gene*gridTot +bin]=%u \t\t",
                    //       particle, fbuf.bufI(FPARTICLE_ID)[particle], gene, bin, gene_counter[gene], fbuf.bufI (FGRIDCNT_ACTIVE_GENES)[gene*gridTot +bin]);
                    */
            }else if (fparam.debug>2 && gene==2 && particle%1000==0)printf("*");
        }
    }
/* 
     * debug chk 
    if (fparam.debug>2){
    uint particle=grdoffset, gene=2;
            if(particle<10 && gene==2) {
                lists[gene][ offsets[gene][bin] + gene_counter[gene] ]=particle;   
                printf("\ncountingSortDenseLists: gene=%u, bin=%u, lists[gene][ offsets[gene][bin] + gene_counter[gene] ] = %u,  offsets[gene][bin]=%u,  gene_counter[gene]=%u ", 
                       gene, bin, lists[gene][ offsets[gene][bin] + gene_counter[gene] ],  offsets[gene][bin], gene_counter[gene]++ );
                gene_counter[gene]++;
            } 
    }
*/
}

extern "C" __global__ void countingSortChanges ( int pnum )
{
    uint bin = bin = threadIdx.x + blockIdx.x * SCAN_BLOCKSIZE/2;  //__mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index
    {   // debug chk
   /* 
    uint bin2=bin;
    float3 gridDelta = fparam.gridDelta;                                //  even if other variable have to be moved to slower 'local' memory  
    int3 gridRes =		fparam.gridRes;   
	float3 gc, binPos;
    gc.x=bin2%gridRes.x;
    bin2-=gridRes.x;
    bin2=bin2/gridRes.x;
    gc.z=bin2%gridRes.z;
    gc.y=bin2/gridRes.z;
    binPos=gc/gridDelta;
   */
  // if (fparam.debug>2 && threadIdx.x==0) printf("\nblockIdx.x=,%u \t",blockIdx.x);
    
    //unsigned int bin = threadIdx.x + blockIdx.x * SCAN_BLOCKSIZE/2;     // NB have to searach all particles => use main list bins. 
    }
    register int gridTot =      fparam.gridTotal;
	if ( bin >= gridTot ) return;                                    // for each bin, for each particle, for each change_list, 
                                                                     // if change_list active, then write to dense list 
    uint count = fbuf.bufI (FGRIDCNT/*_CHANGES*/)[bin];
    //if (count==0) return; 
    {   // debug chk
    /*if (fparam.debug>2 && threadIdx.x==0 && blockIdx.x%32==0)*///if(count!=0)printf("\ncountingSortChanges: bin=%u, gridTot=%u, count=%u, blockIdx.x=%u,  blockDim.x=%u, threadIdx.x=%u \t",bin, gridTot, count, blockIdx.x , blockDim.x, threadIdx.x );
    //if (fparam.debug>2 &&  bin==471311 /*blockIdx.x<100 && bin%32==0*/)printf("\n\n###countingSortChanges: bin=,%u, binPos=(,%f,%f,%f,) gridTot=,%u, count=,%u, blockIdx.x=,%u,  blockDim.x=,%u, threadIdx.x=,%u \t\n",
    //     bin, binPos.x, binPos.y, binPos.z, gridTot, count, blockIdx.x , blockDim.x, threadIdx.x );
    }
                                               // return here means that if all bins in this threadblock are empty,
                                                                     // then this multiprocessor is free for the next threadblock.
  //if (fparam.debug>2)printf("\ncountingSortChanges: bin=%u, count=%u \t",bin,count);
    uint grdoffset = fbuf.bufI (FGRIDOFF)[bin];
    uint change_list_counter[NUM_CHANGES]={0};                       // holds off set within the change bin for this change type, for the particles added so far.  
    
    register uint* lists[NUM_CHANGES];
    for (int change_list=0; change_list<NUM_CHANGES;change_list++) lists[change_list]=fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list];           // This element entry is a pointer
    
    //if(fparam.debug>2 && bin == 1/*change_list>6*/) for (int change_list=0; change_list<NUM_CHANGES;change_list++) printf("\nPointer to lists[%u] = %p,",change_list, lists[change_list]);
    if (count==0) return; 
    
    register uint list_length[NUM_CHANGES];
    for (uint change_list=0; change_list<NUM_CHANGES;change_list++) list_length[change_list]=fbuf.bufI(FDENSE_BUF_LENGTHS_CHANGES)[change_list];/*FDENSE_LIST_LENGTHS_CHANGES*/
/*
    if (bin==0){
        printf("\n");
        for(uint k=0; k<9; k++){
            printf("\n##countingSortChanges1: k=%u, list_length[%u]=%u, fbuf.bufI(FDENSE_BUF_LENGTHS_CHANGES/_*FDENSE_LIST_LENGTHS_CHANGES*_/)[change_list]=%u,  &fbuf.bufI(FDENSE_BUF_LENGTHS_CHANGES /_*FDENSE_LIST_LENGTHS_CHANGES*_/)[%u]=%p \t",
                k, k, list_length[k], fbuf.bufI(FDENSE_BUF_LENGTHS_CHANGES/_*FDENSE_LIST_LENGTHS_CHANGES*_/)[k], k, &fbuf.bufI(FDENSE_BUF_LENGTHS_CHANGES/_*FDENSE_LIST_LENGTHS_CHANGES*_/)[k]);
        }
    }
*/ 
    register uint* offsets[NUM_CHANGES];
    for (int change_list=0; change_list<NUM_CHANGES; change_list++)   offsets[change_list] = &fbuf.bufI(FGRIDOFF_CHANGES)[change_list * gridTot];   // The address of this element
/*
  //if (fparam.debug>2)printf("\ncountingSortChanges: grdoffset=%u, count=%u, pnum=%u \t",grdoffset, count, pnum);
*/
    if (grdoffset+count > pnum){ /* if (fparam.debug>2){printf("\n\n!!Overflow,  countingSortChanges: (grdoffset+count > pnum), bin=%u \n",bin);}  */   return;}
    
    for(uint particle=grdoffset; particle<grdoffset+count; particle++){                                                             // loop through particleIDx in bin in main particle list
/*
 * if(particle==grdoffset){
    float3 pos = fbuf.bufF3(FPOS)[particle];
    uint ID = fbuf.bufI(FPARTICLE_ID)[particle];
    printf("\ncountingSortChanges: bin=%u, particle=%u, ID=%u\t pos.x=%f, pos.y=%f, pos.z=%f",bin, particle, ID, pos.x, pos.y, pos.z);
    }
*/
        for(uint bond=0; bond<BONDS_PER_PARTICLE; bond++){                                                                          // loop through bonds on particle
            uint change = fbuf.bufI(FELASTIDX) [particle*BOND_DATA + bond*DATA_PER_BOND + 8];                                       // binary change indicator per bond.
          //if (fparam.debug>2)printf("\ncountingSortChanges: change=%u \t",change);
            if(change) {
                for (uint change_type=1, change_list=0; change_list<NUM_CHANGES; change_type*=2, change_list++){                    // loop through change indicator  
                  /*  
                   //printf("\nparticle=,%u, change_list=,%u, countingSortChanges: change=,%u, change_type=,%u, (change & change_type)=,%u, \t",
                   //       particle, change_list, change,change_type, (change & change_type) ); 
                   */ 
                    if(change & change_type){                                                                                       // bit mask to ID change type due to this bond
                        //if (fparam.debug>2)printf("\n\ncountingSortChanges: particle=%u, bond=%u \n\n",particle,bond);
                        lists[change_list] [( offsets[change_list][bin] + change_list_counter[change_list] )]                               = particle;   // write particleIdx to change list
                        lists[change_list] [( offsets[change_list][bin] + change_list_counter[change_list] + list_length[change_list] )]    = bond;       // write bondIdx to change list
                        
                        /*
                        if(change_list==1) printf("\ncountingSortChanges, change_list==1: particle=%u, bond=%u, particle_index=%u \t",
                            lists[change_list] [( offsets[change_list][bin] + change_list_counter[change_list] )],
                            lists[change_list] [( offsets[change_list][bin] + change_list_counter[change_list] + list_length[change_list] )],
                            offsets[change_list][bin] + change_list_counter[change_list]
                        );
                        */
                        
                         /*
                         //printf("[%u](%u,%u),",change_list,particle, bond);  // && threadIdx.x==0   if(change_list>6  )
                         //if(change_list==7 && particle==0) printf("\n\n[7](0,%u),", bond);
                         //if(change_list==8) printf("\n\n[8](%u,%u),",particle, bond);
                         */
                        {   // debug chk
                        /*
                        printf("\ncountingSortChanges: change_list=%u, \tlists[change_list]=%p, \tlist_length[change_list]=%u, \t&lists[change_list][particle]=%p, \t&lists[change_list][bond]=%p     \t", 
                               change_list, lists[change_list], list_length[change_list],
                               &lists[change_list][ (offsets[change_list][bin] + change_list_counter[change_list]) ], 
                               &lists[change_list][ (offsets[change_list][bin] + change_list_counter[change_list] + list_length[change_list]) ]
                              );
                        */
                        /*
                        if (particle==0){
                            printf("\ncountingSortChanges2, : lists[0]=%p, lists[1]=%p, &lists[0][0]=%p, &lists[0][1]=%p \n", lists[0] , lists[1], &lists[0][0], &lists[0][1] ); 
                            printf("\ncountingSortChanges3, :change_list=%u, bin=%u,  offsets[%u][%u]=%u, change_list_counter[%u]=%u, list_length[%u]=%u \n",
                                   change_list, bin, change_list, bin, offsets[change_list][bin], change_list, change_list_counter[change_list], change_list, list_length[change_list] );
                            for(int k=0; k<9; k++)
                                printf("\ncountingSortChanges4, :list_length[%u]=%u, fbuf.bufI(FDENSE_LIST_LENGTHS_CHANGES)[change_list]=%u,\t",
                                    k, list_length[k], fbuf.bufI(FDENSE_LIST_LENGTHS_CHANGES)[k]);
                        }
                        */
                        //if (fparam.debug>2 && change_type==2)printf("\ncountingSortChanges, : particle=%u, bond=%u, change=%u, change_type=%u, list_length[%u]=%u,  (offsets[change_list][bin] + change_list_counter[change_list])=%u  \t", 
                        //   particle, bond, change, change_type, change_list, list_length[change_list],  (offsets[change_list][bin] + change_list_counter[change_list]) );
                        /*
                        if (particle==0){
                            printf("\ncountingSortChanges2:  ");
                            for(int k=0; k<NUM_CHANGES; k++){
                                printf("\nlists[%u]=%p,  list_length[%u]=%u,  step=%ld", k, lists[k], k, list_length[k], (lists[k+1]-lists[k])/2  );
                            }
                        }
                        */
                       /* 
                       if (fparam.debug>2 && particle<10/_*00*_/) printf("\ncountingSortChanges()1: debug chk: particle=%u, bond=%u, change=%u, change_list=%u, change_list_counter[change_list]=%u, offsets[change_list][bin]=%u \t\t fbuf.bufI(FGRIDCNT_CHANGES)[ 0*gridTot + fbuf.bufI(FGCELL)[particle] ] =%u, fbuf.bufI(FGCELL)[particle]=%u, \t\t particleIndx=%u, bondIndx=%u \t", 
                            particle, bond, change, change_list, change_list_counter[change_list], offsets[change_list][bin],
                            fbuf.bufI(FGRIDCNT_CHANGES)[ 0*gridTot + fbuf.bufI(FGCELL)[particle] ],
                            fbuf.bufI(FGCELL)[particle],
                            lists[change_list][ (offsets[change_list][bin] + change_list_counter[change_list]) ],
                            lists[change_list][ (offsets[change_list][bin] + change_list_counter[change_list] + list_length[change_list]) ]   // NB only heal : change_list=0
                        );
                        */
                        }
                        change_list_counter[change_list]++;
                    }
                }
            }
        }
    }
 {   // debug chk
  //      for(uint particle=grdoffset; particle<grdoffset+count; particle++){ // ? has found particle in change list, _not_ index in main list  ?     // loop through particles in bin
  //      for(uint bond=0; bond<BONDS_PER_PARTICLE; bond++){                                                                                      // loop through bonds on particle
  //          uint change = fbuf.bufI(FELASTIDX) [particle*BOND_DATA + bond*DATA_PER_BOND + 8];                                                  // binary change indicator per bond.
          //if (fparam.debug>2)printf("\ncountingSortChanges: change=%u \t",change);
  //          if(change) {
  //              for (uint change_type=1, change_list=0; change_list<NUM_CHANGES; change_type*=2, change_list++){                               // loop through change indicator  
                   //if (fparam.debug>2)printf("\ncountingSortChanges: change=%u, change_type=%u, (change & change_type)=%u \t",change,change_type, (change & change_type) ); 
                    
  //                  if(change & change_type){                                                                                                  // bit mask to ID change type due to this bond
                        //if (fparam.debug>2)printf("\n\ncountingSortChanges: particle=%u, bond=%u \n\n",particle,bond);
                      //  lists[change_list][ (offsets[change_list][bin] + change_list_counter[change_list]) ]=particle;                         // write particleIdx to change list
                      //  lists[change_list][ (offsets[change_list][bin] + change_list_counter[change_list] + list_length[change_list]) ]=bond;  // write bondIdx to change list
                        /*
                        printf("\ncountingSortChanges: change_list=%u, \tlists[change_list]=%p, \tlist_length[change_list]=%u, \t&lists[change_list][particle]=%p, \t&lists[change_list][bond]=%p     \t", 
                               change_list, lists[change_list], list_length[change_list],
                               &lists[change_list][ (offsets[change_list][bin] + change_list_counter[change_list]) ], 
                               &lists[change_list][ (offsets[change_list][bin] + change_list_counter[change_list] + list_length[change_list]) ]
                              );
                        */
                        /*
                        if (particle==0){
                            printf("\ncountingSortChanges2, : lists[0]=%p, lists[1]=%p, &lists[0][0]=%p, &lists[0][1]=%p \n", lists[0] , lists[1], &lists[0][0], &lists[0][1] ); 
                            printf("\ncountingSortChanges3, :change_list=%u, bin=%u,  offsets[%u][%u]=%u, change_list_counter[%u]=%u, list_length[%u]=%u \n",
                                   change_list, bin, change_list, bin, offsets[change_list][bin], change_list, change_list_counter[change_list], change_list, list_length[change_list] );
                            for(int k=0; k<9; k++)
                                printf("\ncountingSortChanges4, :list_length[%u]=%u, fbuf.bufI(FDENSE_LIST_LENGTHS_CHANGES)[change_list]=%u,\t",
                                    k, list_length[k], fbuf.bufI(FDENSE_LIST_LENGTHS_CHANGES)[k]);
                        }
                        */
                        //if (fparam.debug>2 && change_type==2)printf("\ncountingSortChanges, : particle=%u, bond=%u, change=%u, change_type=%u, list_length[%u]=%u,  (offsets[change_list][bin] + change_list_counter[change_list])=%u  \t", 
                        //   particle, bond, change, change_type, change_list, list_length[change_list],  (offsets[change_list][bin] + change_list_counter[change_list]) );
                        /*
                        if (particle==0){
                            printf("\ncountingSortChanges2:  ");
                            for(int k=0; k<NUM_CHANGES; k++){
                                printf("\nlists[%u]=%p,  list_length[%u]=%u,  step=%ld", k, lists[k], k, list_length[k], (lists[k+1]-lists[k])/2  );
                            }
                        }
                        */
                        
                        /*
                         * printf("\ncountingSortChanges()2: debug chk: particle=%u, bond=%u, change=%u, change_list=%u, bin=%u, \t\t offsets[change_list][bin+1] - offsets[change_list][bin]=%u,  fbuf.bufI(FGRIDCNT_CHANGES)[ 0*gridTot + fbuf.bufI(FGCELL)[particle] ] =%u, fbuf.bufI(FGCELL)[particle]=%u, \t\t change_list_counter[change_list]=%u, list_length[change_list]=%u, particleIndx=%u, bondIndx=%u \t", 
                            particle, bond, change, change_list, bin, offsets[change_list][bin+1] - offsets[change_list][bin],
                            fbuf.bufI(FGRIDCNT_CHANGES)[ 0*gridTot + fbuf.bufI(FGCELL)[particle] ],
                            fbuf.bufI(FGCELL)[particle],
                            change_list_counter[change_list], list_length[change_list],
                            lists[change_list][ (offsets[change_list][bin] + change_list_counter[change_list]) ],
                            lists[change_list][ (offsets[change_list][bin] + change_list_counter[change_list] + list_length[change_list]) ]   // NB only heal : change_list=0
                        );*/
    //                }
    //            }
    //        }
    //    }
    //}
    /*
    for(uint particle=grdoffset; particle<grdoffset+count; particle++){ // ? has found particle in change list, _not_ index in main list  ?     // loop through particles in bin
        for(uint bond=0; bond<BONDS_PER_PARTICLE; bond++){                                                                                      // loop through bonds on particle
            uint change = fbuf.bufI(FELASTIDX) [particle*BOND_DATA + bond*DATA_PER_BOND + 8];
            if(change==1) {
                for (uint counter=0;counter<change_list_counter[0];counter++){
                    if (fparam.debug>2)printf("\ncountingSortChanges()2: debug chk: particle=%u, bond=%u, change=%u, particleIndx=%u, bondIndx=%u \t", 
                       particle, bond, change,
                       lists[0][ (offsets[0][bin] + change_list_counter[0]) ],
                       lists[0][ (offsets[0][bin] + change_list_counter[0] + list_length[0]) ]                                                  // NB only heal : change_list=0
                    );
                }
            }else if (fparam.debug>2)printf("\n#countingSortChanges(): debug chk: particle=%u, bond=%u, change=%u \t",particle, bond, change );
            
        }
    }// end debug chk
    */
 }
}

extern "C" __device__ float contributePressure ( int i, float3 p, int cell, float &sum_p6k )  
// pressure due to particles in 'cell'. NB for each particle there are 27 cells in which interacting particles might be.
{			
	if ( fbuf.bufI(FGRIDCNT)[cell] == 0 ) return 0.0;                       // If the cell is empty, skip it.

	float3 dist;
	float dsq, r, q, b, c, sum = 0.0;//, sum_p6k = 0.0;
	register float d2 = fparam.psimscale * fparam.psimscale;                // max length in simulation space
	register float r2 = fparam.r2 / d2;                                     // = m_FParams.psmoothradius^2 / m_FParams.psimscale^2
    register float H  = fparam.H;                                           // = m_FParams.psmoothradius / m_FParams.psimscale;
    register float sr = fparam.psmoothradius;
	
	int clast = fbuf.bufI(FGRIDOFF)[cell] + fbuf.bufI(FGRIDCNT)[cell];      // off set of this cell in the list of particles,  PLUS  the count of particles in this cell.

	for ( int cndx = fbuf.bufI(FGRIDOFF)[cell]; cndx < clast; cndx++ ) {    // For particles in this cell.
		int pndx = fbuf.bufI(FGRID) [cndx];                                 // index of this particle
		dist = p - fbuf.bufF3(FPOS) [pndx];                                 // float3 distance between this particle, and the particle for which the loop has been called.
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);              // scalar distance squared
        
        // From https://github.com/DualSPHysics/DualSPHysics/wiki/3.-SPH-formulation#31-smoothing-kernel 
        /*
         * q=r/h, where r=dist between particles, h=smoothing length
         * 
         * W(r,h) = alpha_D(1-q/2)**4 *(2q+1) for 0<=q<=2
         * 
         * where alpha_D = 21/(16*Pi*h**3)  , the normalization kernel in 3D,
         * i.e. 1/integral_(0,2){kernel * area of a sphere}dr 
         * 
         */
        
		if ( dsq < r2 && dsq > 0.0) {                                       // if(in-range && not the same particle) ie unused particles can be stored at one point.
            r=sqrt(dsq);
            q=r/sr;                                                         //r/H; i.e ss:=1
            b=(1-q/2.0);
            b*=b; 
            b*=b;
            sum  += b*(2*q +1);//(H+4*r);                                   // Wendland C^2 quintic kernel for 3 dimensions.
            /*
            if (i<10)printf("\n contribPressure()1: i=,%u, ,j=,%u,\t ,r=sqrt(dsq)=,%f, ,H=sr/ss=,%f, q=r/H=,%f, ,b=(1-q/2.0)^3,%f,\t ,pressure= 1-q/2.0)^3*(2*q +1)=,%f  ",i, pndx, r, H, q, b, b*(2*q +1) );

			c = (r2 - dsq)*d2;
			sum_p6k += c * c * c;
            if (i<10)printf("\ncontribPressure()2: i=,%u, ,j=,%u, r2=sr^2/ss^2=,%f, dsq=,%f, d2=ss^2=,%f,\t\t,c=(r2-dsq)*d2=,%f, ,,,,pressure_p6k=c^3=,%f, ", i, pndx,  r2, dsq, d2, c,  c*c*c );
            */
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
	float sum = 0.0, sum_p6k = 0.0;
	for (int c=0; c < fparam.gridAdjCnt; c++) {                                    
		sum += contributePressure ( i, pos, gc + fparam.gridAdj[c], sum_p6k );
	}
	__syncthreads();
    
	// Compute Density & Pressure
    float old_sum=sum,  old_sum_p6k=sum_p6k;
    float rest_dens = 0.0015;/*fparam.prest_dens*/
	sum = sum * fparam.pmass * fparam.wendlandC2kern;
	//sum_p6k = sum_p6k * fparam.pmass * fparam.poly6kern;
    
	if ( sum == 0.0 ) sum = 1.0;
	fbuf.bufF(FPRESS)  [ i ] = ( sum - rest_dens ) * fparam.pintstiff;   // pressure = (diff from rest density) * stiffness
	fbuf.bufF(FDENSITY)[ i ] = 1.0f / sum;
    /*
    if (i<10)printf("\n computePressure()2: i=,%u, ,old_sum=,%f, ,old_sum_p6k=,%-20.20f, ,sum*=pmass*wendlandC2kern=,%.32f, ,sum_p6k*=pmass*poly6kern=,%f,\t ,wendlandC2kern=,%f, poly6kern=,%f, ,pmass=,%f, ,prest_dens=,%f, ,pintstiff=,%f,\t ,Pressure=(sum-prest_dens)*pintstiff=,%f  ", 
        i, old_sum, old_sum_p6k, sum, sum_p6k, fparam.wendlandC2kern, fparam.poly6kern, fparam.pmass, rest_dens, fparam.pintstiff, fbuf.bufF(FPRESS)[i]  );
    */
}

extern "C" __global__ void computeGeneAction ( int pnum, int gene, uint list_len )  //NB here pnum is for the dense list NB Must zero ftemp.bufI(FEPIGEN) and ftemp.bufI(FCONC) before calling.
{
    uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                                         // particle index
    if ( i >= list_len ) return;
    uint particle_index = fbuf.bufII(FDENSE_LISTS)[gene][i];
    /*if (particle_index >= pnum){
        printf("\ncomputeGeneAction: (particle_index >= pnum),  gene=%u, i=%u, list_len=%u, particle_index=%u, pnum=%u .\t",
            gene, i, list_len, particle_index, pnum);
    } */   
    int delay = (int)fbuf.bufI(FEPIGEN)[gene*fparam.maxPoints + particle_index];                                // Change in _epigenetic_ activation of this particle
    //printf("\nDelay=%i, particle_index=%u\t", delay, particle_index);
    if (0 < delay && delay < INT_MAX){                                                                           // (FEPIGEN==INT_MAX) => active & not counting down.
        fbuf.bufI(FEPIGEN)[gene*fparam.maxPoints + particle_index]--;                                           // (FEPIGEN<1) => inactivated @ insertParticles(..)
        if (delay==1  &&  gene<NUM_GENES && fbuf.bufI(FEPIGEN)[ (gene+1)*fparam.maxPoints + particle_index ] )  // If next gene is active, start count down to inactivate it.
            fbuf.bufI(FEPIGEN)[(gene+1)*fparam.maxPoints + particle_index] = fgenome.delay[gene+1] ;            // Start countdown to silence next gene.
    }                                                                                               // (fgenome.delay[gene+1]==INT_MAX) => barrier to spreading inactivation.
    uint sensitivity[NUM_GENES];                                                                    // TF sensitivities : broadcast to threads
    #pragma unroll                                                                                  // speed up by eliminating loop logic.
    for(int j=0;j<NUM_GENES;j++) sensitivity[j]= fgenome.sensitivity[gene][j];                      // for each gene, its sensitivity to each TF or morphogen
    /*if(i==list_len-1)printf("\ncomputeGeneAction Chk : gene=%u, i=%u, list_len=%u, particle_index=%u, pnum=%u ,  sensitivity[15]=%u.\t",
            gene, i, list_len, particle_index, pnum, sensitivity[15]); */                             // debug chk 
    float activity=0;                                                                               // compute current activity of gene
    #pragma unroll
    for (int tf=0;tf<NUM_TF;tf++){                                                                  // read FCONC
        if(sensitivity[tf]!=0){                                                                     // skip reading unused fconc[]
            activity +=  sensitivity[tf] * fbuf.bufI(FCONC)[particle_index + fparam.maxPoints*tf];
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
        atomicAdd( &ftemp.bufI(FEPIGEN)[other_gene*fparam.maxPoints + particle_index], 1);   // what should be the initial state of other_gene when activated ?
    }
}

extern "C" __global__ void tallyGeneAction ( int pnum, int gene, uint list_length ){// called by ComputeGenesCUDA () after computeGeneAction (..) & synchronize().
    uint particle_index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                            // particle index
    if ( particle_index >= list_length ) return;                                                    // pnum should be length of list.
    // ## TODO convert i to particle index _iff_ not called for all particles : use a special dense list for "living tissue", made at same time as gene lists
    uint i = fbuf.bufII(FDENSE_LISTS)[2][particle_index];                                           // call for dense list of living cells (gene'2'living/telomere (has genes))
    if ( i >= pnum ) return; 
    
    float * fbufFCONC = &fbuf.bufF(FCONC)[i*NUM_TF];
    float * ftempFCONC = &ftemp.bufF(FCONC)[i*NUM_TF];
    uint * fbufFEPIGEN = &fbuf.bufI(FEPIGEN)[i]; //*NUM_GENES                                             // TODO FEPIGEN is a uint here. May need to pack binaries for spread & stop. See paper.
    uint * ftempFEPIGEN = &ftemp.bufI(FEPIGEN)[i]; //*NUM_GENES   // ## need to zero ftemp after counting sort full
    
    for(int j=0; j<NUM_TF;j++)      fbufFCONC[j] += ftempFCONC[j];  // *fparam.maxPoints
    for(int j=0; j<NUM_GENES;j++) fbufFEPIGEN[j*fparam.maxPoints] += ftempFEPIGEN[j*fparam.maxPoints];
}


extern "C" __global__ void computeNerveActivation ( int pnum ) //TODO computeNerveActivation    // initially simple sparse random connections + STDP, later neurogenesis
{                                                                 // NB (i) sensors concetrated in hands & feet (ii)stimuls from womb wall 
    
}

extern "C" __global__ void computeMuscleContraction ( int pnum ) //TODO computeMuscleContraction  // read attached nerve, compute force  
{
    
}

extern "C" __global__ void assembleMuscleFibresOutGoing ( int pnum, uint list, uint list_length ) // used for muscle, elastic ligg, and tendon
{
    uint particle_index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                            // particle index
    if ( particle_index >= list_length )return;
    uint i = fbuf.bufII(FDENSE_LISTS)[list][particle_index];
    if (i > fparam.maxPoints) return;
    
    printf("\nassembleMuscleFibresOutGoing() chk1: i=%u   ",i  );
    
    ///////////// Swap outgoing bond indicies 
    // Find highest stress incoming bond
    float maxStress         = 0.0;
    uint maxStressBondIdx   = 0;
    float stress;
    char* bond_char_ptr = &fbuf.bufC(FELASTIDX)[i*BOND_DATA];
    uint* bond_uint_ptr = &fbuf.bufI(FELASTIDX)[i*BOND_DATA];                                       //fbuf.bufI(FELASTIDX)[i*BOND_DATA + bond*DATA_PER_BOND +  ] ;
    float*bond_flt_ptr  = &fbuf.bufF(FELASTIDX)[i*BOND_DATA];                                       //FELASTIDX   [0]current index, [1]elastic limit, [2]restlength, [3]modulus,
                                                                                                                //[4]damping coeff, [5]particle ID,   [6]bond index, 
                                                                                                                //[7]stress integrator,  [8]change-type binary indicator
    for(int bond=0; bond<BONDS_PER_PARTICLE; bond++){                                               // find highest stress bond
        float stress = bond_flt_ptr[7 + bond*DATA_PER_BOND] ;
        if (stress>maxStress){
            maxStress = stress;
            maxStressBondIdx = bond;        // TODO chk vs null bonds or particles
        }
    }
    printf("\nassembleMuscleFibresOutGoing() chk2: i=%u   ",i  );
    if (maxStressBondIdx!=0){
        printf("\nassembleMuscleFibresOutGoing()  called for :  i=%u   ",i );
        // store high stress bond
        uint bytes = sizeof(uint)*DATA_PER_BOND;
        char temp[sizeof(uint)*DATA_PER_BOND /* DATA_PER_BOND*4 */] = {0};                       // NB sensitive to size of uint and float
        uint bondStep = maxStressBondIdx*DATA_PER_BOND;
        //memcpy(&bond_char_ptr[bondStep], temp, bytes);
        uint  currIdx           = bond_uint_ptr[0 + bondStep];  // could be done faster with a memcpy()  of bytes to a void or char pointer
        float elastLim          = bond_flt_ptr [1 + bondStep];  // NB existing memcpy causes data corruption, probable indexing error.
        float restLength        = bond_flt_ptr [2 + bondStep];
        float modulus           = bond_flt_ptr [3 + bondStep];
        float dampingCoeff      = bond_flt_ptr [4 + bondStep];
        uint  particleID        = bond_uint_ptr[5 + bondStep]; 
        uint  bondIndex         = bond_uint_ptr[6 + bondStep]; 
        float stressIntegrator  = bond_flt_ptr [7 + bondStep];
        uint  changeIndicator   = bond_uint_ptr[8 + bondStep];
        printf("\nassembleMuscleFibresOutGoing() chk1 called for :  i=%u   ",i );
        
        // move low stress bond
        //memcpy(bond_char_ptr, &bond_char_ptr[bondStep*sizeof(uint)], bytes);
        uint otherParticle          = bond_uint_ptr [0];
        uint otherParticleBondIDx   = bond_uint_ptr [6];
        if(otherParticle < fparam.maxPoints && otherParticleBondIDx < BONDS_PER_PARTICLE){
            bond_uint_ptr[0 + bondStep] = bond_uint_ptr[0] ;
            bond_flt_ptr [1 + bondStep] = bond_flt_ptr [1];
            bond_flt_ptr [2 + bondStep] = bond_flt_ptr [2];
            bond_flt_ptr [3 + bondStep] = fgenome.param[1][fgenome.default_modulus];  // change modulus elastic fibre type
            bond_flt_ptr [4 + bondStep] = bond_flt_ptr [4];
            bond_uint_ptr[5 + bondStep] = bond_uint_ptr[5]; 
            bond_uint_ptr[6 + bondStep] = bond_uint_ptr[6]; 
            bond_flt_ptr [7 + bondStep] = bond_flt_ptr [7];
            bond_uint_ptr[8 + bondStep] = bond_uint_ptr[8];

            // update reciprocal record
            printf("\nassembleMuscleFibresOutGoing() chk2 called for :  i=%u  otherParticle=%u  otherParticleBondIDx=%u ", i, otherParticle, otherParticleBondIDx );
            printf(".\n");//flush hopefully...
    
            fbuf.bufI(FELASTIDX)[otherParticle*BOND_DATA + otherParticleBondIDx*DATA_PER_BOND + 6]  = 0;  
        }
        printf("\nassembleMuscleFibresOutGoing() chk3 called for :  i=%u   ",i );    
        
        // write high stress bond
        //memcpy(temp, bond_char_ptr, bytes);
        bond_uint_ptr[0] = currIdx;
        bond_flt_ptr [1] = elastLim;
        bond_flt_ptr [2] = restLength;
        bond_flt_ptr [3] = fgenome.param[0][fgenome.collagen];   //modulus;    // change modulus collagen fibre type
        bond_flt_ptr [4] = dampingCoeff;
        bond_uint_ptr[5] = particleID; 
        bond_uint_ptr[6] = bondIndex; 
        bond_flt_ptr [7] = stressIntegrator;
        bond_uint_ptr[8] = changeIndicator;
        printf("\nassembleMuscleFibresOutGoing() chk4 called for :  i=%u   ",i );
        
        // update reciprocal record
        otherParticle          = bond_uint_ptr [0];
        otherParticleBondIDx   = bond_uint_ptr [6];
        if(otherParticle < fparam.maxPoints && otherParticleBondIDx < BONDS_PER_PARTICLE)
            fbuf.bufI(FELASTIDX)[otherParticle*BOND_DATA + otherParticleBondIDx*DATA_PER_BOND + 6]  = maxStressBondIdx;  
        
    printf("\nassembleMuscleFibresOutGoing() chk5 called for :  i=%u   ",i );
    }
    printf("\nassembleMuscleFibresOutGoing() chk3: i=%u   ",i  );
    
}

    
extern "C" __global__ void assembleMuscleFibresInComing ( int pnum, uint list, uint list_length ) // used for muscle, elastic ligg, and tendon
{
    uint particle_index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                            // particle index
    if ( particle_index >= list_length )return;
    uint i = fbuf.bufII(FDENSE_LISTS)[list][particle_index];
    if (i > fparam.maxPoints) return;
    
    printf("\nassembleMuscleFibresInComing() chk1: i=%u   ",i  );
    
    ///////////// Swap outgoing bond indicies 
    
    // find highest stress incoming bond
    uint incomingParticleIdx      ;//  = fbuf.bufI(FPARTICLEIDX)[i*2*BONDS_PER_PARTICLE];
    uint incomingParticleBondIDx  ;//  = fbuf.bufI(FPARTICLEIDX)[i*2*BONDS_PER_PARTICLE +1];
    
    float maxStress         = 0.0;
    uint maxStressBondIdx   = 0;
    
    
    
    
    for(int bond=0; bond<BONDS_PER_PARTICLE; bond++){                                                                                   // find highest stress bond
        incomingParticleIdx        = fbuf.bufI(FPARTICLEIDX)[i*2*BONDS_PER_PARTICLE + bond*2];
        incomingParticleBondIDx    = fbuf.bufI(FPARTICLEIDX)[i*2*BONDS_PER_PARTICLE + bond*2 +1];
        
        
        if ( incomingParticleIdx < fparam.maxPoints  && incomingParticleBondIDx < BONDS_PER_PARTICLE ) {                                // chk vs null bonds or particles
            float stress = fbuf.bufF(FELASTIDX)[incomingParticleIdx*BOND_DATA +  incomingParticleBondIDx*DATA_PER_BOND + 7]; 
            if (stress>maxStress){
                maxStress = stress;
                maxStressBondIdx = bond; 
            }
        }
    }
    
    
    printf("\nassembleMuscleFibresInComing() chk4: i=%u  maxStressBondIdx=%u ",i ,maxStressBondIdx );
    if (maxStressBondIdx!=0 ){
        // store high stress bond
        incomingParticleIdx        = fbuf.bufI(FPARTICLEIDX)[i*2*BONDS_PER_PARTICLE + maxStressBondIdx*2];
        incomingParticleBondIDx    = fbuf.bufI(FPARTICLEIDX)[i*2*BONDS_PER_PARTICLE + maxStressBondIdx*2 +1];
        
        printf("\nassembleMuscleFibresInComing() chk4.1: i=%u  incomingParticleIdx=%u,  incomingParticleBondIDx=%u ",i ,incomingParticleIdx, incomingParticleBondIDx );
        
        uint lowStressIncomingParticleIdx          = fbuf.bufI(FPARTICLEIDX)[i*2*BONDS_PER_PARTICLE ];
        uint lowStressIncomingParticleBondIDx      = fbuf.bufI(FPARTICLEIDX)[i*2*BONDS_PER_PARTICLE +1];
        
        printf("\nassembleMuscleFibresInComing() chk4.2: i=%u  lowStressIncomingParticleIdx=%u,  lowStressIncomingParticleBondIDx=%u ",i ,lowStressIncomingParticleIdx, lowStressIncomingParticleBondIDx );
        
        // move low stress bond
        fbuf.bufI(FPARTICLEIDX)[i*2*BONDS_PER_PARTICLE + maxStressBondIdx*2]       =  lowStressIncomingParticleIdx;
        fbuf.bufI(FPARTICLEIDX)[i*2*BONDS_PER_PARTICLE + maxStressBondIdx*2 +1]    =  lowStressIncomingParticleBondIDx;
         
        // update reciprocal record
        if(lowStressIncomingParticleIdx<fparam.maxPoints && lowStressIncomingParticleBondIDx<BONDS_PER_PARTICLE){
            fbuf.bufF(FELASTIDX)[lowStressIncomingParticleIdx*BOND_DATA +  lowStressIncomingParticleBondIDx*DATA_PER_BOND + 6] =  maxStressBondIdx;
        }
        
        
        // write high stress bond
        //if(incomingParticleIdx>fparam.maxPoints || incomingParticleBondIDx>BONDS_PER_PARTICLE){
        //    incomingParticleIdx=UINT_MAX;
        //    incomingParticleBondIDx=UINT_MAX;
        //}
        fbuf.bufI(FPARTICLEIDX)[i*2*BONDS_PER_PARTICLE ]      =  incomingParticleIdx;
        fbuf.bufI(FPARTICLEIDX)[i*2*BONDS_PER_PARTICLE +1]    =  incomingParticleBondIDx;
        
       
        // update reciprocal record
        if(incomingParticleIdx<fparam.maxPoints && incomingParticleBondIDx<BONDS_PER_PARTICLE){
            fbuf.bufF(FELASTIDX)[incomingParticleIdx*BOND_DATA +  incomingParticleBondIDx*DATA_PER_BOND + 6] =  0;
        }
    }
    printf("\nassembleMuscleFibresInComing() chk5: i=%u   ",i  );
    
    /////////// connect contractile fibres (bond[1])  // replace bond[1], and leave other particles to heal.
    // if (tendon) return;
    // cudaThreadSynchronize(); // is this correct ?
    
    
    
    
    
    // if(muscle) connect nerves
    // if (elastic ligg) return;
    
    
    
    
    
}


extern "C" __global__ void assembleFasciaFibres ( int pnum, uint list, uint list_length ) // used for muscle and elastic ligg
{
    uint particle_index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                            // particle index
    if ( particle_index >= list_length )return;
    uint i = fbuf.bufII(FDENSE_LISTS)[list][particle_index];
    if (i > fparam.maxPoints) return;
    
    float maxStress0         = 0.0;
    uint maxStressBondIdx0   = 0;
    float maxStress1         = 0.0;
    uint maxStressBondIdx1   = 0;
    float stress0 = 0.0, stress1 = 0.0;
    char* bond_char_ptr = &fbuf.bufC(FELASTIDX)[i*BOND_DATA];
    uint* bond_uint_ptr = &fbuf.bufI(FELASTIDX)[i*BOND_DATA];                                       //fbuf.bufI(FELASTIDX)[i*BOND_DATA + bond*DATA_PER_BOND +  ] ;
    float*bond_flt_ptr  = &fbuf.bufF(FELASTIDX)[i*BOND_DATA];                                       //FELASTIDX   [0]current index, [1]elastic limit, [2]restlength, [3]modulus,
                                                                                                                //[4]damping coeff, [5]particle ID,   [6]bond index, 
                                                                                                                //[7]stress integrator,  [8]change-type binary indicator
    for(int bond=0; bond<BONDS_PER_PARTICLE; bond++){                                               // find highest stress bond
        float stress = bond_flt_ptr[7 + bond*DATA_PER_BOND] ;
        if (stress>maxStress0){
            maxStress0 = stress;
            maxStressBondIdx0 = bond;
            
            
        }else if (stress>maxStress1){
            maxStress1 = stress;
            maxStressBondIdx1 = bond;
        
        }
        
    }
    //  if (maxStressBondIdx0==0) return;
    // Swap bond indicies 
    // store high stress bond
    uint bytes = sizeof(uint)*DATA_PER_BOND;
    char temp[DATA_PER_BOND*4] = {0};                       // NB sensitive to size of uint and float
    uint  bondStep = maxStressBondIdx0*DATA_PER_BOND;
    
    memcpy(&bond_char_ptr[bondStep], temp, bytes);
    /*
    uint  currIdx           = bond_uint_ptr[0 + bondStep];  // could be done faster with a memcpy()  of bytes to a void or char pointer
    float elastLim          = bond_flt_ptr [1 + bondStep];
    float restLength        = bond_flt_ptr [2 + bondStep];
    float modulus           = bond_flt_ptr [3 + bondStep];
    float dampingCoeff      = bond_flt_ptr [4 + bondStep];
    uint  particleID        = bond_uint_ptr[5 + bondStep]; 
    uint  bondIndex         = bond_uint_ptr[6 + bondStep]; 
    float stressIntegrator  = bond_flt_ptr [7 + bondStep];
    uint  changeIndicator   = bond_uint_ptr[8 + bondStep];
    */
    
    // move low stress bond
    memcpy(bond_char_ptr, &bond_char_ptr[bondStep*sizeof(uint)], bytes);
    /*
    bond_uint_ptr[0 + bondStep] = bond_uint_ptr[0] ;
    bond_flt_ptr [1 + bondStep] = bond_flt_ptr [1];
    bond_flt_ptr [2 + bondStep] = bond_flt_ptr [2];
    bond_flt_ptr [3 + bondStep] = bond_flt_ptr [3];  // change modulus elastic fibre type
    bond_flt_ptr [4 + bondStep] = bond_flt_ptr [4];
    bond_uint_ptr[5 + bondStep] = bond_uint_ptr[5]; 
    bond_uint_ptr[6 + bondStep] = bond_uint_ptr[6]; 
    bond_flt_ptr [7 + bondStep] = bond_flt_ptr [7];
    bond_uint_ptr[8 + bondStep] = bond_uint_ptr[8];
    */
//    bond_flt_ptr [3 + bondStep] =    ;  // change modulus elastic fibre type
    // update reciprocal record
    
    
    
    // write high stress bond
    memcpy(temp, bond_char_ptr, bytes);
    /*
    bond_uint_ptr[0] = currIdx;
    bond_flt_ptr [1] = elastLim;
    bond_flt_ptr [2] = restLength;
    bond_flt_ptr [3] = modulus;                 // change modulus collagen fibre type
    bond_flt_ptr [4] = dampingCoeff;
    bond_uint_ptr[5] = particleID; 
    bond_uint_ptr[6] = bondIndex; 
    bond_flt_ptr [7] = stressIntegrator;
    bond_uint_ptr[8] = changeIndicator;
    */
//    bond_flt_ptr [3] =  ;   // change modulus to  collagen fibre type. Leave elastlim and restlength to computeBondChanges().
    // update reciprocal record
    
    
    
    // again for 2nd fibre
    
}


extern "C" __global__ void computeBondChanges ( int pnum, uint list_length )// Given the action of the genes, compute the changes to particle properties & splitting/combining 
{                                                                                                   // Also "inserts changes" 
    uint particle_index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                            // particle index
    if ( particle_index >= list_length ) {/*if (fparam.debug>2)printf("\tcomputeBondChanges:particle_index %u>= %u list_length.\t",particle_index, list_length);*/ return;}                                                    // pnum should be length of list.
    // ## TODO convert i to particle index _iff_ not called for all particles : use a special dense list for "living tissue", made at same time as gene lists
    uint i = fbuf.bufII(FDENSE_LISTS)[2][particle_index];                                           // call for dense list of living cells (gene'2'living/telomere (has genes))
    //if ( i >= pnum || i==0 ) {printf("\tcomputeBondChanges:i %u>=%u pnum\t",i,pnum);   return;} 
    
    if ( i >= pnum ) {printf("\tcomputeBondChanges:i %u>=%u pnum\t",i,pnum);   return;} 
    if ( i==0 ) {printf("\tcomputeBondChanges:i=%u,  pnum=%u, fparam.maxPoints=%u \t",i, pnum, fparam.maxPoints);}

    float * fbufFCONC = &fbuf.bufF(FCONC)[i*NUM_TF];
    //float * ftempFCONC = &ftemp.bufF(FCONC)[i*NUM_TF];
    uint  * fbufFEPIGEN = &fbuf.bufI(FEPIGEN)[i];   /*  *NUM_GENES  */                              // TODO FEPIGEN is a uint here. May need to pack binaries for spread & stop. See paper.
    //uint  * ftempFEPIGEN = &ftemp.bufI(FEPIGEN)[i*NUM_GENES];    // ## need to zero ftemp after counting sort full
    
    //for(int j=0; j<NUM_TF;j++)      fbufFCONC[j] += ftempFCONC[j];                                  // list of transcription factor conc for this particle
    //for(int j=0; j<NUM_GENES;j++) fbufFEPIGEN[j] += ftempFEPIGEN[j];                                // list of epigenetic activations for this particle 
                                                                                                    // NB modification were writtent to ftemp, now added to fbuf here.
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
    uint* bond_uint_ptr = &fbuf.bufI(FELASTIDX)[i*BOND_DATA];                                       //fbuf.bufI(FELASTIDX)[i*BOND_DATA + bond*DATA_PER_BOND +  ] ;
    float*bond_flt_ptr  = &fbuf.bufF(FELASTIDX)[i*BOND_DATA];                                       //FELASTIDX   [0]current index, [1]elastic limit, [2]restlength, [3]modulus,
                                                                                                                //[4]damping coeff, [5]particle ID,   [6]bond index, 
                                                                                                                //[7]stress integrator,  [8]change-type binary indicator
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
    
    // 11 fixed particles
    // 12 externally actuated 
    */
    uint bond_type[BONDS_PER_PARTICLE] = {0};                                                       //  0=elastin, 1=collagen, 2=apatite
    // calculate material type for bond
    for (int bond=0; bond<BONDS_PER_PARTICLE; bond++) bond_type[bond] = 2*(fbufFEPIGEN[9*fparam.maxPoints]>0/*bone*/);
    if (fbufFEPIGEN[6*fparam.maxPoints]>0/*tendon*/||fbufFEPIGEN[7*fparam.maxPoints]>0/*muscle*/||fbufFEPIGEN[10*fparam.maxPoints]>0/*elast lig*/) {bond_type[0] = 1; bond_type[3] = 1;}
    for (int bond=0; bond<BONDS_PER_PARTICLE; bond++) bond_type[bond] = 1*(fbufFEPIGEN[6*fparam.maxPoints]>0/*cartilage*/);
    
    //if (fparam.debug>2 && i%1000==0)printf(",%u,",i);
    
    for (uint bond=0; bond<BONDS_PER_PARTICLE;bond++, bond_uint_ptr+=DATA_PER_BOND, bond_flt_ptr+=DATA_PER_BOND ){
        if (bond_flt_ptr[2]>0){                                                                     // NB (rest_length==0) => bond broken, do not modify.
            float strain_integrator = bond_flt_ptr[7];
            float * param_ptr = fgenome.param[bond_type[bond]];
            float restln_multiplier   = (strain_integrator - param_ptr[fgenome.elongation_threshold]) * param_ptr[fgenome.elongation_factor];
            float strength_multiplier = (strain_integrator - param_ptr[fgenome.strength_threshold])   * param_ptr[fgenome.strengthening_factor];
            float integ_elong_thresh = (strain_integrator - param_ptr[fgenome.elongation_threshold]);
            float integ_stren_thresh = (strain_integrator - param_ptr[fgenome.strength_threshold]);
            
            bond_flt_ptr[2]/*rest length*/ +=  bond_flt_ptr[2] * (strain_integrator - param_ptr[fgenome.elongation_threshold]) * param_ptr[fgenome.elongation_factor];
            bond_flt_ptr[3]/*modulus*/     +=  bond_flt_ptr[3] * (strain_integrator - param_ptr[fgenome.strength_threshold])   * param_ptr[fgenome.strengthening_factor];
            /*
            if (fparam.debug>2 && fbuf.bufI(FPARTICLE_ID)[i]<10){
                //printf("\ncomputeBondChanges(): ParticleID=%u,  bond=%u, bond_type=%u, fbufFEPIGEN[9]=%2.2f, [6]=%2.2f, [7]=%2.2f, [10]=%2.2f,  rest_length=%f,  modulus=%f\t, strain_integrator=%f, elongation_threshold=%f,\t integ-elong_thresh=%f elongation_factor=%f, \t restln_multiplier=%f \t\t strength_threshold=%f, integ_stren_thresh=%f, strengthening_factor=%f, strength_multiplier=%f",
                //   fbuf.bufI(FPARTICLE_ID)[i], bond, bond_type[bond], fbufFEPIGEN[9], fbufFEPIGEN[6], fbufFEPIGEN[7], fbufFEPIGEN[10], bond_flt_ptr[2], bond_flt_ptr[3], 
                //       strain_integrator, param_ptr[fgenome.elongation_threshold], integ_elong_thresh, param_ptr[fgenome.elongation_factor], restln_multiplier, param_ptr[fgenome.strength_threshold], integ_stren_thresh, param_ptr[fgenome.strengthening_factor], strength_multiplier );
            }
            */
        }
        // "insert changes"
        uint * fbufFGRIDCNT_CHANGES = fbuf.bufI(FGRIDCNT_CHANGES);
        int m = 1 + ((bond==0)&&(fbufFEPIGEN[7*fparam.maxPoints]>0/*muscle*/||fbufFEPIGEN[10*fparam.maxPoints]>0));        
                                                                                                    // i.e. if (bond==0 && fbufFEPIGEN[7]>0/*muscle*/) m=2 else m=1;
                                                                                                    // NB two different lists for each change, for (muscle & elastic ligg  vs other tissues)
        bond_uint_ptr[8]=0;                                                                         // Need to zero the indicator.
        /*
        //if (fparam.debug>2 && i%1000==0)if(bond_flt_ptr[2]!=0.0)printf(",");  //("\tcomputeBondChanges:(bond_flt_ptr[2]!=0.0): =i%u \t", i);
        //if (fparam.debug>2 && i%1000==0)if(!(bond < 3 || fbufFEPIGEN[8]>0 ||  fbufFEPIGEN[9]>0))printf("'");   //("\tcomputeBondChanges:!(bond < 3 || fbufFEPIGEN[8]>0 ||  fbufFEPIGEN[9]>0): =i%u \t", i);   
        */
        // NB heal all bonds as if mesenchyme, then remodel later. This is needed to hold tissue together.
        if (bond_flt_ptr[2]==0.0 /*&& (bond < 3 || fbufFEPIGEN[8]>0 ||  fbufFEPIGEN[9]>0/_*cartilage OR bone*_/)*/  ){  // bond_flt_ptr[2]=restlength==0.0 => bond broken 
           /*   
            // && bond_uint_ptr[0]/_*other particle*_/<pnum/_*bond broken*_/
            //TODO what happens when bond broken vs never existed ?  NB information about direction of broken bond.
             if (fparam.debug>2 && i<10  &&  fbufFGRIDCNT_CHANGES[0*gridTot+fbuf.bufI(FGCELL)[i]]<10  )
                printf("\ncomputeBondChanges()1: i=,%u, particle_index=,%u,  bond_uint_ptr[8]=,%u, fbufFGRIDCNT_CHANGES=,%u, address=,%p, ",
                    i, particle_index , bond_uint_ptr[8], fbufFGRIDCNT_CHANGES[ 0*gridTot  + fbuf.bufI(FGCELL)[i] ],
                    &fbuf.bufII(FDENSE_LISTS)[2][particle_index]
                );
            
            //if (fparam.debug>2)printf(".");
            */
            atomicAdd ( &fbufFGRIDCNT_CHANGES[ 0*gridTot  + fbuf.bufI(FGCELL)[i] ], 1 );            //add to heal list //NB device-wide atomic
            bond_uint_ptr[8]+=1;                                                                    // FELASTIDX [8]change-type binary indicator NB accumulates all changes for this bond
            
            if (bond>BONDS_PER_PARTICLE)//(fbuf.bufI(FPARTICLE_ID)[i]<10) 
                printf("\nError :computeBondChanges:add to heal list: i=%u, ParticleID=%u, bond=%u, bond_uint_ptr[0]=%u, fbufFEPIGEN[8*fparam.maxPoints]=%u, fbufFEPIGEN[9*fparam.maxPoints]=%u "
                ,i,fbuf.bufI(FPARTICLE_ID)[i],bond,bond_uint_ptr[0],fbufFEPIGEN[8*fparam.maxPoints],fbufFEPIGEN[9*fparam.maxPoints]);
            /*
            if(fbufFGRIDCNT_CHANGES[0*gridTot+fbuf.bufI(FGCELL)[i]]<50  && i<10)
                printf("\ncomputeBondChanges()2: i=%u, particle_index=%u,  bond_uint_ptr[8]=%u, fbufFGRIDCNT_CHANGES=%u ",
                    i, particle_index , bond_uint_ptr[8], fbufFGRIDCNT_CHANGES[ 0*gridTot  + fbuf.bufI(FGCELL)[i] ]);
                
          //  if (fparam.debug>2 && i==0)printf("\ncomputeBondChanges()2: i==0, particle_index=%u,  bond_uint_ptr[8]=%u, fbufFGRIDCNT_CHANGES=%u ",
          //      particle_index , bond_uint_ptr[8], fbufFGRIDCNT_CHANGES[ 0*gridTot  + fbuf.bufI(FGCELL)[i] ])
          */
            break;                                                                                  // First, heal one bond per timestep. Remodel only after freeze.
        }else if(fparam.freeze==false){                                                                                      // prevents clash with heal.
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
                bond_uint_ptr[8]+=128*m;
            }
        }
        // bond_uint_ptr[8]+=2^n; is ELASTIDX for binary change indicator per bond. 
    }
}

//////   Particle modification kernels called together. Must make sure that they cannot clash. NB atomic operations. 
extern "C" __device__ void addParticle (uint parent_Idx, uint &new_particle_Idx)                    // Template for stregthening & lengthening kernels
{   
    //printf("\naddParticle()1:  parent_Idx=%u, new_particle_Idx=%u, fbuf.bufI(FPARTICLE_ID)[new_particle_Idx]=%u", parent_Idx, new_particle_Idx, fbuf.bufI(FPARTICLE_ID)[new_particle_Idx] );
    
    atomicCAS(&fbuf.bufI(FPARTICLE_ID)[new_particle_Idx /*_otherParticleBondIndex*/], UINT_MAX, parent_Idx);
    if(fbuf.bufI(FPARTICLE_ID)[new_particle_Idx]==parent_Idx){// TODO set a unique particle ID. 
    
    //int particle_Idx = atomicAdd(&fparam.pnumActive, 1);                              // fparam.pnumActive = mActivePoints from PrefixSumCellsCUDA, set in CountingSortFullCUDA
                                                                                      // NOT safe to use fbuf.bufI(FGRIDOFF)[fparam.gridTotal] as active particle count!
        //if (fparam.debug>2)printf("\naddParticle()2:  parent_Idx=%u, new_particle_Idx=%u", parent_Idx, new_particle_Idx);
    
    //if (particle_Idx >= 0  &&  particle_Idx < fparam.pnum) {
    //    new_particle_Idx                            = particle_Idx;
        fbuf.bufF3(FVEVAL)[new_particle_Idx]        = fbuf.bufF3(FVEVAL)[parent_Idx]; // NB could use average with next row. Prob not needed, because old bond is stretched.
        fbuf.bufF3(FVEL)[new_particle_Idx]          = fbuf.bufF3(FVEL)[parent_Idx];
        fbuf.bufF3(FFORCE)[new_particle_Idx]        = fbuf.bufF3(FFORCE)[parent_Idx];
        fbuf.bufI(FMASS_RADIUS)[new_particle_Idx]   = fbuf.bufI(FMASS_RADIUS)[parent_Idx];
        fbuf.bufI(FAGE)[new_particle_Idx]           = fparam.frame;
        fbuf.bufI(FCLR)[new_particle_Idx]           = fbuf.bufI(FCLR)[parent_Idx];
        fbuf.bufI(FNERVEIDX)[new_particle_Idx]      = fbuf.bufI(FNERVEIDX)[parent_Idx];
        
        //for (int tf=0;tf<NUM_TF;tf++)                   fbuf.bufF(FCONC)[new_particle_Idx*NUM_TF+tf]          = fbuf.bufF(FCONC)[parent_Idx*NUM_TF+tf];
        float* fbuf_conc  = &fbuf.bufF(FCONC)[new_particle_Idx * NUM_TF];
        float* fbuf_parent_conc = &fbuf.bufF(FCONC)[parent_Idx * NUM_TF];
        for (int a=0;a<NUM_TF;a++)     fbuf_conc[a] = fbuf_parent_conc[a]; 
        
        //for (int gene=0;gene<NUM_GENES;gene++)          fbuf.bufI(FEPIGEN)[new_particle_Idx*NUM_GENES+gene]   = fbuf.bufI(FEPIGEN)[parent_Idx*NUM_GENES+gene];
        uint* fbuf_epigen  = &fbuf.bufI(FEPIGEN)[new_particle_Idx];
        uint* fbuf_parent_epigen = &fbuf.bufI(FEPIGEN)[parent_Idx];
        for (int a=0;a<NUM_GENES;a++)  fbuf_epigen[fparam.maxPoints*a]  = fbuf_parent_epigen[fparam.maxPoints*a];
        
        //if (fparam.debug>2)printf("\naddParticle()3:  parent_Idx=%u, new_particle_Idx=%u, fbuf.bufI(FAGE)[new_particle_Idx]=%u,  fparam.maxPoints=%u, \"muscle\"=fbuf.bufI(FEPIGEN)[new_particle_Idx+7*fparam.maxPoints]=%u ", 
        //    parent_Idx, new_particle_Idx, fbuf.bufI(FAGE)[new_particle_Idx], fparam.maxPoints, fbuf.bufI(FEPIGEN)[new_particle_Idx+7*fparam.maxPoints]
        //);
        // TODO should FEPIGEN be float, int or uint?
    } else new_particle_Idx=UINT_MAX;               // else failed.
    __syncwarp;
    //if (fparam.debug>2)printf("\naddParticle()4:  parent_Idx=%u  ", parent_Idx);
    //__syncthreads;
}

extern "C" __device__ void removeParticle (uint particle_Idx)                                                       // Template for weakening & shortening kernels
{   //  active particle count : done automatically by insert_particles(..)
    //  sets values to null particle, => will be sorted to reserve section of particle list in next time step.
    if (fparam.debug>2)printf("\nremoveParticle() particle_Idx=%u \t",particle_Idx);
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
    
    uint *ptr_epigen = &fbuf.bufI(FEPIGEN)[particle_Idx];                                                 // Zero FEPIGEN
    for (int gene=0;gene<NUM_GENES;gene++)  ptr_epigen[gene*fparam.maxPoints]=0;
    
    float *ptr_tf = &fbuf.bufF(FCONC)[particle_Idx*NUM_TF];                                                         // Zero FCONC
    for (int tf=0;tf<NUM_TF;tf++) ptr_tf[tf]=0;
}

extern "C" __device__ void find_potential_bonds (int i, float3 ipos, int cell, uint _bonds[BONDS_PER_PARTICLE][2], float _bond_dsq[BONDS_PER_PARTICLE], float max_len_sq)
{                                                                                                           // Triangulated cubic bond selection...
	if ( fbuf.bufI(FGRIDCNT)[cell] == 0 ) return;                                                           // If the cell is empty, skip it.
	float dsq;//, sdist;//, c;
	float3 dist = make_float3(0,0,0), eterm  = make_float3(0,0,0), force = make_float3(0,0,0);
	uint j;
	int clast = fbuf.bufI(FGRIDOFF)[cell] + fbuf.bufI(FGRIDCNT)[cell];                                      // index of last particle in this cell
    for ( int cndx = fbuf.bufI(FGRIDOFF)[cell]; cndx < clast; cndx++ ) {                                    // For particles in this cell.
		j = fbuf.bufI(FGRID)[ cndx ];
		dist = ( ipos - fbuf.bufF3(FPOS)[ j ] );                                                            // dist in cm (Rama's comment)
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);                                              // scalar distance squared
		if ( dsq < max_len_sq && dsq > 0) {                                                                 // IF in-range && not the same particle
            //sdist = sqrt(dsq * fparam.d2);                                                                // smoothing distance = sqrt(dist^2 * sim_scale^2))
			//c = ( fparam.psmoothradius - sdist ); 
            bool known = false;
            uint bond_index = UINT_MAX;
            for (int a=0; a<BONDS_PER_PARTICLE; a++){                                                       // chk if known, i.e. already bonded 
                    if (fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + a*2] == i        ) known = true;   // particle 'j' has a bond to particle 'i'
                    if (fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + a*2] == UINT_MAX ) bond_index = a; // particle 'j' has an empty bond 'a' : picks last empty bond
                    if (_bonds[a][0] == j )known = true; // needed?                                         // particle 'i' already has a bond to particle 'j'  
                                                                                                            // not req?, _bonds starts empty && only touch 'j' once
            }
            if (known == false && bond_index<UINT_MAX){       
                    //int bond_direction = 1*(dist.x-dist.y+dist.z>0.0) + 2*(dist.x+dist.y-dist.z>0.0);     // booleans divide bond space into quadrants of x>0.
                    float approx_zero    = 0.02*fparam.rd2;
                    int   bond_direction = ((dist.x+dist.y+dist.z)>0) * (1*(dist.x*dist.x>approx_zero) + 2*(dist.y*dist.y>approx_zero) + 4*(dist.z*dist.z>approx_zero)) -1; 
                                                                                                            // booleans select +ve quadrant x,y,z axes and their planar diagonals
                    //if (fparam.debug>2)printf("\ni=%u, bond_direction=%i, dist=(%f,%f,%f), dsq=%f, approx_zero=%f", i, bond_direction, dist.x, dist.y, dist.z, dsq, approx_zero);
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


extern "C" __global__ void initialize_FCURAND_STATE (int pnum)  // designed to use to bootstrap itself. Set j=0 from host, call kernel repeatedly for 256^n threads, n=0-> to pnum threads.
{
    unsigned long long sequence=0, offset=1;//, seed=0;
    uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                 // particle index
    /*
    if(pnum==0 && i==0){ 
        seed = clock64();
        if (fparam.debug>2)printf("\ninitialize_FCURAND_STATE(): seed=%llu,\t  &fbuf.bufCuRNDST(FCURAND_STATE)[0]=%p .\t",seed,&fbuf.bufCuRNDST(FCURAND_STATE)[0]);  // getting (nil) a null pointer.
        curand_init(seed, sequence, offset, &fbuf.bufCuRNDST(FCURAND_STATE)[0]);
    }
    */
	if ( i >= pnum ) return;
    /*
    int j = i/256;
    // atomic lock, to ensure separate random numbers
    seed = curand(&fbuf.bufCuRNDST(FCURAND_STATE)[j]);
    
    seed = seed << 32; 
    seed += curand(&fbuf.bufCuRNDST(FCURAND_STATE)[j]);
    */
    curand_init(fbuf.bufI(FCURAND_SEED)[i], sequence, offset, &fbuf.bufCuRNDST(FCURAND_STATE)[i]);
    uint rnd_nm=curand(&fbuf.bufCuRNDST(FCURAND_STATE)[i]);
    
    //if (fparam.debug>2)printf("\n(i=%u,seed=%i, &fbuf.bufCuRNDST(FCURAND_STATE)[i]=%p, rnd_nm=%u),",i,fbuf.bufI(FCURAND_SEED)[i] , &fbuf.bufCuRNDST(FCURAND_STATE)[i], rnd_nm);
}


extern "C" __device__ void find_potential_bond (int i, float3 ipos, uint _thisParticleBonds[BONDS_PER_PARTICLE], float3 tpos, int gc, uint &_otherParticleIdx, uint &_otherParticleBondIdx, float &_bond_dsq, float max_len)                                                                                      // Used when just one bond, near a target location "tpos" is sought.
{
    int nadj = (1*fparam.gridRes.z + 1)*fparam.gridRes.x + 1;
    gc -= nadj; 
    int cell;
    _bond_dsq=max_len*max_len;//FLT_MAX; Better to use breaking length.  (FELASTIDX)[1]elastic limit : depends on type of new bond. Could set _bond_dsq when calling find_potential_bond().
    float max_len_sq = max_len*max_len;
    uint rnd_nmbr = curand(&fbuf.bufCuRNDST(FCURAND_STATE)[i]);                                                 // NB bitshift and mask to get rand bool to choose bond
    /*
    float3 old_tpos=tpos;
    */
    tpos.x += max_len/float(4+(rnd_nmbr&7))     *(-1*float(1&(rnd_nmbr>>3))  );                                 // shift tpos by a random step < max_len, randomises bond.
    tpos.y += max_len/float(4+((rnd_nmbr>>4)&7))*(-1*float(1&(rnd_nmbr>>7))  );
    tpos.z += max_len/float(4+((rnd_nmbr>>8)&7))*(-1*float(1&(rnd_nmbr>>11)) );
/*
    //printf("\ni=%u, &fbuf.bufCuRNDST(FCURAND_STATE)[i]=%p, rnd_nmbr=%u, (1&(rnd_nmbr>>12)=%u, (-1*int(1&(rnd_nmbr>>12))=%d,  float(-1*int(rnd_nmbr&64)=%f  ",i, &fbuf.bufCuRNDST(FCURAND_STATE)[i], rnd_nmbr, 1&(rnd_nmbr>>12), -1*int(1&(rnd_nmbr>>12)), (-1*float(rnd_nmbr&64))  );  // (-1*(rnd_nmbr&64),
*/
/*
     printf("\nold_tpos=(%f,%f,%f), tpos=(%f,%f,%f), max_len=%f.\trnd_nmbr=%u, \trnd_nmbr&7=%u, \t(1&(rnd_nmbr>>3)=%u, \tfloat(1&(rnd_nmbr>>3))*2-1=%f, \t(4+((rnd_nmbr<<3)&7))*(float(1&(rnd_nmbr>>3))*2-1)=%f", 
           old_tpos.x,old_tpos.y,old_tpos.z, tpos.x,tpos.y,tpos.z, max_len, 
           rnd_nmbr, rnd_nmbr&7, 1&(rnd_nmbr>>3), float(1&(rnd_nmbr>>3))*2-1, float(4+(rnd_nmbr&7))*(float(1&(rnd_nmbr>>3))*2-1)
          );
*/
    for (int c=0; c < fparam.gridAdjCnt; c++) { 
        cell = gc + fparam.gridAdj[c];
        float dsq;
        float3 dist = make_float3(0,0,0);
        uint j;
        int clast = fbuf.bufI(FGRIDOFF)[cell] + fbuf.bufI(FGRIDCNT)[cell];                                      // index of last particle in this cell
        for ( int cndx = fbuf.bufI(FGRIDOFF)[cell]; cndx < clast; cndx++ ) {                                    // For particles in this cell.
            j = fbuf.bufI(FGRID)[ cndx ];
            dist = ( ipos - fbuf.bufF3(FPOS)[ j ] );                                                            // dist in cm (Rama's comment)
            dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);                                              // scalar distance squared
            if ( dsq < max_len_sq && dsq > 0) {  // probably wasteful, if tpos is in range.                     // IF in-range && not the same particle
                dist = ( tpos - fbuf.bufF3(FPOS)[ j ] );                                                        // dist to target location
                dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);                                          // scalar distance squared
                if(dsq<_bond_dsq){                                                                              // If closer to tpos than current candidate
                    bool known      = false;
                    uint bond_index = UINT_MAX;
                    for (int a=0; a<BONDS_PER_PARTICLE; a++){                                                   // chk if known, i.e. already bonded 
                        if (fbuf.bufI(FELASTIDX)[i*BOND_DATA*2 + a*DATA_PER_BOND] == j   ) known = true;
                        //if (fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + a*2] == i        ) known = true;   // particle 'j' has a bond to particle 'i'
                        /*if (fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + a*2] == UINT_MAX )*/ bond_index = a; // particle 'j' has an empty bond 'a' : picks last empty bond
                        if (_thisParticleBonds[a] == j )known = true;                                           // particle 'i' already has a bond to particle 'j'  
                    }
                    if (known == false && bond_index<UINT_MAX){
                        _otherParticleIdx = j;                                                                  // index of other particle
                        _otherParticleBondIdx = bond_index;                                                     // FPARTICLEIDX vacancy index of other particle
                        _bond_dsq = dsq;                                                                        // scalar distance squared 
                    }
                }                                                                                               // end of collect potential bonds
            }                                                                                                   // end of: IF in-range && not the same particle
        }                                                                                                       // end of loop round particles in this cell
    }
}


extern "C" __device__ void breakBond (uint thisParticleIdx, uint otherParticleIdx, uint bondIdx, uint otherParticleBondIdx){
    //FBondParams *params_ =  &fgenome.fbondparams[bondType];
    uint*   uint_ptr = &fbuf.bufI(FELASTIDX)[thisParticleIdx*BOND_DATA + bondIdx*DATA_PER_BOND];
    float* float_ptr = &fbuf.bufF(FELASTIDX)[thisParticleIdx*BOND_DATA + bondIdx*DATA_PER_BOND];
    uint_ptr [0]    = UINT_MAX;                                                                             //[0]current index, 
    float_ptr[1]    = 0.0;                                                                                  //[1]elastic limit, 
    float_ptr[2]    = 0.0;                                                                                  //[2]restlength, 
    float_ptr[3]    = 0.0;                                                                                  //[3]modulus, 
    float_ptr[4]    = 0.0;                                                                                  //[4]damping coeff, 
    uint_ptr [5]    = UINT_MAX;                                                                             //[5]particle ID,   
    uint_ptr [6]    = UINT_MAX;                                                                             //[6]bond index 
    float_ptr[7]    = 0;                                                                                    //[7]stress integrator 
    uint_ptr [8]    = 0;                                                                                    //[8]change-type 
                                                                                                            // Connect new particle incoming bonds
    fbuf.bufI(FPARTICLEIDX)[otherParticleIdx*2*BONDS_PER_PARTICLE + otherParticleBondIdx*2]       = UINT_MAX;                  // particle Idx
    fbuf.bufI(FPARTICLEIDX)[otherParticleIdx*2*BONDS_PER_PARTICLE + otherParticleBondIdx*2 +1]    = UINT_MAX;                  // bond Idx 
    
    /*if(thisParticleIdx<20) printf("\nmakeBond: bondtype=%u, default_rest_length=%f, %f, thisParticleIdx=%u, otherParticleIdx=%u, otherParticleBondIdx=%u  \t", 
           bondType, fgenome.param[bondType][fgenome.default_rest_length],  fbuf.bufF(FELASTIDX)[thisParticleIdx*BOND_DATA + bondIdx*DATA_PER_BOND +2], thisParticleIdx, otherParticleIdx, otherParticleBondIdx );
    */
}

extern "C" __device__ void makeBond (uint thisParticleIdx, uint otherParticleIdx, uint bondIdx, uint otherParticleBondIdx, uint bondType /* elastin, collagen, apatite */){
    //FBondParams *params_ =  &fgenome.fbondparams[bondType];
    uint*   uint_ptr = &fbuf.bufI(FELASTIDX)[thisParticleIdx*BOND_DATA + bondIdx*DATA_PER_BOND];
    float* float_ptr = &fbuf.bufF(FELASTIDX)[thisParticleIdx*BOND_DATA + bondIdx*DATA_PER_BOND];
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
    fbuf.bufI(FPARTICLEIDX)[otherParticleIdx*2*BONDS_PER_PARTICLE + otherParticleBondIdx*2]       = thisParticleIdx;           // particle Idx
    fbuf.bufI(FPARTICLEIDX)[otherParticleIdx*2*BONDS_PER_PARTICLE + otherParticleBondIdx*2 +1]    = bondIdx;                   // bond Idx 
    
    /*if(thisParticleIdx<20) printf("\nmakeBond: bondtype=%u, default_rest_length=%f, %f, thisParticleIdx=%u, otherParticleIdx=%u, otherParticleBondIdx=%u  \t", 
           bondType, fgenome.param[bondType][fgenome.default_rest_length],  fbuf.bufF(FELASTIDX)[thisParticleIdx*BOND_DATA + bondIdx*DATA_PER_BOND +2], thisParticleIdx, otherParticleIdx, otherParticleBondIdx );
    */
}


extern "C" __device__ int atomicMakeBond(uint thisParticleIndx,  uint otherParticleIdx, uint bondIdx, uint otherParticleBondIndex, uint bond_type){
    int _otherParticleBondIndex = otherParticleIdx*2*BONDS_PER_PARTICLE + otherParticleBondIndex*2;//BONDS_PER_PARTICLE*2*otherParticleIdx + 2*bondIdx;
{// debug
    //printf("\natomicMakeBond1: ftemp.bufI(FPARTICLEIDX)[_otherParticleBondIndex]=%u \t",ftemp.bufI(FPARTICLEIDX)[_otherParticleBondIndex]);
    //do {} while( atomicCAS(&fbuf.bufI(FPARTICLEIDX)[_otherParticleBondIndex], UINT_MAX, thisParticleIndx) );                                               // lock ////// ###### //  if (not locked) write zero to 'ftemp' to lock.
    
    //printf("\natomicMakeBond2: ftemp.bufI(FPARTICLEIDX)[_otherParticleBondIndex]=%u, \tfbuf.bufI(FPARTICLEIDX)[_otherParticleBondIndex]=%u \t"
    //    ,ftemp.bufI(FPARTICLEIDX)[_otherParticleBondIndex], fbuf.bufI(FPARTICLEIDX)[_otherParticleBondIndex]);
    
    /*if (fbuf.bufI(FPARTICLEIDX)[_otherParticleBondIndex]==UINT_MAX)*/  
    //fbuf.bufI(FPARTICLEIDX)[_otherParticleBondIndex]  = thisParticleIndx;                                   //  if (bond is unoccupied) write to 'fbuf' to assign this bond
    //ftemp.bufI(FPARTICLEIDX)[_otherParticleBondIndex] = UINT_MAX;                                                                            // release lock // ######
}
    atomicCAS(&fbuf.bufI(FPARTICLEIDX)[_otherParticleBondIndex], UINT_MAX, thisParticleIndx);
    if (fbuf.bufI(FPARTICLEIDX)[_otherParticleBondIndex] == thisParticleIndx){                                                               // if (this bond is assigned) write bond data
        makeBond ( thisParticleIndx, otherParticleIdx /*candidate_target_pIDx*/, bondIdx, otherParticleBondIndex, bond_type);
        return 0;
    }else return 1;
}

extern "C" __device__ int findBondAxis(float3 pos, uint j ){
    float3 dist     = ( pos - fbuf.bufF3(FPOS)[ j ] );                                                             // dist in cm (Rama's comment)                                   
    float distxsq   = dist.x*dist.x, distysq=dist.y*dist.y, distzsq=dist.z*dist.z;
    float dsq       = distxsq + distysq + distzsq;                                                                  // scalar distance squared
            
            //printf("\n(B:particle=%u,j=%u,dist=(%f,%f,%f),dsq=%f),", particle, j, dist.x, dist.y, dist.z, dsq );
    int axis =  1*(distxsq>distysq && distxsq>distzsq) + 2*(distysq>=distxsq && distysq>distzsq) +3*(distzsq>=distxsq && distzsq>=distysq);   
    if ((axis==1 && dist.x>0.0) || (axis==2 && dist.y>0.0) || (axis==3 && dist.z>0.0)) axis +=2; else axis--;    // sort by longest axis +/-ve 
    
    return axis;
}


extern "C" __device__ void find_closest_particle_per_axis(uint particle, float3 pos, uint neighbours[6]){       // Used by "insertNewParticle()"
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
	
	float neighbours_dsq[6] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};
    
    //printf("\n\n neighbours_dsq=(%f,%f,%f,%f,%f,%f)  \n", neighbours_dsq[0], neighbours_dsq[1], neighbours_dsq[2], neighbours_dsq[3], neighbours_dsq[4], neighbours_dsq[5]  );
	
	for (int c=0; c < fparam.gridAdjCnt; c++) {                                                                 // For adjacent cells
        uint cell = gs + fparam.gridAdj[c];
        if ( fbuf.bufI(FGRIDCNT)[cell] == 0 ) continue;                                                         // If the cell is empty, skip it.
        //printf(" (A:particle=%u,c=%u,cell=%u),",particle, c, cell);
        float dsq = FLT_MAX;
        float3 dist = make_float3(0,0,0);
        uint j = UINT_MAX;
        int clast = fbuf.bufI(FGRIDOFF)[cell] + fbuf.bufI(FGRIDCNT)[cell];                                      // index of last particle in this cell
        
        for ( int cndx = fbuf.bufI(FGRIDOFF)[cell]; cndx < clast; cndx++ ) {                                    // For particles in this cell.
            j = fbuf.bufI(FGRID)[ cndx ];
            if (j==particle)continue;
            
            dist = ( pos - fbuf.bufF3(FPOS)[ j ] );                                                             // dist in cm (Rama's comment)                                   
            float distxsq=dist.x*dist.x, distysq=dist.y*dist.y, distzsq=dist.z*dist.z;
            dsq = distxsq + distysq + distzsq;                                                                  // scalar distance squared
            
            //printf("\n(B:particle=%u,j=%u,dist=(%f,%f,%f),dsq=%f),", particle, j, dist.x, dist.y, dist.z, dsq );
            
            int axis =  1*(distxsq>distysq && distxsq>distzsq) + 2*(distysq>=distxsq && distysq>distzsq) +3*(distzsq>=distxsq && distzsq>=distysq);
            
            if ((axis==1 && dist.x>0.0) || (axis==2 && dist.y>0.0) || (axis==3 && dist.z>0.0)) axis +=2; else axis--;    // sort by longest axis +/-ve 
            
            //printf("\n(C:particle=%u, j=%u, axis=%i, dsq=%f, neighbours_dsq[axis]=%f ),",particle,j,axis,dsq,neighbours_dsq[axis] );
            if ( dsq>0 && dsq < neighbours_dsq[axis]) {                                                                  // IF in-range && not the same particle
                neighbours_dsq[axis] = dsq;
                neighbours[axis] = j;
                //printf("\n\n(D:particle=%u,dsq=%f,j=%u)\n",particle,dsq,j );
            }                                                                                                   // end of: IF in-range && not the same particle
        }                                                                                                       // end of loop round particles in this cell
    }
    //printf("\nfind_closest_particle_per_axis: particle=%u, pos=(%f,%f,%f), neighbours=(%u, %u, %u, %u, %u, %u), neighbours_dsq=(%f, %f, %f, %f, %f, %f)  ", 
    //       particle, pos.x,pos.y,pos.z, neighbours[0], neighbours[2], neighbours[2], neighbours[3], neighbours[4], neighbours[5], neighbours_dsq[0], neighbours_dsq[1], neighbours_dsq[2], neighbours_dsq[3], neighbours_dsq[4], neighbours_dsq[5]   );
}


extern "C" __device__ void find_bonds_to_redistribute(uint new_particle_Idx, float3 newParticlePos, uint neighbours[6], uint neighboursBondIdx[6], uint neighbours2[6]){
    float neighbours_dsq[6] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};                                                                        // Used by "insertNewParticle()"
    
    //if (fparam.debug>2 /*&& (threadIdx.x==0 || particle_index==0)*/ ) printf("\nfind_bonds_to_redistribute()1.0:  new_particle_Idx=%u, neighbours=(%u,%u,%u,%u,%u,%u) ", 
    //    new_particle_Idx, neighbours[0], neighbours[1], neighbours[2], neighbours[3], neighbours[4], neighbours[5]);
    
    for (int neighbour=0; neighbour<6;neighbour++){
        for (int bond =0; bond<BONDS_PER_PARTICLE; bond++){
            
            //printf("\nfind_bonds_to_redistribute()1.1:  neighbours[neighbour]=%u",neighbours[neighbour]);
            if(neighbours[neighbour]>fparam.maxPoints) continue;// not a valid particle
            uint otherParticle = fbuf.bufI(FELASTIDX)[neighbours[neighbour]*BOND_DATA + bond*DATA_PER_BOND];
            
            //if (fparam.debug>2 /*&& (threadIdx.x==0 || particle_index==0)*/ ) printf("\nfind_bonds_to_redistribute()1.2:  otherParticle=%u",otherParticle);
            if (otherParticle>fparam.maxPoints) continue;
            int chk =0;
            for (; chk<6; chk++) if (otherParticle==neighbours[chk] || otherParticle==neighbours2[chk]) chk =7; // not one of neighbours[6] or neighbours2[6]
            if (chk==7) continue;
            float3 dist = fbuf.bufF3(FPOS)[otherParticle] - newParticlePos ;
            
            float dsq = dist.x*dist.x+dist.y*dist.y+dist.z*dist.z;
            //if (fparam.debug>2  ) printf("\nfind_bonds_to_redistribute()1.3: otherParticle=%u, dsq=%f, neighbours_dsq[neighbour]=%f, neighbour=%u ", 
            //    otherParticle, dsq, neighbours_dsq[neighbour], neighbour);
            
            if (dsq < neighbours_dsq[neighbour]){
                neighbours_dsq[neighbour] = dsq;
                neighbours2[neighbour] = otherParticle;
                neighboursBondIdx[neighbour] = bond;
            }
        }
    }
    //if (fparam.debug>2 /*&& (threadIdx.x==0 || particle_index==0)*/ ) printf("\nfind_bonds_to_redistribute()1.4:  new_particle_Idx=%u,  neighbours2=(%u,%u,%u,%u,%u,%u)", 
    //    new_particle_Idx, neighbours2[0], neighbours2[1], neighbours2[2], neighbours2[3], neighbours2[4], neighbours2[5]);
}

extern "C" __device__ void makeBondIndxMap( uint parentParticleIndx, int bondInxMap[6]){ // A tractable way to approximately map rotation of bonds wrt the world frame.
    uint bond0otherPartlicleIdx = fbuf.bufI(FELASTIDX)[parentParticleIndx*BOND_DATA];                           // Used by "insertNewParticle()"
    uint bond1otherPartlicleIdx = fbuf.bufI(FELASTIDX)[parentParticleIndx*BOND_DATA+DATA_PER_BOND];
    uint bond2otherPartlicleIdx = fbuf.bufI(FELASTIDX)[parentParticleIndx*BOND_DATA+2*DATA_PER_BOND];
    float3 pos      = fbuf.bufF3(FPOS)[parentParticleIndx]; 
    float3 bond0    = fbuf.bufF3(FPOS)[bond0otherPartlicleIdx] - pos;
    float3 bond1    = fbuf.bufF3(FPOS)[bond1otherPartlicleIdx] - pos;
    float3 bond2    = fbuf.bufF3(FPOS)[bond2otherPartlicleIdx] - pos;
/*
    // int axis =  1*(distxsq<distysq && distxsq<distzsq) + 2*(distysq<=distxsq && distysq<distzsq) +3*(distzsq<=distxsq && distzsq<=distysq);
    // if ((axis==1 && dist.x) || (axis==2 && dist.y) || (axis==3 && dist.z)) axis +=2; else axis--;       // sort by longest axis +/-ve 
*/
    float distxsq=bond0.x*bond0.x,  distysq=bond0.y*bond0.y,  distzsq=bond0.z*bond0.z;
    float dsq = distxsq + distysq + distzsq;         
    int axis0 = 1*(distxsq>distysq && distxsq>distzsq) + 2*(distysq>=distxsq && distysq>distzsq) +3*(distzsq>=distxsq && distzsq>=distysq);
    
    distxsq   = bond1.x*bond1.x*(axis0!=1), distysq=bond1.y*bond1.y*(axis0!=2), distzsq=bond1.z*bond1.z*(axis0!=3);
    int axis1 = 1*(distxsq>distysq && distxsq>distzsq) + 2*(distysq>=distxsq && distysq>distzsq) +3*(distzsq>=distxsq && distzsq>=distysq);
    
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
    printf("\nmakeBondIndxMap: parentParticleIndx=%u, bondInxMap=(%i,%i,%i,%i,%i,%i) ", parentParticleIndx, bondInxMap[0], bondInxMap[1], bondInxMap[2], bondInxMap[3], bondInxMap[4], bondInxMap[5]   );
}


extern "C" __device__ void redistribute_bonds(uint new_particle_Idx, float3 newParticlePos, uint neighbours[6], uint neighboursBondIdx[6], uint neighbours2[6]){
    // for particle removal, given list of bonds ... 
    // for each bond 
    
    
}


extern "C" __device__ int insertNewParticle(uint new_particle_Idx, float3 newParticlePos, uint parentParticleIndx, uint bondIdx, uint secondParticleIdx, uint otherParticleBondIndex, uint bond_type[BONDS_PER_PARTICLE]){
    printf ("\ninsertNewParticle1: new_particle_Idx=%u,", new_particle_Idx);                                    // Inserts particle at newParticlePos AND redistributes bonds with neighbours.
  //  addParticle(parentParticleIndx, new_particle_Idx);                                                         // Used by lengthen_tissue(), also for strengthen_tissue(), & muscle...
    
    // cut the old bond here 
    //breakBond (uint thisParticleIdx, uint otherParticleIdx, uint bondIdx, uint otherParticleBondIdx)
    breakBond(parentParticleIndx, secondParticleIdx, bondIdx, otherParticleBondIndex);   
    // may still need to be atomic, and close to original readng, to avoid alteration by oter threads.
    
    // //makeBond (uint thisParticleIdx, uint otherParticleIdx, uint bondIdx, uint otherParticleBondIdx, uint bondType /* elastin, collagen, apatite */)
    if(new_particle_Idx>fparam.maxPoints||secondParticleIdx>fparam.maxPoints||bondIdx>BONDS_PER_PARTICLE||otherParticleBondIndex>BONDS_PER_PARTICLE)return 1;
    makeBond (parentParticleIndx, new_particle_Idx, bondIdx, otherParticleBondIndex*2*BONDS_PER_PARTICLE +0 /*+ otherParticleBondIndex*2 */, bond_type[bondIdx] );
    makeBond (new_particle_Idx, secondParticleIdx, 0/*bondIdx*/, otherParticleBondIndex*2*BONDS_PER_PARTICLE + otherParticleBondIndex*2, bond_type[bondIdx] );
    // NB These two lines(above) replace the parent bond, IFF all indices are valid.
    
    fbuf.bufF3(FPOS)[new_particle_Idx] = newParticlePos;
    uint neighbours[6]          = {UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX}, 
         neighboursBondIdx[6]   = {UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX}, 
         neighbours2[6]         = {UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX};
    
    find_closest_particle_per_axis(new_particle_Idx, newParticlePos, neighbours);
    //find_bonds_to_redistribute(new_particle_Idx, newParticlePos, neighbours, neighboursBondIdx, neighbours2);
    
    //neighbours[bondIdx]= parentParticleIndx;
    //neighboursBondIdx[bondIdx] = bondIdx;
    //neighbours2[otherParticleBondIndex] = otherParticleBondIndex;
    
    int ret1=0, ret2=0, ret3=0;
    int bondInxMap[6]={0,1,2,3,4,5};// no change map    // map parent particle orientation // UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX
    //makeBondIndxMap( parentParticleIndx, bondInxMap);
    
    // need to find the axis of the inherited bonds & swap in bondInxMap
    uint InheritedBondAxis = findBondAxis(newParticlePos, secondParticleIdx); 
    bondInxMap[InheritedBondAxis] = 0; // NB inherited bond is on bondIndex 0.
    
    
    printf("\ninsertNewParticle1.1: new_particle_Idx=%u, \tneighbours[]=(%u,%u,%u,%u,%u,%u), \tneighboursBondIdx[]=(%u,%u,%u,%u,%u,%u), \tneighbours2[]=(%u,%u,%u,%u,%u,%u), \tbondInxMap[]=(%u,%u,%u,%u,%u,%u) ",
       new_particle_Idx,
       neighbours[0],neighbours[1],neighbours[2],neighbours[3],neighbours[4],neighbours[5],  neighboursBondIdx[1],neighboursBondIdx[2],neighboursBondIdx[2],neighboursBondIdx[3],neighboursBondIdx[4],neighboursBondIdx[5],
       neighbours2[0],neighbours2[1],neighbours2[2],neighbours2[3],neighbours2[4],neighbours2[5],
       bondInxMap[0],bondInxMap[1],bondInxMap[2],bondInxMap[3],bondInxMap[4],bondInxMap[5]
    );
                                                                            // ? how to insert the bond being lengthened or strengthened ? 
                                                                            // should occur implicitly due to orientation & placement wrt parent particle.
    for (int bond=1; bond<6; bond++){
        if (neighboursBondIdx[bondInxMap[bond]]<BONDS_PER_PARTICLE 
            && neighbours[bondInxMap[bond]]<fparam.maxPoints){
            //atomicMakeBond(neighbours[bondInxMap[bond]],  new_particle_Idx, neighboursBondIdx[bondInxMap[bond]], bond, bond_type[bond]); 
                                                                                                                // does not need to be atomic
            //atomicMakeBond(uint thisParticleIndx,  uint otherParticleIdx, uint bondIdx, uint otherParticleBondIndex, uint bond_type)
            int _otherParticleBondIndex = new_particle_Idx*2*BONDS_PER_PARTICLE + new_particle_Idx*2;
            makeBond (neighbours[bondInxMap[bond]],  _otherParticleBondIndex, neighboursBondIdx[bondInxMap[bond]], bond, bond_type[bond]);
        }
    }
    
    /*
    for (int bond=1/_*0*_/; bond<6; bond++){
        if (neighboursBondIdx[bondInxMap[bond]]<BONDS_PER_PARTICLE 
            && neighbours[bondInxMap[bond]]<fparam.maxPoints
            && neighbours2[bondInxMap[bond]]<fparam.maxPoints){                                            // suitable bond to redistribute was found)
            ret1 = atomicMakeBond(neighbours[bondInxMap[bond]],  new_particle_Idx, neighboursBondIdx[bondInxMap[bond]], bond, bond_type[bond]);   // new outging bond 
            if (ret1 == 0){                                                                                     // NB ret == 0 : success
                int _otherParticleBondIndex = neighbours2[bondInxMap[bond]]*2*BONDS_PER_PARTICLE + otherParticleBondIndex*2;
                makeBond ( new_particle_Idx, neighbours2[bondInxMap[bond]], bond, _otherParticleBondIndex, bond_type[bond]);
                printf("\ninsertNewParticle2:  new_particle_Idx=%u, neighbours2[bondInxMap[bond]]=%u, bond=%u, _otherParticleBondIndex=%u, bond_type[bond]=%u   ",
                    new_particle_Idx, neighbours2[bondInxMap[bond]], bond, _otherParticleBondIndex, bond_type[bond] );

            }
            if (ret1 || ret2) ret3++;
        }else ret3++;
        printf ("\ninsertNewParticle2.1: bond=%u, new_particle_Idx=%u ret1=%i,ret2=%i,ret3=%i",bond,new_particle_Idx,ret1,ret2,ret3);
    }
    */
    
    printf ("\ninsertNewParticle3: new_particle_Idx=%u, ,ret3=%i", new_particle_Idx, ret3);
    return ret3;                                                                                                //NB causes incoming bonds to fluid particles -> non-adherent surface.
}

extern "C" __global__ void cleanBonds (int pnum){                                   // Called by CleanBondsCUDA (); for use after ComputeParticleChangesCUDA (); Only in Run(), not Run(...)?
    uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                         // particle index
    if ( i >= pnum ) return;
    uint gc = fbuf.bufI(FGCELL)[ i ];                                               // Get search cell	
    if ( gc == GRID_UNDEF ) return;                                                 // particle out-of-range

    gc -= (1*fparam.gridRes.z + 1)*fparam.gridRes.x + 1;
    /*
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
    */                                                                              // Check for broken incomming bonds //////////////////
    for (int a=0; a<BONDS_PER_PARTICLE;a++){                                        // loop round this particle's list of _incomming_ bonds /////
        bool intact = false;
        uint k = fbuf.bufI(FPARTICLEIDX)[i*BONDS_PER_PARTICLE*2 + a*2];
        uint b = fbuf.bufI(FPARTICLEIDX)[i*BONDS_PER_PARTICLE*2 + a*2 +1];          // chk bond intact. nb short circuit evaluation of if conditions.
        // k is a particle, bond_idx is in range, AND k's reciprocal record matches i's record of the bond
        if(k<pnum && b<BONDS_PER_PARTICLE 
            && i==fbuf.bufI(FELASTIDX)[k*BOND_DATA + b*DATA_PER_BOND] 
            && a==fbuf.bufI(FELASTIDX)[k*BOND_DATA + b*DATA_PER_BOND +6] 
            && 0.0<fbuf.bufF(FELASTIDX)[k*BOND_DATA + b*DATA_PER_BOND +2])  intact=true;   
        if(i==k)intact=false;
        if(k<pnum && i!=fbuf.bufI(FELASTIDX)[k*BOND_DATA + b*DATA_PER_BOND] ) printf("\ncleanBonds1: incomming bond not intact : i!=fbuf.bufI(FELASTIDX)[k*BOND_DATA + b*DATA_PER_BOND] ");
        if(k<pnum && a!=fbuf.bufI(FELASTIDX)[k*BOND_DATA + b*DATA_PER_BOND +6]  ) printf("\ncleanBonds1: incomming bond not intact : a!=fbuf.bufI(FELASTIDX)[k*BOND_DATA + b*DATA_PER_BOND +6] ");
        if(k<pnum && 0.0>=fbuf.bufF(FELASTIDX)[k*BOND_DATA + b*DATA_PER_BOND +2] ) printf("\ncleanBonds1: incomming bond not intact : 0.0>=fbuf.bufF(FELASTIDX)[k*BOND_DATA + b*DATA_PER_BOND +2] ");
        if(k<pnum)for(int j=0;j<BONDS_PER_PARTICLE;j++){                                       // check for double bonds, and remove one of them.
           if(k<i && k== fbuf.bufI(FELASTIDX)[i*BOND_DATA + j*DATA_PER_BOND])   intact=false;  // check for reciprocal bonds
           if(j>a && k== fbuf.bufI(FPARTICLEIDX)[i*BONDS_PER_PARTICLE*2 + a*2]) intact=false;  // bonds in the same direction 
        }
        if(intact==false){                                                          // remove broken/missing _incomming_ bonds
            //fbuf.bufI(FPARTICLEIDX)[i*BONDS_PER_PARTICLE*2 + a*2] = UINT_MAX;     // particle NB retain bond direction info
            fbuf.bufI(FPARTICLEIDX)[i*BONDS_PER_PARTICLE*2 + a*2 +1] = UINT_MAX;    // bond index
        }
    }// FELASTIDX //# currently [0]current index, [1]elastic limit, [2]restlength, [3]modulus, [4]damping coeff, [5]particle ID, [6]bond index [7]stress integrator [8]change-type binary indicator
    for (int a=0; a<BONDS_PER_PARTICLE;a++){                                        // loop round this particle's list of _outgoing_ bonds /////
        bool intact = false;
        uint j = fbuf.bufI(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND];
        uint bond_idx = fbuf.bufI(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND + 6];      // chk bond intact nb short circuit evaluation of if conditions.
        // j is a particle, bond_idx is in range, AND j's reciprocal record matches i's record of the bond
        if(j<pnum 
            && bond_idx<BONDS_PER_PARTICLE 
            && i==fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + bond_idx*2] 
            && a==fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + bond_idx*2 +1])  intact=true; 
        if(i==j){
            fbuf.bufI(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND]   =UINT_MAX;
            fbuf.bufF(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND+1] =0;                 // bond to self not allowed, 
            printf("\ncleanBonds2: i=j=%u ",i );
        }
        if(j<pnum && bond_idx>=BONDS_PER_PARTICLE)
            printf("\ncleanBonds3: outgoing bond not intact (bond_idx>=BONDS_PER_PARTICLE) i=%u, j=%u, a=%u bond_idx=%u \t",
                   i,j,a, bond_idx);
        if(j<pnum && i!=fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + bond_idx*2])
            printf("\ncleanBonds3: outgoing bond not intact (i!=fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + bond_idx*2]) i=%u, \t\tfbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + bond_idx*2]=%u, \tj=%u, a=%u bond_idx=%u \t",
                   i,fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + bond_idx*2], j,a, bond_idx);
        if(j<pnum && a!=fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + bond_idx*2 +1])
            printf("\ncleanBonds3: outgoing bond not intact (a!=fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + bond_idx*2 +1]) i=%u, j=%u, a=%u, \t\tfbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + bond_idx*2 +1]=%u, \tbond_idx=%u \t",
                   i,j,a, fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + bond_idx*2 +1], bond_idx);
        
        if(intact==false)fbuf.bufF(FELASTIDX)[i*BOND_DATA+a*DATA_PER_BOND+2] =0.0;  // [2]rest_length  // remove missing _outgoing_ bonds
    }
}


extern "C" __device__ void contribFindBonds ( int i, float3 ipos, int cell, int bond, uint _bondToIdx[BONDS_PER_PARTICLE], float*_bond_dsq, float*_best_theta, uint _pnum)
{
    if ( fbuf.bufI(FGRIDCNT)[cell] == 0 ) return;                                                   // If the cell is empty, skip it.
    uint    j;
    float   dsq;
    float3  dist    = make_float3(0,0,0); 
    int     clast   = fbuf.bufI(FGRIDOFF)[cell] + fbuf.bufI(FGRIDCNT)[cell];                        // index of last particle in this cell
    
    for ( int cndx = fbuf.bufI(FGRIDOFF)[cell]; cndx < clast; cndx++ ) {                            // For particles in this cell.
        j       = fbuf.bufI(FGRID)[ cndx ];
        dist    = ( ipos - fbuf.bufF3(FPOS)[ j ] );                                                 // dist in cm (Rama's comment)
        dsq     = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);                                  // scalar distance squared
        
        if ( dsq < fparam.rd2 && dsq > 0) {                                                         // IF in-range && not the same particle
            float3 u,v;
            u               = dist;
            float theta     = 0;
            uint  bondCount = 0;
            for(int b=0;b<BONDS_PER_PARTICLE;b++){                                                  // Loop round existing outgoing bonds
                uint k      = fbuf.bufI(FELASTIDX)[i*BOND_DATA + b*DATA_PER_BOND];
                if(j==k){theta=FLT_MAX;break;}
                if(k<_pnum){
                    bondCount++;
                    v        = ipos - fbuf.bufF3(FPOS)[ k ];
                    theta    += abs(1.91 - acos( dot(u,v) / (length(u)*length(v)) ) );//*dsq;          // 1.91rad=109.5deg, ideal tetrahedral bond angle.
                }                                                                                   // theta = sum (differences from ideal bond angle)
            }
            if (bond==0) theta      = dsq;
            if (theta<*_best_theta){                                                                // if better than best candidate so far
                *_best_theta        = theta;
                _bondToIdx[bond]    = j;
                *_bond_dsq          = dsq;
            }
        }
    }
    return;
}


extern "C" __global__ void initialize_bonds (int ActivePoints, uint list_length, int gene) {        // Bond angle based search for new bonds.
    uint particle_index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if ( particle_index >= list_length ) return;                                                   
    uint i = fbuf.bufII(FDENSE_LISTS)[gene][particle_index];                                        // call for dense list of solid particles (gene==1)
    if ( i >= ActivePoints ) return;
    
    //printf("\ninitialize_bonds()1: i=%u,  ",i);
    
    uint buf_length = fbuf.bufI(FDENSE_BUF_LENGTHS)[gene];
    uint gc = fbuf.bufI(FGCELL)[ i ];
    uint bondToIdx[BONDS_PER_PARTICLE]; for(int bond=0; bond<BONDS_PER_PARTICLE; bond++) bondToIdx[bond]=UINT_MAX;
    
    //printf("\ninitialize_bonds()2: i=%u,  ",i);
    
    float3 tpos         = fbuf.bufF3(FPOS)[ i ];
    uint  * uintptr     = &fbuf.bufI(FELASTIDX)[i*BOND_DATA];
    float * floatptr    = &fbuf.bufF(FELASTIDX)[i*BOND_DATA];
    
    //printf("\ninitialize_bonds()3: i=%u,  ",i);
    
    uint  elastin       = fgenome.elastin;
    float damping       = fgenome.param[elastin][fgenome.default_damping];
    float modulus       = fgenome.param[elastin][fgenome.default_modulus];
    float rest_length   = fgenome.param[elastin][fgenome.default_rest_length];
    float elastLim      = fgenome.param[elastin][fgenome.elastLim];
    
    //printf("\ninitialize_bonds()4: i=%u,  ",i);
    
    for (int bond=0; bond<BONDS_PER_PARTICLE; bond++){
        float best_theta    = FLT_MAX, bond_dsq = fparam.rd2;                                      // used to compare potential bonds
        //printf("\ninitialize_bonds()4.1: i=%u,  bond=%u",i,bond);
        
        for (int c=0; c < fparam.gridAdjCnt; c++) contribFindBonds ( i, tpos, gc + fparam.gridAdj[c], bond, bondToIdx, &bond_dsq, &best_theta, fparam.maxPoints);
        //if(bondToIdx[bond]>=ActivePoints)printf("\ninitialize_bonds()4.2: i=%u, bond=%u,  bondToIdx[bond]=%u      ",i,bond, bondToIdx[bond] );
        if(bondToIdx[bond]<ActivePoints){ 
            uintptr [bond*DATA_PER_BOND +0] = bondToIdx[bond];
            floatptr[bond*DATA_PER_BOND +1] = elastLim;
            floatptr[bond*DATA_PER_BOND +2] = rest_length;
            floatptr[bond*DATA_PER_BOND +3] = modulus;
            floatptr[bond*DATA_PER_BOND +4] = damping;
            uintptr [bond*DATA_PER_BOND +5] = fbuf.bufI(FPARTICLE_ID)[bondToIdx[bond]];
            uintptr [bond*DATA_PER_BOND +6] = 0;
            uintptr [bond*DATA_PER_BOND +7] = 0;
        }
    }
    
    //printf("\ninitialize_bonds()5: i=%u,  ",i);
}    


extern "C" __global__ void heal (int ActivePoints, uint list_length, int change_list, uint startNewPoints, uint mMaxPoints) { 
    uint particle_index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if ( particle_index >= list_length ) return;                                                   // pnum should be length of list.
    uint i = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index];                        // call for dense list of broken bonds
    if ( i >= ActivePoints ) return;
    uint buf_length = fbuf.bufI(FDENSE_BUF_LENGTHS_CHANGES)[change_list];
    uint bond = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index+buf_length];             //bondIdx
    if (bond>BONDS_PER_PARTICLE)return;
    
    // Bond angle based search for new bond.
    uint gc = fbuf.bufI(FGCELL)[ i ];
    uint bondToIdx[BONDS_PER_PARTICLE]; for(int bond=0; bond<BONDS_PER_PARTICLE; bond++) bondToIdx[bond]=UINT_MAX;
    
    float best_theta= FLT_MAX, bond_dsq = fparam.rd2;                                               // used to compare potential bonds
    
    float3 tpos = fbuf.bufF3(FPOS)[ i ];
    float3 ipos = tpos;
    uint rnd_nmbr = curand(&fbuf.bufCuRNDST(FCURAND_STATE)[i]);                                     // NB bitshift and mask to get rand bool to choose bond
    float max_len = sqrt(fparam.rd2);
    
    tpos.x += max_len/float(4+(rnd_nmbr&7))     *(-1*float(1&(rnd_nmbr>>3))  );                     // shift tpos by a random step < max_len, randomises bond.
    tpos.y += max_len/float(4+((rnd_nmbr>>4)&7))*(-1*float(1&(rnd_nmbr>>7))  );
    tpos.z += max_len/float(4+((rnd_nmbr>>8)&7))*(-1*float(1&(rnd_nmbr>>11)) );
    
 //   printf("\nheal: i=%u, max_len=%f, ipos=(%f,%f,%f), tpos=(%f,%f,%f)",i,max_len, ipos.x,ipos.y,ipos.z, tpos.x,tpos.y,tpos.z);
    
    for (int c=0; c < fparam.gridAdjCnt; c++) contribFindBonds ( i, tpos, gc + fparam.gridAdj[c], bond, bondToIdx, &bond_dsq, &best_theta, fparam.maxPoints);
    /*
    for (int c=0; c < fparam.gridAdjCnt; c++) {                                     // Call contributeForce(..) for fluid forces AND potential new bonds
        contribFindBonds ( i, tpos, gc + fparam.gridAdj[c], bondToIdx, &bond_dsq, &best_theta, fparam.maxPoints);
    }
    */
    if(bondToIdx[bond]<fparam.maxPoints){
        // many are made in 1 step because each broken bond calls heal.
        uint    j_ID         = fbuf.bufI(FPARTICLE_ID)[bondToIdx[bond]];
        float   bond_length  = sqrt(bond_dsq);
        float   modulus      = 100000;       // 100 000 000                                              // 1000000 = min for soft matter integrity // 
        uint *  uintptr      = &fbuf.bufI(FELASTIDX)[i*BOND_DATA + bond*DATA_PER_BOND +0];
        float*  floatptr     = &fbuf.bufF(FELASTIDX)[i*BOND_DATA + bond*DATA_PER_BOND +0];
        
        uintptr[0]  = bondToIdx[bond];                        // [0]current index,
        floatptr[1] = 2 * bond_length ;                 // [1]elastic limit  = 2x restlength i.e. %100 strain
        floatptr[2] = 0.5*bond_length;                  // [2]restlength = initial length  
        floatptr[3] = modulus;                          // [3]modulus
        floatptr[4] = 2*sqrt(fparam.pmass*modulus);     // [4]damping_coeff = optimal for mass-spring pair.
        uintptr[5]  = j_ID;                             // [5]save particle ID of the other particle NB for debugging
        uintptr[6]  = 0;                                // [6]bond index at the other particle 'j's incoming bonds // TODO remove [6] deprecated 
        uintptr[7]  = 0;                                // [7]stress integrator
        uintptr[8]  = 0;                                // [8]change-type binary indicator
    }
}


extern "C" __global__ void lengthen_muscle ( int ActivePoints, int list_length, int change_list, uint startNewPoints, uint mMaxPoints) { //Only for Bond[0] collagen chain //NB elastic tissues (yellow ligments) are non-innervated muscle 
    // TODO consider divergently and convergently branching cases of lengthen_muscle ( int pnum )
    uint particle_index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                             // particle index
    if ( particle_index >= list_length ) return; // pnum should be length of list.
    uint i = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index]; // call for dense list of lengthen_muscle  // NB can come from multiple bonds of same particle.
    
    if (fparam.debug>2 /*&& (threadIdx.x==0 || particle_index==0)*/ ) printf("\nlengthen_muscle()1:  particle_index=%u, list_length=%u, i=%u, ActivePoints=%u, fparam.frame=%u \t", particle_index, list_length, i, ActivePoints, fparam.frame);
    
return;  // suspend use of this kernel for now.
    if ( i >= ActivePoints ) return; 
    //if (fparam.debug>2 /*&& (threadIdx.x==0 || particle_index==0)*/ ) printf("\nlengthen_muscle()1.1:  particle_index=%u  ",particle_index);
    uint buf_length = fbuf.bufI(FDENSE_BUF_LENGTHS_CHANGES)[change_list];
    uint bondIdx            = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index+buf_length]; // bondIdx, NB FDENSE_LISTS_CHANGES [2][list_length] 
    uint secondParticleIdx  = fbuf.bufI(FELASTIDX)[i*BOND_DATA+bondIdx*DATA_PER_BOND];
    uint bondIdx_reciprocal = fbuf.bufI(FELASTIDX)[i*BOND_DATA+bondIdx*DATA_PER_BOND+6]; //[6]bond index // i.e. the incoming bondindx on next_particle_Idx
    // NB (bondIdx_reciprocal > bondIdx)  => convergent branching, (bondIdx_reciprocal < bondIdx) => divergent branching , (0==bondIdx_reciprocal == bondIdx)=> nonbranching, else => error
    
    if (bondIdx!=0)printf("\nlengthen_muscle(): (bondIdx!=0) particle_index=%u ", particle_index);
    
    if (bondIdx_reciprocal>BONDS_PER_PARTICLE || bondIdx>BONDS_PER_PARTICLE || secondParticleIdx>fparam.maxPoints){                                     // corrupt data.
        printf("\nlengthen_muscle, corrupt: bondIdx_reciprocal=%u, bondIdx=%u, secondParticleIdx=%u ", bondIdx_reciprocal, bondIdx, secondParticleIdx ); 
        return; 
    }
/*
    //if (fparam.debug>2 /_*&& (threadIdx.x==0 || particle_index==0)*_/ ) printf("\nlengthen_muscle()1.2:  particle_index=%u, i=%u, bondIdx=%u, bondIdx_reciprocal=%u  ",
    //    particle_index, i, bondIdx, bondIdx_reciprocal);
*/
    uint new_particle_Idx  =  startNewPoints + particle_index;
/*
    // addParticle(i, new_particle_Idx);
*/
    
    float3 newParticlePos =  fbuf.bufF3(FPOS)[i] - 0.5*(fbuf.bufF3(FPOS)[i] - fbuf.bufF3(FPOS)[secondParticleIdx]); // placed near second particle to ensure selection of this bond
    fbuf.bufF3(FPOS)[new_particle_Idx] = newParticlePos;
    
    printf("\nlengthen_muscle:  bondIdx_reciprocal=%u, newParticlePos=(%f,%f,%f)  ",bondIdx_reciprocal, newParticlePos.x, newParticlePos.y, newParticlePos.z );
    
    addParticle(i, new_particle_Idx);   
    uint bond_type[BONDS_PER_PARTICLE] = {0};  bond_type[0] = 1;        //  0=elastin, 1=collagen, 2=apatite
/*
    //makeBond (uint thisParticleIdx, uint otherParticleIdx, uint bondIdx, uint otherParticleBondIdx, uint bondType /_* elastin, collagen, apatite *_/)
    //atomicMakeBond(uint thisParticleIndx,  uint otherParticleIdx, uint bondIdx, uint otherParticleBondIndex, uint bond_type)
*/
    makeBond (i,                new_particle_Idx,  bondIdx, bondIdx_reciprocal, bond_type[bondIdx] ); // making the collagen chain
    makeBond (new_particle_Idx, secondParticleIdx, bondIdx, bondIdx_reciprocal, bond_type[bondIdx] );
    
    float bondRestLength = 0.5 * fbuf.bufI(FELASTIDX)[i*BOND_DATA+bondIdx*DATA_PER_BOND+6];
    fbuf.bufI(FELASTIDX)[i*BOND_DATA                +bondIdx*DATA_PER_BOND+6]   = bondRestLength;
    fbuf.bufI(FELASTIDX)[new_particle_Idx*BOND_DATA +bondIdx*DATA_PER_BOND+6]   = bondRestLength;

    if (fparam.debug>2 /*&& (threadIdx.x==0 || particle_index==0)*/ ) printf("\nlengthen_muscle()1.3:  particle_index=%u, i=%u, bondIdx=%u, bondIdx_reciprocal=%u  ",
        particle_index, i, bondIdx, bondIdx_reciprocal);

    
    // Re-organize bonds to make a contractile chain. Bond[1] must link to next particle but one in the chain. 
    uint Particle[5];
    Particle[0] =  fbuf.bufI(FPARTICLEIDX)[i*2*BONDS_PER_PARTICLE];                                                     // i.e. ParticleIdx of incoming bond[0] 
    Particle[1] =  i ;
    Particle[2] =  new_particle_Idx;
    Particle[3] =  secondParticleIdx;
    Particle[4] =  fbuf.bufI(FELASTIDX)[secondParticleIdx*BOND_DATA];                                                   // i.e. ParticleIdx of outgoing bond[0]
    
    if (fparam.debug>2 /*&& (threadIdx.x==0 || particle_index==0)*/ ) printf("\nlengthen_muscle()1.3.1:  particle_index=%u, i=%u, bondIdx=%u, bondIdx_reciprocal=%u  ",
        particle_index, i, bondIdx, bondIdx_reciprocal);
    
    int start = 1 - (Particle[0]<fparam.maxPoints);
    int stop  = 2 + (Particle[4]<fparam.maxPoints);
    for (int j=start; j<stop; j++){
        uint oldTargetIdx  = fbuf.bufI(FELASTIDX)[Particle[j]*BOND_DATA  +1*DATA_PER_BOND  +bondIdx*DATA_PER_BOND    ]; // i.e. ParticleIdx of outgoing bond[1]
        uint oldTargetBond = fbuf.bufI(FELASTIDX)[Particle[j]*BOND_DATA  +1*DATA_PER_BOND  +bondIdx*DATA_PER_BOND  +1];
        if (oldTargetIdx < fparam.maxPoints && oldTargetBond<BONDS_PER_PARTICLE){
            fbuf.bufI(FPARTICLEIDX)[oldTargetIdx*2*BONDS_PER_PARTICLE + 2*oldTargetBond   ] = UINT_MAX;  
            fbuf.bufI(FPARTICLEIDX)[oldTargetIdx*2*BONDS_PER_PARTICLE + 2*oldTargetBond +1] = UINT_MAX;
        }
        
        if (fparam.debug>2 /*&& (threadIdx.x==0 || particle_index==0)*/ ) printf("\nlengthen_muscle()1.3.2:  particle_index=%u, i=%u, bondIdx=%u, bondIdx_reciprocal=%u  ",
        particle_index, i, bondIdx, bondIdx_reciprocal);
        
        oldTargetIdx  = fbuf.bufI(FPARTICLEIDX)[Particle[j+2]*2*BONDS_PER_PARTICLE + 2*1     ];                         // i.e. ParticleIdx of incoming bond[1]
        oldTargetBond = fbuf.bufI(FPARTICLEIDX)[Particle[j+2]*2*BONDS_PER_PARTICLE + 2*1  +1 ];
        if (oldTargetIdx < fparam.maxPoints && oldTargetBond<BONDS_PER_PARTICLE){
            fbuf.bufI(FELASTIDX)[oldTargetIdx*BOND_DATA  +oldTargetBond*DATA_PER_BOND    ] = UINT_MAX;                      // [0] particle index
            fbuf.bufI(FELASTIDX)[oldTargetIdx*BOND_DATA  +oldTargetBond*DATA_PER_BOND  +6] = UINT_MAX;                      // [6] bond index
        }
        if (fparam.debug>2 /*&& (threadIdx.x==0 || particle_index==0)*/ ) printf("\nlengthen_muscle()1.3.3:  particle_index=%u, i=%u, bondIdx=%u, bondIdx_reciprocal=%u  ",
        particle_index, i, bondIdx, bondIdx_reciprocal);
        
        makeBond (Particle[j], Particle[j+2],  1, 1, 0 ); // making the elastin chain.
        //makeBond(thisParticleIdx, otherParticleIdx, bondIdx, otherParticleBondIdx, bondType /*0 elastin, 1 collagen, 2 apatite*/)
        // ## TODO connect nerves for actuation and sensation:
        // ...
        
    }
    // ## TODO connect new_particle_Idx bonds[2-5].  Need to specify bondsto fill OR change to check for existing bonds.
    //insertNewParticle(new_particle_Idx, newParticlePos, i, bondIdx, secondParticleIdx, bondIdx_reciprocal, bond_type);


    
    if (fparam.debug>2 /*&& (threadIdx.x==0 || particle_index==0)*/ ) printf("\nlengthen_muscle()1.4:  particle_index=%u, i=%u, bondIdx=%u, bondIdx_reciprocal=%u  ",
        particle_index, i, bondIdx, bondIdx_reciprocal);

    // colour particles to indicate replacement of original bond
    uint increment                                      = fparam.frame*10000 + particle_index*10;
    for (int n=0;n<5;n++){fbuf.bufI(FCLR)[Particle[n]]  = increment + n;}
/*    
    fbuf.bufI(FCLR)[new_particle_Idx]   = increment + 4;
    fbuf.bufI(FCLR)[i]                  = increment + 1;
    fbuf.bufI(FCLR)[secondParticleIdx]  = increment + 7;
*/
 if (fparam.debug>2 /*&& (threadIdx.x==0 || particle_index==0)*/ ) printf("\nlengthen_muscle()end:  particle_index=%u,  i=%u, ",  particle_index, i );
}



extern "C" __global__ void lengthen_tissue ( int ActivePoints, int list_length, int change_list, uint startNewPoints, uint mMaxPoints) { //TODO lengthen_tissue ( int pnum )  // add particle in bond
    uint particle_index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                             // particle index
    if ( particle_index >= list_length ) return; // pnum should be length of list.
    uint i = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index]; // call for dense list of lengthen_tissue
    
    //if (fparam.debug>2 /* && (threadIdx.x==0 || particle_index==0) */ ) printf("\nlengthen_tissue() i=%u \t",i);
  return;  
    if ( i >= ActivePoints ) return; 
    uint bondIdx = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index+list_length]; 
    uint secondParticleIdx  = fbuf.bufI(FELASTIDX)[i*BOND_DATA+bondIdx*DATA_PER_BOND];
    // Insert 1 particle on axis of strecthed bond & share existing/new lateral bonds
    // It would help to know which bond. => where to add new particle
    
    uint next_particle_Idx  = fbuf.bufI(FELASTIDX)[i*BOND_DATA+bondIdx*DATA_PER_BOND];
    uint bondIdx_reciprocal = fbuf.bufI(FELASTIDX)[i*BOND_DATA+bondIdx*DATA_PER_BOND+6]; //[6]bond index // i.e. the incoming bondindx on next_particle_Idx
    
    if (next_particle_Idx>fparam.maxPoints) return; 
    
    //create new particle at mid point of bond => uint new_particle_Idx[3] 
    uint new_particle_Idx  =  startNewPoints + particle_index;
    if (new_particle_Idx>fparam.maxPoints)return; // i.e. if run out of spare particles.
    addParticle(i, new_particle_Idx);
    if (new_particle_Idx>fparam.maxPoints)return; // i.e. if addParticle() failed.
    
    printf("\nlengthen_tissue chk0:  i=%u,  next_particle_Idx=%u, fbuf.bufF3(FPOS)[i]=(%f,%f,%f) ",
            i ,next_particle_Idx, fbuf.bufF3(FPOS)[i].x, fbuf.bufF3(FPOS)[i].y, fbuf.bufF3(FPOS)[i].z );
    __syncthreads;
    
    printf("\nlengthen_tissue chk0.1:  i=%u, fbuf.bufF3(FPOS)[next_particle_Idx]=(%f,%f,%f) ",
            i, fbuf.bufF3(FPOS)[next_particle_Idx].x, fbuf.bufF3(FPOS)[next_particle_Idx].y, fbuf.bufF3(FPOS)[next_particle_Idx].z );
    __syncthreads;
    
    //fbuf.bufF3(FPOS)[new_particle_Idx]          = fbuf.bufF3(FPOS)[i] + (fbuf.bufF3(FPOS)[i] - fbuf.bufF3(FPOS)[next_particle_Idx])/2;
    float3 newParticlePos  = fbuf.bufF3(FPOS)[i] + (fbuf.bufF3(FPOS)[next_particle_Idx]  -  fbuf.bufF3(FPOS)[i])/2;
    
    printf("\nlengthen_tissue chk0.2:  i=%u, next_particle_Idx=%u, newParticlePos=(%f,%f,%f) ",
            i, next_particle_Idx, newParticlePos.x, newParticlePos.y, newParticlePos.z );
    __syncthreads;
    
    // Determine bond type from binary change-type indicator
    uint * fbufFEPIGEN = &fbuf.bufI(FEPIGEN)[i]; //*fparam.maxPoints  *NUM_GENES
    uint bond_type[BONDS_PER_PARTICLE] = {0};                          //  0=elastin, 1=collagen, 2=apatite
    
    //printf("\nlengthen_tissue chk1:  i=%u ",i );
    //__syncthreads;
    
    // Calculate material type for bond
    if (fbufFEPIGEN[9*fparam.maxPoints]/*bone*/) for (int bond=0; bond<BONDS_PER_PARTICLE; bond++) bond_type[bond] = 2;
    else if (fbufFEPIGEN[6*fparam.maxPoints]/*tendon*/||fbufFEPIGEN[7*fparam.maxPoints]/*muscle*/||fbufFEPIGEN[10*fparam.maxPoints]/*elast lig*/) {bond_type[0] = 1; bond_type[3] = 1;}
    //NB muscle& elast should not occur here, they have their own list & kernel.
    else if (fbufFEPIGEN[8*fparam.maxPoints]/*cartilage*/)for (int bond=0; bond<BONDS_PER_PARTICLE; bond++) bond_type[bond] = 1;
    
    //printf("\nlengthen_tissue chk2:  i=%u ",i );
    //__syncthreads;
    
    
    int ret = insertNewParticle(new_particle_Idx, newParticlePos, i, bondIdx, secondParticleIdx, bondIdx_reciprocal,  bond_type);
    
    
    
    
    
    
    //__device__ int  insertNewParticle(uint new_particle_Idx, float3 newParticlePos, uint parentParticleIndx, uint bondIdx, uint secondParticleIdx, uint otherParticleBondIndex, uint bond_type[BONDS_PER_PARTICLE]);
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
        
        if (fparam.debug>2 && (uint)bonds[a][0]==i) printf("\n (uint)bonds[a][0]==i, i=%u a=%u",i,a);  // float bonds[BONDS_PER_PARTICLE][3];  [0] = index of other particle, [1] = dsq, [2] = bond_index
                                                                                    // If outgoing bond empty && proposed bond for this quadrant is valid
        if (fparam.debug>2 && fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +1] == 0.0  &&  bonds[a][0] < pnum  && bonds[a][0]!=i  && bond_dsq[a]<3 ){  // ie dsq < 3D diagonal of cube ##### hack #####
                                                                                    // NB "bonds[b][0] = UINT_MAX" is used to indicate no candidate bond found
                                                                                    //    (FELASTIDX) [1]elastic limit = 0.0 isused to indicate out going bond is empty
            //if (fparam.debug>2)printf("\nBond making loop i=%u, a=%i, bonds[a][1]=%u, bond_dsq[a]=%f",i,a,bonds[a][1],bond_dsq[a]);
            
            
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
                //if (fparam.debug>2)printf("\nNew Bond a=%u, i=%u, j=%u, bonds[a][1]=%u, fromPID=%u, toPID=%u,, fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex]=%u, otherParticleBondIndex=%u",
                //       a,i,bonds[a][0],bonds[a][1],i_ID,j_ID, fbuf.bufI(FPARTICLEIDX)[otherParticleBondIndex], otherParticleBondIndex);
            }            
        }// end if 
        __syncthreads();    // NB applies to all threads _if_ the for loop runs, i.e. if(freeze==true)
    }           // TODO make this work with incoming & outgoing bonds, NB preserve existing bonds                    // end loop around FELASTIDX bonds

    */
}


extern "C" __global__ void shorten_muscle ( int ActivePoints, int list_length, int change_list, uint startNewPoints, uint mMaxPoints) { //TODO shorten_muscle ( int pnum )  // remove particle in chain & update contractile bonds
    uint particle_index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                             // particle index
    if ( particle_index >= list_length ) return; // pnum should be length of list.
    uint i = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index]; // call for dense list of shorten_muscle
    
    if (fparam.debug>2 && (threadIdx.x==0 || particle_index==0) ) printf("\nshorten_muscle() i=%u \t",i);
    if ( i >= ActivePoints ) return; 
    uint bondIdx = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index+list_length]; 
    // Need to remove 3 particles, and close the gap.
    
    
    
}

extern "C" __global__ void shorten_tissue ( int ActivePoints, int list_length, int change_list, uint startNewPoints, uint mMaxPoints) { //TODO shorten_tissue ( int pnum )  // remove particle and connect bonds along their axis
    uint particle_index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                             // particle index
    if ( particle_index >= list_length ) return; // pnum should be length of list.
    uint i = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index]; // call for dense list of shorten_tissue
    
    if (fparam.debug>2 && (threadIdx.x==0 || particle_index==0) ) printf("\nshorten_tissue() i=%u \t",i);
    if ( i >= ActivePoints ) return; 
    uint bondIdx = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index+list_length]; 
    // Need to remove 1 particle and close the gap
    // It would help to know which bond. => how to close the gap
    
    
}

extern "C" __global__ void strengthen_muscle ( int ActivePoints, int list_length, int change_list, uint startNewPoints, uint mMaxPoints) { //TODO strengthen_muscle ( int pnum )  // NB Y branching etc
    uint particle_index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                             // particle index
    if ( particle_index >= list_length ) return; // pnum should be length of list.
    uint i = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index]; // call for dense list of strengthen_muscle
    
    if (fparam.debug>2 && (threadIdx.x==0 || particle_index==0) ) printf("\nstrengthen_muscle() i=%u \t",i);
    if ( i >= ActivePoints ) return; 
    uint bondIdx = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index+list_length]; 
    // Need to doulble up the helix i.e. add particles and contractile bonds in parallel.
    // Q Induced by ?
    // Q double up How ? 
    // NB difference between a helix and a zig-zag is only that the contractile bonds reach 2 particles ahead.
    
    
    
}

extern "C" __global__ void strengthen_tissue ( int ActivePoints, int list_length, int change_list, uint startNewPoints, uint mMaxPoints) { //TODO strengthen_tissue ( int pnum )  // add particle and bonds in parallel AND reduce original bon's modulus
    uint particle_index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                             // particle index
    if ( particle_index >= list_length ) return; // pnum should be length of list.
    uint i = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index]; // call for dense list of strengthen_tissue
    
    if (fparam.debug>2 && (threadIdx.x==0 || particle_index==0) ) printf("\nstrengthen_tissue() i=%u \t",i);
    if ( i >= ActivePoints ) return; 
    uint bondIdx = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index+list_length]; 
    // Need to double up articles and bonds in parallel wrt the affected bond
    // It would help to know which bond. => where to place the new particle i.e. orthogonal to the bond NB place where there is space in the plane.
    
    
}

extern "C" __global__ void weaken_muscle ( int ActivePoints, int list_length, int change_list, uint startNewPoints, uint mMaxPoints) { //TODO weaken_muscle ( int pnum )  // NB Y branching etc
    uint particle_index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                             // particle index
    if ( particle_index >= list_length ) return; // pnum should be length of list.
    uint i = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index]; // call for dense list of weaken_muscle
    
    if (fparam.debug>2 && (threadIdx.x==0 || particle_index==0) ) printf("\nweaken_muscle() i=%u \t",i);
    if ( i >= ActivePoints ) return; 
    uint bondIdx = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index+list_length]; 
    // Need to remove a row of particles in parallel - i.e. form/propagate a branch 
    // 
    
    
}

extern "C" __global__ void weaken_tissue ( int ActivePoints, int list_length, int change_list, uint startNewPoints, uint mMaxPoints) { //TODO weaken_tissue ( int pnum )  // remove particle & transfer bonds laterally  
    uint particle_index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;                             // particle index
    if ( particle_index >= list_length ) return; // pnum should be length of list.
    uint i = fbuf.bufII(FDENSE_LISTS_CHANGES)[change_list][particle_index]; // call for dense list of weaken_tissue
    
    if (fparam.debug>2 && (threadIdx.x==0 || particle_index==0) ) printf("\nweaken_tissue() i=%u \t",i);
    if ( i >= ActivePoints ) return; 
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

/*
Wendland C2 kernel for 3D,  

    Phi_(1,3) := ((1-r)**4)*(1+4r),     for range 0<=r<=1, where r is the distance between two particles.

Used to prevent particle clumping/pairing instability under tension. 

for which:
    C = 21/2PI,              C := "normalization constant"
    sigma**2/H**2 = 1/15,    sigma := std dev,    sigma**2 := variance,     H := kernel basis,   h := 2*sigma
    
    1st differential wrt r,         Del.Phi_(1,3)   =   4(1-r)**4   -  4(1-r)**3 * (1+4r)
    2nd differential wrt r,     Del.Del.Phi_(1,3)   = -32(1-r)**3   + 12(1-r)**2 * (1+4r)  
    
    (i.e. the Laplacian differential. Not to be confused with the Laplacian integral transform.)

    From (Dehnen & Al 2012) "Improving convergence in smoothed particle hydrodynamics simulations without pairing instability",
    Differentials computed with Sympy.

##
    
Construction of SPH equations from (Muller et al 2003), 

Navier-Stokes eq for conservation of momentum, in SPH becomes:

    Rho(delta.v/delta.t + v.del.v) = -del.p + rho.g + mu.del.del.v      where p:=pressure, g:=gravity, mu:=coeff viscosity, v:=relative velocity between particles
    
    particle accel:= a_i = dv_i/dt = ( -del.p + rho.g + mu.del.del.v + surface_tension ) / rho_i 
    
    Forces:
    
    f_i^surface_tension := sigma.k.n = -sigma.del.del.Cs.n/|n|                  where n = del.Cs,  Cs(r):= SUM_j{  m_j.(1/rho_j).W(r-r_j,h) }
    
                                                                                so n/|n| is the direction vector, of the gradient of the density field (??)
    
                                     = -sigma.del.del. SUM_j{  m_j.(1/rho_j).W(r-r_j,h) } .n/|n| 

                                                                                where sigma:=surf_tension const.  W():= smoothing kernel.
                                                                        
    f_i^viscosity := mu.SUM_j{ m_j .((v_j-v_i)/rho_j).del.del.W(r_i-r_j,h) }    where mu:= coeff viscosity.
    
                                                                                where del.del.W(..)  is the Laplacian differential of the smoothing kernel.
    
    f_i^pressure  := -SUM_j{ m_j.((p_i+p_j)/2rho_j) .del.W(r_i-r_j,h) }         NB Muller uses W_poly6kern for pressure and W_spiky for force due to pressure. 
                                                                                We use Wendland C^2 for pressure, viscosity and surface tension.
                                                                                
                                                                                NB Fluids-v3 used +ve atmospheric pressure and no surface tension.
                                                                                
    
Notes on possible meanings of the Del operator:

gradient of scalar,          grad_f := del.f
divergence vector field,     div_v  := del.v
curl vector field,           curl_v := del cross v
directoinal derivative     a.grad_f := a.(del.f)
Laplacian                    Delta  := del.del = del^2 
Hessian                      H      := del^2 = del.del^T
Tensor derivative                   := del circle_cross v
    
*/
extern "C" __device__ float3 contributeForce ( int i, float3 ipos, float3 iveleval, float ipress, float idens, int cell)
{			
	if ( fbuf.bufI(FGRIDCNT)[cell] == 0 ) return make_float3(0,0,0);                                        // If the cell is empty, skip it.
	float  dsq, sdist, c, r, sr=1.0;//fparam.psmoothradius;
    float3 pterm= make_float3(0,0,0), sterm= make_float3(0,0,0), vterm= make_float3(0,0,0), forcej= make_float3(0,0,0), delta_v= make_float3(0,0,0);                                                              // pressure, surface tension and viscosity terms.
	float3 dist     = make_float3(0,0,0),      eterm = make_float3(0,0,0),    force = make_float3(0,0,0);
	uint   j;
	int    clast    = fbuf.bufI(FGRIDOFF)[cell] + fbuf.bufI(FGRIDCNT)[cell];                                // index of last particle in this cell
    
    for (int cndx = fbuf.bufI(FGRIDOFF)[cell]; cndx < clast; cndx++ ) {                                     // For particles in this cell.
		j           = fbuf.bufI(FGRID)[ cndx ];
		dist        = ( ipos - fbuf.bufF3(FPOS)[ j ] );                                                     // dist in cm (Rama's comment)
		dsq         = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);                                      // scalar distance squared
		r           = sqrt(dsq);
        
        // From https://github.com/DualSPHysics/DualSPHysics/wiki/3.-SPH-formulation#31-smoothing-kernel 
        /*
         * q=r/h, where r=dist between particles, h=smoothing length
         * 
         * W(r,h) = alpha_D(1-q/2)**4 *(2q+1) for 0<=q<=2
         * 
         * where alpha_D = 21/(16*Pi*h**3)  , the normalization kernel in 3D,
         * i.e. 1/integral_(0,2){kernel * area of a sphere}dr 
         * 
         */
        /* My new kernels
         * sr=1
         * # NB gives equi-pressure radius = 0.5.
         * 
         * Continuous Pressure, i.e. hyrostatic + surface tension or vapour pressure
         * CPkern = (sr - r )**3 - (1/2)*(sr - r )**2
         * 
         * Viscous kernel
         * vkern = (1/2)*(sr - r )**3
         */
        
        if ( dsq < 1 /*fparam.rd2*/ && dsq > 0) {                                                                 // IF in-range && not the same particle
            float kern = pow((sr - r),3);
            pterm = 1000.0* (dist/r) *(kern - (0.5)*pow((sr - r),2));       // 1000 = hydroststic stiffness      
            delta_v = fbuf.bufF3(FVEVAL)[j] - iveleval;
            vterm =  100000.0* delta_v * kern;// (1/2)*pow((sr - r),3) ;
            
            /*
             sdist   = sqrt(dsq * fparam.d2);                                                                // smoothing distance = sqrt(dist^2 * sim_scale^2))
             c       = ( fparam.psmoothradius - sdist );
             pterm   = (dist/sdist) * pow((fparam.psmoothradius - sqrt(dsq)), 3) * (fparam.psmoothradius - dsq) ;
             * fparam.psimscale * -0.5f * c * fparam.spikykern   * ( ipress + fbuf.bufF(FPRESS)[ j ] )/ sdist )  ;       // pressure term
            //sterm   = (dist/dsq) * fparam.sterm * cos(3*CUDART_PI_F*r/(2*fparam.psmoothradius));  // can we use sdist in placeof r ?  or in place od dsq? What about pressure?
			//vterm   =  fparam.vterm * ( fbuf.bufF3(FVEVAL)[ j ] - iveleval );  // make_float3(0,0,0);//
			forcej  += ( pterm + sterm + vterm) * c * idens * (fbuf.bufF(FDENSITY)[ j ] );  // fluid force
            */
            force   +=  vterm + pterm ;
            /*if(i<10)  printf("\ncontribForce : i=%u, r=,%f, sr=,%f, (sr-r)^3=,%f, delta_v=,(%f,%f,%f), vterm=(%f,%f,%f), pterm(%f,%f,%f)  ",i, r, sr, kern, delta_v.x,delta_v.y,delta_v.z, vterm.x,vterm.y,vterm.z, pterm.x,pterm.y,pterm.z);*/
            /*
            if(i<10) printf("\ncontribForce() : i=,%u, ,cell=,%u,  ,cndx=,%u, ,r=,%f, ,sqrt(fparam.rd2)=r_basis=,%f, ,fparam.psmoothradius=,%f,,sdist=,%f, ,(fparam.psmoothradius-sdist)= c =,%f, \t,ipress=,%f, ,jpress=,%f, ,idens=,%f, ,jdens=,%f,       \t ,pterm=(,%f,%f,%f,),  ,sterm=(,%f,%f,%f,), ,vterm=(,%f,%f,%f,), ,forcej=(,%f,%f,%f,) ,  ,fparam.vterm=,%f, ,fbuf.bufF3(FVEVAL)[ j ]=(,%f,%f,%f,), ,iveleval=(,%f,%f,%f,) ", 
                i, cell, cndx, r, sqrt(fparam.rd2), fparam.psmoothradius, sdist, c,  ipress, fbuf.bufF(FPRESS)[j], idens, fbuf.bufF(FDENSITY)[j],    pterm.x,pterm.y,pterm.z, sterm.x,sterm.y,sterm.z, vterm.x,vterm.y,vterm.z, forcej.x,forcej.y,forcej.z, 
                fparam.vterm, fbuf.bufF3(FVEVAL)[j].x, fbuf.bufF3(FVEVAL)[j].y, fbuf.bufF3(FVEVAL)[j].z, iveleval.x, iveleval.y, iveleval.z
            );
            */
        }                                                                                                   // end of: IF in-range && not the same particle
    }                                                                                                       // end of loop round particles in this cell
    //if(i<10)  printf("\ncontribForce : i=%u, force=(%f,%f,%f)  ",i, force.x,force.y,force.z  );
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
    
    //if(fbuf.bufI(FPARTICLE_ID)[i]<10) printf("\ncomputeForce() chk2: ParticleID=%u  ",fbuf.bufI(FPARTICLE_ID)[i] );  
    //__syncthreads();
    
    float3  pvel = {fbuf.bufF3(FVEVAL)[ i ].x,  fbuf.bufF3(FVEVAL)[ i ].y,  fbuf.bufF3(FVEVAL)[ i ].z}; // copy i's FEVAL to thread memory
    for (int a=0;a<BONDS_PER_PARTICLE;a++){                                         // compute elastic force due to bonds /////////////////////////////////////////////////////////
        uint bond                   = i*BOND_DATA + a*DATA_PER_BOND;                // bond's index within i's FELASTIDX 
        uint j                      = fbuf.bufI(FELASTIDX)[bond];                   // particle IDs   i*BOND_DATA + a
        float restlength        = fbuf.bufF(FELASTIDX)[bond + 2];                   // NB fbuf.bufF() for floats, fbuf.bufI for uints.
        if(j<pnum && restlength>0){                                                 // copy FELASTIDX to thread memory for particle i.
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
            #define DECAY_FACTOR 0.8                                                                                   // could be a gene.
            fbuf.bufF(FELASTIDX)[bond + 7] = (fbuf.bufF(FELASTIDX)[bond + 7] + spring_strain) * DECAY_FACTOR;           // spring strain integrator
          
          //if(fbuf.bufI(FPARTICLE_ID)[i]<10) printf("\ncomputeForce() chk3: ParticleID=%u, bond=%u, restlength=%f, modulus=%f , abs_dist=%f , spring_strain=%f , strain_integrator=%f  ",fbuf.bufI(FPARTICLE_ID)[i], a, restlength , modulus , abs_dist , spring_strain , fbuf.bufF(FELASTIDX)[bond + 7]  );  
            
            eterm = ((float)(abs_dist < elastic_limit)) * ( ((dist/abs_dist) * spring_strain * modulus) - damping_coeff*rel_vel); // Elastic force due to bond ####
            
            //if(i<10) printf("\ncomputeForce() : i=,%u, bond=,%u, eterm=(,%f,%f,%f,) ",i, a, eterm.x,eterm.y,eterm.z);
            
            force -= eterm;                                                         // elastic force towards other particle, if (rest_len -abs_dist) is -ve
            atomicAdd( &fbuf.bufF3(FFORCE)[ j ].x, eterm.x);                        // NB Must send equal and opposite force to the other particle
            atomicAdd( &fbuf.bufF3(FFORCE)[ j ].y, eterm.y);
            atomicAdd( &fbuf.bufF3(FFORCE)[ j ].z, eterm.z);                        // temporary hack, ? better to write a float3 attomicAdd using atomicCAS  #########

            if (abs_dist >= elastic_limit){                                         // If (out going bond broken)
                fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +2]=0.0;         // remove broken bond by setting rest length to zero.
                //fbuf.bufF(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +3]=0;         // set modulus to zero
                
                uint bondIndex_ = fbuf.bufI(FELASTIDX)[i*BOND_DATA + a*DATA_PER_BOND +6];
                //fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + bondIndex_+1] = UINT_MAX ;// set the reciprocal bond index to UINT_MAX, but leave the old particle ID for bond direction.
                //fbuf.bufI(FELASTIDX)[bond] = UINT_MAX;
                if (fparam.debug>2)printf("\n#### Set to broken, i=,%i, j=,%i, b=,%i, fbuf.bufI(FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + b]=UINT_MAX\t####",i,j,bondIndex_);
                bondsToFill++;
            }
        }
        //__syncthreads();    // when is this needed ? ############
    }   

    //if (fparam.debug>2)printf("\nComputeForce chk4: i=%u, bondsToFill=%u,  gc=%u,  fparam.gridTotal=%u", i, bondsToFill, gc, fparam.gridTotal);  // was always zero . why ?
    //__syncthreads();
    
    //if(i<10) printf("\n computeForce()1: i=,%u, elastic force=(,%f,%f,%f,) ",i, force.x,force.y,force.z);
	
    bondsToFill=BONDS_PER_PARTICLE; // remove and use result from loop above ? ############
    for (int c=0; c < fparam.gridAdjCnt; c++) {                                 // Call contributeForce(..) for fluid forces AND potential new bonds /////////////////////////
        
        float3 fluid_force = make_float3(0,0,0);
        fluid_force = contributeForce ( i, fbuf.bufF3(FPOS)[ i ], fbuf.bufF3(FVEVAL)[ i ], fbuf.bufF(FPRESS)[ i ], fbuf.bufF(FDENSITY)[ i ], gc + fparam.gridAdj[c]); 
        //if (freeze==true) fluid_force *=0.1;                                        // slow fluid movement while forming bonds
        force += fluid_force;
    }
    //if(i<10) printf("\nComputeForce 2: i=,%u, force=(,%f,%f,%f,) ", i,force.x,force.y,force.z);
    //printf(".\n");
    //__syncthreads();
    
    
    //if (fparam.debug>2)printf("\ni=%u, bond_dsq=(%f,%f,%f,%f,%f,%f),",i,bond_dsq[0],bond_dsq[1],bond_dsq[2],bond_dsq[3],bond_dsq[4],bond_dsq[5]);

	//__syncthreads();   // when is this needed ? ############
    atomicAdd(&fbuf.bufF3(FFORCE)[ i ].x, force.x);                                 // atomicAdd req due to other particles contributing forces via incomming bonds. 
    atomicAdd(&fbuf.bufF3(FFORCE)[ i ].y, force.y);                                 // NB need to reset FFORCE to zero in  CountingSortFull(..)
    atomicAdd(&fbuf.bufF3(FFORCE)[ i ].z, force.z);                                 // temporary hack, ? better to write a float3 atomicAdd using atomicCAS ?  ########

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
	//diff = fparam.pradius - (pos.x - (fparam.pboundmin.x + (sin(time*fparam.pforce_freq)+1)*0.5 * fparam.pforce_min))*ss;  //wave machine NB fparam.pforce_freq
	diff = fparam.pradius - (pos.x - fparam.pboundmin.x ) * ss;
	if ( diff > EPSILON ) {
		norm = make_float3( 1, 0, 0);
		adj = (fparam.pforce_min+1) * fparam.pextstiff * diff - fparam.pdamp * dot(norm, veval );
		norm *= adj; accel += norm;
	}
	//diff = fparam.pradius - ( (fparam.pboundmax.x - (sin(time*fparam.pforce_freq)+1)*0.5*fparam.pforce_max) - pos.x)*ss;  //wave machine
	diff = fparam.pradius - ( fparam.pboundmax.x - pos.x )*ss;
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
	diff = 2*fparam.pradius - (dist.x + dist.y + dist.z) * ss;                  // use Manhatan norm for speed & 2*pradius for safety
	if ( diff > EPSILON ) {
        norm = make_float3( 1, 1, 1 );                                          // NB planar norm for speed, not spherical
        adj = fparam.pextstiff * diff - fparam.pdamp * dot(norm, veval );
		norm *= adj; accel += norm;
    }
	
		
	// Gravity
	accel += fparam.pgravity;
    
    // NB Accel & Vel limits prevent visible instability, but produce thoroughly non-physical behaviour.
    // For quasi-physical simulations we want to avoid triggering these limits. 
    
	// Accel Limit
	speed = accel.x*accel.x + accel.y*accel.y + accel.z*accel.z;
    if(i<10)printf("\nadvanceParticles()1: i=,%u,  mass=,%f,  accel=(,%f,%f,%f,),\t  accel^2=,%f,\t fparam.AL2=,%f,\t  fparam.pgravity=,(,%f,%f,%f,) ",
        i, fparam.pmass, accel.x,accel.y,accel.z, speed, fparam.AL2, fparam.pgravity.x, fparam.pgravity.y, fparam.pgravity.z
    );
	if ( speed > fparam.AL2 ) {
		accel *= fparam.AL / sqrt(speed);     // reduces accel to fparam.AL, while preserving direction. 
	}

	// Velocity Limit
	float3 vel = fbuf.bufF3(FVEL)[i];
    
	speed = vel.x*vel.x + vel.y*vel.y + vel.z*vel.z;
    if(i<10)printf("\nadvanceParticles()2: i=,%u, accel=(,%f,%f,%f,),  vel=(,%f,%f,%f,),  vel^2=,%f,  fparam.VL2=,%f, ",
        i, accel.x,accel.y,accel.z, vel.x,vel.y,vel.z,  speed, fparam.VL2
    );
	if ( speed > fparam.VL2 ) {
		speed = fparam.VL2;
		vel *= fparam.VL / sqrt(speed);       // reduces vel to fparam.VL , while preerving direction.
	}
	
	// Leap-frog Integration                                                    // Write to ftemp.buf*(FEVEL/FVEL/FPOS)
                                                                                // Allows specialParticles() to read old values.
	float3 vnext = accel*dt + vel;                                              // v(t+1/2) = v(t-1/2) + a(t) dt		
	ftemp.bufF3(FVEVAL)[i] = (vel + vnext) * 0.5;                               // v(t+1) = [v(t-1/2) + v(t+1/2)] * 0.5			
	ftemp.bufF3(FVEL)[i] = vnext;
	ftemp.bufF3(FPOS)[i] += vnext * (dt/ss);                                    // p(t+1) = p(t) + v(t+1/2) dt		
    
    
    if (i<10 ){  // fparam.debug>2 && i==0
        printf("\nadvanceParticles()3: i=,%u, accel.x==(,%f,%f,%f,),  vel=(,%f,%f,%f,),  dt==%f, vnext.x==(,%f,%f,%f,), ss==%f",
              i,  accel.x,accel.y,accel.z,  vel.x,vel.y,vel.z,    dt,   vnext.x,vnext.y,vnext.z,   ss
              );
/*
         * printf("\naccel.x==%f",accel.x);
        printf("\ndt==%f",dt);
        printf("\nvnext.x==%f",vnext.x);
        printf("\nss==%f",ss);
*/
    }
    
}


extern "C" __global__ void externalActuation (uint list_len,  float time, float dt, float ss, int numPnts )
{		
	uint particle_index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index
	if ( particle_index >= list_len ) return;
    uint i = fbuf.bufII(FDENSE_LISTS)[12][particle_index];
	if ( i >= numPnts ) return;
  //if (fparam.debug>2)printf("\nexternalActuation(): i=%u\t",i);
    
    // Get particle vars
	register float3 accel;//, norm;
	register float speed; //diff, adj, 
	register float3 pos = fbuf.bufF3(FPOS)[i];
	register float3 veval = fbuf.bufF3(FVEVAL)[i];

	// Leapfrog integration						
	accel = fbuf.bufF3(FFORCE)[i];
	accel *= fparam.pmass;	

    // // Gravity
    // accel += fparam.pgravity;
    
    // External force   // How best to define this ?
    // For now, take sine wave on time.
    // Later, 
    // 1) take cmdln input for period & force vector
    // 2) model input
    // 3) simulation interaction (SOFA)
    
#define FACTOR  10
#define PERIOD  20
    accel +=  fparam.pgravity * FACTOR * sin(time/PERIOD) ;
    
    
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
	
	// Leap-frog Integration                                                    // Write to ftemp.buf*(FEVEL/FVEL/FPOS)
                                                                                // Allows specialParticles() to read old values.
	float3 vnext = accel*dt + vel;					// v(t+1/2) = v(t-1/2) + a(t) dt		
	ftemp.bufF3(FVEVAL)[i] = (vel + vnext) * 0.5;	// v(t+1) = [v(t-1/2) + v(t+1/2)] * 0.5			
	ftemp.bufF3(FVEL)[i] = vnext;
	ftemp.bufF3(FPOS)[i] += vnext * (dt/ss);		// p(t+1) = p(t) + v(t+1/2) dt		
    //if (fparam.debug>2)printf("\nexternalActuation(): i=%u (FVEL)[i]=(%f,%f,%f), (FPOS)[i]=(%f,%f,%f) ",
    //       i, ftemp.bufF3(FVEL)[i].x, ftemp.bufF3(FVEL)[i].y, ftemp.bufF3(FVEL)[i].z,
    //       ftemp.bufF3(FPOS)[i].x, ftemp.bufF3(FPOS)[i].y, ftemp.bufF3(FPOS)[i].z  
    //      );
}


extern "C" __global__ void fixedParticles (uint list_len, int numPnts )
{		
	uint particle_index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index
	if ( particle_index >= list_len ) return;
    uint i = fbuf.bufII(FDENSE_LISTS)[11][particle_index];
	if ( i >= numPnts ) return;
  //if (fparam.debug>2)printf("\nfixedParticles(): i=%u\t",i);

	ftemp.bufF3(FVEVAL)[i] = fbuf.bufF3(FVEVAL)[i];
	ftemp.bufF3(FVEL)[i]   = fbuf.bufF3(FVEL)[i];
	ftemp.bufF3(FPOS)[i]   = fbuf.bufF3(FPOS)[i];
}


/*
 * NB Mechanism of atomicCAS :
 * 
int  atomicCAS( int  *p,  intcmp,  intv )
{
	exclusive_single_thread
	{
		int  old = *p;
		if (cmp== old) *p = v;
	}
	return old;
}

*/
