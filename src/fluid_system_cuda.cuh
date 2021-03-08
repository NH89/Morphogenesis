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

#ifndef DEF_KERN_CUDA
	#define DEF_KERN_CUDA

	#include <curand.h>
	#include <curand_kernel.h>
	#include <stdio.h>
	#include <math.h>
    #include "/usr/local/cuda/include/math_constants.h"  // need better <path> . In CMakeLists.txt "include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})"

	#define CUDA_KERNEL
	#include "fluid.h"

	#define EPSILON				0.00001f
	#define GRID_UCHAR			0xFF
	#define GRID_UNDEF			4294967295// = 2^32 -1
	#define TOTAL_THREADS		1000000
	#define BLOCK_THREADS		256
	#define MAX_NBR				80		
	#define FCOLORA(r,g,b,a)	( (uint((a)*255.0f)<<24) | (uint((b)*255.0f)<<16) | (uint((g)*255.0f)<<8) | uint((r)*255.0f) )
    //#define TWO_POW_31 0x80000000       // Binary 10000000000000000000000000000000  // used to store data in top bit of uint
    //#define TWO_POW_24_MINUS_1 0xFFFFFF // Binary 111111111111111111111111          // used to bit mask 1st 24 bits of uint

    
	typedef unsigned int		uint;
	typedef unsigned short int	ushort;
	typedef unsigned char		uchar;
	
	extern "C" {
        __global__ void initialize_FCURAND_STATE (int pnum);
		__global__ void insertParticles ( int pnum );	
        __global__ void tally_denselist_lengths(int num_lists, int fdense_list_lengths, int fgridcnt, int fgridoff);
		__global__ void countingSortFull ( int pnum );		
        __global__ void countingSortEPIGEN ( int pnum );
        
		__global__ void computeQuery ( int pnum );	
		__global__ void computePressure ( int pnum );		
		__global__ void computeForce ( int pnum , bool freeze = false, uint frame =20);	          // skip CAS lock if frame>10
        __global__ void computeDiffusion ( int pnum );
        __global__ void computeGeneAction ( int pnum, int gene, uint list_len );                  //NB here pnum is for the dense list
        __global__ void computeBondChanges ( int pnum, uint list_length );
        
        //__global__ void computeAutomata ( int pnum );
        //__global__ void freeze ( int pnum);                                                     // new freeze kernel, to generate elastic bonds.
		__global__ void advanceParticles ( float time, float dt, float ss, int numPnts );
		__global__ void emitParticles ( float frame, int emit, int numPnts );
		__global__ void randomInit ( int seed, int numPnts );
		__global__ void sampleParticles ( float* brick, uint3 res, float3 bmin, float3 bmax, int numPnts, float scalar );	
		__global__ void prefixFixup ( uint *input, uint *aux, int len);
		__global__ void prefixSum ( uint* input, uint* output, uint* aux, int len, int zeroff );
		//__global__ void countActiveCells ( int pnum );		
        
        __global__ void insertChanges ( int pnum );
        __global__ void prefixFixupChanges(uint *input, uint *aux, int len);
        __global__ void prefixSumChanges(uint* input, uint* output, uint* aux, int len, int zeroff);
        __global__ void tally_changelist_lengths( );
        __global__ void countingSortChanges ( int pnum );
        __global__ void computeNerveActivation ( int pnum );
        
        __global__ void computeMuscleContraction ( int pnum );
        __device__ void addParticle         (uint parent_Idx, uint &new_particle_Idx);
        __device__ void find_potential_bonds (int i, float3 ipos, int cell, uint _bonds[BONDS_PER_PARTICLE][2], float _bond_dsq[BONDS_PER_PARTICLE], float max_len);
        
        __device__ void find_potential_bond(int i, float3 ipos, uint _thisParticleBonds[6], float3 tpos, int gc, uint& _otherParticleIdx, uint& _otherParticleBondIdx, float& _bond_dsq, float max_len/*_sq*/);
        
        __device__ void makeBond (uint thisParticleIdx, uint otherParticleIdx, uint bondIdx, uint otherParticleBondIdx, uint bondType /* elastin, collagen, apatite */);
        __device__ int  atomicMakeBond(uint thisParticleIndx,  uint otherParticleIdx, uint bondIdx, uint otherParticleBondIndex, uint bond_type);
        __device__ int  insertNewParticle(uint new_particle_Idx, float3 newParticlePos, uint parentParticleIndx, uint bondIdx, uint secondParticleIdx, uint otherParticleBondIndex, uint bond_type[BONDS_PER_PARTICLE]);
        __device__ void find_closest_particle_per_axis(uint particle, float3 pos, uint neighbours[6]);
        __device__ void makeBondIndxMap( uint parentParticleIndx, int bondInxMap[6]);
        __global__ void cleanBonds (int pnum);
        
        __global__ void initialize_bonds    (int ActivePoints, uint list_length, int gene);
        
        __global__ void heal                ( int ActivePoints, uint list_length, int change_list, uint startNewPoints, uint mMaxPoints);
        __global__ void lengthen_muscle     ( int ActivePoints, int list_length, int change_list, uint startNewPoints, uint mMaxPoints);
        __global__ void lengthen_tissue     ( int ActivePoints, int list_length, int change_list, uint startNewPoints, uint mMaxPoints);
        __global__ void shorten_muscle      ( int ActivePoints, int list_length, int change_list, uint startNewPoints, uint mMaxPoints);
        __global__ void shorten_tissue      ( int ActivePoints, int list_length, int change_list, uint startNewPoints, uint mMaxPoints);
        
        __global__ void strengthen_muscle   ( int ActivePoints, int list_length, int change_list, uint startNewPoints, uint mMaxPoints);
        __global__ void strengthen_tissue   ( int ActivePoints, int list_length, int change_list, uint startNewPoints, uint mMaxPoints);
        __global__ void weaken_muscle       ( int ActivePoints, int list_length, int change_list, uint startNewPoints, uint mMaxPoints);
        __global__ void weaken_tissue       ( int ActivePoints, int list_length, int change_list, uint startNewPoints, uint mMaxPoints);
        
	}

#endif
