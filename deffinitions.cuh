#ifndef DEFFINITIONS_H
#define DEFFINITIONS_H
#include "stdio.h" //for CHECK_CUDA_ERROR
#include "stdlib.h" //for CHECK_CUDA_ERROR


//Define this macro to enable 'questionable' optimisations. Usually used to enable direct PTX assembly optimisations. Those may be beneficial
// in some cases but that heavily depends on the surrounding compilation of code (!) and GPU. They may also hinder the performance of the system
// so do some experiments before enabling :-)
#define NQ_ENABLE_EXPERIMENTAL_OPTIMISATIONS

// The minimum compute capability required. atomicAdd() is not available in less than 6 (600)
#define MINIMUM_COMPUTE_CAPABILITY 600

//The number N for the puzzle.
#define N 16

// Number of kernel launches for a given input to run. This is to be used when timing the kernels only. 
// Since results are not cleared or accumulated between runs, this may result in incorrect (inflated) results.
//#define PROFILING_ROUNDS 50

//The first N nsb's set. 
#define N_MASK ((1U<<N)-1U)

// If set to 1, state generation will stop increasing the locked row as soon as doing so results in fewer states being generated.
// If set to 0, state generation will continue exploring until it gets as close to the target limit, or until the board is explored almost completely. 
// For isntance, with this set to 1, and a high enough limit, generating for N=14 will stop when locking at row 10 since locking at row 11 produces fewer states
// With this set to 0 however and a high limit, the full board for N=14 will be explored producing solved states...
#define STATE_GENERATION_DOWNSLOPE_BOUNDED 1

//A number from 0 to 1 used by state generation. If a number of states X is requested, the generator will attempt to generate as many as possible near X
//but some times this may be there will be slightly more than X. Therefore X is a soft limit, and the permissible hard limit is determined as X*(1+PLAY_FACTOR)
#define STATE_GENERATION_LIMIT_PLAY_FACTOR 0.2

//If defined, state generation is done accross multiple threads on the host. This requires pthread (a port exists for windows called pthread4w which can be used)
//If undefined, state generation is serial (hence slower)
#define ENABLE_THREADED_STATE_GENERATION

//Shared memory size. Though variable for CC 8+ and potentially configurable in earlier cc's, we fix it for constant expression evaluation bellow.
//Change to match device's CC! If underestimating, multi-block residency per SM will likely compensate (CC 5.0+) but it's best to align to be a multiple of device's shared memory allocation!
#define SMEM_SIZE 49152 //48kib

//Used for optimisation (loop unrolling) WARN: MAKE SURE TO UPDATE IF NEEDED IN THE FUTURE
//Dynamic alternative: warpSize (predefined variable)
#define WARP_SIZE 32

//Number of warps in each block. 
#define WARPS_PER_BLOCK 32

//When using the single 'do it all' kernel, blocks are 1-dimensional. The size of the block impacts shared memory allocation. THIS VALUE MUST BE DIVISIBLE BY 32!
//Shared memory requirements are this many nq_state_t's plus 32 unsigned ints (aligned) per block. The size of nq_state_t varies depending on N.
//A stable definition was the bellow which acomodates all N up to 31, but produces smaller-than-possible blocks for smaller N. This in itself is no problem, and even desirable in some devices
//#define COMPLETE_KERNEL_BLOCK_THREAD_COUNT 736
//In other cases, the bellow dynamically computes the block size based on relative shared memory requirements (hopefully, the compiler will notice this is a constant expression and will replace by result)
#define COMPLETE_KERNEL_BLOCK_THREAD_COUNT MIN(((SMEM_SIZE - (WARPS_PER_BLOCK * sizeof(unsigned))) / sizeof(nq_state_t)) - (((SMEM_SIZE - (WARPS_PER_BLOCK * sizeof(unsigned))) / sizeof(nq_state_t)) % WARP_SIZE), 1024)

//Number of nq_state_t's in the initial buffer used for state generation. Pool will initially contain this many elements
// and will be resized as needed in a linear fashion, in increments of this many elements (i.e. after the first resize it will be X+X many 
// elements big where X is this constant.)
#define STATE_GENERATION_POOL_COUNT 262144

// If using the kernel-driven solver, probe the GPU for results every X many runs of the kernels.
#define READ_SUMMARY_FROM_GPU_EVERY_RUNS 5000

#define CEILING(x,y) (((x) + (y) - 1) / (y)) // ty: https://stackoverflow.com/q/11363304

#define THREAD_INDX_2D_BLOCK (threadIdx.y * blockDim.x + threadIdx.x)
#define THREAD_INDX_1D_GRID(block_id) (blockIdx.x * blockDim.x * blockDim.y + (block_id))
#define THREAD_INDX_2D_BLOCK_1D_GRID (blockIdx.x * blockDim.x * blockDim.y + THREAD_INDX_2D_BLOCK)

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < MINIMUM_COMPUTE_CAPABILITY
#error "Unsupported compute capability! Compile for compute capability >= 600"
#endif

#if N <= 0 || N >= 32
#error "N must be between 1 and 31!"
#endif

#if (BLOCK_DIM*BLOCK_DIM) % WARP_SIZE != 0
#pragma message("WARN: The current block dim does not allow 2D blocks to be partitioned uniformly to warps of 32 threads.")
#endif

#if READ_SUMMARY_FROM_GPU_EVERY_RUNS < 0
#error
#endif

#ifdef __INTELLISENSE__ //Workarround to convince VS everything is under control...
#include "intelisense_cuda_intrinsics.h"
#define CUDA_KERNEL(...)
#define __CUDA_ARCH__ 0
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif

#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))

#define BELL "\a"

#define FAIL_IF(a) {if(a) {fprintf(stderr,BELL "FAILED: Condition not met in %s at line %u.\n", __FILE__, __LINE__); exit(1);}}

//Error check for CUDA errors that exits with an error message etc.
#define CHECK_CUDA_ERROR(expr) { cudaError_t err = expr; \
								 if(err != cudaSuccess) { \
								 	fprintf(stderr, "\nERROR on line %d file %s: [%s]\n", __LINE__, __FILE__, cudaGetErrorString(err)); \
								 	exit(err); \
								 } \
								}

#endif