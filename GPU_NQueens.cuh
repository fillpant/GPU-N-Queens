#ifndef GPU_NQUEENS_CUH
#define GPU_NQUEENS_CUH
#include "n_queens.cuh"
#include "driver_types.h"
#include "inttypes.h"
#include "imath.h"
#include "nq_gpu_intrinsics.cuh"



typedef struct {
	unsigned int device_id;
	bool async;
} gpu_config_t;

__host__ mpz_t gpu_solver_driver(nq_state_t* const states, const uint_least32_t state_cnt, const unsigned locked_row_end, const gpu_config_t* const configs, const unsigned config_cnt);
__host__ nq_state_t* copy_states_to_gpu(const nq_state_t* const states, const uint64_t state_count, const gpu_config_t* const config);
__host__ void copy_states_from_gpu(nq_state_t* host_states, nq_state_t* device_states, const uint64_t state_count, const gpu_config_t* const config);

#ifdef USE_REGISTER_ONLY_KERNEL
__global__ void kern_doitall_v2_regld(const nq_state_t* const __restrict__ states, const uint_least32_t state_cnt, unsigned* const __restrict__ sols);
#else
__global__ void kern_doitall_v2_smem(const nq_state_t* const __restrict__ states, const uint_least32_t state_cnt, unsigned* const __restrict__ sols);
#endif
//Inline helpers
__device__ __forceinline__ unsigned block_reduce_sum_shfl_variwarp(register unsigned threads_val, unsigned int* __restrict__ smem) {
	// Index of thread IN BLOCK
	const unsigned local_idx = threadIdx.x;
	// Index of thread IN WARP (0-31 for warps of size 32. AKA 'lane id')
	const unsigned thread_lane_in_warp = local_idx % warpSize;
	// The index of THE WARP in the block (w.r.t. the current thread)
	const unsigned warp_id = local_idx / warpSize;

#if  __CUDA_ARCH__  >= 800
	threads_val = __reduce_add_sync(0xFFFFFFFF, threads_val);
#else
#pragma message("Fallback warp shuffling employed for block sum reduction. Compile for compute >=8.0 for quicker alternative")
	threads_val += __shfl_down_sync(0xFFFFFFFF, threads_val, 16);
	threads_val += __shfl_down_sync(0xFFFFFFFF, threads_val, 8);
	threads_val += __shfl_down_sync(0xFFFFFFFF, threads_val, 4);
	threads_val += __shfl_down_sync(0xFFFFFFFF, threads_val, 2);
	threads_val += __shfl_down_sync(0xFFFFFFFF, threads_val, 1);
#endif
	if (thread_lane_in_warp == 0) {
		smem[warp_id] = threads_val;
	}
	__syncthreads();
	if (!warp_id) {
		// Likely predicated load instruction = no divergence.
		threads_val = local_idx < COMPLETE_KERNEL_BLOCK_THREAD_COUNT / WARP_SIZE ? smem[local_idx] : 0;

#if __CUDA_ARCH__  >= 800
		threads_val = __reduce_add_sync(0xFFFFFFFF, threads_val);
#else
		threads_val += __shfl_down_sync(0xFFFFFFFF, threads_val, 16);
		threads_val += __shfl_down_sync(0xFFFFFFFF, threads_val, 8);
		threads_val += __shfl_down_sync(0xFFFFFFFF, threads_val, 4);
		threads_val += __shfl_down_sync(0xFFFFFFFF, threads_val, 2);
		threads_val += __shfl_down_sync(0xFFFFFFFF, threads_val, 1);
#endif
	}
	return threads_val;
}


#endif