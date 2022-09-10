#include "GPU_NQueens.cuh"
#include "deffinitions.cuh"
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <string.h>
#include <math.h>
#include "imath.h"
#include "n_queens.cuh"
#include "assert.h"
#include "nq_utils.cuh"


__constant__ unsigned locked_row_end;

static bool check_gpu_compatibility(unsigned id, size_t size_of_states) {
	struct cudaDeviceProp gpuprop;
	CHECK_CUDA_ERROR(cudaGetDeviceProperties(&gpuprop, id));

	//alignment penalty is usually 256
	if (gpuprop.totalGlobalMem - 256 <= size_of_states) {
		fprintf(stderr, "Device %u (%s) does not have enough memory space (needed: %s).\n", id, gpuprop.name, util_size_to_human_readable(size_of_states));
		return false;
	}

	if (gpuprop.major * 100 + gpuprop.minor * 10 < MINIMUM_COMPUTE_CAPABILITY) {
		fprintf(stderr, "This program requires compute %.1f but device %u (%s) supports up to %u.%u.\n", MINIMUM_COMPUTE_CAPABILITY / 100.0, id, gpuprop.name, gpuprop.major, gpuprop.minor);
		return false;
	}

	if (gpuprop.warpSize != WARP_SIZE) {
		fprintf(stderr, "Device %u (%s) has a warp size of %u, however this program is compiled under the assumption that a warp contains %u threads.\n", id, gpuprop.name, gpuprop.warpSize, WARP_SIZE);
		return false;
	}

	if (gpuprop.computeMode == cudaComputeModeProhibited) {
		fprintf(stderr, "Device %u (%s) cannot be accessed by this process.\n", id, gpuprop.name);
		return false;
	}

	if (gpuprop.sharedMemPerBlock < sizeof(nq_state_t) * COMPLETE_KERNEL_BLOCK_THREAD_COUNT) {
		fprintf(stderr, "Device %u (%s) doesn't have enough shared memory per block for %u states.\n", id, gpuprop.name, COMPLETE_KERNEL_BLOCK_THREAD_COUNT);
		return false;
	}

	return true;
}

__host__ mpz_t gpu_solver_driver(nq_state_t* const states, const uint_least32_t state_cnt, const unsigned row_locked, const gpu_config_t* const configs, const unsigned config_cnt) {
	FAIL_IF(!states);
	FAIL_IF(state_cnt == 0);
	FAIL_IF(!configs);
	FAIL_IF(config_cnt == 0);

	// Make sure all GPUs are capable of running this computation
	for (unsigned gpuc = 0; gpuc < config_cnt; ++gpuc) {
		//Code dup...
		const unsigned states_per_device = (unsigned)floor(state_cnt / config_cnt) + (gpuc == config_cnt - 1 ? state_cnt % config_cnt : 0);
		const unsigned padded_states_per_device = (states_per_device % 32 == 0 ? states_per_device : (states_per_device + (WARP_SIZE - states_per_device % WARP_SIZE)));
		FAIL_IF(!check_gpu_compatibility(configs[gpuc].device_id, sizeof(nq_state_t) * padded_states_per_device));
	}

	typedef struct { nq_state_t* d_states; unsigned d_statecnt, d_statecnt_padded, block_count; unsigned* d_results; } gpudata_t;

	gpudata_t* gdata = (gpudata_t*)calloc(config_cnt, sizeof(gpudata_t));

	nq_state_t* tmp_states = states;

	// Prepare and launch on each gpu. 
	for (unsigned gpuc = 0; gpuc < config_cnt; ++gpuc) {
		CHECK_CUDA_ERROR(cudaSetDevice(configs[gpuc].device_id));

		//TODO check device capabilities match requirements (compute, memory, etc)!
		//TODO error handling leaves allocated memory on some devices.

		// Last device gets extra workload.
		const unsigned states_per_device = (unsigned)floor(state_cnt / config_cnt) + (gpuc == config_cnt - 1 ? state_cnt % config_cnt : 0);
		const unsigned padded_states_per_device = (states_per_device % 32 == 0 ? states_per_device : (states_per_device + (WARP_SIZE - states_per_device % WARP_SIZE)));
		const unsigned block_cnt_doublesweep_light_adv = (unsigned)ceil(states_per_device / (double)COMPLETE_KERNEL_BLOCK_THREAD_COUNT);

		printf("Preparing device %u...\n", configs[gpuc].device_id);

		nq_state_t* d_states = 0;
		unsigned* d_result = 0;
		CHECK_CUDA_ERROR(cudaMalloc(&d_states, sizeof(nq_state_t) * padded_states_per_device));
		CHECK_CUDA_ERROR(cudaMalloc(&d_result, sizeof(unsigned) * block_cnt_doublesweep_light_adv));
		if (configs[gpuc].async) {
			CHECK_CUDA_ERROR(cudaMemsetAsync(d_states, 0, sizeof(nq_state_t) * padded_states_per_device));
			CHECK_CUDA_ERROR(cudaMemcpyAsync(d_states, tmp_states, sizeof(nq_state_t) * states_per_device, cudaMemcpyHostToDevice));
			CHECK_CUDA_ERROR(cudaMemsetAsync(d_result, 0, sizeof(unsigned) * block_cnt_doublesweep_light_adv));
		} else {
			CHECK_CUDA_ERROR(cudaMemset(d_states, 0, sizeof(nq_state_t) * padded_states_per_device));
			CHECK_CUDA_ERROR(cudaMemcpy(d_states, tmp_states, sizeof(nq_state_t) * states_per_device, cudaMemcpyHostToDevice));
			CHECK_CUDA_ERROR(cudaMemset(d_result, 0, sizeof(unsigned) * block_cnt_doublesweep_light_adv));
		}
		tmp_states += states_per_device;

#ifndef __INTELLISENSE__ //Suppressing VS error...
		CHECK_CUDA_ERROR(cudaMemcpyToSymbol(locked_row_end, &row_locked, sizeof(unsigned)));
#endif
		gdata[gpuc].d_states = d_states;
		gdata[gpuc].d_statecnt = states_per_device;
		gdata[gpuc].d_statecnt_padded = padded_states_per_device;
		gdata[gpuc].d_results = d_result;
		gdata[gpuc].block_count = block_cnt_doublesweep_light_adv;
	}

	printf("Starting...\n");
	mpz_t result;
	mp_int_init(&result);
	cudaEvent_t ev = util_start_cuda_timer();
	unsigned max_blocks = 0;

#ifdef PROFILING_ROUNDS 
	float profiling_times[PROFILING_ROUNDS];
	for (unsigned profiling_run_counter = 0; profiling_run_counter < PROFILING_ROUNDS; ++profiling_run_counter) {
		max_blocks = 0;
		ev = util_start_cuda_timer();

#endif
		for (unsigned gpuc = 0; gpuc < config_cnt; ++gpuc) {
			CHECK_CUDA_ERROR(cudaSetDevice(configs[gpuc].device_id));
			max_blocks = MAX(max_blocks, gdata[gpuc].block_count);
#ifdef USE_REGISTER_ONLY_KERNEL
			kern_doitall_v2_regld CUDA_KERNEL(gdata[gpuc].block_count, COMPLETE_KERNEL_BLOCK_THREAD_COUNT)(gdata[gpuc].d_states, gdata[gpuc].d_statecnt_padded, gdata[gpuc].d_results);
#else
			kern_doitall_v2_smem CUDA_KERNEL(gdata[gpuc].block_count, COMPLETE_KERNEL_BLOCK_THREAD_COUNT)(gdata[gpuc].d_states, gdata[gpuc].d_statecnt_padded, gdata[gpuc].d_results);
#endif

		}
		for (unsigned gpuc = 0; gpuc < config_cnt; ++gpuc) {
			CHECK_CUDA_ERROR(cudaSetDevice(configs[gpuc].device_id));
			CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		}
		float time = util_end_event_get_time(ev);
		char* tmp_buf = util_milliseconds_to_duration(time);
		printf("\n\nComputation completed. Time taken: %.4fms. (%s)\n", time, tmp_buf);
		free(tmp_buf);
#ifdef PROFILING_ROUNDS
		profiling_times[profiling_run_counter] = time;
	}
	printf(">>> DATA:  ");
	float sum = 0;
	printf("%u %llu %u %u ", N, state_cnt, row_locked, COMPLETE_KERNEL_BLOCK_THREAD_COUNT);
	for (unsigned time = 0; time < PROFILING_ROUNDS; ++time) {
		printf("%.2f%c", profiling_times[time], time + 1 < PROFILING_ROUNDS ? ' ' : '\n');
		sum += profiling_times[time];
	}
	printf(">>> Total time: %.2fms Avg time: %.2fms\n", sum, sum / PROFILING_ROUNDS);
	// We don't clear the gpu buffers or anything so results are likely a multiple of the profiling rounds.
	printf(">>> No host-result summarisation. During profiling results may be inaccurate!\n");
	return result;
#else
		unsigned* per_block_results;
		CHECK_CUDA_ERROR(cudaMallocHost(&per_block_results, sizeof(unsigned) * max_blocks));
		for (unsigned gpuc = 0; gpuc < config_cnt; ++gpuc) {
			CHECK_CUDA_ERROR(cudaSetDevice(configs[gpuc].device_id));
			CHECK_CUDA_ERROR(cudaMemcpy(per_block_results, gdata[gpuc].d_results, sizeof(unsigned) * gdata[gpuc].block_count, cudaMemcpyDeviceToHost));
			for (unsigned a = 0; a < gdata[gpuc].block_count; ++a)
				mp_int_add_value(&result, per_block_results[a], &result);
			CHECK_CUDA_ERROR(cudaFree(gdata[gpuc].d_states));
			CHECK_CUDA_ERROR(cudaFree(gdata[gpuc].d_results));
		}
		CHECK_CUDA_ERROR(cudaFreeHost(per_block_results));
		char res[1024];
		mp_int_to_string(&result, 10, res, 1024);
		printf("Result: %s\n", res);
		return result;
#endif
}
#ifdef USE_REGISTER_ONLY_KERNEL




	__global__ void kern_doitall_v2_regld(const nq_state_t* const __restrict__ states, const unsigned state_cnt, unsigned* const __restrict__ sols) {
		const unsigned local_idx = threadIdx.x;
		const unsigned global_idx = blockIdx.x * blockDim.x + local_idx;
		__shared__ unsigned char smem[COMPLETE_KERNEL_BLOCK_THREAD_COUNT * N + sizeof(unsigned int) * WARP_SIZE];
		register unsigned t_sols = 0;

		if (global_idx < state_cnt) {
			unsigned char* const __restrict__ l_smem = smem + local_idx * N;
			// Since we have relatively low register pressure (on tested architectures) we can make use of the spare registers as 'memory space' for each thread 
			// instead of shared memory. Struct is broken down to components (hopefully) placed in registers as below:
			register bitset32_t queens_in_columns = states[global_idx].queens_in_columns;
			register uint64_t diagonal = states[global_idx].diagonals.diagonal, antidiagonal = states[global_idx].diagonals.antidiagonal;
			register int curr_row = states[global_idx].curr_row;
			//The queens at index array cannot be placed in a register (without a lot of effort and preprocessor 'hacks' that is) so it stays in smem.
#pragma unroll
			for (int i = 0; i < N; ++i)
				l_smem[i] = states[global_idx].queen_at_index[i];

			do {
				int res = curr_row >= locked_row_end;
				if (!__ballot_sync(0xFFFFFFFF, res))
					break; // Whole warp finished
				if (res) {
					//NOTE: In an effort to speed 
					// using 'find nth bit' (FNB) results in significantly poorer performance than conditionally shifting
					//Advance the state
					while (curr_row >= locked_row_end) {
						const register unsigned queen_index = l_smem[curr_row];
						bitset32_t free_cols = (~(queens_in_columns | dad_extract_explicit(diagonal, antidiagonal, curr_row)) & N_MASK);
						if (queen_index != UNSET_QUEEN_INDEX) {
							// Tried to change the logic to issue a single FNB (Find Nth Bit) instruction, depending on the position of the queen 
							free_cols &= (N_MASK << (queen_index + 1));
							queens_in_columns = bs_clear_bit(queens_in_columns, queen_index);
							l_smem[curr_row] = UNSET_QUEEN_INDEX;
							diagonal &= ~((1LLU << queen_index) << curr_row);
							antidiagonal &= ~((1LLU << queen_index) << (64 - N - curr_row));
						}
						if (!free_cols) {
							--curr_row;
						} else {
							//direct ffs is okay here, free_cols will have at least one set bit.
							const unsigned col = intrin_ffs_nosub(free_cols);
							queens_in_columns = bs_set_bit(queens_in_columns, col);
							l_smem[curr_row] = col;
							diagonal |= ((uint64_t)1U << col) << curr_row;
							antidiagonal |= ((uint64_t)1U << col) << (64 - N - curr_row);
							if (curr_row < N - 1)
								++curr_row;
							break;
						}
					}
				}

				__syncwarp();

				if (res) {
					while (l_smem[curr_row] == UNSET_QUEEN_INDEX) {
						const bitset32_t free_cols = (~(queens_in_columns | dad_extract_explicit(diagonal, antidiagonal, curr_row)) & N_MASK);
						const int POPCNT(free_cols, popcnt);
						if (popcnt == 1) {
#ifdef NQ_ENABLE_EXPERIMENTAL_OPTIMISATIONS
							const unsigned col = intrin_find_leading_one_u32(free_cols);
#else
							const unsigned col = __ffs(free_cols) + 1;
#endif
							queens_in_columns = bs_set_bit(queens_in_columns, col);
							l_smem[curr_row] = col;
							diagonal |= ((uint64_t)1U << col) << curr_row;
							antidiagonal |= ((uint64_t)1U << col) << (64 - N - curr_row);
							if (curr_row < N - 1) ++curr_row;
						} else break;
					}
				}
				__syncwarp();
				t_sols += (queens_in_columns == N_MASK);
			} while (1);
		}
		__syncthreads();
		t_sols = block_reduce_sum_shfl_variwarp((unsigned)t_sols, (unsigned int*)&smem[COMPLETE_KERNEL_BLOCK_THREAD_COUNT * N]);

		if (!local_idx)
			sols[blockIdx.x] += t_sols;
	}




#else 
	// Warning: state_cnt MUST be a multiple of 32 and states must be padded respectively.
	__global__ void kern_doitall_v2_smem(const nq_state_t* const __restrict__ states, const uint_least32_t state_cnt, unsigned* const __restrict__ sols) {
		const uint_least32_t local_idx = threadIdx.x;
		const uint_least32_t global_idx = blockIdx.x * blockDim.x + local_idx;
		__shared__ nq_state_t smem[COMPLETE_KERNEL_BLOCK_THREAD_COUNT + CEILING((sizeof(unsigned int) * WARP_SIZE), sizeof(nq_state_t))];
		register unsigned t_sols = 0;

		if (global_idx < state_cnt) {
			smem[local_idx].queens_in_columns = states[global_idx].queens_in_columns;
			smem[local_idx].diagonals = states[global_idx].diagonals;
			smem[local_idx].curr_row = states[global_idx].curr_row;
#pragma unroll
			for (int i = 0; i < N; ++i) {
				smem[local_idx].queen_at_index[i] = states[global_idx].queen_at_index[i];
			}
			__syncthreads();
			do {
				int res = smem[local_idx].curr_row >= locked_row_end;
				bool any_alive = __ballot_sync(0xFFFFFFFF, res);
				if (!any_alive) // Whole warp finished
					break;
				if (res) device_advance_nq_state(&smem[local_idx], locked_row_end);
				__syncwarp(); // Threads made to converge before doublesweep_light
				if (res)	device_doublesweep_light_nq_state(&smem[local_idx]);
				//__syncwarp(); // Threads made to converge before the following line
				t_sols += (smem[local_idx].queens_in_columns == N_MASK);//Non divergent
			//}
			//__syncwarp();
			} while (1);
		}
		__syncthreads();
		t_sols = block_reduce_sum_shfl_variwarp((unsigned)t_sols, (unsigned int*)&smem[COMPLETE_KERNEL_BLOCK_THREAD_COUNT]);

		if (!local_idx) {
			sols[blockIdx.x] += t_sols;
		}
	}
#endif

	__host__ nq_state_t* copy_states_to_gpu(const nq_state_t* const states, const uint64_t state_count, const gpu_config_t* const config) {
		CHECK_CUDA_ERROR(cudaSetDevice(config->device_id));
		nq_state_t* d_states;
		CHECK_CUDA_ERROR(cudaMalloc(&d_states, sizeof(nq_state_t) * state_count));
		if (config->async) {
			CHECK_CUDA_ERROR(cudaMemcpy(d_states, states, sizeof(nq_state_t) * state_count, cudaMemcpyHostToDevice));
		} else {
			CHECK_CUDA_ERROR(cudaMemcpyAsync(d_states, states, sizeof(nq_state_t) * state_count, cudaMemcpyHostToDevice));
		}
		return d_states;
	}

	__host__ void copy_states_from_gpu(nq_state_t* host_states, nq_state_t* device_states, const uint64_t state_count, const gpu_config_t* const config) {
		if (config->async) {
			CHECK_CUDA_ERROR(cudaMemcpyAsync(host_states, device_states, state_count * sizeof(nq_state_t), cudaMemcpyDeviceToHost));
		} else {
			CHECK_CUDA_ERROR(cudaMemcpy(host_states, device_states, state_count * sizeof(nq_state_t), cudaMemcpyDeviceToHost));
		}
	}
