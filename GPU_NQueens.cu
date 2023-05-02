#include "GPU_NQueens.cuh"
#include "deffinitions.cuh"
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <string.h>
#include <math.h>
#include "n_queens.cuh"
#include "assert.h"
#include "nq_utils.cuh"

/* Warnings and Notes:
 * - On Compute Capability (CC) 8.6 devices the performance may be impacted by cvt instructions
 * between 32 and 64 types! CC 8.0 can do 16/clock tick/sm whereas 8.6 can do 2/clock tick/sm!!!
 * - Current version of kern_doitall_v2_regld performs an intermediate between doublesweep light and DoubleSweep
 *
 *
 */

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
		fprintf(stderr, "Device %u (%s) doesn't have enough shared memory per block for %llu states.\n", id, gpuprop.name, (unsigned long long int)COMPLETE_KERNEL_BLOCK_THREAD_COUNT);
		return false;
	}

	return true;
}

__host__ uint64_t gpu_solver_driver(nq_state_t* const states, const uint_least32_t state_cnt, const unsigned row_locked, const gpu_config_t* const configs, const unsigned config_cnt) {
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

	typedef struct { nq_state_t* d_states; unsigned d_statecnt, d_statecnt_padded, block_count; nq_result_t* d_results; } gpudata_t;

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
		nq_result_t* d_result = 0;
		CHECK_CUDA_ERROR(cudaMalloc(&d_states, sizeof(nq_state_t) * padded_states_per_device));
		CHECK_CUDA_ERROR(cudaMalloc(&d_result, sizeof(nq_result_t) * block_cnt_doublesweep_light_adv));
		if (configs[gpuc].async) {
			CHECK_CUDA_ERROR(cudaMemsetAsync(d_states, 0, sizeof(nq_state_t) * padded_states_per_device));
			CHECK_CUDA_ERROR(cudaMemcpyAsync(d_states, tmp_states, sizeof(nq_state_t) * states_per_device, cudaMemcpyHostToDevice));
			CHECK_CUDA_ERROR(cudaMemsetAsync(d_result, 0, sizeof(nq_result_t) * block_cnt_doublesweep_light_adv));
		}
		else {
			CHECK_CUDA_ERROR(cudaMemset(d_states, 0, sizeof(nq_state_t) * padded_states_per_device));
			CHECK_CUDA_ERROR(cudaMemcpy(d_states, tmp_states, sizeof(nq_state_t) * states_per_device, cudaMemcpyHostToDevice));
			CHECK_CUDA_ERROR(cudaMemset(d_result, 0, sizeof(nq_result_t) * block_cnt_doublesweep_light_adv));
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
	uint64_t result = 0;
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
			//int tc = COMPLETE_KERNEL_BLOCK_THREAD_COUNT;
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
		char* tmp_buf = util_milliseconds_to_duration((uint64_t)time);
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
		unsigned long long int* per_block_results;
		CHECK_CUDA_ERROR(cudaMallocHost(&per_block_results, sizeof(nq_result_t) * max_blocks));
		for (unsigned gpuc = 0; gpuc < config_cnt; ++gpuc) {
			CHECK_CUDA_ERROR(cudaSetDevice(configs[gpuc].device_id));
			CHECK_CUDA_ERROR(cudaMemcpy(per_block_results, gdata[gpuc].d_results, sizeof(nq_result_t) * gdata[gpuc].block_count, cudaMemcpyDeviceToHost));
			for (unsigned a = 0; a < gdata[gpuc].block_count; ++a)
				result += per_block_results[a];
			CHECK_CUDA_ERROR(cudaFree(gdata[gpuc].d_states));
			CHECK_CUDA_ERROR(cudaFree(gdata[gpuc].d_results));
		}
		CHECK_CUDA_ERROR(cudaFreeHost(per_block_results));
		return result;
#endif
}
#ifdef USE_REGISTER_ONLY_KERNEL

	// Perform the steps required to place a queen at column 'col', row 'row'. 
	// Update queens_in_columns with the new positions, update queen_indexes at row to have the new value
	// Finally, update  diagonal and antidiagonal bit vectors with the new information.
	// Assumptions:
	// - col is an unsigned int
	// - row is a positive integer
	// - queens_in_columns is a 32(or more) bit unsigned integer
	// - queen_indexes is a pointer to an unsigned integer type capable of storing 'col' with at least 'row'+1 many spaces, 
	// - diagonal is a 64(or more) bit unsigned integer
	// - antidiagonal is a 64(or more) bit unsigned integer
#define PLACE_QUEEN_AT(col, row, queens_in_columns, queen_indexes, diagonal, antidiagonal) { \
	queens_in_columns = bs_set_bit(queens_in_columns, (col)); \
	queen_indexes[(row)] = (col); \
	diagonal |= (1LLU << (col)) << (row); \
	antidiagonal |= (1LLU << (col)) << (64 - N - (row)); \
}

// Performs the inverse operation to the above ^
// Assumptions are the same as above.
#define REMOVE_QUEEN_AT(col, row, queens_in_columns, queen_indexes, diagonal, antidiagonal) { \
	queens_in_columns = bs_clear_bit(queens_in_columns, (col));\
	queen_indexes[(row)] = UNSET_QUEEN_INDEX;\
	diagonal &= ~((1LLU << (col)) << (row));\
	antidiagonal &= ~((1LLU << (col)) << (64 - N - (row))); \
}

// TODO to avoid multiplication, we can switch sign as ((~X)-1). THIS ASSUMES TWOS COMPLEMENT.
// WARN: DO NOT apply CALC_CURRENT_ROW on signed row_marker! NVCC doesn't specify what behaviour to expect.
// row_marker being a signed integer results in a implementation-defined right shift operation (!). Maybe cast row_marker to unsigned for shift provided it is not signed.
#define CALC_CURRENT_ROW(center_row, row_marker) ((center_row) + (((row_marker) >> 1) + ((row_marker) & 1)) * (((row_marker) & 1) ? 1 : -1)) //Tested, works.

#define IS_QUEEN_DERIVED(idx) (((idx)&0x80))
#define MARK_QUEEN_DERIVED(idx) ((idx)|0x80)
#define GET_QUEEN_INDEX_WITHOUT_FLAG(idx) ((idx)&0x7F)


//	__global__ void kern_doitall_v2_regld_full_ds(const nq_state_t* const __restrict__ states, const unsigned state_cnt, unsigned* const __restrict__ sols) {
//		const unsigned local_idx = threadIdx.x;
//		const unsigned global_idx = blockIdx.x * blockDim.x + local_idx;
//		__shared__ unsigned char smem[COMPLETE_KERNEL_BLOCK_THREAD_COUNT * N + sizeof(unsigned int) * WARP_SIZE];
//		register unsigned t_sols = 0;
//
//		if (global_idx < state_cnt) {
//			unsigned char* const __restrict__ l_smem = smem + local_idx * N;
//			//When the iteration counter hits this value, it means it's now outside the valid board area.
//			const int iteration_counter_max = N - locked_row_end;
//			// Since we have relatively low register pressure (on tested architectures) we can make use of the spare registers as 'memory space' for each thread 
//			// instead of shared memory. Struct is broken down to components (hopefully) placed in registers as below:
//			register bitset32_t queens_in_columns = states[global_idx].queens_in_columns; //1 where there is a queen, 0 where not.
//			register uint64_t diagonal = states[global_idx].diagonals.diagonal; //DO NOT initialise on the same line. Some undefined behaviour kicks in (?)
//			register uint64_t antidiagonal = states[global_idx].diagonals.antidiagonal; //1 where queen, 0 where not
//			// Precondition: locked_row_end < N.
//			// Assumption: we start from center_row and we go DOWN first, then up, then down ... 
//			register const unsigned center_row = (unsigned)(locked_row_end + (ceil((N - locked_row_end) / (float)2) - 1));
//			// May become -1, when backtracking beyond limits. Signed overflow is undefined, hopefully yielding better perofrmance for +/-
//			register int iteration_counter = 0;
//			register int curr_row = CALC_CURRENT_ROW(center_row, iteration_counter/*was: iteration_counter. If it is that, we would start off from the centre row.*/);
//
//
//			register int backtracking = 0;
///*        */#pragma unroll
//			for (int i = 0; i < N; ++i)	l_smem[i] = ((i < locked_row_end) ? states[global_idx].queen_at_index[i] : UNSET_QUEEN_INDEX);
//
//			do {
//				//If iteration counter < 0 
//				int res = iteration_counter >= 0;
//				if (!__ballot_sync(0xFFFFFFFF, res))
//					break; // Whole warp 
//				if (res) {
//					// Advance state (i.e. place a single queen)
//					// When iteration_counter hits N-locked_row_end, then curr_row maps outside the usable area.
//					// When iteration_counter goes below 0 we've backtracked too far, nothing can save us now.
//					do {
//						register unsigned const queen_index = l_smem[curr_row];
//						// free_cols has 1 in cols which are free, 0 in cols which are not to be used.
//						register bitset32_t free_cols = (~(queens_in_columns | dad_extract_explicit(diagonal, antidiagonal, curr_row)) & N_MASK);
//						if (backtracking) {
//							//Slide the window of view over the remaining available queen positions in the current row (to eliminate all places
//							//already tried)
//							free_cols &= (N_MASK << (queen_index + 1));
//							//Remove the queen.
//							REMOVE_QUEEN_AT(queen_index, curr_row, queens_in_columns, l_smem, diagonal, antidiagonal);
//							//If there's no free column, backtrack further.
//							if (!free_cols) {
//								--iteration_counter;
//								curr_row = CALC_CURRENT_ROW(center_row, (unsigned)iteration_counter);
//								continue;
//							}
//							else { //if there is, place a queen!
//								const unsigned col = intrin_ffs_nosub(free_cols);
//								PLACE_QUEEN_AT(col, curr_row, queens_in_columns, l_smem, diagonal, antidiagonal);
//								if (iteration_counter + 1 < iteration_counter_max) {
//									++iteration_counter;
//									curr_row = CALC_CURRENT_ROW(center_row, (unsigned)iteration_counter);
//									backtracking = 0; // TODO: When this was outside the if, the code was much faster (se below comment). Has to be here, but why performance drop?
//								}
//								//backtracking=0 should be in the if statement^. If we are on the last row, move a queen, and then disable backtracking,
//								//then we go to "increment" mode afterward which means we likely double-place a queen (!)
//								break; //We successfully advanced.
//							}
//						}
//						else {
//							// (1) Problematic scenario: The last row is populated with a queen.
//							// free_cols are non-existent for that row.
//							// Backtracking is initiated, WITHOUT removing the queen in that row! -- ADDRESSED.
//							//
//							//FIX: Before advancing that state, ensure the backtracking flag is set if 
//							// the last row is populated. 
//							if (!free_cols) {
//								--iteration_counter;
//								curr_row = CALC_CURRENT_ROW(center_row, (unsigned)iteration_counter);
//								backtracking = 1;
//								continue;
//							}
//							//TODO what if we find another queen? Either derived, ... or not (which happens on the last row, since we don't increment past it) -- NOT ANYMORE
//							const unsigned col = intrin_ffs_nosub(free_cols);
//							PLACE_QUEEN_AT(col, curr_row, queens_in_columns, l_smem, diagonal, antidiagonal);
//							if ((iteration_counter + 1) < iteration_counter_max) {
//								++iteration_counter;
//								curr_row = CALC_CURRENT_ROW(center_row, (unsigned)iteration_counter);
//							}
//							else {
//								//To address Problematic Scenario (1). Set backtracking flag.
//								//Next time advancement of the current state is called, it will start by backtracking right away.
//								backtracking = 1;
//							}
//							break; //We successfully advanced.
//						}
//					} while (iteration_counter >= 0);
//				}
//				//__syncwarp();
//				/*if (res) {
//					while (l_smem[curr_row] == UNSET_QUEEN_INDEX) {
//						const bitset32_t free_cols = (~(queens_in_columns | dad_extract_explicit(diagonal, antidiagonal, curr_row)) & N_MASK);
//						const int POPCNT(free_cols, popcnt);
//						if (popcnt == 1) {
//#ifdef NQ_ENABLE_EXPERIMENTAL_OPTIMISATIONS
//							const unsigned col = intrin_find_leading_one_u32(free_cols);
//#else
//							const unsigned col = __ffs(free_cols) + 1;
//#endif
//							PLACE_QUEEN_AT(col, curr_row, queens_in_columns, l_smem, diagonal, antidiagonal);
//							if (curr_row < N - 1) ++curr_row;
//						} else break;
//					}
//				}*/
//				__syncwarp();
//#error does not consider NQ_ENABLE_EXPERIMENTAL_OPTIMISATIONS.
//				//TODO all evaluate this. Can we reduce the number of pointless additions? most of the time == evals to 0.
//				//POTENTIAL ERROR: If the advance call does not perform a change, then we are double-counting solutions.
//				t_sols += (queens_in_columns == N_MASK);
//			} while (1);
//		}
//		__syncthreads();
//		t_sols = block_reduce_sum_shfl_variwarp((unsigned)t_sols, (unsigned int*)&smem[COMPLETE_KERNEL_BLOCK_THREAD_COUNT * N]);
//
//		if (!local_idx)
//			sols[blockIdx.x] += t_sols;
//	}

	__global__ void kern_doitall_v2_regld(const nq_state_t* const __restrict__ states, const unsigned state_cnt, nq_result_t* const __restrict__ sols) {
		const unsigned local_idx = threadIdx.x;
		const unsigned global_idx = blockIdx.x * blockDim.x + local_idx;
		__shared__ unsigned char smem[COMPLETE_KERNEL_BLOCK_THREAD_COUNT * N + sizeof(unsigned int) * WARP_SIZE];
		register nq_result_t t_sols = 0;

		if (global_idx < state_cnt) {
			unsigned char* const __restrict__ l_smem = smem + local_idx * N;
			// Since we have relatively low register pressure (on tested architectures) we can make use of the spare registers as 'memory space' for each thread 
			// instead of shared memory. Struct is broken down to components (hopefully) placed in registers as below:
			register bitset32_t queens_in_columns = states[global_idx].queens_in_columns;
			register uint64_t diagonal = states[global_idx].diagonals.diagonal, antidiagonal = states[global_idx].diagonals.antidiagonal;
			register int curr_row = states[global_idx].curr_row;
			//The queens at index array cannot be placed in a register (without a lot of effort and preprocessor 'hacks' that is) so it stays in smem.
#pragma unroll
			for (int i = 0; i < N; ++i)	l_smem[i] = states[global_idx].queen_at_index[i];
#ifdef ENABLE_STATIC_HALF_SEARCHSPACE_REFLECTION_ELIMINATION
			//TODO assumes the solver will never be given a state where locked_row_end is <1
			register int mult_res_2 = l_smem[0] < (N >> 1);
#endif
			do {
				int res = curr_row >= locked_row_end;
				if (!__ballot_sync(0xFFFFFFFF, res)) break; // Whole warp finished
				if (res) {
					//Advance state (i.e. place a single queen)
					while (curr_row >= locked_row_end) {
						// Queen idx in current row
						const register unsigned queen_index = l_smem[curr_row];
						// Free columns across board
						register bitset32_t free_cols = (~(queens_in_columns | dad_extract_explicit(diagonal, antidiagonal, curr_row)) & N_MASK);
						// If there's a queen in the current row, means we're back tracking...
						if (queen_index != UNSET_QUEEN_INDEX) {
							// Remove said queen from free cols.
							free_cols &= (N_MASK << (queen_index + 1));
							REMOVE_QUEEN_AT(queen_index, curr_row, queens_in_columns, l_smem, diagonal, antidiagonal);
						}
						//If there are no more free cols, we need to backtrack (further)
						if (!free_cols) {
							--curr_row;
						}
						else {
							// If there are free cols however, we place a queen! 
							// Work out the column (first available)
							const unsigned col = intrin_ffs_nosub(free_cols);
							PLACE_QUEEN_AT(col, curr_row, queens_in_columns, l_smem, diagonal, antidiagonal);
							// If the current row is not past the end of the board, move to next.
							// This is compiled to [setp, selp, add] so no branching.
							if (curr_row < N - 1)
								++curr_row;
						done: break;
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
							PLACE_QUEEN_AT(col, curr_row, queens_in_columns, l_smem, diagonal, antidiagonal);
							if (curr_row < N - 1) ++curr_row;
						}
						else break;
					}
				}
				__syncwarp();
#if defined(NQ_ENABLE_EXPERIMENTAL_OPTIMISATIONS) && defined(USE_64_BIT_RESULT_COUNTERS)
				static_assert(sizeof(queens_in_columns) == 4, "Experimental optimisation: queens_in_columns MUST be a 32 bit type.");
				static_assert(sizeof(t_sols) == 8, "Experimental optimisation: t_sols MUST be a 64 bit type.");
				//Compiler produces setp, selp, add for the desired operation.
				//The added instruction (selp) takes extra time for no reason.
				//Eliminate it using predication! Predicate the add to only touch data
				//when the predicate holds, without converting it to an integer!
				//Result: setp followed by predicated add.
				//On isolated tests for sm_80 on 3090, compiler's version took 97 clock 
				//cycles, and mine took 95.

				asm("{\n\t"
				    ".reg .pred %p;\n\t"
					"setp.eq.u32 %p, %1, %2;\n\t"
					"@%p add.u64 %0, %0, 1;\n"
				"}" : "+l"(t_sols) : "r" (queens_in_columns) , "r"(N_MASK));
				//Curious observation: 
				//The regular snipper bellow compiled to setp.eq.s32, setlp.u64, add.s64
				//The registers operated upon were all unsigned, so the mixing of signed
				//addition and equality checks are strange in this situation?
				//Probably influenced by the compiler's internal knowledge of these
				//instructions implementations, but strange regardless. For safety, I stuck
				//with the docs and used u64/u32 where appropriate

#else 
				t_sols += (queens_in_columns == N_MASK);
#endif
			} while (1);
#ifdef ENABLE_STATIC_HALF_SEARCHSPACE_REFLECTION_ELIMINATION
			t_sols <<= mult_res_2;
#endif
		}
#ifdef USE_64_BIT_RESULT_COUNTERS
		//Each thread atomically adds its count to the global memory area.
		atomicAdd(&sols[blockIdx.x], t_sols);
#else
		//Warp shuffling intrinsics with 64 bit integer operands not available in current CUDA versions.
		__syncthreads();
		t_sols = block_reduce_sum_shfl_variwarp((unsigned)t_sols, (unsigned int*)&smem[COMPLETE_KERNEL_BLOCK_THREAD_COUNT * N]);

		if (!local_idx)
			sols[blockIdx.x] += t_sols;
#endif
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
#ifdef ENABLE_STATIC_HALF_SEARCHSPACE_REFLECTION_ELIMINATION
			t_sols <<= (states[global_idx].queen_at_index[0] < N/2);
#endif
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
		}
		else {
			CHECK_CUDA_ERROR(cudaMemcpyAsync(d_states, states, sizeof(nq_state_t) * state_count, cudaMemcpyHostToDevice));
		}
		return d_states;
	}

	__host__ void copy_states_from_gpu(nq_state_t* host_states, nq_state_t* device_states, const uint64_t state_count, const gpu_config_t* const config) {
		if (config->async) {
			CHECK_CUDA_ERROR(cudaMemcpyAsync(host_states, device_states, state_count * sizeof(nq_state_t), cudaMemcpyDeviceToHost));
		}
		else {
			CHECK_CUDA_ERROR(cudaMemcpy(host_states, device_states, state_count * sizeof(nq_state_t), cudaMemcpyDeviceToHost));
		}
	}
