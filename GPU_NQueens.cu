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
 *
 *
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
			int tc = COMPLETE_KERNEL_BLOCK_THREAD_COUNT;
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
		unsigned* per_block_results;
		CHECK_CUDA_ERROR(cudaMallocHost(&per_block_results, sizeof(unsigned) * max_blocks));
		for (unsigned gpuc = 0; gpuc < config_cnt; ++gpuc) {
			CHECK_CUDA_ERROR(cudaSetDevice(configs[gpuc].device_id));
			CHECK_CUDA_ERROR(cudaMemcpy(per_block_results, gdata[gpuc].d_results, sizeof(unsigned) * gdata[gpuc].block_count, cudaMemcpyDeviceToHost));
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

	//	__global__ void kern_doitall_v2_regld2(const nq_state_t* const __restrict__ states, const unsigned state_cnt, unsigned* const __restrict__ sols) {
	//		const unsigned local_idx = threadIdx.x;
	//		const unsigned global_idx = blockIdx.x * blockDim.x + local_idx;
	//		__shared__ unsigned char smem[COMPLETE_KERNEL_BLOCK_THREAD_COUNT * N + sizeof(unsigned int) * WARP_SIZE];
	//		register unsigned t_sols = 0;
	//
	//		printf("%u,%u\n", global_idx, state_cnt);
	//		if (global_idx < state_cnt) {
	//			unsigned char* const __restrict__ l_smem = smem + local_idx * N;
	//			register bitset32_t queens_in_columns = states[global_idx].queens_in_columns;
	//			register uint64_t diagonal = states[global_idx].diagonals.diagonal, antidiagonal = states[global_idx].diagonals.antidiagonal;
	//			//register int curr_row = states[global_idx].curr_row;
	//			//Equivalent to ceil((N-locked_row_end)/2)-1;
	//			register unsigned center_row = ((N - locked_row_end) >> 1) + (((N - locked_row_end) & 1) - 1);
	//			printf("%u\n", center_row);
	//			//Queens in rows are 1's where there's a queen. 0th,1st,...,curr_row-1'th bits are set to 1's since these rows are 'locked' already.
	//			register bitset32_t queens_in_rows = (~(N_MASK << states[global_idx].curr_row)) & N_MASK;
	//			register int mode = 1;//1=search, 0=backtrack
	//			//The queens at index array cannot be placed in a register (without a lot of effort and preprocessor 'hacks' that is) so it stays in smem.
	///*		  */#pragma unroll
	//			for (int i = 0; i < N; ++i)
	//				l_smem[i] = states[global_idx].queen_at_index[i];
	//			//for (unsigned i = locked_row_end; i < N; ++i)
	//			//	l_smem[i] = UNSET_QUEEN_INDEX;
	//			//////////////////////////////////////////////////////
	//			register unsigned row_marker = 1;
	//			do {
	//				//Basically, center_row + (ceil(iteration/2.0)*(iteration%2==1?1:-1);
	//				//It goes 1,-1,2,-2,3,-3 ...
	//				register int curr_row = center_row + ((row_marker >> 1) + (row_marker & 1)) * ((row_marker & 1) ? 1 : -1);
	//				int res = curr_row >= locked_row_end;
	//				if (!__ballot_sync(0xFFFFFFFF, res))
	//					break; // Whole warp finished
	//				if (res) {
	//					//Place a queen at curr_row, or find next available row following the n1-2-curve pattern:
	//					// Row X then:
	//					// Row X+1 then:
	//					// Row X-1 then:
	//					// Row X+2 then:
	//					// Row X-2 then:
	//					// Row X+3 then:
	//					// Row X-3 then:
	//					// ...
	//
	//					//If empty, place queen
	//					//If queen exists, then it MUST be a derived queen (otherwise we have a logic error somewhere)
	//					//If empty, but can't place queen: backtrack.
	//
	//					//Backtracking: go to curr_row (on iteration -1), perform queen move.
	//					//      if can't move queen, backtrack further.
	//
	//					//If this is a derived queen (i.e. msb is set), continue. 
	//					while (curr_row >= locked_row_end) {
	//						if (mode) { //If searching
	//							const register unsigned queen_index = l_smem[curr_row];
	//							if (queen_index == UNSET_QUEEN_INDEX) {
	//								register bitset32_t free_cols = (~(queens_in_columns | dad_extract_explicit(diagonal, antidiagonal, curr_row)) & N_MASK);
	//								//Place queen
	//								const unsigned col = intrin_ffs_nosub(free_cols);
	//								// Update queens in columns
	//								queens_in_columns = bs_set_bit(queens_in_columns, col);
	//								// Update queen pos in memory
	//								l_smem[curr_row] = col;
	//								// Insert queen in (anti)diagonal
	//								diagonal |= (1LLU << col) << curr_row;
	//								antidiagonal |= (1LLU << col) << (64 - N - curr_row);
	//								++row_marker;
	//								break;
	//							} else if (queen_index & 0x80) { // derived queen
	//								//TODO: do we need to mark this as a "non-derived queen"? 
	//								//Since if we place a queen in the next position after this, then backtrack, this queen will get cleared but our row_marker will be pointing at a row after it.
	//								//Most likely answer: YES, but check logic (i.e. does 'accepting' this propagated queen and moving on result in us not exploring the full search space?)
	//								//LOGIC seems to check out, so doit.
	//								l_smem[curr_row] &= 0x7F;//Accept the derived queen as a 'placed' queen and move on. DO NOT break, we want to find the row without a derived queen.
	//								++row_marker;
	//								curr_row = center_row + ((row_marker >> 1) + (row_marker & 1)) * ((row_marker & 1) ? 1 : -1);
	//								continue;
	//							} else {
	//								printf("We should've never gotten here\n");
	//								//Error.
	//							}
	//						} else { // backtracking
	//							// 1. Clear all derived queens
	//							// 2. Try move current queen to new position
	//							//		On success, set mode to 1, and break.
	//							//		On fail, keep mode to 0, and continue.
	//
	//							//1
	//							//TODO: if we are backtracking for the second,third ... ith time, there's no need to run this. OPTIMISE.
	//							for (register unsigned i = locked_row_end; i < N; ++i) {
	//								if (l_smem[i] & 0x80) {
	//									register unsigned queen_idx = l_smem[i] & 0x7F;
	//									// Remove queen from occupied columns
	//									queens_in_columns = bs_clear_bit(queens_in_columns, queen_idx);
	//									// Remove queen index from smem.
	//									l_smem[i] = UNSET_QUEEN_INDEX;
	//									// Remove queen from (anti)diagonal
	//									diagonal &= ~((1LLU << queen_idx) << i);
	//									antidiagonal &= ~((1LLU << queen_idx) << (64 - N - i));
	//								}
	//							}
	//							//2
	//							--row_marker;
	//							curr_row = center_row + ((row_marker >> 1) + (row_marker & 1)) * ((row_marker & 1) ? 1 : -1);
	//							register unsigned queen_index = l_smem[curr_row];
	//							// Remove queen from occupied columns
	//							queens_in_columns = bs_clear_bit(queens_in_columns, queen_index);
	//							// Remove queen index from smem.
	//							l_smem[curr_row] = UNSET_QUEEN_INDEX;
	//							// Remove queen from (anti)diagonal
	//							diagonal &= ~((1LLU << queen_index) << curr_row);
	//							antidiagonal &= ~((1LLU << queen_index) << (64 - N - curr_row));
	//							register bitset32_t free_cols = (~(queens_in_columns | dad_extract_explicit(diagonal, antidiagonal, curr_row)) & N_MASK);
	//							free_cols &= (N_MASK << (queen_index + 1));
	//							//IF there is a free slot, move the queen to that. 
	//							if (free_cols) {
	//								// Work out the column (first available)
	//								const unsigned col = intrin_ffs_nosub(free_cols);
	//								// Update queens in columns
	//								queens_in_columns = bs_set_bit(queens_in_columns, col);
	//								// Update queen pos in memory
	//								l_smem[curr_row] = col;
	//								// Insert queen in (anti)diagonal
	//								diagonal |= (1LLU << col) << curr_row;
	//								antidiagonal |= (1LLU << col) << (64 - N - curr_row);
	//								mode = 1;
	//								++row_marker;
	//								break;
	//							} else {
	//								continue;
	//							}
	//						}
	//					}
	//
	//					// Propagation:
	//					// - Do a pass from top to bottom, propagating.
	//					// - Do a pass from bottom to top, propagating.
	//					// ! Warning: Propatated queens must have their MSB set. row_marker doesn't move.
	//					// If an impossible row is encountered, set mode to 0 (backtracking) and stop.
	//					// If board is completed, backtrack! (may be the same as above) otherwise, errors...
	//					//TODO no need to go 0-N since we know how many center rows are populated. Do top and bottom parts instead.
	//
	//				}
	//				// Add to t_sols iff queens_in_columns is the same as N_MASK (i.e. every column has a queen
	//				t_sols += (queens_in_columns == N_MASK);
	//			} while (1);
	//		}
	//		__syncthreads();
	//		t_sols = block_reduce_sum_shfl_variwarp((unsigned)t_sols, (unsigned int*)&smem[COMPLETE_KERNEL_BLOCK_THREAD_COUNT * N]);
	//
	//		if (!local_idx)
	//			sols[blockIdx.x] += t_sols;
	//	}




//Tested, works.
#define PLACE_QUEEN_AT(col, row, queens_in_columns, queen_indexes, diagonal, antidiagonal) { \
	queens_in_columns = bs_set_bit(queens_in_columns, (col)); \
	queen_indexes[(row)] = (col); \
	diagonal |= (1LLU << (col)) << (row); \
	antidiagonal |= (1LLU << (col)) << (64 - N - (row)); \
}

//Tested, works.
#define REMOVE_QUEEN_AT(col, row, queens_in_columns, queen_indexes, diagonal, antidiagonal) { \
	queens_in_columns = bs_clear_bit(queens_in_columns, (col));\
	queen_indexes[(row)] = UNSET_QUEEN_INDEX;\
	diagonal &= ~((1LLU << (col)) << (row));\
	antidiagonal &= ~((1LLU << (col)) << (64 - N - (row))); \
}

// TODO to avoid multiplication, we can switch sign as ((~X)-1). THIS ASSUMES TWOS COMPLEMENT.
// WARN: DO NOT apply CALC_CURRENT_ROW on negative row_marker! 
// row_marked being a signed integer results in a implementation-defined right shift operation (!). Maybe cast row_marker to unsigned for shift provided it is not signed.
#define CALC_CURRENT_ROW(center_row, row_marker) ((center_row) + (((row_marker) >> 1) + ((row_marker) & 1)) * (((row_marker) & 1) ? 1 : -1)) //Tested, works.
#define IS_QUEEN_DERIVED(idx) (((idx)&0x80))
#define MARK_QUEEN_DERIVED(idx) ((idx)|0x80)
#define GET_QUEEN_INDEX_WITHOUT_FLAG(idx) ((idx)&0x7F)


	__global__ void kern_doitall_v2_regld(const nq_state_t* const __restrict__ states, const unsigned state_cnt, unsigned* const __restrict__ sols) {
		const unsigned local_idx = threadIdx.x;
		const unsigned global_idx = blockIdx.x * blockDim.x + local_idx;
		__shared__ unsigned char smem[COMPLETE_KERNEL_BLOCK_THREAD_COUNT * N + sizeof(unsigned int) * WARP_SIZE];
		register unsigned t_sols = 0;

		if (global_idx < state_cnt) {
			unsigned char* const __restrict__ l_smem = smem + local_idx * N;
			//When the iteration counter hits this value, it means it's now outside the valid board area.
			const int iteration_counter_max = N - locked_row_end;
			// Since we have relatively low register pressure (on tested architectures) we can make use of the spare registers as 'memory space' for each thread 
			// instead of shared memory. Struct is broken down to components (hopefully) placed in registers as below:
			register bitset32_t queens_in_columns = states[global_idx].queens_in_columns; //1 where there is a queen, 0 where not.
			register uint64_t diagonal = states[global_idx].diagonals.diagonal; //DO NOT initialise on the same line. Some undefined behaviour kicks in (?)
			register uint64_t antidiagonal = states[global_idx].diagonals.antidiagonal; //1 where queen, 0 where not
			// Precondition: locked_row_end < N.
			// Assumption: we start from center_row and we go DOWN first, then up, then down ... 
			register const unsigned center_row = (unsigned) (locked_row_end + (ceil((N - locked_row_end) / (float)2) - 1));
			// May become -1, when backtracking beyond limits. Signed overflow is undefined, hopefully yielding better perofrmance for +/-
			register int iteration_counter = 0;
			register int curr_row = CALC_CURRENT_ROW(center_row, iteration_counter/*was: iteration_counter. If it is that, we would start off from the centre row.*/);


			register int backtracking = 0;
/*        */#pragma unroll
			for (int i = 0; i < N; ++i)	l_smem[i] = ((i < locked_row_end) ? states[global_idx].queen_at_index[i] : UNSET_QUEEN_INDEX);

			do {
				//If iteration counter < 0 
				int res = iteration_counter >= 0;
				if (!__ballot_sync(0xFFFFFFFF, res))
					break; // Whole warp finished

				if (res) {
					/*if (global_idx == 0) {
						printf("Advance  -> lock: %u max_cnt: %d q_cols: %x center: %u cnt: %d btrck: %d curr_row: %u diag: %llx adag: %llx ", locked_row_end, iteration_counter_max, queens_in_columns, center_row, iteration_counter, backtracking,curr_row, diagonal, antidiagonal);
						for (int i = 0; i < N; ++i) printf("q[%d]=%u, ", i, l_smem[i]);
						printf("\n");
					}*/
					//Advance state (i.e. place a single queen)
					// When iteration_counter hits N-locked_row_end, then curr_row maps outside the usable area.
					// When iteration_counter goes below 0 we've backtracked too far, nothing can save us now.
					while (iteration_counter >= 0) {//TODO res was checked above. This could be a do-while to eliminate initial check
						register unsigned const queen_index = l_smem[curr_row];
						//free_cols has 1 in cols which are free, 0 in cols which are not to be used.
						register bitset32_t free_cols = (~(queens_in_columns | dad_extract_explicit(diagonal, antidiagonal, curr_row)) & N_MASK);
						if (backtracking) {
							//Slide the window of view over the remaining available queen positions in the current row (to eliminate all places
							//already tried)
							free_cols &= (N_MASK << (queen_index + 1));
							//Remove the queen.
							REMOVE_QUEEN_AT(queen_index, curr_row, queens_in_columns, l_smem, diagonal, antidiagonal);
							//If there's no free column, backtrack further.
							if (!free_cols) {
								--iteration_counter;
								curr_row = CALC_CURRENT_ROW(center_row, iteration_counter);
								continue;
							} else { //if there is, place a queen!
								const unsigned col = intrin_ffs_nosub(free_cols);
								PLACE_QUEEN_AT(col, curr_row, queens_in_columns, l_smem, diagonal, antidiagonal);
								if (iteration_counter + 1 < iteration_counter_max) {
									++iteration_counter;
									curr_row = CALC_CURRENT_ROW(center_row, iteration_counter);
									backtracking = 0; // TODO: is this right? If not, when do we reset the backtracking flag!?!?
								}
								//backtracking=0 should be in the if statement^. If we are on the last row, move a queen, and then disable backtracking,
								//then we go to "increment" mode afterward which means we likely double-place a queen (!)
								break; //We successfully advanced.
							}
						} else {
							// (1) Problematic scenario: The last row is populated with a queen.
							// free_cols are non-existent for that row.
							// Backtracking is initiated, WITHOUT removing the queen in that row! -- ADDRESSED.
							//
							//FIX: Before advancing that state, ensure the backtracking flag is set if 
							// the last row is populated. 
							if (!free_cols) {
								--iteration_counter;
								curr_row = CALC_CURRENT_ROW(center_row, iteration_counter);
								backtracking = 1;
								continue;
							}
							//TODO what if we find another queen? Either derived, ... or not (which happens on the last row, since we don't increment past it) -- NOT ANYMORE
							const unsigned col = intrin_ffs_nosub(free_cols);
							PLACE_QUEEN_AT(col, curr_row, queens_in_columns, l_smem, diagonal, antidiagonal);
							if ((iteration_counter + 1) < iteration_counter_max) {
								++iteration_counter;
								curr_row = CALC_CURRENT_ROW(center_row, iteration_counter);
								//if (global_idx == 0)printf(" NCORN");
							} else {
								//if (global_idx == 0)printf(" CORN: %d, %d, %d  ", iteration_counter +1, iteration_counter_max, iteration_counter + 1 < iteration_counter_max);
								//To address Problematic Scenario (1). Set backtracking flag.
								//Next time advancement of the current state is called, it will start by backtracking right away.
								backtracking = 1;
							}
							break; //We successfully advanced.
						}
					}
					__syncwarp(); //Dangerous. Only works in pascal, delete for other archs.
					/*if (global_idx == 0) {
						printf("Advanced -> lock: %u max_cnt: %d q_cols: %x center: %u cnt: %d btrck: %d curr_row: %u diag: %llx adag: %llx ", locked_row_end, iteration_counter_max, queens_in_columns, center_row, iteration_counter, backtracking, curr_row, diagonal, antidiagonal);
						for (int i = 0; i < N; ++i) printf("q[%d]=%u, ", i, l_smem[i]);
						printf("\n\n\n");
					}*/
				}
				//__syncwarp();
				/*if (res) {
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
						} else break;
					}
				}*/
				__syncwarp();
				//TODO all evaluate this. Can we reduce the number of pointless additions? most of the time == evals to 0.
				//POTENTIAL ERROR: If the advance call does not perform a change, then we are double-counting solutions.
				t_sols += (queens_in_columns == N_MASK);
			} while (1);
		}
		__syncthreads();
		t_sols = block_reduce_sum_shfl_variwarp((unsigned)t_sols, (unsigned int*)&smem[COMPLETE_KERNEL_BLOCK_THREAD_COUNT * N]);

		if (!local_idx)
			sols[blockIdx.x] += t_sols;
	}

	//	__global__ void kern_doitall_v2_regld(const nq_state_t* const __restrict__ states, const unsigned state_cnt, unsigned* const __restrict__ sols) {
	//		const unsigned local_idx = threadIdx.x;
	//		const unsigned global_idx = blockIdx.x * blockDim.x + local_idx;
	//		__shared__ unsigned char smem[COMPLETE_KERNEL_BLOCK_THREAD_COUNT * N + sizeof(unsigned int) * WARP_SIZE];
	//		register unsigned t_sols = 0;
	//
	//		if (global_idx < state_cnt) {
	//			unsigned char* const __restrict__ l_smem = smem + local_idx * N;
	//			// Since we have relatively low register pressure (on tested architectures) we can make use of the spare registers as 'memory space' for each thread 
	//			// instead of shared memory. Struct is broken down to components (hopefully) placed in registers as below:
	//			register bitset32_t queens_in_columns = states[global_idx].queens_in_columns;
	//			register uint64_t diagonal = states[global_idx].diagonals.diagonal, antidiagonal = states[global_idx].diagonals.antidiagonal;
	//			register int curr_row = states[global_idx].curr_row;
	//			register int direction = 0;
	//			//The queens at index array cannot be placed in a register (without a lot of effort and preprocessor 'hacks' that is) so it stays in smem.
	//#pragma unroll
	//			for (int i = 0; i < N; ++i)	l_smem[i] = states[global_idx].queen_at_index[i];
	//
	//			do {
	//				int res = curr_row >= locked_row_end;
	//				if (!__ballot_sync(0xFFFFFFFF, res)) break; // Whole warp finished
	//				if (res) {
	//					//Advance state (i.e. place a single queen)
	//					while (curr_row >= locked_row_end) {
	//						// Queen idx in current row
	//						const register unsigned queen_index = l_smem[curr_row];
	//						// Free columns across board
	//						register bitset32_t free_cols = (~(queens_in_columns | dad_extract_explicit(diagonal, antidiagonal, curr_row)) & N_MASK);
	//						// If there's a queen in the current row, means we're back tracking...
	//						if (queen_index != UNSET_QUEEN_INDEX) {
	//							// Remove said queen from free cols.
	//							free_cols &= (N_MASK << (queen_index + 1));
	//							REMOVE_QUEEN_AT(queen_index, curr_row, queens_in_columns, l_smem, diagonal, antidiagonal);
	//						}
	//						//If there are no more free cols, we need to backtrack (further)
	//						if (!free_cols) {
	//							--curr_row;
	//						} else {
	//							// If there are free cols however, we place a queen! 
	//							// Work out the column (first available)
	//							const unsigned col = intrin_ffs_nosub(free_cols);
	//							PLACE_QUEEN_AT(col, curr_row, queens_in_columns, l_smem, diagonal, antidiagonal);
	//							// If the current row is not past the end of the board, move to next.
	//							if (curr_row < N - 1)
	//								++curr_row;
	//							break;
	//						}
	//					}
	//				}
	//				__syncwarp();
	//				if (res) {
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
	//				}
	//				__syncwarp();
	//				t_sols += (queens_in_columns == N_MASK);
	//			} while (1);
	//		}
	//		__syncthreads();
	//		t_sols = block_reduce_sum_shfl_variwarp((unsigned)t_sols, (unsigned int*)&smem[COMPLETE_KERNEL_BLOCK_THREAD_COUNT * N]);
	//
	//		if (!local_idx)
	//			sols[blockIdx.x] += t_sols;
	//	}


		//	__global__ void kern_doitall_v2_regld(const nq_state_t* const __restrict__ states, const unsigned state_cnt, unsigned* const __restrict__ sols) {
		//		const unsigned local_idx = threadIdx.x;
		//		const unsigned global_idx = blockIdx.x * blockDim.x + local_idx;
		//		__shared__ unsigned char smem[COMPLETE_KERNEL_BLOCK_THREAD_COUNT * N + sizeof(unsigned int) * WARP_SIZE];
		//		register unsigned t_sols = 0;
		//
		//		if (global_idx < state_cnt) {
		//			unsigned char* const __restrict__ l_smem = smem + local_idx * N;
		//			// Since we have relatively low register pressure (on tested architectures) we can make use of the spare registers as 'memory space' for each thread 
		//			// instead of shared memory. Struct is broken down to components (hopefully) placed in registers as below:
		//			register bitset32_t queens_in_columns = states[global_idx].queens_in_columns;
		//			register uint64_t diagonal = states[global_idx].diagonals.diagonal, antidiagonal = states[global_idx].diagonals.antidiagonal;
		//			register int curr_row = states[global_idx].curr_row;
		//			register int direction = 0;
		//			//The queens at index array cannot be placed in a register (without a lot of effort and preprocessor 'hacks' that is) so it stays in smem.
		//#pragma unroll
		//			for (int i = 0; i < N; ++i)
		//				l_smem[i] = states[global_idx].queen_at_index[i];
		//
		//			do {
		//				int res = curr_row >= locked_row_end;
		//				if (!__ballot_sync(0xFFFFFFFF, res))
		//					break; // Whole warp finished
		//				if (res) {
		//
		//					//Advance state (i.e. place a single queen)
		//					while (curr_row >= locked_row_end) {
		//						// Queen idx in current row
		//						const register unsigned queen_index = l_smem[curr_row];
		//						// Free columns across board
		//						register bitset32_t free_cols = (~(queens_in_columns | dad_extract_explicit(diagonal, antidiagonal, curr_row)) & N_MASK);
		//						// If there's a queen in the current row, means we're back tracking...
		//						if (queen_index != UNSET_QUEEN_INDEX) {
		//							// Remove said queen from free cols.
		//							free_cols &= (N_MASK << (queen_index + 1));
		//							// Remove queen from occupied columns
		//							queens_in_columns = bs_clear_bit(queens_in_columns, queen_index);
		//							// Remove queen index from smem.
		//							l_smem[curr_row] = UNSET_QUEEN_INDEX;
		//							// Remove queen from (anti)diagonal
		//							diagonal &= ~((1LLU << queen_index) << curr_row);
		//							antidiagonal &= ~((1LLU << queen_index) << (64 - N - curr_row));
		//						}
		//
		//						//If there are no more free cols, we need to backtrack (further)
		//						if (!free_cols) {
		//							--curr_row;
		//						} else {
		//							// If there are free cols however, we place a queen! 
		//							// Work out the column (first available)
		//							const unsigned col = intrin_ffs_nosub(free_cols);
		//							// Update queens in columns
		//							queens_in_columns = bs_set_bit(queens_in_columns, col);
		//							// Update queen pos in memory
		//							l_smem[curr_row] = col;
		//							// Insert queen in (anti)diagonal
		//							diagonal |= (1LLU << col) << curr_row;
		//							antidiagonal |= (1LLU << col) << (64 - N - curr_row);
		//							// If the current row is not past the end of the board, move to next.
		//							if (curr_row < N - 1)
		//								++curr_row;
		//							break;
		//						}
		//						///__syncwarp();
		//					}
		//				}
		//
		//				__syncwarp();
		//
		//				if (res) {
		//					while (l_smem[curr_row] == UNSET_QUEEN_INDEX) {
		//						const bitset32_t free_cols = (~(queens_in_columns | dad_extract_explicit(diagonal, antidiagonal, curr_row)) & N_MASK);
		//						const int POPCNT(free_cols, popcnt);
		//						if (popcnt == 1) {
		//#ifdef NQ_ENABLE_EXPERIMENTAL_OPTIMISATIONS
		//							const unsigned col = intrin_find_leading_one_u32(free_cols);
		//#else
		//							const unsigned col = __ffs(free_cols) + 1;
		//#endif
		//							queens_in_columns = bs_set_bit(queens_in_columns, col);
		//							l_smem[curr_row] = col;
		//							diagonal |= ((uint64_t)1U << col) << curr_row;
		//							antidiagonal |= ((uint64_t)1U << col) << (64 - N - curr_row);
		//							if (curr_row < N - 1) ++curr_row;
		//						} else break;
		//					}
		//				}
		//				__syncwarp();
		//				t_sols += (queens_in_columns == N_MASK);
		//			} while (1);
		//		}
		//		__syncthreads();
		//		t_sols = block_reduce_sum_shfl_variwarp((unsigned)t_sols, (unsigned int*)&smem[COMPLETE_KERNEL_BLOCK_THREAD_COUNT * N]);
		//
		//		if (!local_idx)
		//			sols[blockIdx.x] += t_sols;
		//	}


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
		//t_sols = block_reduce_sum_shfl_variwarp((unsigned)t_sols, (unsigned int*)&smem[COMPLETE_KERNEL_BLOCK_THREAD_COUNT]);

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
