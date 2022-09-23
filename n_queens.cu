#include "deffinitions.cuh"
#include "cuda_runtime.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "n_queens.cuh"
#include "diagonals.cuh"
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include "nq_utils.cuh"

#ifdef ENABLE_THREADED_STATE_GENERATION
#include <pthread.h>

typedef struct {
	unsigned first_queen_idx, lock_at_row, error_code;
	uint64_t absolute_max_count, result_len;
	nq_state_t* result_ptr;
} internal_state_gen_thread_data_t;


//TODO code duplciation
static void* internal_gen_states_rowlocked_thread(void* datptr) {
	internal_state_gen_thread_data_t* dat = (internal_state_gen_thread_data_t*)datptr;
	dat->error_code = 0;
	nq_state_t master = init_nq_state();
	place_queen_at(&master, 0, dat->first_queen_idx);
	++master.curr_row;

	nq_state_t* approx_states = (nq_state_t*)malloc(sizeof(nq_state_t) * STATE_GENERATION_POOL_COUNT);
	uint64_t buf_len = STATE_GENERATION_POOL_COUNT;

	if (!approx_states) {
		dat->error_code = 1;
		pthread_exit(NULL);
		return NULL;
	}

	uint64_t cnt;
	for (cnt = 0; cnt < dat->absolute_max_count; ++cnt) {
		bool advanced;
		do {
			advanced = locked_advance_nq_state(&master, 1, dat->lock_at_row - 1);
		} while (advanced && (!NQ_FREE_COLS_AT(&master, master.curr_row + 1) || master.queen_at_index[master.curr_row] == UNSET_QUEEN_INDEX));
		if (advanced) {
			memcpy(&approx_states[cnt], &master, sizeof(nq_state_t));
			approx_states[cnt].curr_row++;

			if (buf_len - 1 == cnt) {
				buf_len += STATE_GENERATION_POOL_COUNT;
				nq_state_t* reallocd_approx_states = (nq_state_t*)realloc(approx_states, sizeof(nq_state_t) * buf_len);
				if (!reallocd_approx_states) {
					free(approx_states);
					dat->error_code = 2;
					pthread_exit(NULL);
					return NULL;
				} else {
					approx_states = reallocd_approx_states;
				}
			}
		} else {
			break;
		}
	}
	if (dat->absolute_max_count == cnt) {
		free(approx_states);
		dat->error_code = 3;
		pthread_exit(NULL);
		return NULL;
	}
	dat->result_len = cnt;
	dat->result_ptr = approx_states;
	pthread_exit(NULL);
	return NULL;
}

static nq_state_t* gen_states_threaded(const uint64_t how_many, uint64_t* const __restrict__ returned_count, unsigned* __restrict__ locked_row_end) {
	FAIL_IF(!returned_count);
	const uint64_t absolute_max_states = (uint64_t)(how_many * (1 + STATE_GENERATION_LIMIT_PLAY_FACTOR));
	unsigned int lock_at_row = (unsigned int)(log((double)how_many) / log(N));
	lock_at_row = lock_at_row >= N ? N - 1 : lock_at_row;
	FAIL_IF(!lock_at_row);

	typedef struct { uint64_t total_len; internal_state_gen_thread_data_t thread_data[N]; unsigned locked_at; } buf_t;

	buf_t tmp, actual; tmp = actual = { 0, {}, 0 };

	internal_state_gen_thread_data_t threads[N];
	pthread_t thread_ids[N];
	memset(&actual, 0, sizeof(buf_t));

	do {

		printf("Lock at %u ... ", lock_at_row);
		//Free last 'actual' result
		for (unsigned i = 0; i < N; ++i)
			if (actual.thread_data[i].result_ptr) {
				free(actual.thread_data[i].result_ptr);
				actual.thread_data[i].result_ptr = 0;
			}
		actual = tmp;
		memset(&tmp, 0, sizeof(buf_t));

		// Clear previous thread data and re-configure
		memset(threads, 0, sizeof(internal_state_gen_thread_data_t));
		for (unsigned i = 0; i < N; ++i) {
			threads[i].absolute_max_count = absolute_max_states;
			threads[i].first_queen_idx = i;
			threads[i].lock_at_row = lock_at_row;
			pthread_create(&thread_ids[i], NULL, internal_gen_states_rowlocked_thread, &threads[i]);
		}

		// Wait for each thread to finish
		for (unsigned i = 0; i < N; ++i)
			pthread_join(thread_ids[i], NULL);

		// Collect data from each thread.
		for (unsigned i = 0; i < N; ++i) {
			if (threads[i].error_code) {
				fprintf(stderr, "Thread %u failed with error code %u.\n", i, threads[i].error_code);
				goto loopend;
			}
			tmp.total_len += threads[i].result_len;
			tmp.thread_data[i] = threads[i];
			tmp.locked_at = lock_at_row;
		}
		printf("%llu states generated\n", tmp.total_len);
		lock_at_row++;
	} while (
#if STATE_GENERATION_DOWNSLOPE_BOUNDED == 1
		tmp.total_len >= actual.total_len &&
#endif
		lock_at_row <= N && tmp.total_len <= absolute_max_states
		);
loopend:

	//If first is null, actual wasn't initialized by the above loop
	FAIL_IF(!actual.thread_data[0].result_ptr);

	// Cleanup tmp 
	for (unsigned i = 0; i < N; ++i)
		if (tmp.thread_data[i].result_ptr)
			free(tmp.thread_data[i].result_ptr);

	nq_state_t* states = (nq_state_t*)malloc(sizeof(nq_state_t) * actual.total_len);

	uint64_t offset = 0;
	for (unsigned i = 0; i < N; ++i) {
		memcpy(states + offset, actual.thread_data[i].result_ptr, sizeof(nq_state_t) * actual.thread_data[i].result_len);
		offset += actual.thread_data[i].result_len;
		free(actual.thread_data[i].result_ptr);
	}
	FAIL_IF(offset != actual.total_len)
		* returned_count = actual.total_len;
	*locked_row_end = actual.locked_at;
	return states;
}

#else 
static nq_state_t* internal_gen_states_rowlocked(const unsigned int lock_at_row, uint64_t* const returned_count, nq_state_t* __restrict__ existing_states, const uint64_t existing_cnt, const uint64_t absolute_max_count) {
	nq_state_t master = init_nq_state();

	nq_state_t* approx_states;
	uint64_t buf_len = 0;
	if (existing_states) {
		approx_states = existing_states;
		buf_len = existing_cnt;
	} else {
		approx_states = (nq_state_t*)malloc(sizeof(nq_state_t) * STATE_GENERATION_POOL_COUNT);
		buf_len = STATE_GENERATION_POOL_COUNT;
	}
	if (!approx_states)
		return NULL;

	uint64_t cnt;
	for (cnt = 0; cnt < absolute_max_count; ++cnt) {
		bool advanced;
		do {
			advanced = locked_advance_nq_state(&master, 0, lock_at_row - 1);
		} while (advanced && //While we have made an advancement and that advancement is either:
			(!NQ_FREE_COLS_AT(&master, master.curr_row + 1) || //unsolvable further or
				master.queen_at_index[master.curr_row] == UNSET_QUEEN_INDEX //or not fully explored (i.e. unset queen at last locked row)
				));

		if (advanced) {
			memcpy(&approx_states[cnt], &master, sizeof(nq_state_t));
			approx_states[cnt].curr_row++; //We need this to be pointing at the first empty row (which is not the current value
			//Resize buffer if needed
			if (buf_len - 1 == cnt) {
				//Linear incremental resize. Whilst this means more calls to realloc, it gives finer controll over the buffer allowing
				//us to use full memory if needs be without tremendous overallocation
				buf_len += STATE_GENERATION_POOL_COUNT;
				nq_state_t* reallocd_approx_states = (nq_state_t*)realloc(approx_states, sizeof(nq_state_t) * buf_len);
				if (!reallocd_approx_states) {
					free(approx_states);
					return NULL;
				} else {
					approx_states = reallocd_approx_states;
				}
			}
		} else {
			break;
		}
	}
	if (absolute_max_count == cnt) {
		free(approx_states);
		return NULL;
	}

	*returned_count = cnt;
	return approx_states;
}

static nq_state_t* gen_states(const unsigned int how_many, uint64_t* const __restrict__ returned_count, unsigned* __restrict__ locked_row_end) {
	FAIL_IF(!returned_count);
	const uint64_t absolute_max_states = (how_many * ((uint64_t)(1 + STATE_GENERATION_LIMIT_PLAY_FACTOR)));
	unsigned int lock_at_row = (unsigned int)(log(how_many) / log(N));
	lock_at_row = lock_at_row >= N ? N - 1 : lock_at_row;
	FAIL_IF(!lock_at_row);

	typedef struct { uint64_t len; nq_state_t* ptr; unsigned locked_at; } buf_t;

	buf_t tmp, actual, dead;
	tmp = actual = dead = { 0, NULL , 0 };

	do {
		FAIL_IF(tmp.ptr && tmp.ptr == actual.ptr);
		if (actual.ptr)
			dead = actual;
		actual = tmp;

		tmp.locked_at = lock_at_row;
		tmp.ptr = internal_gen_states_rowlocked(lock_at_row, &tmp.len, dead.ptr, dead.len, absolute_max_states);
		if (!tmp.ptr) {
			printf("Failed to allocate memory or allocation surrpaces maximum state count when locking at row %u\n", lock_at_row);
			break;
		}
		lock_at_row++;
	} while (
#if STATE_GENERATION_DOWNSLOPE_BOUNDED == 1
		tmp.total_len >= actual.total_len &&
#endif
		lock_at_row <= N && tmp.len <= absolute_max_states);

	FAIL_IF(!actual.ptr);
	if (tmp.ptr)
		free(tmp.ptr);

	nq_state_t* states = NULL;
	CHECK_CUDA_ERROR(cudaMallocHost(&states, sizeof(nq_state_t) * actual.len));
	memcpy(states, actual.ptr, sizeof(nq_state_t) * actual.len);
	free(actual.ptr);
	*returned_count = actual.len;
	*locked_row_end = actual.locked_at;
	return states;
}


#endif


__host__ nq_state_t* nq_generate_states(const uint64_t how_many, uint64_t* const __restrict__ returned_count, unsigned* __restrict__ locked_row_end) {
	FAIL_IF(!returned_count);
	return
#ifdef ENABLE_THREADED_STATE_GENERATION
		gen_states_threaded
#else
		gen_states
#endif
		(how_many, returned_count, locked_row_end);
}


// Perform the same function as the GPU in advancing a state except the end point is 'fixed' and may not be the end of the board.
__host__ bool locked_advance_nq_state(nq_state_t* __restrict__ s, const unsigned locked_top_cols, const unsigned int lock) {
	while (s->curr_row >= (signed char)locked_top_cols) {
		const unsigned char queen_index = s->queen_at_index[s->curr_row];
		bitset32_t free_cols = NQ_FREE_COLS(s);

		if (queen_index != UNSET_QUEEN_INDEX) {
			free_cols &= (N_MASK << (queen_index + 1));
			remove_queen_at(s, s->curr_row, queen_index);
		}

		if (free_cols == 0)
			--(s->curr_row);
		else {
			bit_index_t	FFS(free_cols, place_at);
			place_at--;
			place_queen_at(s, s->curr_row, (unsigned char)place_at);
			if (s->curr_row < (int)lock)
				s->curr_row++;
			return true;
		}
	}
	return false;
}

// NOTE: Assumes s is not an empty state!! There must be at least one queen on it.
__host__ bool host_doublesweep_light_nq_state(nq_state_t* __restrict__ s) {
	bool changed = false;
	do {
		const unsigned char queen_index = s->queen_at_index[s->curr_row];
		if (queen_index != UNSET_QUEEN_INDEX)
			break;

		bitset32_t free_cols = NQ_FREE_COLS(s);
		const int POPCNT(free_cols, popcnt);

		//Exactly one place we can use. 
		if (popcnt == 1) {
			bit_index_t FFS(free_cols, place_at);
			place_queen_at(s, s->curr_row, (unsigned char)place_at - 1);
			if (s->curr_row < N - 1)
				s->curr_row++;
			changed = true;
		} else
			break;
	} while (1);
	return changed;
}

__host__ nq_state_t init_nq_state(void) {
	nq_state_t state;
	clear_nq_state(&state);
	return state;
}

__host__ void clear_nq_state(nq_state_t* state) {
	state->queens_in_columns = 0;
	state->diagonals = dad_init_blank();
	state->curr_row = 0;
	memset(state->queen_at_index, UNSET_QUEEN_INDEX, sizeof(state->queen_at_index));

}
