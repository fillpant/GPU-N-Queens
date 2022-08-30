#ifndef NQ_UTILS_CUH
#define NQ_UTILS_CUH
#include <time.h>
#include "n_queens.cuh"
#include "bitsets.cuh"

#define NQ_VALIDATION_BUFFER_OK 0
#define NQ_VALIDATION_INVALID_STATE 1
#define NQ_VALIDATION_UNADVANCEABLE_STATE 2
#define NQ_VALIDATION_NULL_POINTER 4
#define NQ_VALIDATION_DUPLICATE_STATES 8

inline cudaEvent_t util_start_cuda_timer(void) {
	cudaEvent_t e;
	CHECK_CUDA_ERROR(cudaEventCreate(&e));
	CHECK_CUDA_ERROR(cudaEventRecord(e, 0));
	return e;
}

inline float util_end_event_get_time(cudaEvent_t s) {
	cudaEvent_t e = util_start_cuda_timer();
	CHECK_CUDA_ERROR(cudaEventRecord(e, 0));
	CHECK_CUDA_ERROR(cudaEventSynchronize(e));
	float time = -1;
	CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, s, e));
	return time;
}

inline uint64_t util_curr_time_ns(void) {
	struct timespec ts;
	if (timespec_get(&ts, TIME_UTC) != TIME_UTC)
		return 0;
	return (uint64_t)(1e9 * ts.tv_sec + ts.tv_nsec);
}

int util_write_nq_state_buf_to_stream(FILE* const out, const nq_state_t* const states, const uint64_t len, const unsigned char locked_at_row);
int util_read_nq_state_buf_from_stream(FILE* const in, nq_state_t** states, uint64_t* len, unsigned char* locked_at_row);
int util_minified_write_nq_states_to_stream(FILE* const out, nq_state_t* states, const uint64_t len, const unsigned char locked_at_row);
int util_minified_read_nq_states_from_stream(FILE* const in, nq_state_t** states, uint64_t* len, unsigned char* locked_at_row);
uint32_t crc32_chsm(const void* const buf, const size_t element_size, const size_t element_count);
bitset32_t util_validate_state_buffer(nq_state_t* const buf, const uint64_t buf_len, const unsigned lock_at_row);
char* util_size_to_human_readable(size_t siz);
char* util_large_integer_string_decimal_sep(char* num);
void util_visualise_nq_state(const nq_state_t* const what, const bool show_blocked);
void util_prettyprint_nq_state(nq_state_t* s);
char* util_bits_to_string(unsigned long long num, unsigned bits);
char* util_milliseconds_to_duration(uint64_t ms);
bool util_nq_states_equal(nq_state_t* one, nq_state_t* two);
__host__ __device__ void util_bits_to_buf(unsigned long long num, unsigned bits, char arr[]);
#endif