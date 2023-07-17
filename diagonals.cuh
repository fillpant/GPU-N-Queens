#ifndef DIAGONALS_CUH
#define DIAGONALS_CUH

#include <inttypes.h>
#include "bitsets.cuh"
#include "deffinitions.cuh"

typedef struct {
	uint64_t diagonal, antidiagonal;
} diagonals_t;


__host__ __device__ inline diagonals_t dad_init_blank(void) {
	return { 0,0 };
}

__host__ __device__ __forceinline__ bitset32_t dad_extract_explicit(const uint64_t diagonal, const uint64_t antidiagonal, const unsigned i) {
	//Warning: 64-N-i is a signed subtraction in PTX (!)
	return (uint32_t)((diagonal >> i) | (antidiagonal >> (N - i))) & UINT32_MAX;//UINT32_MAX dropped in PTX code.
}

__host__ __device__ inline bitset32_t dad_extract(const diagonals_t* const from, const unsigned i) {
	return dad_extract_explicit(from->diagonal, from->antidiagonal, i);
}

__host__ __device__ inline void dad_add(diagonals_t* const to, const bitset32_t row, const unsigned i) {
	to->diagonal |= ((uint64_t)row) << i;
	to->antidiagonal |= ((uint64_t)row) << (N - i);
}

__host__ __device__ inline void dad_remove(diagonals_t* const from, const bitset32_t row, const unsigned i) {
	from->diagonal &= ~(((uint64_t)row) << i);
	from->antidiagonal &= ~(((uint64_t)row) << (N - i));
}

#endif