#ifndef BITSET32_H
#define BITSET32_H

#include <inttypes.h>
#include "nq_gpu_intrinsics.cuh"

#define BS_GET_BIT(bw,idx) (0x01&((bw)>>(idx)))
#define BS_SET_BIT(bw,idx) ((bw)|(1<<(idx)))
#define BS_CLEAR_BIT(bw,idx) ((bw)&~(1<<(idx)))

#define BITSET8_MAX UINT8_MAX
#define BITSET16_MAX UINT16_MAX 
#define BITSET32_MAX UINT32_MAX
#define BITSET64_MAX UINT64_MAX

typedef uint8_t bitset8_t;
typedef uint16_t bitset16_t;
typedef uint32_t bitset32_t;
typedef uint64_t bitset64_t;

#ifdef __CUDA_VER__
extern __device__ int __popc(unsigned int);
#endif

__device__ __host__ __forceinline__ bitset32_t bs_set_bit(bitset32_t bw, const unsigned idx) {
#if defined(__CUDA_VER__) && defined(NQ_ENABLE_EXPERIMENTAL_OPTIMISATIONS)
	return intrin_bit_field_insert_b32(1, bw, idx, 1);
#else
	return BS_SET_BIT(bw, idx);
#endif
}

__device__ __host__ __forceinline__ bitset32_t bs_get_bit(bitset32_t bw, const unsigned idx) {
#if defined(__CUDA_VER__) && defined(NQ_ENABLE_EXPERIMENTAL_OPTIMISATIONS)
	return intrin_bit_field_extract_u32(bw, idx, 1);
#else
	return BS_GET_BIT(bw, idx);
#endif
}

__device__ __forceinline__ int bs_is_amo(bitset32_t bs) {
	//Option 1: fls == ffs
	//If ffs finds the bit at position x, then fls finds it at 31-x, then it's only 1 bit.
	//i.e. if ffs finds the bit at 0 (i.e. lsb set) then fls finds it at 31, 31-31==0. 
	// if other bits are set, then the msb set != lsb set.
	//return intrin_ffs_nosub(bs) == (31 - intrin_fls_nosub(bs));
	//Option 2: popc...
#ifdef __CUDA_VER__
	return __popc(bs) == 1;
#else
	return 0;
#endif
}

__device__ __host__ __forceinline__ bitset32_t bs_clear_bit(bitset32_t bw, const unsigned idx) {
#if defined(__CUDA_VER__) && defined(NQ_ENABLE_EXPERIMENTAL_OPTIMISATIONS)
	return intrin_bit_field_insert_b32(0, bw, idx, 1);
#else
	return BS_CLEAR_BIT(bw, idx);
#endif
}

#endif