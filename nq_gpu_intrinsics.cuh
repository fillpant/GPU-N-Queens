

#ifndef NQ_GPU_INTRINSICS_CUH
#define NQ_GPU_INTRINSICS_CUH

#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#ifndef __CUDA_ARCH__
//It appears that during non-device code compilation, MSVC attempts to compile asm() calls (!) from the bellow
//functions despite them being __device__ only declared. This 'tricks' MSVC into thinking asm is some weird preprocessor definition.
#define asm(...) printf("ASM override -- You shouldn't be seeing this");
#endif

//PTX intrinsics.
// +------------------------------+
// |      SM_20 + intrinsics      |
// |                              |
// +------------------------------+
__device__ __forceinline__ unsigned int intrin_bit_field_extract_u32(unsigned bits, unsigned from, unsigned cnt) {
	unsigned res;
	asm("bfe.u32 %0, %1, %2, %3;" : "=r"(res) : "r"(bits), "r"(from), "r"(cnt));
	return res;
}
__device__ __forceinline__ unsigned long long intrin_bit_field_extract_u64(unsigned long long bits, unsigned from, unsigned cnt) {
	unsigned long long res;
	asm("bfe.u64 %0, %1, %2, %3;" : "=l"(res) : "l"(bits), "r"(from), "r"(cnt));
	return res;
}

__device__ __forceinline__ unsigned intrin_bit_field_insert_b32(unsigned bitfield_from, unsigned bitfield_to, unsigned start, unsigned cnt) {
	//bfi.b32  d,a,b,start,len;
	unsigned res;
	asm("bfi.b32 %0, %1, %2, %3, %4;" : "=r"(res) : "r"(bitfield_from), "r"(bitfield_to), "r"(start), "r"(cnt));
	return res;
}

__device__ __forceinline__ unsigned long long intrin_bit_field_insert_b64(unsigned long long bitfield_from, unsigned long long bitfield_to, unsigned start, unsigned cnt) {
	unsigned long long res;
	asm("bfi.b64 %0, %1, %2, %3, %4;" : "=l"(res) : "l"(bitfield_from), "l"(bitfield_to), "r"(start), "r"(cnt));
	return res;
}

// The intrinsic __ffs performs the following two instructions but manipulates the result to be off by 1, which then needs to be countered by subtracting 1. Sometimes, this
// operation does not get optimised out by the compiler. This function does not suffer from the problem, provided some bit is set. If no bit is set, 0xFFFFFFFF is retruend.
__device__ __forceinline__ unsigned intrin_ffs_nosub(unsigned of) {
	unsigned res = 1;
	asm("brev.b32 %0, %1;" : "=r"(res) : "r"(of));
	asm("bfind.shiftamt.u32 %0, %1;" : "=r"(res) : "r"(res));
	return res;
}

// Find the ammount of left shifts needed to put the most significant set bit at the most significant position
// i.e in 0b00000000000000000000000000000001 the result is 31, since 31 left shifts move the set bit to the top.
// in 0b00000000000000000000000000010001 the result is 27, since then the most significant set bit  is at the top.
__device__ __forceinline__ unsigned intrin_find_leading_one_shiftamt_u32(unsigned of) {
	unsigned res;
	asm("bfind.shiftamt.u32 %0, %1;" : "=r"(res) : "r"(of));
	return res;
}

// Find the position of the most significant set bit from the right. For example:
// 0b00000000000000000000000000000001 the result is 0 -- most significant set bit is at index 0
// 0b10000000000000000000000000000000 the result is 31 -- most significant set bit is at index 31
// 0b00000000010000000000000000000001 the result is 22 -- the most significant set bit (leftmost) is at 22.
__device__ __forceinline__ unsigned intrin_find_leading_one_u32(unsigned of) {
	unsigned res;
	asm("bfind.u32 %0, %1;" : "=r"(res) : "r"(of));
	return res;
}

// +------------------------------+
// |      SM_30 + intrinsics      |
// |                              |
// +------------------------------+

// WARNING: If base is >31 this results in  undefined behaviour.
// Find the nth bit in mask that is a 1, starting from base.
// If n is positive, then the search starts from base and moves up to the most significant bit. 
// If n is negative, then the search goes from base towards the least significant bit.
// Returns the position of the nth bit from base that is a 1.
// For example, in 0111101011 with base=2 and n=3, the search will start on the 2nd bit (0) and move to the left
// until it hits the 3rd bit that is a 1, which in this case is index 6.
// Second exmaple, in mask 110011101010101011 with base=8 and n=-2 the returned index is 5 since the 2nd set bit 
// to the right of base is at 5. 
__device__ __forceinline__ unsigned intrin_find_nth_set(unsigned mask, unsigned base, unsigned n) {
	unsigned res;
	asm("fns.b32 %0, %1, %2, %3;" : "=r"(res) : "r"(mask), "r"(base), "r"(n));
	return res;
}


// +------------------------------+
// |      SM_70 + intrinsics      |
// |                              |
// +------------------------------+

// Constructs a bitmask starting at bit_idx_from with how_many bits set to one to the left.
// I.e. intrin_bit_mask_32_clamp(5,3) will start at index 5 (0 indexed) and append 3 1's to the left: 11100000
__device__ __forceinline__ unsigned intrin_bit_mask_32_clamp(unsigned bit_idx_from, unsigned how_many) {
	unsigned res;
	asm("bmsk.clamp.b32 %0, %1, %2;" : "=r"(res) : "r"(bit_idx_from), "r"(how_many));
	return res;
}

#endif