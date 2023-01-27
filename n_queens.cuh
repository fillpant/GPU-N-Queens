#ifndef N_QUEENS_CUH
#define N_QUEENS_CUH
#include "deffinitions.cuh"
#include "device_launch_parameters.h"
#include "bitsets.cuh"
#include "diagonals.cuh"
#include "nq_mem_utilities.cuh"
#include <inttypes.h>
#include <errno.h>


#define UNSET_QUEEN_INDEX 0xFF
#define NQ_FREE_COLS(state) NQ_FREE_COLS_AT(state, (state->curr_row))
#define NQ_FREE_COLS_AT(state,row) (~((state)->queens_in_columns | dad_extract(&(state)->diagonals, (row))) & N_MASK)

//Define CLZ(LL) and FFS(LL) macros cross implementation
//Macros format: param 1 is the input to the function, param 2 is the result holder variable.
//Result holder must be of type BITINDEX
#ifdef __CUDA_ARCH__ 
	//TODO examine instruction FLO.U32 <R1>, <R2>; (Find Leading One) available in Kepler+!
#define UNROLL_LOOP(i) _Pragma("unroll") 
typedef int bit_index_t;
#define CLZLL(i,res) res = __clzll(i)
#define CLZ(i,res)   res = __clz(i)
#define BREVLL(i,res) res = __brevll(i)
#define BREV(i,res) res = __brev(i)
#define POPCNT(i,res) res = __popc(i) //Native instruction
#define POPCNTLL(i,res) res = __popcll(i) //Native instruction
//Sets the value to a num representing the position of the first bit set 1-indexed. FFS(0) returns 0.
//DOES NOT CURRENTLY HAVE HARDWARE SUPPORT!
#define FFSLL(i,res) res = __ffsll(i) 
#define FFS(i,res)   res = __ffs(i)
#elif defined(_MSC_VER)
typedef unsigned long bit_index_t;
#include <intrin.h>
#define UNROLL_LOOP(i) // N/A
#pragma message ( "CLZ(LL) definitions rely on intrinsics specific to Intel Haswell / AMD ABM supporting ISAs" )
#define CLZLL(i,res) res = __lzcnt64(i)
#define CLZ(i,res)   res = __lzcntcnt(i)
#define NEED_OWN_BIT_REVERSE
#define BREV(i,res)   res = bit_reverse_lut_32(i)
#define BREVLL(i,res)   res = bit_reverse_lut_64(i)
#define POPCNT(i,res) res = __popcnt(i)
#define POPCNTLL(i,res) res = __popcnt64(i)
// It has to be different for MSVC... This expression aligns the behaviour with the CUDA/GNUCC compilers.
// Of course, you need pointers in an intrinsic and of course, it has to be a pointer to an unsigned long (???)
#define FFSLL(i,res) res = (_BitScanForward64(&res,i) ? (res + 1) : 0)
#define FFS(i,res)   res = (_BitScanForward64(&res,i)   ? (res + 1) : 0)
#elif defined(__GNUC__)
#warning "Untested builtin intrinsics!"
#define UNROLL_LOOP(i) //_Pragma("GCC unroll " #i)
typedef int bit_index_t;
#define CLZLL(i,res) res = __clzll(i)
#define CLZ(i,res)   res = __clz(i)	
#define NEED_OWN_BIT_REVERSE
//#define BREVLL(i,res) res = (((uint64_t)bit_reverse_lut_32(0xFFFFFFFF & (i >> 32))) << 32) | bit_reverse_lut_32(0xFFFFFFFF & i)
#define BREV(i,res)   res = bit_reverse_lut_32(i)
//Sets the value to a num representing the position of the first bit set 1-indexed. FFS(0) returns 0.
#define FFSLL(i,res) res = __builtin_ffsll(i)
#define FFS(i,res)   res = __builtin_ffs(i)
#define POPCNT(i,res) res = __builtin_popcount(i)
#define POPCNTLL(i,res) res = __builtin_popcountll(i)
#else
#error "UNSUPPORTED COMPILER! Intrinsic macros (ffs,clz,brev,etc.) were not been defined."
#endif

#ifdef NEED_OWN_BIT_REVERSE
#define BIT_REVERSE_ARR { \
		0x00, 0x80, 0x40, 0xC0, 0x20, 0xA0, 0x60, 0xE0, 0x10, 0x90, 0x50, 0xD0, 0x30, 0xB0, 0x70, 0xF0,\
		0x08, 0x88, 0x48, 0xC8, 0x28, 0xA8, 0x68, 0xE8, 0x18, 0x98, 0x58, 0xD8, 0x38, 0xB8, 0x78, 0xF8,\
		0x04, 0x84, 0x44, 0xC4, 0x24, 0xA4, 0x64, 0xE4, 0x14, 0x94, 0x54, 0xD4, 0x34, 0xB4, 0x74, 0xF4,\
		0x0C, 0x8C, 0x4C, 0xCC, 0x2C, 0xAC, 0x6C, 0xEC, 0x1C, 0x9C, 0x5C, 0xDC, 0x3C, 0xBC, 0x7C, 0xFC,\
		0x02, 0x82, 0x42, 0xC2, 0x22, 0xA2, 0x62, 0xE2, 0x12, 0x92, 0x52, 0xD2, 0x32, 0xB2, 0x72, 0xF2,\
		0x0A, 0x8A, 0x4A, 0xCA, 0x2A, 0xAA, 0x6A, 0xEA, 0x1A, 0x9A, 0x5A, 0xDA, 0x3A, 0xBA, 0x7A, 0xFA,\
		0x06, 0x86, 0x46, 0xC6, 0x26, 0xA6, 0x66, 0xE6, 0x16, 0x96, 0x56, 0xD6, 0x36, 0xB6, 0x76, 0xF6,\
		0x0E, 0x8E, 0x4E, 0xCE, 0x2E, 0xAE, 0x6E, 0xEE, 0x1E, 0x9E, 0x5E, 0xDE, 0x3E, 0xBE, 0x7E, 0xFE,\
		0x01, 0x81, 0x41, 0xC1, 0x21, 0xA1, 0x61, 0xE1, 0x11, 0x91, 0x51, 0xD1, 0x31, 0xB1, 0x71, 0xF1,\
		0x09, 0x89, 0x49, 0xC9, 0x29, 0xA9, 0x69, 0xE9, 0x19, 0x99, 0x59, 0xD9, 0x39, 0xB9, 0x79, 0xF9,\
		0x05, 0x85, 0x45, 0xC5, 0x25, 0xA5, 0x65, 0xE5, 0x15, 0x95, 0x55, 0xD5, 0x35, 0xB5, 0x75, 0xF5,\
		0x0D, 0x8D, 0x4D, 0xCD, 0x2D, 0xAD, 0x6D, 0xED, 0x1D, 0x9D, 0x5D, 0xDD, 0x3D, 0xBD, 0x7D, 0xFD,\
		0x03, 0x83, 0x43, 0xC3, 0x23, 0xA3, 0x63, 0xE3, 0x13, 0x93, 0x53, 0xD3, 0x33, 0xB3, 0x73, 0xF3,\
		0x0B, 0x8B, 0x4B, 0xCB, 0x2B, 0xAB, 0x6B, 0xEB, 0x1B, 0x9B, 0x5B, 0xDB, 0x3B, 0xBB, 0x7B, 0xFB,\
		0x07, 0x87, 0x47, 0xC7, 0x27, 0xA7, 0x67, 0xE7, 0x17, 0x97, 0x57, 0xD7, 0x37, 0xB7, 0x77, 0xF7,\
		0x0F, 0x8F, 0x4F, 0xCF, 0x2F, 0xAF, 0x6F, 0xEF, 0x1F, 0x9F, 0x5F, 0xDF, 0x3F, 0xBF, 0x7F, 0xFF \
	};
inline uint32_t bit_reverse_lut_32(const uint32_t i) {
	static uint8_t byte_reverse_lookup[256] = BIT_REVERSE_ARR;
	return (byte_reverse_lookup[i & 0xFF] << 24) |
		(byte_reverse_lookup[(i >> 8) & 0xFF] << 16) |
		(byte_reverse_lookup[(i >> 16) & 0xFF] << 8) |
		(byte_reverse_lookup[(i >> 24) & 0xFF]);
}
inline uint64_t bit_reverse_lut_64(const uint64_t i) {
	static uint8_t byte_reverse_lookup[256] = BIT_REVERSE_ARR;
	return ((uint64_t)bit_reverse_lut_32(i & UINT32_MAX)) << 32 | bit_reverse_lut_32(i >> 32);
}
#endif


typedef struct 
#ifndef MEM_SAVING_STATES
	__align__(8) 
#endif
{
	//Current diagonals of board
	diagonals_t diagonals;
	//Bitset where 1 means queen in that column and 0 means no queen
	bitset32_t queens_in_columns;
	//The current row
	char curr_row;
	//Could be a 6-bit partitioned bitset, only represents 0-31 and 1 value for "UNSET"
	unsigned char queen_at_index[N];
} nq_state_t;

__host__ nq_state_t init_nq_state(void);
__host__ void clear_nq_state(nq_state_t* state);
__host__ nq_mem_handle_t* nq_generate_states(const uint64_t how_many, uint64_t* const __restrict__ returned_count, unsigned* __restrict__ locked_row_end);
__host__ nq_mem_handle_t* gen_states_rowlock(const nq_state_t* const __restrict__ master_state, const unsigned lock_at_row, const uint64_t max_number_of_states, uint64_t* __restrict__ const generated_cnt);
__host__ nq_mem_handle_t* nq_generate_states_rowlock(const unsigned row_lock, uint64_t* const __restrict__ returned_count);
__host__ bool locked_advance_nq_state(nq_state_t* __restrict__ s, const unsigned top_locked_row, const unsigned int lock);
__host__ bool host_doublesweep_light_nq_state(nq_state_t* __restrict__ s);


///////////////////Inline queen placement/removal functions (host and device)
__host__ __device__ __forceinline__ void place_queen_at(nq_state_t* __restrict__ s, const unsigned char row, const unsigned char col) {
	//1. set queen_in_column
	s->queens_in_columns = bs_set_bit(s->queens_in_columns, col);
	//2. update row index
	s->queen_at_index[row] = col;
	//3. update diagonals: Construct a row of 0's with index 'col' holding a 1. 
	dad_add(&s->diagonals, 1U << col, row);
}

__host__ __device__ __forceinline__ void remove_queen_at(nq_state_t* __restrict__ s, const unsigned char row, const unsigned char col) {
	//1. clear queen_in_column
	s->queens_in_columns = bs_clear_bit(s->queens_in_columns, col);
	//2. update row index (not too important, anything bellow curr_row is considered uninitialised)
	s->queen_at_index[row] = UNSET_QUEEN_INDEX; //Indicate theres no queen in that row anymore
	//3. remove diagonal/antidiagonal
	dad_remove(&s->diagonals, 1U << col, row);
}

//////////////////Improved (inlined) device-only functions

__device__ __inline__ void device_doublesweep_light_nq_state(nq_state_t* __restrict__ s) {
	while ((s->queen_at_index[s->curr_row] == UNSET_QUEEN_INDEX)) {
		const bitset32_t free_cols = NQ_FREE_COLS(s);
		const int POPCNT(free_cols, popcnt);
		if (popcnt == 1) {
#ifdef NQ_ENABLE_EXPERIMENTAL_OPTIMISATIONS
			// __ffs compiles to brev + bfind.shiftamt.u32 -- We don't need brev here! There's only 1 set bit.
			// bfind.u32 will suffice (we don't need the shiftamt). Indexes are 0-31 hence no need to subtract 1.
			place_queen_at(s, s->curr_row, intrin_find_leading_one_u32(free_cols));
#else
			place_queen_at(s, s->curr_row, __ffs(free_cols) - 1);
#endif
			if (s->curr_row < N - 1)
				++s->curr_row;
		} else {
			break;
		}
	}
}

__device__ __inline__ void device_advance_nq_state(nq_state_t* __restrict__ s, const unsigned locked_row_end) {
	while (s->curr_row >= locked_row_end) {
		const unsigned char queen_index = s->queen_at_index[s->curr_row];
		bitset32_t free_cols = NQ_FREE_COLS(s);
		if (queen_index != UNSET_QUEEN_INDEX) {
#if __CUDA_ARCH__ > 700
			//Sliight improvement on 3090
			free_cols &= intrin_bit_mask_32_clamp(queen_index + 1, N);//TODO why not try register-caching 32 masks/LUT?
#else
			free_cols &= (N_MASK << (queen_index + 1));
#endif
			remove_queen_at(s, s->curr_row, queen_index);
		}
		if (free_cols == 0) {
			--(s->curr_row);
		} else {
			place_queen_at(s, s->curr_row, __ffs(free_cols) - 1);
			if (s->curr_row < N - 1)
				++s->curr_row;
			break;
		}
	}
}

#endif