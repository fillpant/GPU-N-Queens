#include "device_launch_parameters.h"
#include <stdlib.h>
#include <string.h>
#include "diagonals.cuh"
#include "bitsets.cuh"
#include "n_queens.cuh"
#include "deffinitions.cuh"
#include "nq_utils.cuh"
#include <stdio.h>
#include "GPU_NQueens.cuh"
#include <cuda_runtime.h>
#include <math.h>
#include "nq_gpu_intrinsics.cuh"
#include "cargs.h"
#include "nq_mem_utilities.cuh"

static void check_if_known_result(unsigned long long int res) {
	static unsigned long long int results[] = { 1ULL, 0ULL, 0ULL, 2ULL, 10ULL, 4ULL, 40ULL, 92ULL, 352ULL, 724ULL, 2680ULL, 14200ULL, 73712ULL, 365596ULL, 2279184ULL, 14772512ULL, 95815104ULL, 666090624ULL, 4968057848ULL, 39029188884ULL, 314666222712ULL, 2691008701644ULL, 24233937684440ULL, 227514171973736ULL, 2207893435808352ULL, 22317699616364044ULL, 234907967154122528ULL };
	if (N >= sizeof(results) / sizeof(unsigned long long int))
		return;
	if (results[N - 1] != res) {
		printf("\n\n#################################################\n          INCORRECT RESULT!!\n#################################################");
	}
}

void testDAD(void) {
	diagonals_t dad = dad_init_blank();
	dad_add(&dad, 0b0100, 0);
	dad_add(&dad, 0b0001, 1);
	dad_add(&dad, 0b1000, 2);
	dad_add(&dad, 0b0010, 3);


	dad_remove(&dad, 0b0001, 1);


	for (unsigned i = 0; i < N; i++) {
		bitset32_t rslt = dad_extract(&dad, i);
		for (int i = 31; i >= 0; --i)
			printf("%d", BS_GET_BIT(rslt, i));
		printf("\n");
	}
}

void testDiagonalSingleu64() {

	diagonals_t dad = dad_init_blank();
	dad_add(&dad, 0b00000000000000000000000000000001U, 0);
	dad_add(&dad, 0b00000000000000000000000000100000U, 1);
	dad_add(&dad, 0b00000000000000000000000000000100U, 2);
	dad_add(&dad, 0b00000000000000000000010000000001U, 3);
	dad_add(&dad, 0b00000000000000000010000000000001U, 4);
	dad_add(&dad, 0b00000000100000000000000000000001U, 5);
	dad_add(&dad, 0b00000000000000010000000000000001U, 6);

	printf("DIAG: %s\n", util_bits_to_string(dad.diagonal, 64));
	printf("ANTI: %s\n\n\n", util_bits_to_string(dad.antidiagonal, 64));


#define EN 16
	uint64_t d = 0, ad = 0;

#define DAD_ADD(row,i)d |= ((uint64_t)row << i)
	DAD_ADD(0b00000000000000000000000000000001U, 0);
	DAD_ADD(0b00000000000000000000000000100000U, 1);
	DAD_ADD(0b00000000000000000000000000000100U, 2);
	DAD_ADD(0b00000000000000000000010000000001U, 3);
	DAD_ADD(0b00000000000000000010000000000001U, 4);
	DAD_ADD(0b00000000100000000000000000000001U, 5);
	DAD_ADD(0b00000000000000010000000000000001U, 6);

	ad = d << 64 - EN;

	printf("DIAG: %s\n", util_bits_to_string(d, 64));
	printf("ANTI: %s\n\n\n", util_bits_to_string(ad, 64));


	for (unsigned int i = 0; i < 32; i++) {
		unsigned int val = (uint32_t)((d >> i) | (ad >> (64 - N - i))) & UINT32_MAX;
		printf("R: %s Olivers:", util_bits_to_string(val, 32));
		printf(" %s\n", util_bits_to_string(dad_extract(&dad, i), 32));
	}


}

static char* pre_gen_state_files[] = {
	".\\pre_generated_states\\n10_6358_lk7_valid.nqsb",
	".\\pre_generated_states\\n11_30040_lk7_valid.nqsb",
	".\\pre_generated_states\\n12_152604_lk8_valid.nqsb",
	".\\pre_generated_states\\n13_802192_lk9_valid.nqsb",
	".\\pre_generated_states\\n14_4568428_lk10_valid.nqsb",
	".\\pre_generated_states\\n15_27851188_lk11_valid.nqsb",
	".\\pre_generated_states\\n16_44903824_lk9_valid.nqsb",
	".\\pre_generated_states\\n17_35255116_lk8_valid.nqsb",
	".\\pre_generated_states\\n18_75913592_lk8_valid.nqsb",
	".\\pre_generated_states\\n19_26512942_lk7_valid.nqsb",
	".\\pre_generated_states\\n20_45562852_lk7_valid.nqsb",
	".\\pre_generated_states\\n21_75580634_lk7_valid.nqsb",
	".\\pre_generated_states\\n22_13145292_lk6_valid.nqsb",
	".\\pre_generated_states\\n23_18908302_lk6_valid.nqsb",
	".\\pre_generated_states\\n24_26670584_lk6_valid.nqsb",
	".\\pre_generated_states\\n25_36961170_lk6_valid.nqsb",
	".\\pre_generated_states\\n26_50409604_lk6_valid.nqsb"
};
static unsigned file_arr_starting_en = 10;

//static nq_state_t* guarded_get_states(const uint64_t count, uint64_t* actual, unsigned* locked_row_end, bool load_pre_gen) {
//	FAIL_IF(count > UINT_MAX);
//	nq_state_t* states;
//	if (load_pre_gen && N >= file_arr_starting_en && N - file_arr_starting_en < sizeof(pre_gen_state_files) / sizeof(char*)) {
//		FILE* f = fopen(pre_gen_state_files[N - file_arr_starting_en], "rb");
//		if (f) {
//			printf("Loading pre-generated states from file...\n");
//			unsigned char locked;
//			nq_state_t* tmp_buf;
//			int res = util_read_nq_states_from_stream(f, &tmp_buf, actual, &locked, false);
//			fclose(f);
//			FAIL_IF(res);
//			CHECK_CUDA_ERROR(cudaMallocHost(&states, sizeof(nq_state_t) * *actual));
//			// Transfer to page locked memory
//			memcpy(states, tmp_buf, sizeof(nq_state_t) * *actual);
//			free(tmp_buf);
//
//			*locked_row_end = locked;
//			return states;
//		}
//	}
//	nq_mem_handle_t* hand = nq_generate_states((unsigned int)count, actual, locked_row_end);
//	
//	states = (nq_state_t*) nq_mem_transfer_and_free(hand);
//
//	FAIL_IF(!states);
//	return states;
//}
//
//void testGenStates(void) {
//	uint64_t actually_generated = 0;
//	unsigned locked_row;
//	nq_state_t* states = guarded_get_states(4096, &actually_generated, &locked_row, false);
//
//	for (uint64_t i = 0; i < actually_generated; ++i) {
//		util_visualise_nq_state(states + i, true);
//	}
//
//	CHECK_CUDA_ERROR(cudaFreeHost(states));
//}
//
//void testdoublesweep_light(void) {
//	//Targeted testing for N=4
//	// state 1, queens at 1,0 and 3,1, doublesweep_light should fill both rows bellow returning true.
//	nq_state_t one = init_nq_state();
//	place_queen_at(&one, 0, 1);
//	place_queen_at(&one, 1, 3);
//	one.curr_row = 2;
//
//	printf("S1 Before doublesweep_light:\n");
//	util_visualise_nq_state(&one, true);
//
//	bool doublesweep_light = host_doublesweep_light_nq_state(&one);
//	printf("S1 doublesweep_light success? %d. After doublesweep_light:\n", doublesweep_light);
//	util_visualise_nq_state(&one, true);
//	/// state 2, queen at 0,0 only, doublesweep_light should be 'other' for that row
//	clear_nq_state(&one);
//	place_queen_at(&one, 0, 0);
//	one.curr_row = 1;
//	printf("S2 Before doublesweep_light:\n");
//	util_visualise_nq_state(&one, true);
//
//	doublesweep_light = host_doublesweep_light_nq_state(&one);
//	printf("S2 doublesweep_light success? %d. After doublesweep_light:\n", doublesweep_light);
//	util_visualise_nq_state(&one, true);
//
//	/// state 3, queens at 0,0 and 3,1, doublesweep_light should populate the next row only.
//	clear_nq_state(&one);
//	place_queen_at(&one, 0, 0);
//	place_queen_at(&one, 1, 3);
//	one.curr_row = 2;
//	printf("S3 Before doublesweep_light:\n");
//	util_visualise_nq_state(&one, true);
//
//	doublesweep_light = host_doublesweep_light_nq_state(&one);
//	printf("S3 doublesweep_light success? %d. After doublesweep_light:\n", doublesweep_light);
//	util_visualise_nq_state(&one, true);
//
//}
//
//void testDoItAllKernel(void) {
//	printf("Solving N=%u.\n", N);
//	uint64_t actually_generated = 0;
//	unsigned locked_row_end = 0;
//
//	const uint64_t target_state_cnt = 80000000;
//	printf("Generating states... (soft limit: %" PRIu64", hard-limit %" PRIu64")\n", target_state_cnt, (uint64_t)(target_state_cnt * (1 + STATE_GENERATION_LIMIT_PLAY_FACTOR)));
//	nq_state_t* states = guarded_get_states(target_state_cnt, &actually_generated, &locked_row_end, true);
//
//	printf("Generated %" PRIu64" states (Approx. %s in memory)\n", actually_generated, util_size_to_human_readable(sizeof(nq_state_t) * actually_generated));
//	printf("Validating state buffer (this may take a while)...\n");
//	/*bitset32_t problems = util_validate_state_buffer(states, actually_generated, locked_row_end);
//	if (problems) {
//		printf(BELL "!INVALID STATE BUFFER GENERATED!\n");
//		printf("\tIs the buffer pointer null? %3s\n", problems & NQ_VALIDATION_NULL_POINTER ? "Yes" : "No");
//		printf("\tHas duplicate states?       %3s\n", problems & NQ_VALIDATION_DUPLICATE_STATES ? "Yes" : "No");
//		printf("\tHas invalid state(s)?       %3s\n", problems & NQ_VALIDATION_INVALID_STATE ? "Yes" : "No");
//		printf("\tHas unadvanceable state(s)? %3s\n", problems & NQ_VALIDATION_UNADVANCEABLE_STATE ? "Yes" : "No");
//		CHECK_CUDA_ERROR(cudaFreeHost(states));
//		return;
//	} else {
//		printf("Sate buffer is valid.\n");
//	}*/
//	printf("\n");
//
//	gpu_config_t configs[2];
//	configs[0].device_id = 0;
//	configs[0].async = true;
//
//	configs[1].device_id = 1;
//	configs[1].async = true;
//
//	FAIL_IF(actually_generated > UINT_LEAST32_MAX);
//
//	char res[32];
//	uint64_t result = gpu_solver_driver(states, (uint_least32_t)actually_generated, locked_row_end, configs, 1);
//	sprintf(res, "%" PRIu64, result);
//
//	char* formatted_res = util_large_integer_string_decimal_sep(res);
//	printf("solution_count(%u) = %s (%s)\n", N, res, formatted_res);
//	free(formatted_res);
//	CHECK_CUDA_ERROR(cudaFreeHost(states));
//
//	//gpu_config_t config;
//	//config.async = false;
//	//config.device_id = 1;
//	//nq_state_t* const d_states = copy_states_to_gpu(states, actually_generated, &config);
//	//unsigned long long int* d_result, result;
//	//CHECK_CUDA_ERROR(cudaMalloc(&d_result, sizeof(unsigned long long int)));
//	//CHECK_CUDA_ERROR(cudaMemset(d_result, 0, sizeof(unsigned long long int)));
//	//
//	////const dim3 block(BLOCK_DIM, BLOCK_DIM);
//	//const unsigned threads_per_block = COMPLETE_KERNEL_BLOCK_THREAD_COUNT;
//	//const unsigned block_cnt_doublesweep_light_adv = (unsigned)ceil(actually_generated / (double)threads_per_block);
//	//cudaEvent_t ev = util_start_cuda_timer();
//	//kern_doitall_v2 CUDA_KERNEL(block_cnt_doublesweep_light_adv, threads_per_block)(d_states, actually_generated, d_result);
//	//CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//	//float time = util_end_event_get_time(ev);
//	//CHECK_CUDA_ERROR(cudaMemcpy(&result, d_result, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
//	//
//	//printf("\nResult: %llu in %.4fms\n", result, time);
//	//check_result(result);
//	//
//	//CHECK_CUDA_ERROR(cudaFree(d_states));
//	//CHECK_CUDA_ERROR(cudaFree(d_result));
//	//CHECK_CUDA_ERROR(cudaFreeHost(states));
//}

//void testStateBufferValidation() {
//	uint64_t actual;
//	unsigned locked;
//	nq_state_t* states = nq_generate_states(1000, &actual, &locked);
//	bitset32_t res = util_validate_state_buffer(states, actual, locked);
//	printf("\n\nValidation result (Expect all 0's): %s\n", util_bits_to_string(res, 32));
//	CHECK_CUDA_ERROR(cudaFreeHost(states));
//
//	states = nq_generate_states(1000, &actual, &locked);
//	res = util_validate_state_buffer(states, actual, locked);
//	printf("\nValidation result (Expect 1011): %s\n", util_bits_to_string(res, 32));
//	CHECK_CUDA_ERROR(cudaFreeHost(states));
//
//	nq_state_t tmp = init_nq_state();
//	res = util_validate_state_buffer(&tmp, 1, 3);
//	printf("\nValidation result (Expect 11): %s\n", util_bits_to_string(res, 32));
//
//}
//
//__global__ void intrinsic_test_kernel(unsigned mask, unsigned base, unsigned offset) {
//	unsigned i = THREAD_INDX_1D_GRID(threadIdx.x) & (N_MASK << 7);
//	//for (unsigned i = 0; i < UINT_MAX; ++i) {
//		if (intrin_ffs_nosub(i) != intrin_find_nth_set(i, base, offset)) {
//			printf("!%u!\n", i);
//		}
//		//printf("FFSNI: %u\n", intrin_ffs_nosub(i));
//		//printf("FS: %u\n", intrin_find_nth_set(i, base, offset));
//	//}
//}
//
//void testDirectPTXIntrinsics(void) {
//	//Start offset many bits left from base bit in mask, and find the index of the set bit there
//	intrinsic_test_kernel CUDA_KERNEL(4194304, 1024)(0b11111111000111111 & (N_MASK << 6), 0, 1);
//	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//}
//
//nq_state_t* preGenerateStates(char* dir_path, char* fname, uint64_t* cnt, unsigned* lock_at) {
//	unsigned lock;
//	uint64_t target = 80000000, actual = 0;
//	nq_state_t* gen = guarded_get_states(target, &actual, &lock, false);
//
//	printf("Generated %llu states (approx %s) by locking at row %u. Validating...\n", actual, util_size_to_human_readable(sizeof(nq_state_t) * actual), lock);
//	if (gen && util_validate_state_buffer(gen, actual, lock)) {
//		printf("INVALID STATE BUFFER");
//		return NULL;
//	}
//
//	*cnt = actual;
//	*lock_at = lock;
//	char name[2048];
//	memset(name, 0, sizeof(char) * 2048);
//	sprintf(name, "%s\\n%u_%" PRId64"_lk%u_valid.nqsb", dir_path, N, actual, lock);
//
//	strcpy(fname, name);
//
//	FILE* out = fopen(name, "wb");
//	if (!out) {
//		printf("FAILED to open file %s\n", name);
//		exit(-1);
//	}
//	int res = util_write_nq_states_to_stream(out, gen, actual, lock);
//	if (!res) {
//		printf("Wrote successfully.");
//		fclose(out);
//	} else {
//		printf("FAILED");
//	}
//	return gen;
//}
//
//__global__ void sum_kern_shuffle(unsigned int* data) {
//	const uint_least32_t local_idx = threadIdx.x;
//	const uint_least32_t global_idx = blockIdx.x * blockDim.x + local_idx;
//
//	register unsigned int threads_val = data[global_idx];
//	threads_val += __shfl_down_sync(0xFFFFFFFF, threads_val, 16);
//	threads_val += __shfl_down_sync(0xFFFFFFFF, threads_val, 8);
//	threads_val += __shfl_down_sync(0xFFFFFFFF, threads_val, 4);
//	threads_val += __shfl_down_sync(0xFFFFFFFF, threads_val, 2);
//	threads_val += __shfl_down_sync(0xFFFFFFFF, threads_val, 1);
//	//threads_val = __reduce_add_sync(0xFFFFFFFF, threads_val);
//
//	if (local_idx % 32 == 0) {
//		data[global_idx] = threads_val;
//	}
//}
//
//int test_stuff(void) {
//	uint64_t gen;
//	unsigned row;
//	nq_state_t* states = guarded_get_states(99999, &gen, &row, false);
//
//	while (states + --row > states) {
//		util_prettyprint_nq_state(states + row);
//	}
//	/*uint64_t gen;
//	unsigned row;
//	nq_state_t* states = guarded_get_states(36, &gen, &row, false);
//
//	for (uint64_t s = 0; s < gen; ++s) {
//		host_doublesweep_light_nq_state(states + s);
//		if (states[s].curr_row == N - 1) {
//			util_visualise_nq_state(states + s, true);
//		}
//	}*/
//
//	return 0;
//}
//
//int badmain(void) {
//
//	//Hack
//	CHECK_CUDA_ERROR(cudaSetDevice(0));
//	//test_stuff();
//	/*unsigned int* d_ptr;
//	CHECK_CUDA_ERROR(cudaMalloc(&d_ptr, sizeof(unsigned int) * 1024 * 1024 * 1024*2));
//	double time = 0;
//	for (unsigned i = 0; i < 100; i++) {
//		cudaEvent_t ev = util_start_cuda_timer();
//
//		sum_kern_shuffle CUDA_KERNEL(1024 *2* 1024, 1024)(d_ptr);
//		CHECK_CUDA_ERROR(cudaDeviceSynchronize());*/
//
//	//testDirectPTXIntrinsics();
//	testDoItAllKernel();
//
//	//char fname[1024];
//	//uint64_t genlen;
//	//unsigned gen_lock_at;
//	//nq_state_t* generated = preGenerateStates(".\\pre_generated_states", fname, &genlen, &gen_lock_at);
//
//	//printf("name: {%s}\n", fname);
//
//	//unsigned char lock_at;
//	//uint64_t len;
//
//	//nq_state_t* s;
//	//int ret = util_read_nq_states_from_stream(fopen(fname, "rb"), &s, &len, &lock_at, false);
//	//if (ret)
//	//	printf("ERROR: %d\n", ret);
//	//for (uint64_t i = 0; i < len; ++i) {
//	//	//uint32_t crc1 = crc32_chsm(generated + i, sizeof(nq_state_t), 1);
//	//	//uint32_t crc2 = crc32_chsm(s + i, sizeof(nq_state_t), 1);
//
//	//	if (!util_nq_states_equal(generated + i, s + i)) {
//	//		printf("State %u is not the same!\n", i);
//	//		uint8_t* buf = (uint8_t*)generated + i;
//	//		uint8_t* buf2 = (uint8_t*)s + i;
//	//		for (unsigned j = 0; j < sizeof(nq_state_t); ++j)
//	//			printf("%u,%u\n", buf[j], buf2[j]);
//
//	//		printf("Comp: %d\n", strncmp((char*)buf, (char*)buf2, sizeof(nq_state_t)));
//	//	}
//	//}
//	//cudaFreeHost(generated);
//
//
//	//if (ret) {
//	//	printf("Failed. %d\n", ret);
//	//} else if (!util_validate_state_buffer(s, len, lock_at)) {
//	//	printf("valid %llu states locked at %u\n", len, lock_at);
//	//}
//
//	//LINT* b = new_lint_num_degree(35, 0);
//	//LINT* b = new_lint_str("35");
//	//expand_size(b);
//	//expand_size(b);
//	//print_lint(b);
//	//testDAD();
//	//testNQAdvance();
//	//testGenStates();
//	//testdoublesweep_light();
//	//testLINT();
//	//testdoublesweep_lightKernel();
//	//testAdvanceKernel();
//	//testSumarisationKernel2();
//	//testSumarisationKernel();
//	//testGenStates();
//	//testFullSolver();
//	//	shuffleSumReductionTest();
//	//testImath();
//	//stepThroughSolver();
//	//testDiagonalSingleu64();
//
////testDirectPTXIntrinsics();
//
////	testDoItAllKernel();
//	//	testStateBufferValidation();
//	//	testStateBufferValidation();
//	// 
//	//testDirectPTXIntrinsics();
//	//nq_state_t s = init_nq_state();
//	//place_queen_at(&s, 0, 0);
//	//place_queen_at(&s, 1, 7);
//	////place_queen_at(&s, 2, 5);
//	//s.curr_row = 2;
//	//s.locked_row_end = 1;
//	//printf("State:\n");
//	//util_visualise_nq_state(&s, true);
//	//gpu_configuration_t config;
//	//config.device_id = 0;
//	//config.async = false;
//	//config.block_cnt = 1;
//	//nq_state_t* d_s = copy_states_to_gpu(&s, 1, &config);
//	//kern_advance_states CUDA_KERNEL(1,1)(d_s, 1);
//	//printf("CPU:\n");
//	//advance_nq_state(&s);
//	//util_visualise_nq_state(&s, true);
//	////util_prettyprint_nq_state(&s);
//	//
//	//printf("\nGPU:\n");
//	//CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//	//copy_states_from_gpu(&s, d_s, 1, &config);
//	//util_visualise_nq_state(&s, true);
//	////util_prettyprint_nq_state(&s);
//	cag_option_get(NULL);
//	///////uint64_t result = 2279184;
//	///////
//	///////
//	///////mpz_t  a;
//	///////mp_int_init(&a);
//	/////////TODO BAD
//	///////uint64_t tmp = result;
//	///////mpz_t tmp_mp;
//	///////mp_int_init(&tmp_mp);
//	///////mp_int_set_uvalue(&tmp_mp, MP_SMALL_MAX);
//	///////
//	///////while (tmp > MP_SMALL_MAX) {
//	///////	mp_int_add(&a, &tmp_mp, &a);
//	///////	tmp -= MP_SMALL_MAX;
//	///////}
//	///////mp_int_set_uvalue(&tmp_mp, (mp_usmall)tmp);
//	///////mp_int_add(&a, &tmp_mp, &a);
//	///////
//	///////char* buf;
//	///////int    len;
//	///////
//	///////
//	///////
//	///////
//	///////
//	////////* Print out the result */
//	///////len = mp_int_string_len(&a, 10);
//	///////buf = (char*)calloc(len, sizeof(*buf));
//	///////mp_int_to_string(&a, 10, buf, len);
//	///////printf("result = %s\n", buf);
//	///////free(buf);
//	///////
//	////////* Release memory occupied by mpz_t structures when finished */
//	///////mp_int_clear(&a);
//	///////
//	//Hack
//	CHECK_CUDA_ERROR(cudaSetDevice(0));
//	//	const float ms = util_end_event_get_time(ev);
//	//	time += ms;
//	//	char* pretty_buf = util_milliseconds_to_duration((uint64_t)ms);
//	//	printf("\n>>>>> Opperation completed in %.4fms (%s)\n", ms, pretty_buf);
//	//	free(pretty_buf);
//	//}
//	//printf("\n%.2lfms avg\n", time / 100);
//	return 0;
//}
