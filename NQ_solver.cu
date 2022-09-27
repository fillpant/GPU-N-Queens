#include "cargs.h"
#include <stdlib.h>
#include <stdio.h>
#include "deffinitions.cuh"
#include <errno.h>
#include <limits.h>
#include "n_queens.cuh"
#include "nq_utils.cuh"
#include "GPU_NQueens.cuh"
#include <inttypes.h>
#include <math.h>
#include <string.h>

typedef struct cag_option cag_opt_t;
static cag_opt_t opts[] = {
	//-t --try-solve <num_of_states>
	{'t',"tT","try-solve","limit","For testing only! Specify the max number of states in generate-and-solve configuration"},
	//-g --gen <limit>
	{'g',"gG","gen-states","limit","Generate states. Limit sets a soft limit of how many states to generate (for the compiled N). This flag must be used in conjuction with one state output flag."},
	//-w --states-to-file <file>
	{'w',"wW","write-states","file/dir","Write the generated states to the given file. This must be a directory if -s is specified, or a file otherwise."},
	//-s --split-pool
	{'s',"sS","split-states","count","Split generated states into multiple files, prefixed with X_ where X is an integer in range [0,count)"},
	//-m --mix-states 
	{'m',"mM","mix-states",NULL,"Shuffle the state buffer using a random seed. This option applies both to state loading and state generation."},
	//-r --rand-seed
	{'r',"rR","random-seed","seed","Use this seed to initialize random."},
	//-l --load-states <file>
	{'l',"lL","load-states","file", "Load states from the given file, instead of generating a new pool"},
	//-h/-H/-? --help 
	{'h',"?hH","help",NULL,"Show help options"},
	//-d --devices <gpu1,gpu2,gpu3,...>
	{'d',"dD","devices","gpu1,gpu2,...","A comma separated array of device IDs as shown in nvidia-smi, to be used for the computation. If not specified all available devices are used."}
};


static void shuffle_states(nq_state_t* const states, const uint64_t len) {
	//Flexible type in union is non-standard for C99/C11
	union { uint64_t rval; int16_t arr[4];  } r;
	nq_state_t tmp;
	for (uint64_t i = 0; i < len - 2; ++i) {
		r.arr[0] = rand();
		r.arr[1] = rand();
		r.arr[2] = rand();
		r.arr[3] = rand();
		r.rval = ((r.rval % len) + i) % len;
		if (r.rval == i) continue;
		memcpy(&tmp, states + i, sizeof(nq_state_t));
		memmove(states + i, states + r.rval, sizeof(nq_state_t));
		memcpy(states + r.rval, &tmp, sizeof(nq_state_t));
	}
}

static int generate_states_to_files(long long int state_count, const char* file_out, int file_chunks, bool shuffle) {
	uint64_t actual_cnt;
	unsigned locked_row;
	FAIL_IF(state_count > UINT64_MAX);
	nq_state_t* buf = nq_generate_states((uint64_t)state_count, &actual_cnt, &locked_row);
	if (!buf) {
		fprintf(stderr, "Failed to generate states!");
		return EXIT_FAILURE;
	}

	if (shuffle)
		shuffle_states(buf, actual_cnt);

	printf("Generated %" PRIu64" states by locking at row %u\n", actual_cnt, locked_row);
	if (file_chunks) {
		uint64_t per_chunk = (uint64_t)ceil(actual_cnt / (double)file_chunks);
		int left_overs = (int)(actual_cnt % file_chunks);

		if (!per_chunk) {
			fprintf(stderr, "Cannot divide state pool across requested number of chunks! More chunks than states!");
			return EXIT_FAILURE;
		}

		printf("Writing approx. %llu states per file across %d chunk files in %s...\n", (unsigned long long) per_chunk, file_chunks, file_out);

		char fpath_buf[MAX_PATH + 1];
		uint64_t written = 0;
		for (int i = 0; i < file_chunks; ++i) {
			if (!sprintf_s(fpath_buf, sizeof(fpath_buf), "%s" FILE_SEPARATOR"chunk_%d_for_n%u_cnt_%llu.nqsf", file_out, i, N, per_chunk)) {
				fprintf(stderr, "Failed to construct chunk file path. Max file path length exceeded!");
				return EXIT_FAILURE;
			}
			printf("\tWriting %" PRIu64" states for chunk %d in file [%s]...", per_chunk, i, fpath_buf);
			FILE* chunk = fopen(fpath_buf, "wb");
			if (!chunk) {
				fprintf(stderr, "\nFailed to open chunk file!");
				return EXIT_FAILURE;
			}

			if (util_write_nq_states_to_stream(chunk, buf + written, per_chunk, locked_row)) {
				fprintf(stderr, "\nFailed to write chunk!");
				fclose(chunk);
				return EXIT_FAILURE;
			}
			written += per_chunk;
			if (i == left_overs - 1) --per_chunk;
			printf("Done\n");
			fclose(chunk);
		}
		FAIL_IF(written != actual_cnt);
	} else {
		printf("Writing generated states to file...");
		FILE* out = fopen(file_out, "wb");
		if (!out) {
			fprintf(stderr, "Failed to open file %s for writing.", file_out);
			return EXIT_FAILURE;
		}
		if (util_write_nq_states_to_stream(out, buf, actual_cnt, locked_row)) {
			fprintf(stderr, "Failed to write states to file!");
			fclose(out);
			return EXIT_FAILURE;
		}
	}
	printf("Success! Exiting.");
	return EXIT_SUCCESS;
}

static int parse_gpu_configs(const char* const to_parse, gpu_config_t** const configs, unsigned* const size) {
	int elements = 1;
	const char* str = to_parse;
	for (; *str; ++str)
		elements += *str == ',';

	gpu_config_t* confs = (gpu_config_t*)calloc(elements, sizeof(gpu_config_t));
	if (!confs)
		return 1;

	char tmp_parse[20];
	const char* end = to_parse;
	str = to_parse;
	for (int cidx = 0;; ++end) {
		if (*end == ',' || !*end) {
			ptrdiff_t len = end - str;
			if (len && len < sizeof(tmp_parse)) {
				memcpy(tmp_parse, str, len);
				long long int id = strtoll(tmp_parse, NULL, 10);
				if (errno) {
					free(confs);
					return errno;
				}
				if (id > UINT_MAX)
					return 3;
				confs[cidx++].device_id = (unsigned int)id; //Safe cast as per above.
				str = end + 1;
			} else
				return 2;
		}
		if (!*end) break;
	}
	*size = elements;
	*configs = confs;
	return 0;
}

static int get_all_gpus(gpu_config_t** const configs, unsigned* gpu_config_count) {
	int cnt;
	CHECK_CUDA_ERROR(cudaGetDeviceCount(&cnt));
	*configs = (gpu_config_t*)calloc(cnt, sizeof(gpu_config_t));
	if (!*configs) return 1;
	*gpu_config_count = (unsigned)cnt;
	for (--cnt; cnt >= 0; --cnt)
		(*configs)[cnt].device_id = cnt;
	//todo asybc flag
	return 0;
}

static uint64_t solve(nq_state_t* const states, const uint64_t len, const unsigned lock_at_row, gpu_config_t* configs, unsigned gpu_config_count) {
	nq_state_t* pinned = (nq_state_t*)util_copy_to_pinned_mem(states, sizeof(nq_state_t), len);
	free(states);
	FAIL_IF(len > UINT_MAX);
	FAIL_IF(!configs && get_all_gpus(&configs, &gpu_config_count));

	printf("Solving...\n");
	uint64_t result = gpu_solver_driver(pinned, (uint_least32_t)len, lock_at_row, configs, gpu_config_count);

	char res[32] = {};
	sprintf(res, "%" PRIu64, result);
	char* formatted_res = util_large_integer_string_decimal_sep(res);
	printf("solution_count(%u) = %s (%s)\n", N, res, formatted_res);
	free(formatted_res);
	CHECK_CUDA_ERROR(cudaFreeHost(pinned));
	return 0;
}



int main(int argc, char** argv) {

	//unsigned char rla;
	//uint64_t w_gen_cnt, r.;
	//unsigned w_locked_at;
	//nq_state_t* sbuf = nq_generate_states(50000000, &w_gen_cnt, &w_locked_at), *rbuf;

	////FILE* fo = fopen("R:/tmp", "wb");
	////util_write_nq_states_to_stream(fo, sbuf, w_gen_cnt, w_locked_at);
	////fclose(fo);

	//FILE* fo = fopen("R:/a.nqsf", "rb");
	//util_read_nq_states_from_stream(fo, &rbuf, &r, &rla, false);

	//for (uint64_t i = 0; i < r; ++i) {
	//	if (!util_nq_states_equal(sbuf + i, rbuf + i)) {
	//		printf("FAIL: %llu\n", i);
	//	}
	//}


	//exit(0);
	cag_option_context ctxt;

	const char* gen_states_file = NULL;
	long long int state_count = 0;
	long long int try_with_statecnt = 0;
	const char* input_states_file = NULL;
	int state_split_count = 0;
	gpu_config_t* gpu_configs = NULL;
	unsigned gpu_conf_size = 0;
	bool shuffle = 0;
#ifndef PROFILING_ROUNDS
	cag_option_prepare(&ctxt, opts, CAG_ARRAY_SIZE(opts), argc, argv);
	while (cag_option_fetch(&ctxt)) {
		const char* val;
		long long res;
		char optn = cag_option_get(&ctxt);
		switch (optn) {
		case 'r':
			val = cag_option_get_value(&ctxt);
			res = strtol(val, NULL, 10);
			FAIL_IF(res > UINT_MAX || res < 0);
			srand((unsigned int)res);
			break;
		case 'g':
			val = cag_option_get_value(&ctxt);
			res = strtoll(val, NULL, 10);
			if (errno || res <= 0) {
				fprintf(stderr, "Cannot generate '%s' many states!", val);
				if (gpu_configs) free(gpu_configs);
				return EXIT_FAILURE;
			}
			state_count = res;
			break;
		case 'w':
			gen_states_file = cag_option_get_value(&ctxt);
			break;
		case 's':
			val = cag_option_get_value(&ctxt);
			res = strtol(val, NULL, 10);
			if (errno || res <= 0 || res > INT_MAX) {
				fprintf(stderr, "Count value '%s' is invalid.", val);
				if (gpu_configs) free(gpu_configs);
				return EXIT_FAILURE;
			}
			state_split_count = (int)res; //Safe cast as per above check.
			break;
		case 'l':
			input_states_file = cag_option_get_value(&ctxt);
			break;
		case 'm':
			shuffle = 1;
			break;
		case 'h':
			printf("Usage: %s <option 1> <option 2> ...\n", argv[0]);
			printf("Note: Windows style flags (/) will not parse!\n");
			cag_option_print(opts, CAG_ARRAY_SIZE(opts), stdout);
			if (gpu_configs) free(gpu_configs);
			return EXIT_SUCCESS;
		case 'd':
			if (parse_gpu_configs(cag_option_get_value(&ctxt), &gpu_configs, &gpu_conf_size)) {
				fprintf(stderr, "Failed to parse GPU list!");
				return EXIT_FAILURE;
			}
			break;
		case 't':
			val = cag_option_get_value(&ctxt);
			res = strtoll(val, NULL, 10);
			if (errno || res <= 0) {
				fprintf(stderr, "Cannot try with '%s' many states!", val);
				if (gpu_configs) free(gpu_configs);
				return EXIT_FAILURE;
			}
			try_with_statecnt = res;
			break;
		default:
			fprintf(stderr, "Unsupported argument '%c'!", optn);
			if (gpu_configs) free(gpu_configs);
			break;
		}

	}

	printf("\nCompiled for N=%u\n", N);
#ifdef NQ_ENABLE_EXPERIMENTAL_OPTIMISATIONS
	printf("\tExperimental optimizations enabled!\n");
#endif
#ifdef USE_REGISTER_ONLY_KERNEL
	printf("\tUsing register-based kernel\n");
#else
	printf("\tUsing shared memory based kernel\n");
#endif
	printf("\tState generation downslope bounded: %s\n", STATE_GENERATION_DOWNSLOPE_BOUNDED ? "Yes" : "No");
#ifdef ENABLE_THREADED_STATE_GENERATION
	printf("\tThreaded state generation\n");
#endif
	printf("\tFixed SMEM size: %u\n", SMEM_SIZE);
	printf("\tFixed Warp size: %u\n", WARP_SIZE);
	printf("\tState generation play factor: %lf\n", STATE_GENERATION_LIMIT_PLAY_FACTOR);
	printf("\tCurrent block size: %llu\n", COMPLETE_KERNEL_BLOCK_THREAD_COUNT);
	printf("\tState generation memblock size: %llu\n", STATE_GENERATION_POOL_COUNT);


	printf("\n\n");
	//State generation
	if (state_count) {
		if (!gen_states_file) {
			fprintf(stderr, "State generation selected, but no output file supplied!");
			return EXIT_FAILURE;
		}
		if (try_with_statecnt) {
			printf("Ignoring solving state count flag -- This is not applicable here.\n");
		}
		printf("Starting state generation for up to %lld states. Writing to: %s.\n", state_count, gen_states_file);
		generate_states_to_files(state_count, gen_states_file, state_split_count, shuffle);
	} else if (input_states_file) { // Read states from file and solve
		if (try_with_statecnt) {
			printf("Ignoring solving state count flag -- This is not applicable here.\n");
		}
		printf("Loading states from file...\n");
		FILE* in = fopen(input_states_file, "rb");
		if (!in) {
			fprintf(stderr, "Failed to read state file [%s]!", input_states_file);
			return EXIT_FAILURE;
		}
		nq_state_t* sbuf;
		uint64_t len;
		unsigned char lock_at;
		if (util_read_nq_states_from_stream(in, &sbuf, &len, &lock_at, false)) {
			fprintf(stderr, "Failed to read states from file!");
			fclose(in);
			return EXIT_FAILURE;
		}
		printf("Loaded %" PRIu64" states\n", len);
		fclose(in);
		if (shuffle) {
			printf("Shuffling loaded states...\n");
			shuffle_states(sbuf, len);
		}
		solve(sbuf, len, lock_at, gpu_configs, gpu_conf_size);
	} else { // Just solve without reading from file
#endif
		if (!try_with_statecnt) {
			printf("WARN: Test-solve run without set number of states. Defaulting to 50M\n");
			try_with_statecnt = 50000000;
		}
		uint64_t gen_cnt;
		unsigned locked_at;
		nq_state_t* sbuf = nq_generate_states(try_with_statecnt, &gen_cnt, &locked_at);
		if (!sbuf) {
			fprintf(stderr, "Failed to generate states!");
			return EXIT_FAILURE;
		}
		solve(sbuf, gen_cnt, locked_at, gpu_configs, gpu_conf_size);
#ifndef PROFILING_ROUNDS
	}
#endif
	if (gpu_configs)
		free(gpu_configs);
	return EXIT_SUCCESS;
}