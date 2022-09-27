#include <string.h>
#include <stdio.h>
#include "deffinitions.cuh"
#include "diagonals.cuh"
#include "bitsets.cuh"
#include "n_queens.cuh"
#include "nq_utils.cuh"
#include "string.h"
#include "math.h"
#include "assert.h"

#define NQ_STATE_UTIL_CMP_INTEGERS_ORDERING(a,b) (((a)>(b))-((a)<(b)))

const static uint32_t crc_mtable[] = { // source: https://web.mit.edu/freebsd/head/sys/libkern/crc32.c
	0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f,
	0xe963a535, 0x9e6495a3,	0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
	0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91, 0x1db71064, 0x6ab020f2,
	0xf3b97148, 0x84be41de,	0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
	0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec,	0x14015c4f, 0x63066cd9,
	0xfa0f3d63, 0x8d080df5,	0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172,
	0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b,	0x35b5a8fa, 0x42b2986c,
	0xdbbbc9d6, 0xacbcf940,	0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
	0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423,
	0xcfba9599, 0xb8bda50f, 0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924,
	0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d,	0x76dc4190, 0x01db7106,
	0x98d220bc, 0xefd5102a, 0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
	0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818, 0x7f6a0dbb, 0x086d3d2d,
	0x91646c97, 0xe6635c01, 0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e,
	0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457, 0x65b0d9c6, 0x12b7e950,
	0x8bbeb8ea, 0xfcb9887c, 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
	0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2, 0x4adfa541, 0x3dd895d7,
	0xa4d1c46d, 0xd3d6f4fb, 0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0,
	0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9, 0x5005713c, 0x270241aa,
	0xbe0b1010, 0xc90c2086, 0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
	0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4, 0x59b33d17, 0x2eb40d81,
	0xb7bd5c3b, 0xc0ba6cad, 0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a,
	0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683, 0xe3630b12, 0x94643b84,
	0x0d6d6a3e, 0x7a6a5aa8, 0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
	0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe, 0xf762575d, 0x806567cb,
	0x196c3671, 0x6e6b06e7, 0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc,
	0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5, 0xd6d6a3e8, 0xa1d1937e,
	0x38d8c2c4, 0x4fdff252, 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
	0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60, 0xdf60efc3, 0xa867df55,
	0x316e8eef, 0x4669be79, 0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236,
	0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f, 0xc5ba3bbe, 0xb2bd0b28,
	0x2bb45a92, 0x5cb36a04, 0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
	0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a, 0x9c0906a9, 0xeb0e363f,
	0x72076785, 0x05005713, 0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38,
	0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21, 0x86d3d2d4, 0xf1d4e242,
	0x68ddb3f8, 0x1fda836e, 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
	0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c, 0x8f659eff, 0xf862ae69,
	0x616bffd3, 0x166ccf45, 0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2,
	0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db, 0xaed16a4a, 0xd9d65adc,
	0x40df0b66, 0x37d83bf0, 0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
	0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6, 0xbad03605, 0xcdd70693,
	0x54de5729, 0x23d967bf, 0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94,
	0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d
};


static int nq_state_sort_cmp(const void* a, const void* b) {
	return memcmp(a, b, sizeof(nq_state_t));
}

char* util_large_integer_string_decimal_sep(char* num) {
	size_t len = strlen(num);
	if (len <= 3) {
		return strdup(num);
	}

	size_t ret_len = len + (len / 3) - (!(len % 3)) + 1;
	char* ret = (char*)malloc(ret_len * sizeof(char));
	ret[ret_len - 1] = 0;
	for (size_t j = ret_len - 2, i = len; i > 0; --i, --j) {
		size_t l = len - i;
		if (l && !(l % 3))
			ret[j--] = ',';
		ret[j] = num[i - 1];
	}
	return ret;
}

// Allocates space in pinned memory via cudaMallocHost, and copies the specified element count to it.
// The original buffer is not free'd
void* util_copy_to_pinned_mem(void* data, size_t element_size, size_t element_count) {
	void* p;
	CHECK_CUDA_ERROR(cudaMallocHost(&p, element_size * element_count));
	memcpy(p, data, element_size * element_count);
	return p;
}

//File format:
//1 byte: N
//1 byte: Locked row
//8 bytes: Total states written
//
// Subheader data follows. Up to N sub headers.
// Each subheader preceeds a number of state data that all have been populated up to row 'x':
//1 byte: latest populated row 'lr'
//8 bytes: number 'z' of state data that follows.
// <z many queen position arryas each of length lr>
//1 byte: latest populated row 'lr'
//8 bytes: number 'z' of state data that follows.
// <z many queen position arryas each of length lr>
// ............
int util_write_nq_states_to_stream(FILE* const stream, nq_state_t* states, uint64_t len, const unsigned char locked_at_row) {
	if (stream && len) {
		const unsigned char en = N;
		bool doHeadersFor[N] = {};
		nq_state_t* ptr = states;
		for (; ptr < states + len; ++ptr)
			doHeadersFor[ptr->curr_row] = 1;

		fwrite(&en, sizeof(unsigned char), 1, stream);
		fwrite(&locked_at_row, sizeof(unsigned char), 1, stream);
		fwrite(&len, sizeof(uint64_t), 1, stream);

		//Write subheader followed by data for each set of states.
		UNROLL_LOOP(N);
		for (char h = 0; h < N; ++h) {
			if (doHeadersFor[h]) {
				//Stop if error occured in previous write(s)
				if (ferror(stream))
					return ferror(stream);

				//Write partial header
				fwrite(&h, sizeof(char), 1, stream);
				//We don't know the length yet, keep the position and come back after writing the states.
				const long pos = ftell(stream);
				fseek(stream, sizeof(uint64_t), SEEK_CUR);
				uint64_t cnt = 0;
				for (ptr = states; ptr < states + len; ++ptr) {
					if (ptr->curr_row == h) {
						++cnt;
						fwrite(ptr->queen_at_index, sizeof(unsigned char), h, stream);
					}
				}
				//Go back to header and write the count of states that follows.
				fseek(stream, pos, SEEK_SET);
				fwrite(&cnt, sizeof(uint64_t), 1, stream);
				fseek(stream, 0, SEEK_END); //Go back to end of file.
			}
		}
		return ferror(stream);
	}
	return 1;
}

int util_read_nq_states_from_stream(FILE* const stream, nq_state_t** states, uint64_t* len, unsigned char* const locked_at_row, bool skip_n_check) {
	if (stream && len && states && locked_at_row) {
		unsigned char en = 0;

		fread(&en, sizeof(unsigned char), 1, stream);
		fread(locked_at_row, sizeof(unsigned char), 1, stream);
		fread(len, sizeof(uint64_t), 1, stream);

		// Check N is correct
		assert(skip_n_check || N == en);

		*states = (nq_state_t*)malloc(sizeof(nq_state_t) * (*len));
		if (!*states) return -1;

		nq_state_t* bpos = *states;
		unsigned char tmp_buff[N];
		while (1) {
			//Read header
			char curr_row = 0;
			uint64_t cnt = 0;
			fread(&curr_row, sizeof(unsigned char), 1, stream);

			//FEOF won't return true until we have read past the end of the file. Hence why infinite loop with abrupt break.
			if (feof(stream))
				break;

			fread(&cnt, sizeof(uint64_t), 1, stream);

			//Make sure writing cnt many states to bpos won't push us out of bounds
			assert(bpos + cnt <= ((*states) + *len));

			//Either empty segment or segment non empty and we have stuff to read.
			assert(!cnt || cnt > 0 && !feof(stream));

			//Begin re-creating the states.
			nq_state_t temp = init_nq_state();
			for (uint64_t c = 0; c < cnt; ++c) {
				clear_nq_state(&temp);
				temp.curr_row = curr_row;
				fread(tmp_buff, sizeof(unsigned char), curr_row, stream);
				for (char i = 0; i < curr_row; ++i)
					place_queen_at(&temp, i, tmp_buff[i]);
				memcpy(bpos + c, &temp, sizeof(nq_state_t));
			}
			//Move to the next segment on the buffer
			bpos += cnt;
		}
		return ferror(stream);
	}
	return 1;
}

uint32_t crc32_chsm(const void* const buf, const size_t element_size, const size_t element_count) {
	const uint8_t* const b = (uint8_t*)buf;
	bitset32_t sum = BITSET32_MAX;
	for (size_t i = 0; i < element_size * element_count; ++i)
		sum = crc_mtable[0xFF & (sum ^ b[i])] ^ (sum >> 8);
	return sum ^ BITSET32_MAX;
}

static bitset32_t is_nq_state_valid(const nq_state_t* const st, const unsigned lock_at_row) {
	bitset32_t result = 0;
	if (st->curr_row < 0 || st->curr_row >= N || (signed char)lock_at_row > st->curr_row || !NQ_FREE_COLS_AT(st, st->curr_row)) {
		util_visualise_nq_state(st, true);
		result |= NQ_VALIDATION_UNADVANCEABLE_STATE;
	}

	int POPCNT(st->queens_in_columns, occupied_cols);

	if ((unsigned)occupied_cols != lock_at_row)
		return result | NQ_VALIDATION_INVALID_STATE;

	for (unsigned i = 0; i < N; ++i) {
		if ((signed char)i < st->curr_row) {
			if (st->queen_at_index[i] == UNSET_QUEEN_INDEX) {
				util_visualise_nq_state(st, true);
				return result | NQ_VALIDATION_INVALID_STATE;
			}
		} else {
			if (st->queen_at_index[i] != UNSET_QUEEN_INDEX) {
				util_visualise_nq_state(st, true);
				return result | NQ_VALIDATION_INVALID_STATE;
			}
		}
	}
	return result;
}

// Checks the buffer of states for consistency. 
// Returns a bitset32 with the value: NQ_VALIDATION_BUFFER_OK if no issues where found.
// If validation identifies an issue, a bitset32 is returned which is a mask of one or more of the following:
// - NQ_VALIDATION_INVALID_STATE
// - NQ_VALIDATION_UNADVANCEABLE_STATE
// - NQ_VALIDATION_NULL_POINTER
// - NQ_VALIDATION_DUPLICATE_STATES
// NOTE: This function may modify the order of elements in the buffer!
__host__ bitset32_t util_validate_state_buffer(nq_state_t* const buf, const uint64_t buf_len, const unsigned lock_at_row) {
	if (!buf)
		return NQ_VALIDATION_NULL_POINTER;
	const bitset32_t all_set = NQ_VALIDATION_DUPLICATE_STATES | NQ_VALIDATION_INVALID_STATE | NQ_VALIDATION_NULL_POINTER | NQ_VALIDATION_UNADVANCEABLE_STATE;

	bitset32_t result = NQ_VALIDATION_BUFFER_OK;
	qsort(buf, buf_len, sizeof(nq_state_t), &nq_state_sort_cmp);

	for (uint64_t i = 0; i < buf_len && result ^ all_set; ++i) {
		if (i && !(result & NQ_VALIDATION_DUPLICATE_STATES) && !nq_state_sort_cmp(buf + (i - 1), buf + i))
			result |= NQ_VALIDATION_DUPLICATE_STATES;

		if (!(result & NQ_VALIDATION_UNADVANCEABLE_STATE) || !(result & NQ_VALIDATION_INVALID_STATE))
			result |= is_nq_state_valid(buf + i, lock_at_row);
	}
	return result;
}

__host__ void util_visualise_nq_state(const nq_state_t* const what, const bool show_blocked) {
	static char header[N * 4 + 3] = { 0 };
	if (!header[0]) {
		for (unsigned i = 0; i < N; ++i)
			strcpy(&header[i * 4], "+---");
		strcpy(&header[N * 4], "+\n\0");
	}

	char strbuf[4] = {};
	for (unsigned i = 0; i < N; ++i) {
		printf("%s",header);
		for (unsigned j = 0; j < N; ++j) {
			if (show_blocked) {
				strbuf[0] = BS_GET_BIT(dad_extract(&what->diagonals, i), j) ? 'd' : ' ';
				strbuf[1] = BS_GET_BIT(what->queens_in_columns, j) ? 'c' : ' ';
				//strbuf[2] = BS_GET_BIT(what->queens_in_rows, i) ? 'r' : ' ';
				strbuf[2] = what->queen_at_index[i] != UNSET_QUEEN_INDEX ? 'r' : ' ';
			}
			printf("|%s", what->queen_at_index[i] == j ? "[Q]" : strbuf);
		}
		printf("|\n");
	}
	printf("%s",header);
	printf("\n\n");
}

char* util_size_to_human_readable(size_t siz) {
	static char* units[] = { "B","KB","MB","GB","TB","PB" };//Waste of memory but oh well :-)
	static char buf[128];

	unsigned unit;
	for (unit = 0; unit < sizeof(units) / sizeof(char*) && siz / pow(1000, unit) >= 1000; ++unit);

	sprintf_s(buf, sizeof(buf), "%.2f%s", ((double)siz) / pow(1000, unit), units[unit]);
	return buf;
}

void util_prettyprint_nq_state(nq_state_t* s) {
	printf("Queens at index:\n");
	for (unsigned i = 0; i < N; i++) {
		if (s->queen_at_index[i] == UNSET_QUEEN_INDEX)
			printf("\t[%u] -> UNSET\n", i);
		else
			printf("\t[%u] -> %3u\n", i, s->queen_at_index[i]);
	}
	printf("\tQueens in cols: %*sb\n", N, util_bits_to_string(s->queens_in_columns, N));
	//printf("\tLocked row end: %*uu\n", N, s->locked_row_end);
	printf("\t      Curr row: %*uu\n", N, s->curr_row);
	printf("\tDiagonals:\n");
	for (unsigned i = 0; i < N; i++) {
		bitset32_t rslt = dad_extract(&s->diagonals, i);
		printf("\t\t%sb\n", util_bits_to_string(rslt, N));
	}
}

char* util_milliseconds_to_duration(uint64_t ms) {
	//XXXXX Years, XX Months, XX Weeks, XX Days, XX Hours, XX Minutes, XX.XXX Seconds 
	const unsigned years = (unsigned)(ms / 31536000000LLU);
	ms -= 31536000000LLU * years;
	const unsigned months = (unsigned)(ms / 2628288000LLU);
	ms -= 2628288000LLU * months;
	const unsigned weeks = (unsigned)(ms / 604800000LLU);
	ms -= 604800000LLU * weeks;
	const unsigned days = (unsigned)(ms / 86400000LLU);
	ms -= 86400000LLU * days;
	const unsigned hours = (unsigned)(ms / 3600000LLU);
	ms -= 3600000LLU * hours;
	const unsigned minutes = (unsigned)(ms / 60000LLU);
	ms -= 60000LLU * minutes;
	const unsigned seconds = (unsigned)(ms / 1000LLU);
	ms -= 1000LLU * seconds;

	bitset8_t fields = 0;
	size_t buffer_len = 0;
	buffer_len += years ? fields = BS_SET_BIT(fields, 6), 1 + (unsigned long)log10(years ? years : 1) + 8 : 0; // 8 for ' Years, '
	buffer_len += months || buffer_len ? fields = BS_SET_BIT(fields, 5), 1 + (unsigned long)log10(months ? months : 1) + 9 : 0; // 9  for ' Months, '
	buffer_len += weeks || buffer_len ? fields = BS_SET_BIT(fields, 4), 1 + (unsigned long)log10(weeks ? weeks : 1) + 8 : 0; // 8 for ' Weeks, '
	buffer_len += days || buffer_len ? fields = BS_SET_BIT(fields, 3), 1 + (unsigned long)log10(days ? days : 1) + 7 : 0; // 7 for ' Days, '
	buffer_len += hours || buffer_len ? fields = BS_SET_BIT(fields, 2), 1 + (unsigned long)log10(hours ? hours : 1) + 8 : 0; // 7 for ' Hours, '
	buffer_len += minutes || buffer_len ? fields = BS_SET_BIT(fields, 1), 1 + (unsigned long)log10(minutes ? minutes : 1) + 10 : 0; // 10 for ' Minutes, '
	buffer_len += seconds || buffer_len ? fields = BS_SET_BIT(fields, 0), 1 + (unsigned long)log10(seconds ? seconds : 1) + 10 : 0; // 10 for ' Seconds, '
	buffer_len += 1 + (unsigned long)log10(ms ? ms : 1) + 14; //for "milliseconds.".
	++buffer_len; //null byte

	char* ret = (char*)malloc(sizeof(char) * buffer_len);
	char* buf = ret;
	if (BS_GET_BIT(fields, 6))
		buf += sprintf(buf, "%u Years, ", years);
	if (BS_GET_BIT(fields, 5))
		buf += sprintf(buf, "%u Months, ", months);
	if (BS_GET_BIT(fields, 4))
		buf += sprintf(buf, "%u Weeks, ", weeks);
	if (BS_GET_BIT(fields, 3))
		buf += sprintf(buf, "%u Days, ", days);
	if (BS_GET_BIT(fields, 2))
		buf += sprintf(buf, "%u Hours, ", hours);
	if (BS_GET_BIT(fields, 1))
		buf += sprintf(buf, "%u Minutes, ", minutes);
	if (BS_GET_BIT(fields, 0))
		buf += sprintf(buf, "%u Seconds, ", seconds);
	buf += sprintf(buf, "%u Milliseconds", (unsigned)ms);
	*buf = 0;
	return ret;
}

char* util_bits_to_string(unsigned long long num, unsigned bits) {
	static char buf[65] = {};
	util_bits_to_buf(num, bits, buf);
	return buf;
}

__host__ __device__ void util_bits_to_buf(unsigned long long num, unsigned bits, char buf[]) {
	unsigned i;
	for (i = 0; i < (bits > 64 ? 64 : bits); i++) {
		buf[i] = '0' + BS_GET_BIT(num, bits - 1 - i);
	}
	buf[i] = '\0';
}

__host__ bool util_nq_states_equal(nq_state_t* one, nq_state_t* two) {
	return one->curr_row == two->curr_row
		&& one->diagonals.diagonal == two->diagonals.diagonal
		&& one->diagonals.antidiagonal == two->diagonals.antidiagonal
		&& one->queens_in_columns == two->queens_in_columns
		&& !memcmp(one->queen_at_index, two->queen_at_index, sizeof(one->queen_at_index));
}


