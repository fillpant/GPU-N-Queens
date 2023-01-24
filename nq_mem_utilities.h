#ifndef NQ_MEM_UTILITIES_H
#define NQ_MEM_UTILITIES_H
#include "deffinitions.cuh"

#ifdef ENABLE_PERSISTENT_BACKED_MEMORY
// Brings the given size_t s to be a multiple of the page size on the system.
#define SIZE_TO_PAGESIZE_MULTIPLE(s) ((s) % sysconf(_SC_PAGE_SIZE) ? s + (sysconf(_SC_PAGE_SIZE) - (s % sysconf(_SC_PAGE_SIZE))) : s) 
#endif
// Types of memory allocation permitted through this utility. Internal use only.
enum nq_mem_allocation_type {
#ifdef ENABLE_PERSISTENT_BACKED_MEMORY
	NQ_MEM_PERSISTENT_BACKED, 
#endif
	NQ_MEM_REGULAR
};

// Holds the relevant information used to allocate/resize/free memory. The holder of a copy
// of this struct must not modify any of the values within it.
typedef struct {
	void* mem_ptr;
	enum nq_mem_allocation_type alloc_type;
	size_t mem_size;
	char* backing_file_path;
} nq_mem_handle_t;


#ifdef ENABLE_PERSISTENT_BACKED_MEMORY
void nq_mem_init_persistent(nq_mem_handle_t* handle);
void nq_mem_init_persistent_f(nq_mem_handle_t* handle, char* fpath);
#endif
void nq_mem_init_regular(nq_mem_handle_t* handle);
void nq_mem_init(nq_mem_handle_t* handle);
void* nq_mem_alloc(nq_mem_handle_t* handle, size_t mem_size);
void* nq_mem_realloc(nq_mem_handle_t* handle, size_t mem_size);
int nq_mem_free(nq_mem_handle_t* handle);
#endif
