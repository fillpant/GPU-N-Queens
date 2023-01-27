//Required for MREMAP_MAYMOVE/mremap. Must be atop includes...
#include "deffinitions.cuh"
#include "bitsets.cuh"
#include "nq_mem_utilities.cuh"
#include <inttypes.h>
#include <string.h>
#include <stdlib.h>
#ifdef ENABLE_PERSISTENT_BACKED_MEMORY
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif



#ifdef ENABLE_PERSISTENT_BACKED_MEMORY
// Initialises nq_mem_handle_t for 'persistent backed' memory allocation (mmap/mremap/munmap)
// file_path specifies the path to the file backing the memory. If left NULL, a temp file will be used
// Warning: tmpnam is used to create a temp file path. This is unsafe if an other process gets the same
// file name as this one before nq_mem_alloc is called to create and use this file!!
void nq_mem_init_persistent_f(nq_mem_handle_t* handle, char* file_path) {
	char buf[L_tmpnam];
	handle->mem_ptr = NULL;
	handle->alloc_type = NQ_MEM_PERSISTENT_BACKED;
	handle->mem_size = 0;
	if (!file_path)
		file_path = tmpnam(buf);
	// If user provided, or if generated in local buffer, it's a 'must' to transfer to 
	// heap memory controlled by us.
	file_path = strdup(file_path);
	handle->backing_file_path = file_path;
}

// Initialises nq_mem_handle_t for 'persistent backed' memory allocation (mmap/mremap/munmap)
void nq_mem_init_persistent(nq_mem_handle_t* handle) {
	nq_mem_init_persistent_f(handle, NULL);
}
#endif

//initialises nq mem handle such that if persistent memory is enabled, persistent memory initialization is 
//used, and if not, regular memory allocation is used.
void nq_mem_init(nq_mem_handle_t* handle) {
#ifdef ENABLE_PERSISTENT_BACKED_MEMORY
	return nq_mem_init_persistent(handle);
#else
	return nq_mem_init_regular(handle);
#endif // ENABLE_PERSISTENT_BACKED_MEMORY
}

// Initialises nq_mem_handle_t for 'regular' memory allocation (malloc/realloc/free)
void nq_mem_init_regular(nq_mem_handle_t* handle) {
	handle->mem_ptr = NULL;
	handle->alloc_type = NQ_MEM_REGULAR;
	handle->mem_size = 0;
	handle->backing_file_path = NULL;
}


// Allocates memory for the given handle of size mem_size and returns a pointer to the allocated memory
// which is of size at least mem_size. 
// If an error occurs, NULL is returned and errno may be set.
// handle must have been initialised before calling this function! 
void* nq_mem_alloc(nq_mem_handle_t* handle, size_t mem_size) {
	// Uninitialised/poorly created handle
	if (handle->mem_ptr || handle->mem_size)
		return NULL;

	void* mem = NULL;
#ifdef ENABLE_PERSISTENT_BACKED_MEMORY
	if (handle->alloc_type == NQ_MEM_PERSISTENT_BACKED) {
		// No backing file is chosen...
		if (!(handle->backing_file_path))
			return NULL;
		// Map the size to a multiple of the pagesize of the system
		handle->mem_size = SIZE_TO_PAGESIZE_MULTIPLE(mem_size);
		// Open file descriptor to backing file with RW permission. Create it if not exists with
		// --rw-rw-rw- mask. The umask set will override this potentially but it's okay.
		const int fd = open(handle->backing_file_path, O_RDWR | O_CREAT | O_TRUNC, S_IWUSR | S_IRUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
		if (fd == -1)
			return NULL;
		// Truncate the file to exactly the required size. Without this mmap will fail.
		int err = ftruncate(fd, handle->mem_size);
		if (err) {
			close(fd);
			return NULL;
		}
		// Mem-map the file to virtual memory with RW protection and the shared flag. Without the shared
		// flag, the system may attempt to place all contents of the file to memory (and fail if not enough mem)
		mem = mmap(NULL, handle->mem_size, PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0);
		if (mem == MAP_FAILED) {
			close(fd);
			return NULL;
		}
		// Update the handle with the allocated ptr
		handle->mem_ptr = mem;
		err = close(fd);
		if (err) {
			//Free the memory. It is assumed that if this function returns null no memory has been allocated
			nq_mem_free(handle);
			return NULL;
		}
	} else
#endif
		if (handle->alloc_type == NQ_MEM_REGULAR) {
			handle->mem_size = mem_size;
			mem = malloc(mem_size);
			handle->mem_ptr = mem;
		}
	return mem;
}

// Reallocate memory to increase/trim the size of an existing allocation
// Returns NULL if the reallocation fails, however the handle and previous pointer
// are both 'alive' and unchanged and must therefore be free'd responsibly.
void* nq_mem_realloc(nq_mem_handle_t* handle, size_t mem_size) {
	if (!(handle->mem_ptr) || !(handle->mem_size))
		return NULL;
	void* mem = NULL;
#ifdef ENABLE_PERSISTENT_BACKED_MEMORY
	//See comments for nq_mem_alloc
	if (handle->alloc_type == NQ_MEM_PERSISTENT_BACKED) {
		if (!(handle->backing_file_path))
			return NULL;
		mem_size = SIZE_TO_PAGESIZE_MULTIPLE(mem_size);
		const int fd = open(handle->backing_file_path, O_RDWR);
		if (fd == -1)
			return NULL;
		int err = ftruncate(fd, mem_size);
		if (err) {
			close(fd);
			return NULL;
		}
		// Remap with maymove to allow max flexibility. This may move the ptr hence it has
		// to be 'replaced' like realloc
		mem = mremap(handle->mem_ptr, handle->mem_size, mem_size, MREMAP_MAYMOVE);
		if (mem == MAP_FAILED) {
			close(fd);
			return NULL;
		}
		// Update handle. If error occurs bellow, the handle would have to be manually free'd
		handle->mem_ptr = mem;
		handle->mem_size = mem_size;
		err = close(fd);
		//Unlike before, if we can't close the file, we don't free the memory.
		//This function fails by returning null but without touching the previous
		//pointer!
		if (err)
			return NULL;
	} else
#endif
		if (handle->alloc_type == NQ_MEM_REGULAR) {
			mem = realloc(handle->mem_ptr, mem_size);
			if (mem == NULL)
				return NULL;
			handle->mem_size = mem_size;
			handle->mem_ptr = mem;
		}
	return mem;
}

// Free a memory handle and its associated memory pointer
// This function will free the memory behind all pointers held relative to this handle!!!
// The pointers are therefore not to be free'd manually!!!
// The handle becomes 'spent' following this operation and must be re-initialised before it can
// safely be used again.
int nq_mem_free(nq_mem_handle_t* handle) {
	int err = -1;
#ifdef ENABLE_PERSISTENT_BACKED_MEMORY
	if (handle->alloc_type == NQ_MEM_PERSISTENT_BACKED) {
		// If the backing file path is specified (assumption is it will be,
		// but have to check because the state of the handle is assumed 'unknown')
		// delete the backing file.
		if (handle->backing_file_path) {
			err = unlink(handle->backing_file_path);
			free(handle->backing_file_path);
		}

		if (!handle->mem_ptr && !handle->mem_size)
			return err;
		// Attempt to unmap the file mapping. mumnap requirs mem_size
		// to be a multiple of page size which isn't checked/enforced here but
		// we just assume the handle won't be *that* corrupted...
		err = munmap(handle->mem_ptr, handle->mem_size);
	} else
#endif
		if (handle->alloc_type == NQ_MEM_REGULAR) {
			if (!handle->mem_ptr)
				return -1;
			free(handle->mem_ptr);
			handle->mem_ptr = NULL;
			handle->mem_size = 0;
			err = 0;
		}
	return err;
}


void* nq_mem_transfer_and_free(nq_mem_handle_t* handle) {
	if (handle->alloc_type == NQ_MEM_REGULAR) {
		if (handle->mem_ptr) {
			//No need to copy.
			return handle->mem_ptr;
		}

#ifdef ENABLE_PERSISTENT_BACKED_MEMORY
	} else if (handle->alloc_type == NQ_MEM_PERSISTENT_BACKED) {
		void* mem = malloc(handle->mem_size);
		if (mem) {
			memcpy(mem, handle->mem_ptr, handle->mem_size);
			nq_mem_free(handle);
		}
#endif
	} else {
		return NULL;
	}
}
