###########################################################
GPU_ARCH=sm_60


NVCC=nvcc
PTXAS_OPTS= -v --opt-level 3 -warn-lmem-usage -warn-spills -opt-fp-atomics
CCFLAGS=-O3 -Wall
NVCC_FLAGS=  -arch $(GPU_ARCH) -use_fast_math -maxrregcount=0 -Xcompiler '$(CCFLAGS)' --ptxas-options='$(PTXAS_OPTS)' --machine 64 -cudart static

CUDA_LINK_LIBS= -lcudart

##########################################################
SRC_DIR = .
#WARN: Clear removes entire OBJ_DIR! DO NOT associate with dir containing important things
OBJ_DIR = obj
#WARN: Clear removes entire OUT_DIR!!! DO NOT associate with dir containing important things
OUT_DIR = bin
#WARN: Clear removes entire LIB_OBJ_DIR! DO NOT associate with dir containing important things
LIB_OBJ_DIR = obj/libs
#WARN: Clear removes entire PTX dir!!!
PTX_DIR = ptx
INCLUDES = libs/cargs
EXE = nqueens


SOURCE_FILES = $(wildcard $(SRC_DIR)/*.c) $(wildcard $(SRC_DIR)/*.cu)
SOURCE_OBJECTS = $(subst .c,.o,$(subst .cu,.o,$(SOURCE_FILES)))
LIB_SOURCES = $(foreach f,$(INCLUDES),$(wildcard $(f)/*.c) $(wildcard $(f)/*.cu))
LIB_OBJECTS = $(subst .c,.o,$(subst .cu,.o,$(LIB_SOURCES)))

##########################################################
all: compile_libs compile link

compile_libs $(INC_DIRS) $(LIB_SOURCES):
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(LIB_OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) --compile -odir $(LIB_OBJ_DIR) $(addprefix -I,$(INCLUDES)) $(LIB_SOURCES)

compile $(SOURCE_FILES):
	$(NVCC) $(NVCC_FLAGS) --compile -odir $(OBJ_DIR) $(addprefix -I,$(INCLUDES)) $(SOURCE_FILES)

link $(LIB_OBJECTS) $(SOURCE_OBJECTS):
	@mkdir -p $(OUT_DIR)
	$(NVCC) --link $(wildcard $(OBJ_DIR)/*.o) $(wildcard $(LIB_OBJ_DIR)/*.o) -o $(OUT_DIR)/$(EXE)
ptx:
	@mkdir -p $(PTX_DIR)
	$(NVCC) $(NVCC_FLAGS) $(addprefix -I,$(INCLUDES)) --ptx $(wildcard $(SRC_DIR)/*.cu) -odir $(PTX_DIR)

clean_lib:
	$(RM) -r $(LIB_OBJ_DIR) $(OBJ_DIR)

clear_ptx:
	$(RM) -r $(PTX_DIR)

clean: clean_lib clear_ptx
	$(RM) -r $(OUT_DIR)

