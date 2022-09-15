# tool macros
GPU_ARCH := sm_60
NVCC := nvcc
PTXAS_OPT := -v

CCFLAGS := -O3 -Wall -Wpedantic
ifeq ($(OS),Windows_NT)
	CCFLAGS := -EHsc -nologo -O2  -FS -GR-
endif

NVCCFLAGS := -arch=$(GPU_ARCH) -x cu -rdc=true -use_fast_math -maxrregcount=0 -Xcompiler '$(CCFLAGS)' --ptxas-options='$(PTXAS_OPT)' --machine 64 -cudart static

INCLUDE_FLAGS_LIN := -I $(SRC_PATH)/libs/cargs/
INCLUDE_FLAGS_WIN := -I '$(SRC_PATH)\libs\cargs\:$(SRC_PATH)\libs\pthreads\'
INCLUDE_FLAGS := $(INCLUDE_FLAGS_LIN)
ifeq ($(OS),Windows_NT)
	INCLUDE_FLAGS := $(INCLUDE_FLAGS_WIN)
endif


BIN_DIR := bin
SRC_PATH := .


TARGET_NAME := nqueens
TARGET := $(BIN_DIR)/$(TARGET_NAME)
SRC := $(foreach x, $(SRC_PATH), $(wildcard $(addprefix $(x)/*,.cu))) $(foreach x, $(SRC_PATH), $(wildcard $(addprefix $(x)/*,.c)))

default: fullcompile

fullcompile $(SRC):
	$(NVCC) $(INCLUDE_FLAGS) $(NVCCFLAGS) $(SRC) -o $(TARGET)