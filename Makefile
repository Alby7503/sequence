# Exact genetic sequence alignment
#
# Parallel computing (Degree in Computer Engineering)
# 2023/2024
#
# (c) 2024 Arturo Gonzalez-Escribano
# Grupo Trasgo, Universidad de Valladolid (Spain)

# Compilers
CC = gcc
OMPFLAG = -fopenmp
MPICC = mpicc
CUDACC = nvcc

# Flags for optimization and external libs
LIBS = -lm
FLAGS = -O3 -Wall
CUDAFLAGS = -O3 -Xcompiler -Wall

# Auto-detect GPU architecture via nvidia-smi
# Fallback to sm_60 if detection fails
GPU_ARCH := $(shell \
  detect=$$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d "."); \
  if [ -z "$$detect" ]; then echo sm_60; else echo sm_$$detect; fi)

# Add -arch flag for nvcc
CUDAFLAGS += -arch=$(GPU_ARCH)

# Targets to build
#OBJS = align_seq align_omp align_mpi align_cuda align_mpi_omp align_seq_new align_mpi_omp_2
OBJS = align_seq_new align_cuda align_mpi_omp_2

# Rules. By default show help
help:
	@echo
	@echo "Exact genetic sequence alignment"
	@echo
	@echo "Group Trasgo, Universidad de Valladolid (Spain)"
	@echo
	@echo "make align_seq	Build only the sequential version"
	@echo "make align_omp	Build only the OpenMP version"
	@echo "make align_mpi	Build only the MPI version"
	@echo "make align_cuda	Build only the CUDA version (auto-detected GPU_ARCH=$(GPU_ARCH))"
	@echo
	@echo "make all	Build all versions (Sequential, OpenMP, MPI, CUDA)"
	@echo "make debug	Build all version with demo output for small sequences"
	@echo "make clean	Remove targets"
	@echo

all: $(OBJS)

align_seq: align.c rng.c
	$(CC) $(FLAGS) $(DEBUG) $< $(LIBS) -o $@

align_seq_new: align_new.c rng.c
	$(CC) $(FLAGS) $(DEBUG) $< $(LIBS) -o $@

align_omp: align_omp.c rng.c
	$(CC) $(FLAGS) $(DEBUG) $(OMPFLAG) $< $(LIBS) -o $@

align_mpi: align_mpi.c rng.c
	$(MPICC) $(FLAGS) $(DEBUG) $< $(LIBS) -o $@

align_mpi_omp: align_mpi_omp.c rng.c
	$(MPICC) $(FLAGS) $(DEBUG) $(OMPFLAG) $< $(LIBS) -o $@

align_cuda: align_cuda.cu rng.c
	@echo "Compiling CUDA version with architecture $(GPU_ARCH)..."
	$(CUDACC) $(CUDAFLAGS) $< $(LIBS) -o $@

align_mpi_omp_2: align_mpi_omp_2.c rng.c
	$(MPICC) $(FLAGS) $(OMPFLAG) $< $(LIBS) -o $@

# Remove the target files
clean:
	rm -rf $(OBJS)

# Compile in debug mode
debug:
	make DEBUG="-DDEBUG -g" all
