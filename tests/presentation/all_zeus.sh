#!/bin/bash

export MPI_NUMBER_PROC="-np 2"
export OMP_NUM_THREADS=64

export MPI_FLAGS="--map-by socket --bind-to socket"

#./tests/presentation/all.sh
./tests/presentation/benchmark.sh