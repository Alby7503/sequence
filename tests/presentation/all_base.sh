#!/bin/bash

export MPI_NUMBER_PROC=4
export OMP_NUM_THREADS=4

export_MPI_FLAGS="--map-by numa --bind-to numa

./all.sh