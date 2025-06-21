#!/bin/bash

export OMP_NUM_THREADS=4  # Imposta il numero di thread OpenMP per processo

make align_mpi_omp_2

# Funzione per eseguire e confrontare un test
run_test() {
    echo "Running test with arguments: $*"

    par_output=$(mpirun ./align_mpi_omp_2 "$@" | tee /dev/tty)
    #par_output=$(mpirun -np 4 ./align_mpi_omp_2 "$@" | tee /dev/tty)

    echo "------------------------------------------------------------"
}

# Test cases
#run_test 300 0.1 0.3 0.35 100 5 5 300 150 50 150 80 M 609823
#run_test 1000 0.35 0.2 0.25 0 0 0 20000 10 0 500 0 M 4353435
#run_test 10000 0.35 0.2 0.25 0 0 0 10000 9000 9000 50 100 M 4353435
#run_test 429496730 0.35 0.2 0.25 0 0 0 1 1 0 4294967298 0 M 683224
run_test 4294967300 0.35 0.2 0.25 0 0 0 1 1 0 4294967298 0 M 683224
