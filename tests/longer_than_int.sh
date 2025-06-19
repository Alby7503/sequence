./align_seq_new 4294967300 0.35 0.2 0.25 0 0 0 1 1 0 4294967298 0 M 683224
./align_cuda 4294967300 0.35 0.2 0.25 0 0 0 1 1 0 4294967298 0 M 683224
export OMP_NUM_THREADS=4  # Imposta il numero di thread OpenMP per processo
mpirun -np 2 ./align_mpi_omp 4294967300 0.35 0.2 0.25 0 0 0 1 1 0 4294967298 0 M 683224