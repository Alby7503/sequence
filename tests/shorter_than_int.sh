echo "Running align_seq"
./align_seq 429496730 0.35 0.2 0.25 0 0 0 1 1 0 4294967298 0 M 683224
echo "Running align_seq_new"
./align_seq_new 429496730 0.35 0.2 0.25 0 0 0 1 1 0 4294967298 0 M 683224
echo "Running align_cuda"
./align_cuda 429496730 0.35 0.2 0.25 0 0 0 1 1 0 4294967298 0 M 683224
echo "Running align_mpi_omp"
export OMP_NUM_THREADS=4  # Imposta il numero di thread OpenMP per processo
mpirun ./align_mpi_omp 429496730 0.35 0.2 0.25 0 0 0 1 1 0 4294967298 0 M 683224