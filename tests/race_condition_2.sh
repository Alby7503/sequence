echo "Running align_seq"
./align_seq 10000 0.35 0.2 0.25 0 0 0 10000 9000 9000 50 100 M 4353435
echo "Running align_seq_new"
./align_seq_new 10000 0.35 0.2 0.25 0 0 0 10000 9000 9000 50 100 M 4353435
echo "Running align_cuda"
./align_cuda 10000 0.35 0.2 0.25 0 0 0 10000 9000 9000 50 100 M 4353435
echo "Running align_mpi_omp"
export OMP_NUM_THREADS=4  # Imposta il numero di thread OpenMP per processo
mpirun ./align_mpi_omp 10000 0.35 0.2 0.25 0 0 0 10000 9000 9000 50 100 M 4353435