echo "Running align_seq"
./align_seq 1000 0.35 0.2 0.25 0 0 0 20000 10 0 500 0 M 4353435
echo "Running align_seq_new"
./align_seq_new 1000 0.35 0.2 0.25 0 0 0 20000 10 0 500 0 M 4353435
echo "Running align_cuda"
./align_cuda 1000 0.35 0.2 0.25 0 0 0 20000 10 0 500 0 M 4353435
echo "Running align_mpi_omp"
export OMP_NUM_THREADS=4  # Imposta il numero di thread OpenMP per processo
mpirun ./align_mpi_omp 1000 0.35 0.2 0.25 0 0 0 20000 10 0 500 0 M 4353435