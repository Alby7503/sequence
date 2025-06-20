echo "Running align_seq"
./align_seq 300 0.1 0.3 0.35 100 5 5 300 150 50 150 80 M 609823
echo "Running align_seq_new"
./align_seq_new 300 0.1 0.3 0.35 100 5 5 300 150 50 150 80 M 609823
echo "Running align_cuda"
./align_cuda 300 0.1 0.3 0.35 100 5 5 300 150 50 150 80 M 609823
echo "Running align_mpi_omp"
export OMP_NUM_THREADS=4  # Imposta il numero di thread OpenMP per processo
mpirun ./align_mpi_omp 300 0.1 0.3 0.35 100 5 5 300 150 50 150 80 M 609823