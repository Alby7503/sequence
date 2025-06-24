#!/bin/bash

#==============================================================================
# SBATCH - Job Submission Script for Slurm (DGX Node - 128 Cores)
#==============================================================================

# --- JOB NAME ---
# Nome descrittivo per il job.
#SBATCH --job-name=align_DGX_benchmark

# --- OUTPUT AND ERROR FILES ---
# Usa nomi di file chiari. %j è l'ID del job.
#SBATCH --output=benchmark_output_%j.txt
#SBATCH --error=benchmark_error_%j.txt

# --- RESOURCE REQUESTS (Full DGX Node) ---
# Richiedi 1 solo nodo, ma chiedilo in modo esclusivo se possibile.
#SBATCH --nodes=1
#SBATCH --exclusive

# Richiedi tutti i 128 core su quel nodo.
# Usiamo un approccio ibrido bilanciato:
# 16 processi MPI, ciascuno con 8 thread OpenMP (16 * 8 = 128).
# Questo è un ottimo punto di partenza per l'efficienza.
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8

# --- PARTITION AND TIME ---
# SBATCH --partition=high-prio
# Tempo massimo di esecuzione. Aumentalo se i test sono lunghi.
#SBATCH --time=00:05:00

#==============================================================================
# SCRIPT EXECUTION
#==============================================================================

echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on node: $SLURM_NODELIST"
echo "Total Cores Requested: $(($SLURM_NTASKS * $SLURM_CPUS_PER_TASK))"
echo "MPI tasks: $SLURM_NTASKS"
echo "OpenMP threads per task: $SLURM_CPUS_PER_TASK"
echo "========================================================"

# --- SET UP ENVIRONMENT ---
# Usa 'module avail' e 'module list' per trovare quelli giusti.
echo "Loading modules..."
module purge
module load gcc/64/4.1.5rc2
module load openmpi/gcc/64/4.1.5

# --- SET OPENMP THREADS ---
# Informa OpenMP su quanti thread usare.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "OMP_NUM_THREADS set to: $OMP_NUM_THREADS"

# --- RUN THE BENCHMARK TESTS IN SEQUENCE ---

# Definisci il nome del tuo eseguibile
EXECUTABLE="./align_mpi_omp_2"

echo "--- Test 1: Small scale ---"
srun $EXECUTABLE 300 0.1 0.3 0.35 100 5 5 300 150 50 150 80 M 609823
echo "--- Test 1 finished ---"

echo ""
echo "--- Test 2: Medium scale, sample-heavy ---"
srun $EXECUTABLE 1000 0.35 0.2 0.25 0 0 0 20000 10 0 500 0 M 4353435
echo "--- Test 2 finished ---"

echo ""
echo "--- Test 3: Large scale, large patterns ---"
srun $EXECUTABLE 10000 0.35 0.2 0.25 0 0 0 10000 9000 9000 50 100 M 4353435
echo "--- Test 3 finished ---"

echo ""
echo "--- Test 4: Large sequence, single small pattern ---"
srun $EXECUTABLE 4294967295 0.35 0.2 0.25 0 0 0 1 1 0 2147483647 0 M 683224
echo "--- Test 4 finished ---"

echo ""
echo "--- Test 5: Very large sequence ---"
srun $EXECUTABLE 4294967295 0.35 0.2 0.25 0 0 0 1 1 0 2147483647 0 M 683224
echo "--- Test 5 finished ---"

echo "========================================================"
echo "All benchmark tests finished. Job complete."
echo "========================================================"