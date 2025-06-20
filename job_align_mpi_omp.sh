#!/bin/bash

#==============================================================================
# SBATCH - Job Submission Script for Slurm
#==============================================================================

# --- JOB NAME ---
#SBATCH --job-name=align_mpi_omp_job

# --- OUTPUT AND ERROR FILES ---
#SBATCH --output=output_%j.txt   # File per lo standard output (%j viene sostituito con l'ID del job)
#SBATCH --error=error_%j.txt     # File per lo standard error

# --- RESOURCE REQUESTS ---
#SBATCH --nodes=2                     # Richiedi 2 nodi fisici
#SBATCH --ntasks-per-node=4           # Lancia 4 processi MPI su ciascun nodo
#SBATCH --cpus-per-task=8             # Assegna 8 core (e quindi 8 thread OpenMP) a ciascun processo MPI
#SBATCH --time=00:10:00               # Tempo massimo di esecuzione (10 minuti)
#SBATCH --partition=defq-noprio       # Scegli la partizione (coda).
#SBATCH --workdir=/mnt/beegfs/home/avona/sequence

# --- TOTAL MPI TASKS ---
# Il numero totale di processi MPI sarà: --nodes * --ntasks-per-node = 2 * 4 = 8 processi

# --- TOTAL CORES ---
# Il numero totale di core richiesti sarà: --nodes * --ntasks-per-node * --cpus-per-task = 2 * 4 * 8 = 64 core

#==============================================================================
# SCRIPT EXECUTION
#==============================================================================

echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on nodes: $SLURM_NODELIST"
echo "Number of Nodes: $SLURM_NNODES"
echo "Number of MPI tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "========================================================"

# --- SET UP ENVIRONMENT ---
# Carica gli stessi moduli usati per la compilazione
module purge  # Pulisce i moduli correnti per un ambiente pulito
module load gcc
module load openmpi

# --- SET OPENMP THREADS ---
# Informa OpenMP su quanti thread usare. Slurm imposta la variabile
# d'ambiente SLURM_CPUS_PER_TASK con il valore che abbiamo richiesto.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- DEFINE PROGRAM ARGUMENTS ---
# Esempio di argomenti per un test. Modificali secondo le tue necessità.
SEQ_LENGTH=10000000
PROB_G=0.25
PROB_C=0.25
PROB_A=0.25
PAT_RNG_NUM=500
PAT_RNG_LEN_MEAN=100
PAT_RNG_LEN_DEV=10
PAT_SAMP_NUM=500
PAT_SAMP_LEN_MEAN=100
PAT_SAMP_LEN_DEV=10
PAT_SAMP_LOC_MEAN=5000000
PAT_SAMP_LOC_DEV=1000000
PAT_SAMP_MIX=M
SEED=12345

# --- RUN THE PROGRAM ---
# 'srun' è il comando di Slurm per lanciare task paralleli.
# Distribuirà i processi MPI sui nodi allocati come richiesto.
srun ./align_mpi_omp $SEQ_LENGTH $PROB_G $PROB_C $PROB_A \
                     $PAT_RNG_NUM $PAT_RNG_LEN_MEAN $PAT_RNG_LEN_DEV \
                     $PAT_SAMP_NUM $PAT_SAMP_LEN_MEAN $PAT_SAMP_LEN_DEV \
                     $PAT_SAMP_LOC_MEAN $PAT_SAMP_LOC_DEV \
                     $PAT_SAMP_MIX $SEED

echo "========================================================"
echo "Job finished."
echo "========================================================"