#!/bin/bash

#==============================================================================
# SBATCH - Job Submission Script for Slurm (SAFE CONFIGURATION)
#==============================================================================

# --- JOB NAME ---
#SBATCH --job-name=align_safe_job

# --- OUTPUT AND ERROR FILES ---
# Nome del file di output. %j verrà sostituito con l'ID univoco del job.
#SBATCH --output=output_%j.txt
# Nome del file di errore.
#SBATCH --error=error_%j.txt

# --- RESOURCE REQUESTS (Conservative Configuration) ---
# Richiedi 1 solo nodo fisico per rispettare i limiti della partizione.
#SBATCH --nodes=1

# Richiedi 4 processi MPI. È un numero modesto.
#SBATCH --ntasks-per-node=4

# Assegna 2 core (e quindi 2 thread OpenMP) a ciascun processo MPI.
#SBATCH --cpus-per-task=2

# Il totale dei core richiesti è 4 * 2 = 8. Questo dovrebbe essere disponibile.

# --- PARTITION AND TIME ---
# Specifica la partizione corretta.
#SBATCH --partition=multicore
# Tempo massimo di esecuzione (es. 20 minuti).
#SBATCH --time=00:08:00

#==============================================================================
# SCRIPT EXECUTION
#==============================================================================

echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on node: $SLURM_NODELIST"
echo "Number of Nodes: $SLURM_NNODES"
echo "Number of MPI tasks requested: $SLURM_NTASKS"
echo "CPUs per task requested: $SLURM_CPUS_PER_TASK"
echo "========================================================"

# --- SET UP ENVIRONMENT ---
# È buona pratica pulire i moduli correnti per evitare conflitti.
#module purge
# Carica i moduli necessari. I nomi/versioni potrebbero variare sul tuo cluster.
# Usa 'module avail' per vedere quali sono disponibili.
#module load gcc
#module load openmpi

# --- SET OPENMP THREADS ---
# Informa OpenMP su quanti thread usare. Slurm imposta automaticamente
# la variabile d'ambiente SLURM_CPUS_PER_TASK con il valore che abbiamo richiesto.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "OMP_NUM_THREADS set to: $OMP_NUM_THREADS"

# --- DEFINE PROGRAM ARGUMENTS ---
# Usa i parametri "pesanti" per evidenziare il parallelismo.
#SEQ_LENGTH=200000000
#PROB_G=0.25
#PROB_C=0.25
#PROB_A=0.25
#PAT_RNG_NUM=50000
#PAT_RNG_LEN_MEAN=50
#PAT_RNG_LEN_DEV=5
#PAT_SAMP_NUM=0
#PAT_SAMP_LEN_MEAN=10
#PAT_SAMP_LEN_DEV=1
#PAT_SAMP_LOC_MEAN=100000000
#PAT_SAMP_LOC_DEV=50000000
#PAT_SAMP_MIX=B
#SEED=12345
SEQ_LENGTH=600000
PROB_G=0.35
PROB_C=0.2
PROB_A=0.25
PAT_RNG_NUM=35000
PAT_RNG_LEN_MEAN=1500
PAT_RNG_LEN_DEV=1000
PAT_SAMP_NUM=25000
PAT_SAMP_LEN_MEAN=1500
PAT_SAMP_LEN_DEV=500
PAT_SAMP_LOC_MEAN=500
PAT_SAMP_LOC_DEV=100
PAT_SAMP_MIX=M
SEED=4353435

# --- RUN THE PROGRAM ---
# 'srun' è il comando preferito in Slurm per lanciare task paralleli.
# Distribuirà automaticamente 4 processi MPI, ciascuno con 2 thread OpenMP.
echo "Starting the MPI+OpenMP program..."
srun ./align_mpi_omp_2 300 0.1 0.3 0.35 100 5 5 300 150 50 150 80 M 609823
srun ./align_mpi_omp_2 1000 0.35 0.2 0.25 0 0 0 20000 10 0 500 0 M 4353435
srun ./align_mpi_omp_2 10000 0.35 0.2 0.25 0 0 0 10000 9000 9000 50 100 M 4353435
srun ./align_mpi_omp_2 429496730 0.35 0.2 0.25 0 0 0 1 1 0 4294967298 0 M 683224
srun ./align_mpi_omp_2 4294967300 0.35 0.2 0.25 0 0 0 1 1 0 4294967298 0 M 683224
srun ./align_mpi_omp_2 600000 0.35 0.2 0.25 35000 1500 1000 25000 1500 500 500 100 M 4353435


echo "========================================================"
echo "Job finished with exit code $?."
echo "========================================================"