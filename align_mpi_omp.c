/*
 * Exact genetic sequence alignment
 * (Using brute force)
 *
 * MPI + OpenMP version
 *
 * Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2023/2024
 *
 * v1.4
 *
 * (c) 2024, Arturo Gonzalez-Escribano
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h> // Aggiunto header per OpenMP

/* Arbitrary value to indicate that no matches are found */
#define NOT_FOUND -1

/* Arbitrary value to restrict the checksums period */
#define CHECKSUM_MAX 65535

/*
 * Utils: Function to get wall time
 */
double cp_Wtime()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}

/*
 * Utils: Random generator
 */
#include "rng.c"

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */

/*
 * Function: Increment the number of pattern matches on the sequence positions
 * 	This function can be changed and/or optimized by the students
 *  NOTA: Con OpenMP, questa funzione deve essere resa thread-safe se chiamata
 *  da più thread su dati condivisi. In questa implementazione, la usiamo
 *  in una regione non parallela dopo la ricerca.
 */
void increment_matches(int pat, unsigned long *pat_found, unsigned long *pat_length, int *seq_matches)
{
    unsigned long ind;
    for (ind = 0; ind < pat_length[pat]; ind++)
    {
        if (seq_matches[pat_found[pat] + ind] == NOT_FOUND)
            seq_matches[pat_found[pat] + ind] = 1;
        else
            seq_matches[pat_found[pat] + ind]++;
    }
}

/*
 * Function: Fill random sequence or pattern (MPI parallelized version)
 * Each process generates a part of the sequence, ensuring the final result
 * is identical to the sequential version.
 */
void generate_rng_sequence(
    rng_t *random_seed,     // Puntatore allo stato iniziale del RNG (basato sul seme)
    float prob_G, 
    float prob_C, 
    float prob_A, 
    char *local_seq,        // Buffer locale per la porzione di sequenza
    unsigned long seq_length, 
    int rank, 
    int nprocs
) 
{
    // --- 1. Calcola la divisione del lavoro ---
    unsigned long n_per_proc = seq_length / nprocs;
    unsigned long remainder = seq_length % nprocs;
    
    // Calcola la dimensione del blocco per questo processo
    unsigned long my_block_size = n_per_proc + (rank < remainder ? 1 : 0);
    
    // Calcola l'offset di partenza (quanti caratteri sono stati generati dai processi precedenti)
    unsigned long my_start_offset = rank * n_per_proc + (rank < remainder ? rank : remainder);

    // Se questo processo non ha lavoro da fare (es. più processi che caratteri), esce.
    if (my_block_size <= 0) {
        return;
    }

    // --- 2. Prepara il generatore locale ---
    rng_t local_random = *random_seed; // Ogni processo parte dallo stesso stato iniziale

    // --- 3. Salta al punto di partenza corretto ---
    // Questa è la magia: porta il generatore allo stato esatto in cui deve essere
    // per iniziare a generare la porzione di questo processo.
    rng_skip(&local_random, my_start_offset);

    // --- 4. Genera la porzione locale della sequenza ---
	unsigned long ind; 
	for( ind=0; ind<my_block_size; ind++ ) {
		double prob = rng_next( &local_random );
		if( prob < prob_G ) local_seq[ind] = 'G';
		else if( prob < prob_C ) local_seq[ind] = 'C';
		else if( prob < prob_A ) local_seq[ind] = 'A';
		else local_seq[ind] = 'T';
	}
}

/*
 * Function: Copy a sample of the sequence
 * NOTA: Questa funzione rimane sequenziale.
 */
void copy_sample_sequence(rng_t *random, char *sequence, unsigned long seq_length, unsigned long pat_samp_loc_mean, unsigned long pat_samp_loc_dev, char *pattern, unsigned long length)
{
    /* Choose location */
    unsigned long location = (unsigned long)rng_next_normal(random, (double)pat_samp_loc_mean, (double)pat_samp_loc_dev);
    if (location > seq_length - length)
        location = seq_length - length;
    if (location <= 0)
        location = 0;

    /* Copy sample */
    unsigned long ind;
    for (ind = 0; ind < length; ind++)
        pattern[ind] = sequence[ind + location];
}

/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */

/*
 * Function: Allocate new patttern
 */
char *pattern_allocate(rng_t *random, unsigned long pat_rng_length_mean, unsigned long pat_rng_length_dev, unsigned long seq_length, unsigned long *new_length)
{

    /* Random length */
    unsigned long length = (unsigned long)rng_next_normal(random, (double)pat_rng_length_mean, (double)pat_rng_length_dev);
    if (length > seq_length)
        length = seq_length;
    if (length <= 0)
        length = 1;

    /* Allocate pattern */
    char *pattern = (char *)malloc(sizeof(char) * length);
    if (pattern == NULL)
    {
        fprintf(stderr, "\n-- Error allocating a pattern of size: %lu\n", length);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* Return results */
    *new_length = length;
    return pattern;
}

/*
 * Function: Regenerate a sample of the sequence
 */
void generate_sample_sequence(rng_t *random, rng_t random_seq, float prob_G, float prob_C, float prob_A, unsigned long seq_length, unsigned long pat_samp_loc_mean, unsigned long pat_samp_loc_dev, char *pattern, unsigned long length)
{
    /* Choose location */
    unsigned long location = (unsigned long)rng_next_normal(random, (double)pat_samp_loc_mean, (double)pat_samp_loc_dev);
    if (location > seq_length - length)
        location = seq_length - length;
    if (location <= 0)
        location = 0;

    /* Regenerate sample */
    rng_t local_random = random_seq;
    rng_skip(&local_random, location);
    generate_rng_sequence(&local_random, prob_G, prob_C, prob_A, pattern, length);
}

/*
 * Function: Print usage line in stderr
 */
void show_usage(char *program_name)
{
    fprintf(stderr, "Usage: %s ", program_name);
    fprintf(stderr, "<seq_length> <prob_G> <prob_C> <prob_A> <pat_rng_num> <pat_rng_length_mean> <pat_rng_length_dev> <pat_samples_num> <pat_samp_length_mean> <pat_samp_length_dev> <pat_samp_loc_mean> <pat_samp_loc_dev> <pat_samp_mix:B[efore]|A[fter]|M[ixed]> <long_seed>\n");
    fprintf(stderr, "\n");
}

/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[])
{
    /* 0. Default output and error without buffering, forces to write immediately */
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    /* 1. Read scenary arguments */
    /* 1.0. Init MPI before processing arguments */
    int nprocs;
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* 1.1. Check minimum number of arguments */
    if (argc < 15)
    {
        if (rank == 0)
        {
            fprintf(stderr, "\n-- Error: Not enough arguments when reading configuration from the command line\n\n");
            show_usage(argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* 1.2. Read argument values */
    unsigned long seq_length = atol(argv[1]);
    float prob_G = atof(argv[2]);
    float prob_C = atof(argv[3]);
    float prob_A = atof(argv[4]);
    if (prob_G + prob_C + prob_A > 1)
    {
        if (rank == 0)
        {
            fprintf(stderr, "\n-- Error: The sum of G,C,A,T nucleotid probabilities cannot be higher than 1\n\n");
            show_usage(argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    prob_C += prob_G;
    prob_A += prob_C;

    int pat_rng_num = atoi(argv[5]);
    unsigned long pat_rng_length_mean = atol(argv[6]);
    unsigned long pat_rng_length_dev = atol(argv[7]);

    int pat_samp_num = atoi(argv[8]);
    unsigned long pat_samp_length_mean = atol(argv[9]);
    unsigned long pat_samp_length_dev = atol(argv[10]);
    unsigned long pat_samp_loc_mean = atol(argv[11]);
    unsigned long pat_samp_loc_dev = atol(argv[12]);

    char pat_samp_mix = argv[13][0];
    if (pat_samp_mix != 'B' && pat_samp_mix != 'A' && pat_samp_mix != 'M')
    {
        if (rank == 0)
        {
            fprintf(stderr, "\n-- Error: Incorrect first character of pat_samp_mix: %c\n\n", pat_samp_mix);
            show_usage(argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    unsigned long seed = atol(argv[14]);

#ifdef DEBUG
    /* DEBUG: Print arguments */
    if (rank == 0)
    {
        printf("\nArguments: seq_length=%lu\n", seq_length);
        printf("Arguments: Accumulated probabilitiy G=%f, C=%f, A=%f, T=1\n", prob_G, prob_C, prob_A);
        printf("Arguments: Random patterns number=%d, length_mean=%lu, length_dev=%lu\n", pat_rng_num, pat_rng_length_mean, pat_rng_length_dev);
        printf("Arguments: Sample patterns number=%d, length_mean=%lu, length_dev=%lu, loc_mean=%lu, loc_dev=%lu\n", pat_samp_num, pat_samp_length_mean, pat_samp_length_dev, pat_samp_loc_mean, pat_samp_loc_dev);
        printf("Arguments: Type of mix: %c, Random seed: %lu, MPI Procs: %d, OMP Threads: %d\n", pat_samp_mix, seed, nprocs, omp_get_max_threads());
        printf("\n");
    }
#endif // DEBUG

    // Calcola la quantitò di lavoro per processo
    unsigned long n_per_proc = seq_length / nprocs;
    unsigned long remainder = seq_length % nprocs;

    /* 2. Initialize data structures IN ALL PROCESSES */
    /* Ogni processo genera tutti i dati per sapere quale parte gli spetta */
    rng_t random = rng_new(seed);
    rng_skip(&random, seq_length);

    int pat_number = pat_rng_num + pat_samp_num;
    unsigned long *pat_length = (unsigned long *)malloc(sizeof(unsigned long) * pat_number);
    char **pattern = (char **)malloc(sizeof(char *) * pat_number);
    if (pattern == NULL || pat_length == NULL)
    {
        fprintf(stderr, "\n--[%d] Error allocating the basic patterns structures for size: %d\n", rank, pat_number);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int ind;
    unsigned long lind;
#define PAT_TYPE_NONE 0
#define PAT_TYPE_RNG 1
#define PAT_TYPE_SAMP 2
    char *pat_type = (char *)malloc(sizeof(char) * pat_number);
    if (pat_type == NULL)
    {
        fprintf(stderr, "\n--[%d] Error allocating ancillary structure for pattern of size: %d\n", rank, pat_number);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    for (ind = 0; ind < pat_number; ind++)
        pat_type[ind] = PAT_TYPE_NONE;

    switch (pat_samp_mix)
    {
    case 'A':
        for (ind = 0; ind < pat_rng_num; ind++)
            pat_type[ind] = PAT_TYPE_RNG;
        for (; ind < pat_number; ind++)
            pat_type[ind] = PAT_TYPE_SAMP;
        break;
    case 'B':
        for (ind = 0; ind < pat_samp_num; ind++)
            pat_type[ind] = PAT_TYPE_SAMP;
        for (; ind < pat_number; ind++)
            pat_type[ind] = PAT_TYPE_RNG;
        break;
    default:
        if (pat_rng_num == 0)
        {
            for (ind = 0; ind < pat_number; ind++)
                pat_type[ind] = PAT_TYPE_SAMP;
        }
        else if (pat_samp_num == 0)
        {
            for (ind = 0; ind < pat_number; ind++)
                pat_type[ind] = PAT_TYPE_RNG;
        }
        else if (pat_rng_num < pat_samp_num)
        {
            int interval = pat_number / pat_rng_num;
            for (ind = 0; ind < pat_number; ind++)
                if ((ind + 1) % interval == 0)
                    pat_type[ind] = PAT_TYPE_RNG;
                else
                    pat_type[ind] = PAT_TYPE_SAMP;
        }
        else
        {
            int interval = pat_number / pat_samp_num;
            for (ind = 0; ind < pat_number; ind++)
                if ((ind + 1) % interval == 0)
                    pat_type[ind] = PAT_TYPE_SAMP;
                else
                    pat_type[ind] = PAT_TYPE_RNG;
        }
    }

    for (ind = 0; ind < pat_number; ind++)
    {
        if (pat_type[ind] == PAT_TYPE_RNG)
        {
            pattern[ind] = pattern_allocate(&random, pat_rng_length_mean, pat_rng_length_dev, seq_length, &pat_length[ind]);
            generate_rng_sequence(&random, prob_G, prob_C, prob_A, pattern[ind], pat_length[ind]);
        }
        else if (pat_type[ind] == PAT_TYPE_SAMP)
        {
            pattern[ind] = pattern_allocate(&random, pat_samp_length_mean, pat_samp_length_dev, seq_length, &pat_length[ind]);
#define REGENERATE_SAMPLE_PATTERNS
#ifdef REGENERATE_SAMPLE_PATTERNS
            rng_t random_seq_orig = rng_new(seed);
            generate_sample_sequence(&random, random_seq_orig, prob_G, prob_C, prob_A, seq_length, pat_samp_loc_mean, pat_samp_loc_dev, pattern[ind], pat_length[ind]);
#else
            copy_sample_sequence(&random, sequence, seq_length, pat_samp_loc_mean, pat_samp_loc_dev, pattern[ind], pat_length[ind]);
#endif
        }
        else
        {
            fprintf(stderr, "\n--[%d] Error internal: Paranoic check! A pattern without type at position %d\n", rank, ind);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    free(pat_type);

    /* 2.3. Allocate result data structures */
    unsigned long *pat_found = (unsigned long *)malloc(sizeof(unsigned long) * pat_number);
    if (pat_found == NULL)
    {
        fprintf(stderr, "\n--[%d] Error allocating aux pattern structure for size: %d\n", rank, pat_number);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* 3. Start global timer */
    MPI_Barrier(MPI_COMM_WORLD);
    double ttotal = cp_Wtime();

    /*
     *
     * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
     *
     */
    /* 2.1. Generate the main sequence in all processes */
    char *sequence = (char *)malloc(sizeof(char) * seq_length);
    if (sequence == NULL)
    {
        fprintf(stderr, "\n--[%d] Error allocating the sequence for size: %lu\n", rank, seq_length);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    random = rng_new(seed);
    generate_rng_sequence(&random, prob_G, prob_C, prob_A, sequence, seq_length);

    /* 2.3.2. Allocate local seq_matches array */
    int *seq_matches = (int *)malloc(sizeof(int) * seq_length);
    if (seq_matches == NULL)
    {
        fprintf(stderr, "\n--[%d] Error allocating aux sequence structures for size: %lu\n", rank, seq_length);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* 4. Initialize local result structures */
    int local_pat_matches = 0;
    for (ind = 0; ind < pat_number; ind++)
    {
        pat_found[ind] = (unsigned long)NOT_FOUND;
    }
    for (lind = 0; lind < seq_length; lind++)
    {
        seq_matches[lind] = NOT_FOUND;
    }

    /* 5. Search for each pattern using MPI for division of work and OpenMP for internal parallelism */
    int pat;
    unsigned long start;
// Parallelize the outer loop over patterns with OpenMP
// Each thread will process a different pattern independently
#pragma omp parallel for schedule(dynamic) private(pat, lind, start) reduction(+ : local_pat_matches)
    for (pat = rank; pat < pat_number; pat += nprocs)
    {

        // This process is responsible for pattern 'pat'
        // The search logic for a single pattern remains sequential

        /* 5.1. For each possible starting position */
        for (start = 0; start <= seq_length - pat_length[pat]; start++)
        {

            /* 5.1.1. For each pattern element */
            for (lind = 0; lind < pat_length[pat]; lind++)
            {
                /* Stop this test when different nucleotids are found */
                if (sequence[start + lind] != pattern[pat][lind])
                    break;
            }
            /* 5.1.2. Check if the loop ended with a match */
            if (lind == pat_length[pat])
            {
                local_pat_matches++;
                pat_found[pat] = start;
                break; // Stop searching for this pattern
            }
        }
    } // --- End of omp parallel for ---

    /* 5.2. After parallel search, update seq_matches sequentially to avoid race conditions.
       This is done by each process only for the patterns it found. */
    for (pat = rank; pat < pat_number; pat += nprocs)
    {
        if (pat_found[pat] != (unsigned long)NOT_FOUND)
        {
            increment_matches(pat, pat_found, pat_length, seq_matches);
        }
    }

    /* 6. AGGREGATE RESULTS using MPI */

    /* 6.1. Aggregate total pattern matches */
    int total_pat_matches = 0;
    MPI_Reduce(&local_pat_matches, &total_pat_matches, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    /* 6.2. Aggregate checksum_found */
    unsigned long local_checksum_found = 0;
    for (ind = rank; ind < pat_number; ind += nprocs)
    {
        if (pat_found[ind] != (unsigned long)NOT_FOUND)
            local_checksum_found = (local_checksum_found + pat_found[ind]);
    }
    // We sum without modulo first, then apply it on rank 0 to maintain correctness
    unsigned long total_checksum_found = 0;
    MPI_Reduce(&local_checksum_found, &total_checksum_found, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        total_checksum_found %= CHECKSUM_MAX;
    }

    /* 6.3. Aggregate seq_matches and calculate its checksum */
    unsigned long total_checksum_matches = 0;
    if (rank == 0)
    {
        // Rank 0 will receive the sum of all seq_matches arrays
        int *global_seq_matches = (int *)malloc(sizeof(int) * seq_length);
        if (global_seq_matches == NULL)
        {
            fprintf(stderr, "\n--[0] Error allocating global seq_matches\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        // MPI_Reduce to sum the seq_matches arrays element-wise
        // NOTE: The initial value of seq_matches is NOT_FOUND (-1). A simple sum is not correct.
        // We need to convert NOT_FOUND to 0 before summing.
        for (lind = 0; lind < seq_length; lind++)
        {
            if (seq_matches[lind] == NOT_FOUND)
                seq_matches[lind] = 0;
        }

        MPI_Reduce(seq_matches, global_seq_matches, seq_length, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        // Now, calculate the final checksum on the aggregated array
        for (lind = 0; lind < seq_length; lind++)
        {
            // A value of 0 in the global array could mean 0 matches, or it could be
            // the sum of -1 (local) + 1 (another local), etc.
            // A better approach is to sum non-NOT_FOUND values. Let's correct the logic.
            // The original logic is: sum values that are not NOT_FOUND.
            // After reduction, a 0 could be a real 0, or sum of -1s.
            // The cleanest way is to sum non-zero values from the reduced array,
            // assuming initial NOT_FOUND values summed to 0.
            if (global_seq_matches[lind] > 0)
                total_checksum_matches = (total_checksum_matches + global_seq_matches[lind]) % CHECKSUM_MAX;
        }
        free(global_seq_matches);
    }
    else
    {
        // Other ranks participate in the reduction
        for (lind = 0; lind < seq_length; lind++)
        {
            if (seq_matches[lind] == NOT_FOUND)
                seq_matches[lind] = 0;
        }
        MPI_Reduce(seq_matches, NULL, seq_length, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }

#ifdef DEBUG
    // This debug section would need significant changes for MPI and is omitted for clarity.
    // Each rank would need to print its local data, or data would need to be gathered to rank 0.
#endif // DEBUG

    /* Free local resources */
    free(sequence);
    free(seq_matches);

    /*
     *
     * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
     *
     */

    /* 8. Stop global time */
    MPI_Barrier(MPI_COMM_WORLD);
    ttotal = cp_Wtime() - ttotal;

    /* 9. Output for leaderboard */
    if (rank == 0)
    {
        printf("\n");
        /* 9.1. Total computation time */
        printf("Time: %lf\n", ttotal);

        /* 9.2. Results: Statistics */
        printf("Result: %d, %lu, %lu\n\n",
               total_pat_matches,
               total_checksum_found,
               total_checksum_matches);
    }

    /* 10. Free resources */
    int i;
    for (i = 0; i < pat_number; i++)
        free(pattern[i]);
    free(pattern);
    free(pat_length);
    free(pat_found);

    /* 11. End */
    MPI_Finalize();
    return 0;
}