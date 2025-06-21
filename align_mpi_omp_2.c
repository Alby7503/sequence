/*
 * Exact genetic sequence alignment
 * (Using brute force)
 *
 * MPI version
 *
 * Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2023/2024
 *
 * v1.3
 *
 * (c) 2024, Arturo Gonzalez-Escribano
 */
#include <limits.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

/* Arbitrary value to indicate that no matches are found */
#define NOT_FOUND -1

/* Arbitrary value to restrict the checksums period */
#define CHECKSUM_MAX 65535

/*
 * Utils: Function to get wall time
 */
double cp_Wtime() {
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
 */
void increment_matches(int pat, unsigned long *pat_found, unsigned long *pat_length, int *seq_matches) {
    for (unsigned long ind = 0; ind < pat_length[pat]; ind++) {
#pragma omp atomic update
        seq_matches[pat_found[pat] + ind]++;
    }
}

/*
 * Function: Fill random sequence or pattern
 */
void generate_rng_sequence(rng_t *random, float prob_G, float prob_C, float prob_A, char *seq, unsigned long length) {
    unsigned long ind;
    for (ind = 0; ind < length; ind++) {
        double prob = rng_next(random);
        if (prob < prob_G)
            seq[ind] = 'G';
        else if (prob < prob_C)
            seq[ind] = 'C';
        else if (prob < prob_A)
            seq[ind] = 'A';
        else
            seq[ind] = 'T';
    }
}
void generate_rng_sequence_mpi(rng_t *random, float prob_G, float prob_C, float prob_A, char *seq, int64_t length) {
    int64_t length_per_thread = length / omp_get_max_threads();
#pragma omp parallel
    {
        rng_t local_random = *random;
        rng_skip(&local_random, length_per_thread * omp_get_thread_num());

        int tid = omp_get_thread_num();
        int64_t start = length_per_thread * tid;
        int64_t end = (tid == omp_get_max_threads() - 1) ? length : start + length_per_thread;

        for (int64_t ind = start; ind < end; ind++) {
            double prob = rng_next(&local_random);
            if (prob < prob_G)
                seq[ind] = 'G';
            else if (prob < prob_C)
                seq[ind] = 'C';
            else if (prob < prob_A)
                seq[ind] = 'A';
            else
                seq[ind] = 'T';
        }
    }
}

/*
 * Function: Copy a sample of the sequence
 */
void copy_sample_sequence(rng_t *random, char *sequence, unsigned long seq_length, unsigned long pat_samp_loc_mean, unsigned long pat_samp_loc_dev, char *pattern, unsigned long length) {
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
char *pattern_allocate(rng_t *random, unsigned long pat_rng_length_mean, unsigned long pat_rng_length_dev, unsigned long seq_length, unsigned long *new_length) {

    /* Random length */
    unsigned long length = (unsigned long)rng_next_normal(random, (double)pat_rng_length_mean, (double)pat_rng_length_dev);
    if (length > seq_length)
        length = seq_length;
    if (length <= 0)
        length = 1;

    /* Allocate pattern */
    char *pattern = (char *)malloc(sizeof(char) * length);
    if (pattern == NULL) {
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
void generate_sample_sequence(rng_t *random, rng_t random_seq, float prob_G, float prob_C, float prob_A, unsigned long seq_length, unsigned long pat_samp_loc_mean, unsigned long pat_samp_loc_dev, char *pattern, unsigned long length) {
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
void show_usage(char *program_name) {
    fprintf(stderr, "Usage: %s ", program_name);
    fprintf(stderr, "<seq_length> <prob_G> <prob_C> <prob_A> <pat_rng_num> <pat_rng_length_mean> <pat_rng_length_dev> <pat_samples_num> <pat_samp_length_mean> <pat_samp_length_dev> <pat_samp_loc_mean> <pat_samp_loc_dev> <pat_samp_mix:B[efore]|A[fter]|M[ixed]> <long_seed>\n");
    fprintf(stderr, "\n");
}

/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {
    /* 0. Default output and error without buffering, forces to write immediately */
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    /* 1. Read scenary arguments */
    /* 1.0. Init MPI before processing arguments */
    int provided;
    // MPI_Init(&argc, &argv);
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // if (rank == 0)
    //        printf("START\n");

    /* 1.1. Check minimum number of arguments */
    if (argc < 15) {
        fprintf(stderr, "\n-- Error: Not enough arguments when reading configuration from the command line\n\n");
        show_usage(argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* 1.2. Read argument values */
    unsigned long seq_length = atol(argv[1]);
    float prob_G = atof(argv[2]);
    float prob_C = atof(argv[3]);
    float prob_A = atof(argv[4]);
    if (prob_G + prob_C + prob_A > 1) {
        fprintf(stderr, "\n-- Error: The sum of G,C,A,T nucleotid probabilities cannot be higher than 1\n\n");
        show_usage(argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    double timer = cp_Wtime();
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
    if (pat_samp_mix != 'B' && pat_samp_mix != 'A' && pat_samp_mix != 'M') {
        fprintf(stderr, "\n-- Error: Incorrect first character of pat_samp_mix: %c\n\n", pat_samp_mix);
        show_usage(argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    unsigned long seed = atol(argv[14]);

#ifdef DEBUG
    /* DEBUG: Print arguments */
    if (rank == 0) {
        printf("\nArguments: seq_length=%lu\n", seq_length);
        printf("Arguments: Accumulated probabilitiy G=%f, C=%f, A=%f, T=1\n", prob_G, prob_C, prob_A);
        printf("Arguments: Random patterns number=%d, length_mean=%lu, length_dev=%lu\n", pat_rng_num, pat_rng_length_mean, pat_rng_length_dev);
        printf("Arguments: Sample patterns number=%d, length_mean=%lu, length_dev=%lu, loc_mean=%lu, loc_dev=%lu\n", pat_samp_num, pat_samp_length_mean, pat_samp_length_dev, pat_samp_loc_mean, pat_samp_loc_dev);
        printf("Arguments: Type of mix: %c, Random seed: %lu\n", pat_samp_mix, seed);
        printf("\n");
    }
#endif // DEBUG

    /* 2. Initialize data structures */
    /* 2.1. Skip allocate and fill sequence */
    rng_t random = rng_new(seed);
    rng_skip(&random, seq_length);

    /* 2.2. Allocate and fill patterns */
    /* 2.2.1 Allocate main structures */
    int pat_number = pat_rng_num + pat_samp_num;
    unsigned long *pat_length = (unsigned long *)malloc(sizeof(unsigned long) * pat_number);
    char **pattern = (char **)malloc(sizeof(char *) * pat_number);
    if (pattern == NULL || pat_length == NULL) {
        fprintf(stderr, "\n-- Error allocating the basic patterns structures for size: %d\n", pat_number);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* 2.2.2 Allocate and initialize ancillary structure for pattern types */
    int ind;
    unsigned long lind;
#define PAT_TYPE_NONE 0
#define PAT_TYPE_RNG 1
#define PAT_TYPE_SAMP 2
    char *pat_type = (char *)malloc(sizeof(char) * pat_number);
    if (pat_type == NULL) {
        fprintf(stderr, "\n-- Error allocating ancillary structure for pattern of size: %d\n", pat_number);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    for (ind = 0; ind < pat_number; ind++)
        pat_type[ind] = PAT_TYPE_NONE;

    /* 2.2.3 Fill up pattern types using the chosen mode */
    switch (pat_samp_mix) {
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
        if (pat_rng_num == 0) {
            for (ind = 0; ind < pat_number; ind++)
                pat_type[ind] = PAT_TYPE_SAMP;
        } else if (pat_samp_num == 0) {
            for (ind = 0; ind < pat_number; ind++)
                pat_type[ind] = PAT_TYPE_RNG;
        } else if (pat_rng_num < pat_samp_num) {
            int interval = pat_number / pat_rng_num;
            for (ind = 0; ind < pat_number; ind++)
                if ((ind + 1) % interval == 0)
                    pat_type[ind] = PAT_TYPE_RNG;
                else
                    pat_type[ind] = PAT_TYPE_SAMP;
        } else {
            int interval = pat_number / pat_samp_num;
            for (ind = 0; ind < pat_number; ind++)
                if ((ind + 1) % interval == 0)
                    pat_type[ind] = PAT_TYPE_SAMP;
                else
                    pat_type[ind] = PAT_TYPE_RNG;
        }
    }

    /* 2.2.4 Generate the patterns */
    for (ind = 0; ind < pat_number; ind++) {
        if (pat_type[ind] == PAT_TYPE_RNG) {
            pattern[ind] = pattern_allocate(&random, pat_rng_length_mean, pat_rng_length_dev, seq_length, &pat_length[ind]);
            generate_rng_sequence(&random, prob_G, prob_C, prob_A, pattern[ind], pat_length[ind]);
        } else if (pat_type[ind] == PAT_TYPE_SAMP) {
            pattern[ind] = pattern_allocate(&random, pat_samp_length_mean, pat_samp_length_dev, seq_length, &pat_length[ind]);
#define REGENERATE_SAMPLE_PATTERNS
#ifdef REGENERATE_SAMPLE_PATTERNS
            rng_t random_seq_orig = rng_new(seed);
            generate_sample_sequence(&random, random_seq_orig, prob_G, prob_C, prob_A, seq_length, pat_samp_loc_mean, pat_samp_loc_dev, pattern[ind], pat_length[ind]);
#else
            copy_sample_sequence(&random, sequence, seq_length, pat_samp_loc_mean, pat_samp_loc_dev, pattern[ind], pat_length[ind]);
#endif
        } else {
            fprintf(stderr, "\n-- Error internal: Paranoic check! A pattern without type at position %d\n", ind);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    free(pat_type);

    /* Avoid the usage of arguments to take strategic decisions
     * In a real case the user only has the patterns and sequence data to analize
     */
    argc = 0;
    argv = NULL;
    pat_rng_num = 0;
    pat_rng_length_mean = 0;
    pat_rng_length_dev = 0;
    pat_samp_num = 0;
    pat_samp_length_mean = 0;
    pat_samp_length_dev = 0;
    pat_samp_loc_mean = 0;
    pat_samp_loc_dev = 0;
    pat_samp_mix = '0';

    /* 2.3. Other result data and structures */
    int pat_matches = 0;

    /* 2.3.1. Other results related to patterns */
    unsigned long *pat_found;
    pat_found = (unsigned long *)malloc(sizeof(unsigned long) * pat_number);
    if (pat_found == NULL) {
        fprintf(stderr, "\n-- Error allocating aux pattern structure for size: %d\n", pat_number);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* 3. Start global timer */
    MPI_Barrier(MPI_COMM_WORLD);
    timer = cp_Wtime() - timer;
    // if (rank == 0)
    //        printf("Initialization time: %lf\n", timer);
    double ttotal = cp_Wtime();

    /*
     *
     * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
     *
     */
    // if (rank == 0)
    //        printf("START HERE\n");
    /* 2.1. Allocate and fill sequence */
    void *baseptr = NULL;
    char *sequence = NULL;

    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);

    int shrm_rank; //, shm_nprocs;
    MPI_Comm_rank(shmcomm, &shrm_rank);

    MPI_Aint size = (shrm_rank == 0) ? sizeof(char) * seq_length : 0;

    // MPI_Win matches_shwin;
    MPI_Win sequence_shmwin;
    // MPI_Win
    MPI_Win_allocate_shared(size, sizeof(char), MPI_INFO_NULL, shmcomm, &baseptr, &sequence_shmwin);

    MPI_Aint sizeB;
    int disp_unit;
    MPI_Win_shared_query(sequence_shmwin, 0, &sizeB, &disp_unit, &baseptr);

    sequence = (char *)baseptr;

    if (sequence == NULL) {
        fprintf(stderr, "\n-- Error allocating the sequence of size: %lu\n", seq_length);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Win_sync(sequence_shmwin);
    MPI_Barrier(shmcomm);

    random = rng_new(seed);

    int64_t n_per_proc = seq_length / nprocs;
    int64_t remainder = seq_length % nprocs;

    int64_t *recvcounts = malloc(sizeof(int64_t) * nprocs); // Quanti elementi ricevere da ogni processo
    if (recvcounts == NULL) {
        fprintf(stderr, "\n-- Error allocating recvcounts for size: %d\n", nprocs);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    int64_t *displs = malloc(sizeof(int64_t) * nprocs); // A quale offset posizionare i dati ricevuti
    if (displs == NULL) {
        fprintf(stderr, "\n-- Error allocating displs for size: %d\n", nprocs);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    for (int64_t i = 0; i < nprocs; i++) {
        recvcounts[i] = n_per_proc + (i < remainder ? 1 : 0);
        displs[i] = (i > 0) ? (displs[i - 1] + recvcounts[i - 1]) : 0;
    }

    rng_skip(&random, rank * n_per_proc + (rank < remainder ? rank : remainder));
    generate_rng_sequence_mpi(&random, prob_G, prob_C, prob_A, sequence + displs[rank], recvcounts[rank]);

    MPI_Barrier(MPI_COMM_WORLD);

    /* 2.1.1. Gather the sequence in all processes */
#ifdef DEBUG
    /* DEBUG: Print sequence and patterns */
    printf("-----------------\n");
    printf("Sequence: ");
    for (lind = 0; lind < seq_length; lind++)
        printf("%c", sequence[lind]);
    printf("\n-----------------\n");
    printf("Patterns: %d ( rng: %d, samples: %d )\n", pat_number, pat_rng_num, pat_samp_num);
    int debug_pat;
    for (debug_pat = 0; debug_pat < pat_number; debug_pat++) {
        printf("Pat[%d]: ", debug_pat);
        for (lind = 0; lind < pat_length[debug_pat]; lind++)
            printf("%c", pattern[debug_pat][lind]);
        printf("\n");
    }
    printf("-----------------\n\n");
#endif // DEBUG

    /* 2.3.2. Other results related to the main sequence */
    // int *seq_matches;
    // baseptr = NULL;

    // size = (shrm_rank == 0) ? sizeof(int) * seq_length : 0;

    /*MPI_Win_allocate_shared(
        size,          // Size of the sequence matches array
        sizeof(int),   // Size of each element in the array
        MPI_INFO_NULL, // No special info for the window
        shmcomm,       // Shared memory communicator
        &baseptr,      // Pointer to the base address of the window
        &matches_shwin // Window object
    );*/

    // MPI_Win_shared_query(matches_shwin, 0, &sizeB, &disp_unit, &baseptr);

    // seq_matches = (int *)baseptr;
    //  seq_matches = (int *)malloc(sizeof(int) * seq_length);
    int *seq_matches = (int *)calloc(seq_length, sizeof(int));

    if (seq_matches == NULL) {
        fprintf(stderr, "\n-- Error allocating aux sequence structures for size: %lu\n", seq_length);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int64_t pattern_per_rank = pat_number / nprocs;
    int64_t pattern_remainder = pat_number % nprocs;

    int64_t pattern_start = rank * pattern_per_rank + (rank < pattern_remainder ? rank : pattern_remainder);
    int64_t pattern_end = pattern_start + pattern_per_rank + (rank < pattern_remainder ? 1 : 0);

/* 4. Initialize ancillary structures */
#pragma omp parallel for private(ind)
    for (ind = pattern_start; ind < pattern_end; ind++) {
        pat_found[ind] = NOT_FOUND;
    }
    /*#pragma omp parallel for private(lind)
        for (lind = displs[rank]; lind < displs[rank] + recvcounts[rank]; lind++) {
            seq_matches[lind] = NOT_FOUND;
        }*/
    MPI_Barrier(MPI_COMM_WORLD);

    /* 5. Search for each pattern */
    unsigned long start;

// #pragma omp parallel for private(start, lind) reduction(+ : pat_matches) schedule(dynamic)
#pragma omp parallel for private(start) reduction(+ : pat_matches)
    for (int64_t pat = pattern_start; pat < pattern_end; pat++) {
        const char *current_pattern = pattern[pat];
        const unsigned int current_pat_length = pat_length[pat];
        const unsigned long max_start = seq_length - current_pat_length;
        /* 5.1. For each posible starting position */
        for (start = 0; start <= max_start; start++) {
            //     /* 5.1.1. For each pattern element */
            //     for (lind = 0; lind < pat_length[pat]; lind++) {
            //         /* Stop this test when different nucleotids are found */
            //     if (sequence[start + lind] != pattern[pat][lind])
            //         break;

            if (memcmp(sequence + start, current_pattern, current_pat_length) == 0) {
                pat_matches++;
                pat_found[pat] = start;
                increment_matches(pat, pat_found, pat_length, seq_matches);
                break; // Pattern trovato, passa al successivo
            }
        }
        /* 5.1.2. Check if the loop ended with a match */
        // if (lind == pat_length[pat]) {
        //     pat_matches++;
        //     pat_found[pat] = start;
        //     break;
        // }
    }

    /* 5.2. Pattern found */
    // if (pat_found[pat] != NOT_FOUND) {
    //     /* 4.2.1. Increment the number of pattern matches on the sequence positions */
    //     increment_matches(pat, pat_found, pat_length, seq_matches);
    // }

    MPI_Barrier(MPI_COMM_WORLD);

    /* 7. Check sums */
    unsigned long checksum_matches = 0;
    unsigned long checksum_found = 0;

#pragma omp parallel for private(ind) reduction(+ : checksum_found)
    for (ind = pattern_start; ind < pattern_end; ind++) {
        if (pat_found[ind] != NOT_FOUND)
            checksum_found += pat_found[ind];
    }

    int *total_seq_matches = NULL;
    if (rank == 0) {
        total_seq_matches = (int *)malloc(sizeof(int) * seq_length);
        if (total_seq_matches == NULL) {
            fprintf(stderr, "\n-- Error allocating total sequence matches array of size: %lu\n", seq_length);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    MPI_Reduce(seq_matches,       // Buffer to send (my local sequence matches)
               total_seq_matches, // Buffer to receive (only on rank 0)
               seq_length,        // Number of elements to reduce (the whole sequence)
               MPI_INT,           // Data type
               MPI_SUM,           // Reduction operation
               0,                 // Rank of the process that receives
               MPI_COMM_WORLD);

    if (rank == 0) {
#pragma omp parallel for reduction(+ : checksum_matches)
        for (lind = 0; lind < seq_length; lind++) {
            checksum_matches += total_seq_matches[lind];
        }
    }

    /* 7.1. Reduce results */
    unsigned long total_pat_matches = 0;
    unsigned long total_checksum_found = 0;
    // unsigned long total_checksum_matches = 0;

    MPI_Reduce(&pat_matches,       // Buffer to send (my local pattern matches)
               &total_pat_matches, // Buffer to receive (only on rank 0)
               1,                  // Number of elements to reduce (one only)
               MPI_INT,            // Data type
               MPI_SUM,            // Reduction operation
               0,                  // Rank of the process that receives
               MPI_COMM_WORLD);
    MPI_Reduce(&checksum_found,       // Buffer to send (my local checksum found)
               &total_checksum_found, // Buffer to receive (only on rank 0)
               1,                     // Number of elements to reduce (one only)
               MPI_UNSIGNED_LONG,     // Data type
               MPI_SUM,               // Reduction operation
               0,                     // Rank of the process that receives
               MPI_COMM_WORLD);
    /*MPI_Reduce(&checksum_matches,       // Buffer to send (my local checksum matches)
               &total_checksum_matches, // Buffer to receive (only on rank 0)
               1,                       // Number of elements to reduce (one only)
               MPI_UNSIGNED_LONG,       // Data type
               MPI_SUM,                 // Reduction operation
               0,                       // Rank of the process that receives
               MPI_COMM_WORLD);*/
    if (rank == 0) {
        total_checksum_found %= CHECKSUM_MAX;
        checksum_matches %= CHECKSUM_MAX;
    }

#ifdef DEBUG
    /* DEBUG: Write results */
    printf("-----------------\n");
    printf("Found start:");
    for (debug_pat = 0; debug_pat < pat_number; debug_pat++) {
        printf(" %lu", pat_found[debug_pat]);
    }
    printf("\n");
    printf("-----------------\n");
    printf("Matches:");
    for (lind = 0; lind < seq_length; lind++)
        printf(" %d", seq_matches[lind]);
    printf("\n");
    printf("-----------------\n");
#endif // DEBUG

    /* Free local resources */
    // free(sequence);
    // free(seq_matches);

    /*
     *
     * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
     *
     */

    /* 8. Stop global time */
    MPI_Barrier(MPI_COMM_WORLD);
    ttotal = cp_Wtime() - ttotal;

    /* 9. Output for leaderboard */
    if (rank == 0) {
        printf("\n");
        /* 9.1. Total computation time */
        printf("Time: %lf\n", ttotal);

        /* 9.2. Results: Statistics */
        printf("Result: %lu, %lu, %lu\n\n",
               total_pat_matches,
               total_checksum_found,
               checksum_matches);
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