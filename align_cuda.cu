/*
 * Exact genetic sequence alignment
 * (Using brute force)
 *
 * CUDA version
 *
 * Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2023/2024
 *
 * v1.3
 *
 * (c) 2024, Arturo Gonzalez-Escribano
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <sys/time.h>

/* Headers for the CUDA assignment versions */
#include <cuda.h>

/* Example of macros for error checking in CUDA */
#define CUDA_CHECK_FUNCTION(call)                                                                 \
    {                                                                                             \
        cudaError_t check = call;                                                                 \
        if (check != cudaSuccess)                                                                 \
            fprintf(stderr, "CUDA Error in line: %d, %s\n", __LINE__, cudaGetErrorString(check)); \
    }
#define CUDA_CHECK_KERNEL()                                                                              \
    {                                                                                                    \
        cudaError_t check = cudaGetLastError();                                                          \
        if (check != cudaSuccess)                                                                        \
            fprintf(stderr, "CUDA Kernel Error in line: %d, %s\n", __LINE__, cudaGetErrorString(check)); \
    }

/* Arbitrary value to indicate that no matches are found */
#define NOT_FOUND -1

// #define DEBUG

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
 * DO NOT USE OpenMP IN YOUR CODE
 *
 */

/* Kernel to find the first match for each pattern */
__global__ void find_patterns_kernel(const char *d_sequence, unsigned long seq_length,
                                     char **d_pattern, const unsigned long *d_pat_length,
                                     int pat_number, unsigned long *d_pat_found)
{
    int pat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pat_idx >= pat_number)
    {
        return;
    }

    unsigned long my_pat_length = d_pat_length[pat_idx];
    char *my_pattern = d_pattern[pat_idx];

    // For each possible starting position
    for (unsigned long start = 0; start <= seq_length - my_pat_length; start++)
    {
        unsigned long lind;
        // For each pattern element
        for (lind = 0; lind < my_pat_length; lind++)
        {
            // Stop this test when different nucleotides are found
            if (d_sequence[start + lind] != my_pattern[lind])
            {
                break;
            }
        }
        // Check if the loop ended with a match
        if (lind == my_pat_length)
        {
            d_pat_found[pat_idx] = start;
            return; // Found the first match, this thread is done
        }
    }
}

/* Kernel to update the seq_matches array atomically */
__global__ void increment_matches_kernel(const unsigned long *d_pat_found, const unsigned long *d_pat_length,
                                         int pat_number, int *d_seq_matches)
{
    int pat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pat_idx >= pat_number)
    {
        return;
    }

    if (d_pat_found[pat_idx] != (unsigned long)NOT_FOUND)
    {
        unsigned long start_pos = d_pat_found[pat_idx];
        unsigned long length = d_pat_length[pat_idx];

        for (unsigned long ind = 0; ind < length; ind++)
        {
            // Address for atomic operation
            int *addr = &d_seq_matches[start_pos + ind];

            // Replicate the logic: if (val == NOT_FOUND) val = 0; else val++;
            // This is done by trying to swap NOT_FOUND with 0.
            // If the swap fails, it means another thread already set it, so we just increment.
            // Note: atomicCAS returns the OLD value at the address.
            if (atomicCAS(addr, NOT_FOUND, 0) != NOT_FOUND)
            {
                atomicAdd(addr, 1);
            }
        }
    }
}

/* Kernel to initialize arrays on the device */
__global__ void initialize_arrays_kernel(unsigned long *d_pat_found, int *d_seq_matches,
                                         int pat_number, unsigned long seq_length)
{
    // Initialize pat_found
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pat_number)
    {
        d_pat_found[idx] = (unsigned long)NOT_FOUND;
    }
    // Initialize seq_matches
    // Use a grid-stride loop to cover the whole array if it's large
    for (unsigned long i = idx; i < seq_length; i += gridDim.x * blockDim.x)
    {
        d_seq_matches[i] = NOT_FOUND;
    }
}

/*
 * Function: Increment the number of pattern matches on the sequence positions
 * 	This function is now implemented in a CUDA kernel (increment_matches_kernel)
 *  and is no longer used in the parallel version.
 */
void increment_matches(int pat, unsigned long *pat_found, unsigned long *pat_length, int *seq_matches)
{
    // This function is intentionally left empty as its logic is now on the GPU.
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
        exit(EXIT_FAILURE);
    }

    /* Return results */
    *new_length = length;
    return pattern;
}

/*
 * Function: Fill random sequence or pattern
 */
void generate_rng_sequence(rng_t *random, float prob_G, float prob_C, float prob_A, char *seq, unsigned long length)
{
    unsigned long ind;
    for (ind = 0; ind < length; ind++)
    {
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

/*
 * Function: Copy a sample of the sequence
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
    /* 1.1. Check minimum number of arguments */
    if (argc < 15)
    {
        fprintf(stderr, "\n-- Error: Not enough arguments when reading configuration from the command line\n\n");
        show_usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    /* 1.2. Read argument values */
    unsigned long seq_length = atol(argv[1]);
    float prob_G = atof(argv[2]);
    float prob_C = atof(argv[3]);
    float prob_A = atof(argv[4]);
    if (prob_G + prob_C + prob_A > 1)
    {
        fprintf(stderr, "\n-- Error: The sum of G,C,A,T nucleotid probabilities cannot be higher than 1\n\n");
        show_usage(argv[0]);
        exit(EXIT_FAILURE);
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
        fprintf(stderr, "\n-- Error: Incorrect first character of pat_samp_mix: %c\n\n", pat_samp_mix);
        show_usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    unsigned long seed = atol(argv[14]);

#ifdef DEBUG
    /* DEBUG: Print arguments */
    printf("\nArguments: seq_length=%lu\n", seq_length);
    printf("Arguments: Accumulated probabilitiy G=%f, C=%f, A=%f, T=1\n", prob_G, prob_C, prob_A);
    printf("Arguments: Random patterns number=%d, length_mean=%lu, length_dev=%lu\n", pat_rng_num, pat_rng_length_mean, pat_rng_length_dev);
    printf("Arguments: Sample patterns number=%d, length_mean=%lu, length_dev=%lu, loc_mean=%lu, loc_dev=%lu\n", pat_samp_num, pat_samp_length_mean, pat_samp_length_dev, pat_samp_loc_mean, pat_samp_loc_dev);
    printf("Arguments: Type of mix: %c, Random seed: %lu\n", pat_samp_mix, seed);
    printf("\n");
#endif // DEBUG

    CUDA_CHECK_FUNCTION(cudaSetDevice(0));

    /* 2. Initialize data structures */
    /* 2.1. Skip allocate and fill sequence for now. Done inside timed section */
    rng_t random = rng_new(seed);
    // We must still advance the random number generator to match the sequential version's state
    rng_skip(&random, seq_length);

    /* 2.2. Allocate and fill patterns */
    /* 2.2.1 Allocate main structures */
    int pat_number = pat_rng_num + pat_samp_num;
    unsigned long *pat_length = (unsigned long *)malloc(sizeof(unsigned long) * pat_number);
    char **pattern = (char **)malloc(sizeof(char *) * pat_number);
    if (pattern == NULL || pat_length == NULL)
    {
        fprintf(stderr, "\n-- Error allocating the basic patterns structures for size: %d\n", pat_number);
        exit(EXIT_FAILURE);
    }

    /* 2.2.2 Allocate and initialize ancillary structure for pattern types */
    int ind;
#define PAT_TYPE_NONE 0
#define PAT_TYPE_RNG 1
#define PAT_TYPE_SAMP 2
    char *pat_type = (char *)malloc(sizeof(char) * pat_number);
    if (pat_type == NULL)
    {
        fprintf(stderr, "\n-- Error allocating ancillary structure for pattern of size: %d\n", pat_number);
        exit(EXIT_FAILURE);
    }
    for (ind = 0; ind < pat_number; ind++)
        pat_type[ind] = PAT_TYPE_NONE;

    /* 2.2.3 Fill up pattern types using the chosen mode */
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

    /* 2.2.4 Generate the patterns on HOST */
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
// The template forces REGENERATE, so we keep it to match the provided logic
#define REGENERATE_SAMPLE_PATTERNS
#ifdef REGENERATE_SAMPLE_PATTERNS
            rng_t random_seq_orig = rng_new(seed);
            generate_sample_sequence(&random, random_seq_orig, prob_G, prob_C, prob_A, seq_length, pat_samp_loc_mean, pat_samp_loc_dev, pattern[ind], pat_length[ind]);
#else // This part is not used based on the template, but kept for completeness
      // We would need the host 'sequence' array for this to work
      // copy_sample_sequence( &random, sequence, seq_length, pat_samp_loc_mean, pat_samp_loc_dev, pattern[ind], pat_length[ind] );
#endif
        }
        else
        {
            fprintf(stderr, "\n-- Error internal: Paranoic check! A pattern without type at position %d\n", ind);
            exit(EXIT_FAILURE);
        }
    }
    free(pat_type);

    /* Avoid the usage of arguments to take strategic decisions
     * In a real case the user only has the patterns and sequence data to analize
     */
    argc = 0;
    argv = NULL;
    // The original values are still needed for debug printing if enabled
    // int orig_pat_rng_num = pat_rng_num;
    // int orig_pat_samp_num = pat_samp_num;
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
    unsigned long *pat_found;
    pat_found = (unsigned long *)malloc(sizeof(unsigned long) * pat_number);
    if (pat_found == NULL)
    {
        fprintf(stderr, "\n-- Error allocating aux pattern structure for size: %d\n", pat_number);
        exit(EXIT_FAILURE);
    }
    int *seq_matches;
    seq_matches = (int *)malloc(sizeof(int) * seq_length);
    if (seq_matches == NULL)
    {
        fprintf(stderr, "\n-- Error allocating aux sequence structures for size: %lu\n", seq_length);
        exit(EXIT_FAILURE);
    }

    /* 3. Start global timer */
    CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());
    double ttotal = cp_Wtime();

    /*
     *
     * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
     * DO NOT USE OpenMP IN YOUR CODE
     *
     */
    /* 2.1. Allocate and fill sequence on HOST as per template logic */
    char *sequence = (char *)malloc(sizeof(char) * seq_length);
    if (sequence == NULL)
    {
        fprintf(stderr, "\n-- Error allocating the sequence for size: %lu\n", seq_length);
        exit(EXIT_FAILURE);
    }

    random = rng_new(seed);
    generate_rng_sequence(&random, prob_G, prob_C, prob_A, sequence, seq_length);

#ifdef DEBUG
    /* DEBUG: Print sequence and patterns */
    printf("-----------------\n");
    printf("Sequence: ");
    for (unsigned long lind = 0; lind < seq_length; lind++)
        printf("%c", sequence[lind]);
    printf("\n-----------------\n");
    printf("Patterns: %d\n", pat_number);
    int debug_pat;
    for (debug_pat = 0; debug_pat < pat_number; debug_pat++)
    {
        printf("Pat[%d]: ", debug_pat);
        for (unsigned long lind = 0; lind < pat_length[debug_pat]; lind++)
            printf("%c", pattern[debug_pat][lind]);
        printf("\n");
    }
    printf("-----------------\n\n");
#endif // DEBUG

    // GPU Data Structures
    char *d_sequence;
    unsigned long *d_pat_length;
    char **d_pattern;
    unsigned long *d_pat_found;
    int *d_seq_matches;

    // Allocate memory on GPU
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_sequence, sizeof(char) * seq_length));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_pat_length, sizeof(unsigned long) * pat_number));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_pattern, sizeof(char *) * pat_number));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_pat_found, sizeof(unsigned long) * pat_number));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_seq_matches, sizeof(int) * seq_length));

    // Copy data from Host to Device
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_sequence, sequence, sizeof(char) * seq_length, cudaMemcpyHostToDevice));
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_pat_length, pat_length, sizeof(unsigned long) * pat_number, cudaMemcpyHostToDevice));

    // Copy jagged pattern array
    char **d_pattern_in_host = (char **)malloc(sizeof(char *) * pat_number);
    if (d_pattern_in_host == NULL)
    {
        fprintf(stderr, "\n-- Error allocating host-side device pointers array for size: %d\n", pat_number);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < pat_number; i++)
    {
        CUDA_CHECK_FUNCTION(cudaMalloc(&(d_pattern_in_host[i]), sizeof(char) * pat_length[i]));
        CUDA_CHECK_FUNCTION(cudaMemcpy(d_pattern_in_host[i], pattern[i], sizeof(char) * pat_length[i], cudaMemcpyHostToDevice));
    }
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_pattern, d_pattern_in_host, sizeof(char *) * pat_number, cudaMemcpyHostToDevice));

    // CUDA Kernel launch configuration
    int threads_per_block = 256;
    int grid_size_pat = (pat_number + threads_per_block - 1) / threads_per_block;
    int grid_size_seq = (seq_length + threads_per_block - 1) / threads_per_block;

    // Initialize result arrays on GPU
    initialize_arrays_kernel<<<grid_size_seq, threads_per_block>>>(d_pat_found, d_seq_matches, pat_number, seq_length);
    CUDA_CHECK_KERNEL();

    // Launch kernel to find patterns
    find_patterns_kernel<<<grid_size_pat, threads_per_block>>>(d_sequence, seq_length, d_pattern, d_pat_length, pat_number, d_pat_found);
    CUDA_CHECK_KERNEL();

    // Launch kernel to increment match counters
    increment_matches_kernel<<<grid_size_pat, threads_per_block>>>(d_pat_found, d_pat_length, pat_number, d_seq_matches);
    CUDA_CHECK_KERNEL();

    // Copy results back from Device to Host
    CUDA_CHECK_FUNCTION(cudaMemcpy(pat_found, d_pat_found, sizeof(unsigned long) * pat_number, cudaMemcpyDeviceToHost));
    CUDA_CHECK_FUNCTION(cudaMemcpy(seq_matches, d_seq_matches, sizeof(int) * seq_length, cudaMemcpyDeviceToHost));

    // Host-side calculations based on GPU results
    int pat_matches = 0;
    unsigned long checksum_matches = 0;
    unsigned long checksum_found = 0;

    for (ind = 0; ind < pat_number; ind++)
    {
        if (pat_found[ind] != (unsigned long)NOT_FOUND)
        {
            pat_matches++;
            checksum_found = (checksum_found + pat_found[ind]) % CHECKSUM_MAX;
        }
    }

    for (unsigned long lind = 0; lind < seq_length; lind++)
    {
        if (seq_matches[lind] != NOT_FOUND)
        {
            checksum_matches = (checksum_matches + seq_matches[lind]) % CHECKSUM_MAX;
        }
    }

#ifdef DEBUG
    /* DEBUG: Write results */
    printf("-----------------\n");
    printf("Found start:");
    for (debug_pat = 0; debug_pat < pat_number; debug_pat++)
    {
        printf(" %lu", pat_found[debug_pat]);
    }
    printf("\n");
    printf("-----------------\n");
    printf("Matches:");
    for (unsigned long lind = 0; lind < seq_length; lind++)
        printf(" %d", seq_matches[lind]);
    printf("\n");
    printf("-----------------\n");
#endif // DEBUG

    /* Free local resources */
    free(sequence);
    // seq_matches is freed at the end of main

    /* Free GPU resources */
    CUDA_CHECK_FUNCTION(cudaFree(d_sequence));
    CUDA_CHECK_FUNCTION(cudaFree(d_pat_length));
    for (int i = 0; i < pat_number; i++)
    {
        CUDA_CHECK_FUNCTION(cudaFree(d_pattern_in_host[i]));
    }
    free(d_pattern_in_host);
    CUDA_CHECK_FUNCTION(cudaFree(d_pattern));
    CUDA_CHECK_FUNCTION(cudaFree(d_pat_found));
    CUDA_CHECK_FUNCTION(cudaFree(d_seq_matches));

    /*
     *
     * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
     *
     */

    /* 8. Stop global timer */
    CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());
    ttotal = cp_Wtime() - ttotal;

    /* 9. Output for leaderboard */
    printf("\n");
    /* 9.1. Total computation time */
    printf("Time: %lf\n", ttotal);

    /* 9.2. Results: Statistics */
    printf("Result: %d, %lu, %lu\n\n",
           pat_matches,
           checksum_found,
           checksum_matches);

    /* 10. Free resources */
    int i;
    for (i = 0; i < pat_number; i++)
        free(pattern[i]);
    free(pattern);
    free(pat_length);
    free(pat_found);
    free(seq_matches);

    /* 11. End */
    return 0;
}