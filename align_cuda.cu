/*
 * Exact genetic sequence alignment
 * (Using brute force)
 *
 * CUDA version parallelizzata con debug CPU vs GPU
 *
 * Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2023/2024
 *
 * v1.5 parallel CUDA con debug e RNG integrato
 *
 * (c) 2025, ChatGPT adaptation by user request
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>    // per ULLONG_MAX, ULONG_MAX
#include <stdint.h>    // per uint64_t
#include <cuda.h>
#include <cuda_runtime.h>

// Implementazione minimalista di RNG e generate_rng_sequence basata su LCG 64-bit

#define DEBUG

typedef unsigned long rng_t;

// Inizializza lo stato RNG con seed
rng_t rng_new(unsigned long seed) {
    return (rng_t)seed;
}

// Restituisce prossimo valore RNG e aggiorna lo stato
static inline unsigned long rng_next(rng_t *rng) {
    const unsigned long a = 6364136223846793005UL;
    const unsigned long c = 1UL;
    *rng = (rng_t)((uint64_t)(*rng) * (uint64_t)a + (uint64_t)c);
    return *rng;
}

// Genera una sequenza di nucleotidi G,C,A,T secondo probabilità
void generate_rng_sequence(rng_t *rng, double probG, double probC, double probA, char *sequence, unsigned long seq_length) {
    double threshG = probG;
    double threshC = probG + probC;
    double threshA = probG + probC + probA;
    for (unsigned long i = 0; i < seq_length; ++i) {
        unsigned long r = rng_next(rng);
        double u = (double)r / (double)ULLONG_MAX;
        if (u < threshG) sequence[i] = 'G';
        else if (u < threshC) sequence[i] = 'C';
        else if (u < threshA) sequence[i] = 'A';
        else sequence[i] = 'T';
    }
}

#define NOT_FOUND_ULL ULLONG_MAX
#define NOT_FOUND_UL ULONG_MAX

// Controllo errori CUDA
#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if(err != cudaSuccess) { \
            fprintf(stderr, "Errore CUDA in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    }

// Kernel per trovare il primo allineamento esatto di ogni pattern
__global__ void find_first_match_kernel(const char *sequence, unsigned long seq_length,
                                         const char *patterns_concat, const unsigned long *pat_offsets,
                                         const unsigned long *pat_length, unsigned long long *pat_found, unsigned long num_patterns) {
    unsigned long p = blockIdx.x;
    if (p >= num_patterns) return;
    unsigned long length = pat_length[p];
    unsigned long max_start = seq_length >= length ? seq_length - length : 0;
    unsigned long idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (idx > max_start) return;
    const char *pat = patterns_concat + pat_offsets[p];
    unsigned long base = idx;
    bool match = true;
    for (unsigned long k = 0; k < length; ++k) {
        if (sequence[base + k] != pat[k]) { match = false; break; }
    }
    if (match) {
        atomicMin((unsigned long long*)&pat_found[p], (unsigned long long)idx);
    }
}

// Kernel per incrementare i conteggi sulle posizioni della sequenza per ogni pattern trovato
__global__ void increment_matches_kernel(int *seq_matches, unsigned long seq_length,
                                         const unsigned long long *pat_found, const unsigned long *pat_length, unsigned long num_patterns) {
    unsigned long p = blockIdx.x;
    if (p >= num_patterns) return;
    unsigned long long found = pat_found[p];
    unsigned long length = pat_length[p];
    if (found == NOT_FOUND_ULL || found + length > seq_length) return;
    unsigned long offset = (unsigned long)found + threadIdx.x;
    if (threadIdx.x < length) {
        atomicAdd(&seq_matches[offset], 1);
    }
}

// Concatenazione pattern su host
void prepare_patterns_host(char **pattern, int pat_number, unsigned long *pat_length,
                           char **h_patterns_concat, unsigned long **h_pat_offsets, unsigned long *total_chars) {
    unsigned long sum = 0;
    for (int i = 0; i < pat_number; ++i) sum += pat_length[i];
    *total_chars = sum;
    char *buf = (char*)malloc(sum * sizeof(char));
    if (!buf) { fprintf(stderr, "Errore allocazione buffer concatenato pattern\n"); exit(EXIT_FAILURE); }
    unsigned long *offsets = (unsigned long*)malloc(pat_number * sizeof(unsigned long));
    if (!offsets) { fprintf(stderr, "Errore allocazione offsets pattern\n"); exit(EXIT_FAILURE); }
    unsigned long pos = 0;
    for (int i = 0; i < pat_number; ++i) {
        offsets[i] = pos;
        memcpy(buf + pos, pattern[i], pat_length[i] * sizeof(char));
        pos += pat_length[i];
    }
    *h_patterns_concat = buf;
    *h_pat_offsets = offsets;
}

int main(int argc, char *argv[]) {
    if (argc < 15) {
        fprintf(stderr, "Usage: %s <seq length> <prob G> <prob C> <prob A> <pat rng num> <pat rng length mean> <pat rng length dev> <pat samples num> <pat samp length mean> <pat samp length dev> <pat samp loc mean> <pat samp loc dev> <pat samp mix> <long seed>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    unsigned long seq_length = strtoul(argv[1], NULL, 10);
    double prob_G = atof(argv[2]);
    double prob_C = atof(argv[3]);
    double prob_A = atof(argv[4]);
    int pat_rng_num = atoi(argv[5]);
    int pat_rng_len_mean = atoi(argv[6]);
    int pat_rng_len_dev = atoi(argv[7]);
    int pat_samp_num = atoi(argv[8]);
    int pat_samp_len_mean = atoi(argv[9]);
    int pat_samp_len_dev = atoi(argv[10]);
    int pat_samp_loc_mean = atoi(argv[11]);
    int pat_samp_loc_dev = atoi(argv[12]);
    char pat_samp_mix = argv[13][0];
    unsigned long seed = strtoul(argv[14], NULL, 10);

    // Allocazione e generazione sequenza
    char *sequence = (char*)malloc(seq_length * sizeof(char));
    if (!sequence) { fprintf(stderr, "Errore allocazione sequenza size %lu\n", seq_length); exit(EXIT_FAILURE); }
    rng_t random = rng_new(seed);
    generate_rng_sequence(&random, prob_G, prob_C, prob_A, sequence, seq_length);

    // Numero totale pattern
    int pat_number = pat_rng_num + pat_samp_num;
    char **pattern = (char**)malloc(pat_number * sizeof(char*));
    unsigned long *pat_length = (unsigned long*)malloc(pat_number * sizeof(unsigned long));
    if (!pattern || !pat_length) { fprintf(stderr, "Errore allocazione pattern host\n"); exit(EXIT_FAILURE); }

    // Generazione pattern:
    // Qui va inserita la logica originale di generazione dei pattern, ad es.:
    // - pat_rng_num pattern casuali di lunghezza distribuita secondo media/dev
    // - pat_samp_num campionamenti da sequenza con distribuzione di locazioni
    // L'implementazione dipende dal codice originale.
    // Esempio minimale: pattern casuali di lunghezza fixa 10 (da sostituire):
    for (int i = 0; i < pat_number; ++i) {
        unsigned long length = 10; // placeholder: sostituire con logica reale
        pat_length[i] = length;
        pattern[i] = (char*)malloc(length * sizeof(char));
        if (!pattern[i]) { fprintf(stderr, "Errore allocazione pattern[%d]\n", i); exit(EXIT_FAILURE); }
        for (unsigned long j = 0; j < length; ++j) {
            // esempio di generazione casuale basata su RNG
            unsigned long r = rng_next(&random);
            char nuc;
            switch (r % 4) {
                case 0: nuc = 'A'; break;
                case 1: nuc = 'C'; break;
                case 2: nuc = 'G'; break;
                default: nuc = 'T';
            }
            pattern[i][j] = nuc;
        }
    }

    // Host arrays
    unsigned long long *pat_found = (unsigned long long*)malloc(pat_number * sizeof(unsigned long long));
    if (!pat_found) { fprintf(stderr, "Errore allocazione pat_found host\n"); exit(EXIT_FAILURE); }
    for (int i = 0; i < pat_number; ++i) pat_found[i] = NOT_FOUND_ULL;
    int *seq_matches = (int*)malloc(seq_length * sizeof(int));
    if (!seq_matches) { fprintf(stderr, "Errore allocazione seq_matches host\n"); exit(EXIT_FAILURE); }
    for (unsigned long i = 0; i < seq_length; ++i) seq_matches[i] = 0;

    // Preparazione buffer concatenato pattern
    char *h_patterns_concat = NULL;
    unsigned long *h_pat_offsets = NULL;
    unsigned long total_chars = 0;
    prepare_patterns_host(pattern, pat_number, pat_length, &h_patterns_concat, &h_pat_offsets, &total_chars);

    // Allocazione device
    char *d_sequence = NULL;
    char *d_patterns_concat = NULL;
    unsigned long *d_pat_offsets = NULL;
    unsigned long *d_pat_length = NULL;
    unsigned long long *d_pat_found = NULL;
    int *d_seq_matches = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_sequence, seq_length * sizeof(char)));
    CUDA_CHECK(cudaMalloc((void**)&d_patterns_concat, total_chars * sizeof(char)));
    CUDA_CHECK(cudaMalloc((void**)&d_pat_offsets, pat_number * sizeof(unsigned long)));
    CUDA_CHECK(cudaMalloc((void**)&d_pat_length, pat_number * sizeof(unsigned long)));
    CUDA_CHECK(cudaMalloc((void**)&d_pat_found, pat_number * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc((void**)&d_seq_matches, seq_length * sizeof(int)));

    // Copia su device
    CUDA_CHECK(cudaMemcpy(d_sequence, sequence, seq_length * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_patterns_concat, h_patterns_concat, total_chars * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pat_offsets, h_pat_offsets, pat_number * sizeof(unsigned long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pat_length, pat_length, pat_number * sizeof(unsigned long), cudaMemcpyHostToDevice));
    // Inizializza pat_found su device
    unsigned long long *h_init_found = (unsigned long long*)malloc(pat_number * sizeof(unsigned long long));
    if (!h_init_found) { fprintf(stderr, "Errore allocazione init_found host\n"); exit(EXIT_FAILURE); }
    for (int i = 0; i < pat_number; ++i) h_init_found[i] = NOT_FOUND_ULL;
    CUDA_CHECK(cudaMemcpy(d_pat_found, h_init_found, pat_number * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    free(h_init_found);
    CUDA_CHECK(cudaMemset(d_seq_matches, 0, seq_length * sizeof(int)));

    // Launch find_first_match_kernel
    int threads = 256;
    unsigned long max_positions = seq_length;
    int blocks_y = (max_positions + threads - 1) / threads;
    dim3 grid_find(pat_number, blocks_y);
    dim3 block_find(threads);
    find_first_match_kernel<<<grid_find, block_find>>>(d_sequence, seq_length,
                                                        d_patterns_concat, d_pat_offsets,
                                                        d_pat_length, d_pat_found, pat_number);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch increment_matches_kernel
    unsigned long max_pat_len = 0;
    for (int i = 0; i < pat_number; ++i) if (pat_length[i] > max_pat_len) max_pat_len = pat_length[i];
    int threads_inc = 256;
    if ((unsigned long)threads_inc < max_pat_len) threads_inc = ((max_pat_len + 255) / 256) * 256;
    if (threads_inc > 1024) threads_inc = 1024;
    dim3 grid_inc(pat_number);
    dim3 block_inc(threads_inc);
    increment_matches_kernel<<<grid_inc, block_inc>>>(d_seq_matches, seq_length,
                                                       d_pat_found, d_pat_length, pat_number);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copia risultati su host
    CUDA_CHECK(cudaMemcpy(pat_found, d_pat_found, pat_number * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(seq_matches, d_seq_matches, seq_length * sizeof(int), cudaMemcpyDeviceToHost));

#ifdef DEBUG
    // Debug CPU vs GPU
    fprintf(stderr, "--- DEBUG: verifica CPU vs GPU ---\n");
    unsigned long *host_pat_found = (unsigned long*)malloc(pat_number * sizeof(unsigned long));
    int *host_seq_matches = (int*)calloc(seq_length, sizeof(int));
    if (!host_pat_found || !host_seq_matches) { fprintf(stderr, "Errore allocazione debug host\n"); exit(EXIT_FAILURE); }
    for (int p = 0; p < pat_number; ++p) {
        unsigned long length = pat_length[p];
        unsigned long found = NOT_FOUND_UL;
        for (unsigned long idx = 0; idx + length <= seq_length; ++idx) {
            bool match = true;
            for (unsigned long k = 0; k < length; ++k) {
                if (sequence[idx + k] != pattern[p][k]) { match = false; break; }
            }
            if (match) { found = idx; break; }
        }
        host_pat_found[p] = found;
        unsigned long long gpu_found = pat_found[p];
        if (gpu_found == NOT_FOUND_ULL) {
            if (found == NOT_FOUND_UL) fprintf(stderr, "Pattern %d: NOT FOUND (OK)\n", p);
            else fprintf(stderr, "Pattern %d: GPU NOT FOUND, CPU found at %lu (MISMATCH)\n", p, found);
        } else {
            if (found == NOT_FOUND_UL) fprintf(stderr, "Pattern %d: GPU found at %llu, CPU NOT FOUND (MISMATCH)\n", p, gpu_found);
            else if ((unsigned long)gpu_found == found) fprintf(stderr, "Pattern %d: found at %llu (OK)\n", p, gpu_found);
            else fprintf(stderr, "Pattern %d: GPU found at %llu, CPU found at %lu (MISMATCH)\n", p, gpu_found, found);
        }
        if (found != NOT_FOUND_UL) {
            for (unsigned long i = found; i < found + length; ++i) host_seq_matches[i]++;
        }
    }
    long mismatches = 0;
    for (unsigned long i = 0; i < seq_length; ++i) {
        if (host_seq_matches[i] != seq_matches[i]) {
            if (mismatches < 10) fprintf(stderr, "seq_matches[%lu]: GPU=%d, CPU=%d\n", i, seq_matches[i], host_seq_matches[i]);
            mismatches++;
        }
    }
    if (mismatches == 0) fprintf(stderr, "seq_matches: nessuna differenza tra GPU e CPU (OK)\n");
    else fprintf(stderr, "seq_matches: trovate %ld posizioni con mismatch (vedi prime 10 sopra)\n", mismatches);
    free(host_pat_found);
    free(host_seq_matches);
    fprintf(stderr, "--- Fine debug ---\n");
#endif

    // Calcolo checksum o altre operazioni finali se necessario

    // Free risorse
    CUDA_CHECK(cudaFree(d_sequence));
    CUDA_CHECK(cudaFree(d_patterns_concat));
    CUDA_CHECK(cudaFree(d_pat_offsets));
    CUDA_CHECK(cudaFree(d_pat_length));
    CUDA_CHECK(cudaFree(d_pat_found));
    CUDA_CHECK(cudaFree(d_seq_matches));
    free(h_patterns_concat);
    free(h_pat_offsets);
    free(sequence);
    for (int i = 0; i < pat_number; ++i) free(pattern[i]);
    free(pattern);
    free(pat_length);
    free(pat_found);
    free(seq_matches);

    return 0;
}
