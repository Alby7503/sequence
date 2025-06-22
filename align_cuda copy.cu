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
#include <cuda.h> // Libreria principale per la programmazione CUDA

/* Macro per controllare gli errori delle chiamate a funzioni CUDA.
 * Esegue la funzione e, se l'esito non è 'cudaSuccess', stampa un messaggio di errore
 * con la linea del codice e la descrizione dell'errore. */
#define CUDA_CHECK_FUNCTION(call)                                                                 \
    {                                                                                             \
        cudaError_t check = call;                                                                 \
        if (check != cudaSuccess)                                                                 \
            fprintf(stderr, "CUDA Error in line: %d, %s\n", __LINE__, cudaGetErrorString(check)); \
    }
/* Macro per controllare gli errori dopo il lancio di un kernel CUDA.
 * I kernel vengono eseguiti in modo asincrono, quindi 'cudaGetLastError'
 * è necessario per verificare se il kernel lanciato in precedenza ha generato un errore. */
#define CUDA_CHECK_KERNEL()                                                                              \
    {                                                                                                    \
        cudaError_t check = cudaGetLastError();                                                          \
        if (check != cudaSuccess)                                                                        \
            fprintf(stderr, "CUDA Kernel Error in line: %d, %s\n", __LINE__, cudaGetErrorString(check)); \
    }

/* Valore speciale per indicare che un pattern non è stato trovato nella sequenza. */
#define NOT_FOUND -1

// #define DEBUG // Se decommentata, abilita la stampa di informazioni di debug

/* Valore massimo per il calcolo dei checksum, per evitare overflow e mantenere i valori in un range gestibile. */
#define CHECKSUM_MAX 65535

/*
 * Utils: Funzione per ottenere il tempo di orologio (wall time).
 * Utile per misurare le performance del codice.
 */
double cp_Wtime()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}

/*
 * Utils: Inclusione del generatore di numeri casuali.
 */
#include "rng.c"

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 * DO NOT USE OpenMP IN YOUR CODE
 *
 */

#ifdef __CUDACC__
__host__ __device__
#endif
    inline double
    rng_compute_skip(rng_t seq, uint64_t steps)
{
    uint64_t cur_mult = RNG_MULTIPLIER;
    uint64_t cur_plus = RNG_INCREMENT;

    uint64_t acc_mult = 1u;
    uint64_t acc_plus = 0u;
    while (steps > 0)
    {
        if (steps & 1)
        {
            acc_mult *= cur_mult;
            acc_plus = acc_plus * cur_mult + cur_plus;
        }
        cur_plus = (cur_mult + 1) * cur_plus;
        cur_mult *= cur_mult;
        steps /= 2;
    }

    return (double)ldexpf((acc_mult * seq + acc_plus) * RNG_MULTIPLIER + RNG_INCREMENT, -64);
}

__global__ void generate_rng_sequence_kernel(rng_t *__restrict__ d_random, float prob_G, float prob_C, float prob_A, char *__restrict__ d_seq, uint64_t length)
{
    // Calcola l'ID univoco del thread corrente.
    uint64_t ind = blockIdx.x * blockDim.x + threadIdx.x;

    // Ogni thread genera un carattere della sequenza, se rientra nei limiti della lunghezza.
    if (ind < length)
    {
        double prob = rng_compute_skip(*d_random, ind); // Salta il generatore di numeri casuali per ottenere un carattere unico.
        if (prob < prob_G)
            d_seq[ind] = 'G';
        else if (prob < prob_C)
            d_seq[ind] = 'C';
        else if (prob < prob_A)
            d_seq[ind] = 'A';
        else
            d_seq[ind] = 'T';
    }
}

/*
 * KERNEL CUDA: Cerca la prima corrispondenza per ogni pattern in parallelo.
 * OTTIMIZZAZIONE: Aggiunto __restrict__ per indicare al compilatore che i puntatori non si sovrappongono.
 */
__global__ void find_patterns_kernel(const char *__restrict__ d_sequence, uint64_t seq_length,
                                     char **__restrict__ d_pattern, const uint64_t *__restrict__ d_pat_length,
                                     int pat_number, uint64_t *__restrict__ d_pat_found)
{
    // Calcola l'ID univoco del thread corrente. Questo ID corrisponde all'indice del pattern di cui si occuperà.
    int pat_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // "Guardia": se l'ID del thread è maggiore del numero di pattern, il thread non fa nulla ed esce.
    if (pat_idx >= pat_number)
    {
        return;
    }

    // Ogni thread ottiene i dati specifici per il suo pattern.
    uint64_t my_pat_length = d_pat_length[pat_idx];
    char *my_pattern = d_pattern[pat_idx];

    // Loop principale di ricerca (brute-force).
    for (uint64_t start = 0; start <= seq_length - my_pat_length; start++)
    {
        uint64_t lind;
        // Loop interno: confronta carattere per carattere il pattern con la sottosequenza corrente.
        for (lind = 0; lind < my_pat_length; lind++)
        {
            // Se un carattere non corrisponde, interrompe il confronto per questa posizione di partenza.
            if (d_sequence[start + lind] != my_pattern[lind])
            {
                break;
            }
        }
        // Se il loop interno è terminato perché tutti i caratteri corrispondevano...
        if (lind == my_pat_length)
        {
            // ...il pattern è stato trovato. Salva la posizione di inizio e termina.
            d_pat_found[pat_idx] = start;
            return;
        }
    }
}

/*
 * KERNEL CUDA: Aggiorna l'array seq_matches in modo atomico.
 * OTTIMIZZAZIONE: Questo kernel è stato radicalmente semplificato.
 * Invece di usare un ciclo `atomicCAS` complesso e soggetto a contesa, ora
 * si usa un singolo `atomicAdd`. Questo è possibile perché `d_seq_matches`
 * viene inizializzato a 0 invece che a NOT_FOUND (-1).
 */
__global__ void increment_matches_kernel(const uint64_t *__restrict__ d_pat_found, const uint64_t *__restrict__ d_pat_length,
                                         int pat_number, int *__restrict__ d_seq_matches)
{
    // Calcola l'ID univoco del thread, che corrisponde all'indice del pattern.
    int pat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pat_idx >= pat_number)
    {
        return;
    }

    // Il thread procede solo se il suo pattern è stato effettivamente trovato.
    if (d_pat_found[pat_idx] != (uint64_t)NOT_FOUND)
    {
        uint64_t start_pos = d_pat_found[pat_idx];
        uint64_t length = d_pat_length[pat_idx];

        // Itera su ogni posizione della sequenza coperta da questo pattern.
        for (uint64_t ind = 0; ind < length; ind++)
        {
            // OTTIMIZZAZIONE: Usa atomicAdd, molto più veloce di un ciclo CAS.
            // Incrementa il contatore per la posizione corrente della sequenza.
            atomicAdd(&d_seq_matches[start_pos + ind], 1);
        }
    }
}

/*
 * KERNEL CUDA: Inizializza solo l'array d_pat_found.
 * OTTIMIZZAZIONE: L'inizializzazione di d_seq_matches è stata spostata
 * in una chiamata `cudaMemsetAsync` più efficiente nel `main`.
 */
__global__ void initialize_pat_found_kernel(uint64_t *__restrict__ d_pat_found, int pat_number)
{
    // Calcola l'ID univoco del thread.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Inizializza l'array d_pat_found.
    if (idx < pat_number)
    {
        d_pat_found[idx] = (uint64_t)NOT_FOUND;
    }
}

/*
 * Funzione CPU (ora vuota): L'aggiornamento dei match.
 * Nella versione CUDA, la sua logica è stata spostata nel kernel 'increment_matches_kernel'.
 * Viene lasciata vuota per mantenere la compatibilità con la struttura del template.
 */
void increment_matches(int pat, uint64_t *pat_found, const uint64_t *pat_length, int *seq_matches)
{
    // Questa funzione rimane intenzionalmente vuota.
}

/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */

/*
 * Function: Allocate new patttern
 */
char *pattern_allocate(rng_t *random, uint64_t pat_rng_length_mean, uint64_t pat_rng_length_dev, uint64_t seq_length, uint64_t *new_length)
{

    /* Random length */
    uint64_t length = (uint64_t)rng_next_normal(random, (double)pat_rng_length_mean, (double)pat_rng_length_dev);
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
void generate_rng_sequence(rng_t *random, float prob_G, float prob_C, float prob_A, char *seq, uint64_t length)
{
    uint64_t ind;
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
void copy_sample_sequence(rng_t *random, char *sequence, uint64_t seq_length, uint64_t pat_samp_loc_mean, uint64_t pat_samp_loc_dev, char *pattern, uint64_t length)
{
    /* Choose location */
    uint64_t location = (uint64_t)rng_next_normal(random, (double)pat_samp_loc_mean, (double)pat_samp_loc_dev);
    if (location > seq_length - length)
        location = seq_length - length;
    if (location <= 0)
        location = 0;

    /* Copy sample */
    uint64_t ind;
    for (ind = 0; ind < length; ind++)
        pattern[ind] = sequence[ind + location];
}

/*
 * Function: Regenerate a sample of the sequence
 */
void generate_sample_sequence(rng_t *random, rng_t random_seq, float prob_G, float prob_C, float prob_A, uint64_t seq_length, uint64_t pat_samp_loc_mean, uint64_t pat_samp_loc_dev, char *pattern, uint64_t length)
{
    /* Choose location */
    uint64_t location = (uint64_t)rng_next_normal(random, (double)pat_samp_loc_mean, (double)pat_samp_loc_dev);
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
    /* 0. Disabilita il buffering per stdout e stderr per avere output immediato. */
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    /* 1. Lettura e validazione degli argomenti dalla riga di comando. */
    if (argc < 15)
    {
        fprintf(stderr, "\n-- Error: Not enough arguments when reading configuration from the command line\n\n");
        show_usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    /* 1.2. Lettura dei valori degli argomenti. */
    uint64_t seq_length = atol(argv[1]);
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
    uint64_t pat_rng_length_mean = atol(argv[6]);
    uint64_t pat_rng_length_dev = atol(argv[7]);

    int pat_samp_num = atoi(argv[8]);
    uint64_t pat_samp_length_mean = atol(argv[9]);
    uint64_t pat_samp_length_dev = atol(argv[10]);
    uint64_t pat_samp_loc_mean = atol(argv[11]);
    uint64_t pat_samp_loc_dev = atol(argv[12]);

    char pat_samp_mix = argv[13][0];
    if (pat_samp_mix != 'B' && pat_samp_mix != 'A' && pat_samp_mix != 'M')
    {
        fprintf(stderr, "\n-- Error: Incorrect first character of pat_samp_mix: %c\n\n", pat_samp_mix);
        show_usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    uint64_t seed = atol(argv[14]);

#ifdef DEBUG
    /* DEBUG: Stampa degli argomenti letti, se la modalità DEBUG è attiva. */
    printf("\nArguments: seq_length=%lu\n", seq_length);
    printf("Arguments: Accumulated probabilitiy G=%f, C=%f, A=%f, T=1\n", prob_G, prob_C, prob_A);
    printf("Arguments: Random patterns number=%d, length_mean=%lu, length_dev=%lu\n", pat_rng_num, pat_rng_length_mean, pat_rng_length_dev);
    printf("Arguments: Sample patterns number=%d, length_mean=%lu, length_dev=%lu, loc_mean=%lu, loc_dev=%lu\n", pat_samp_num, pat_samp_length_mean, pat_samp_length_dev, pat_samp_loc_mean, pat_samp_loc_dev);
    printf("Arguments: Type of mix: %c, Random seed: %lu\n", pat_samp_mix, seed);
    printf("\n");
#endif // DEBUG

    /* Seleziona il dispositivo GPU da usare (in questo caso il dispositivo 0). */
    CUDA_CHECK_FUNCTION(cudaSetDevice(0));
    CUDA_CHECK_FUNCTION(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

    /* 2. Inizializzazione delle strutture dati sulla CPU (Host). */
    rng_t random = rng_new(seed);
    // Salta il generatore di numeri casuali per la lunghezza della sequenza, per mantenere
    // lo stato del generatore identico a quello della versione sequenziale.
    rng_skip(&random, seq_length);

    /* 2.2. Alloca e genera i pattern sulla CPU. */
    int pat_number = pat_rng_num + pat_samp_num;
    uint64_t *pat_length = (uint64_t *)malloc(sizeof(uint64_t) * pat_number);
    char **pattern = (char **)malloc(sizeof(char *) * pat_number);
    if (pattern == NULL || pat_length == NULL)
    {
        fprintf(stderr, "\n-- Error allocating the basic patterns structures for size: %d\n", pat_number);
        exit(EXIT_FAILURE);
    }

    /* 2.2.2 Allocazione e inizializzazione di una struttura ausiliaria per il tipo di pattern. */
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

    /* 2.2.3 Determina l'ordine dei pattern (casuali o campioni) in base al parametro 'pat_samp_mix'. */
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

    /* 2.2.4 Genera i pattern sulla CPU in base al tipo determinato in precedenza. */
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
// Il template forza la rigenerazione dei campioni, quindi questa opzione è sempre attiva.
#define REGENERATE_SAMPLE_PATTERNS
#ifdef REGENERATE_SAMPLE_PATTERNS
            rng_t random_seq_orig = rng_new(seed);
            generate_sample_sequence(&random, random_seq_orig, prob_G, prob_C, prob_A, seq_length, pat_samp_loc_mean, pat_samp_loc_dev, pattern[ind], pat_length[ind]);
#else
            // Questa parte non viene usata ma è mantenuta per completezza.
            // copy_sample_sequence( &random, sequence, seq_length, pat_samp_loc_mean, pat_samp_loc_dev, pattern[ind], pat_length[ind] );
#endif
        }
        else
        {
            fprintf(stderr, "\n-- Error internal: Paranoic check! A pattern without type at position %d\n", ind);
            exit(EXIT_FAILURE);
        }
    }
    free(pat_type); // Libera la memoria della struttura ausiliaria.

#ifdef DEBUG
    printf("\n--- CUDA DEBUG ---\n");
    if (pat_number > 0)
    {
        printf("CUDA_DEBUG: Pattern 0[0] = %c\n", pattern[0][0]);
    }
    if (pat_number > 1)
    {
        printf("CUDA_DEBUG: Pattern 1[0] = %c\n", pattern[1][0]);
    }
    printf("--- END PATTERN DEBUG ---\n");
#endif // DEBUG

    /* Azzera gli argomenti per non usarli più (buona pratica richiesta dal problema). */
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

    /* 2.3. Alloca memoria sulla CPU per gli array che conterranno i risultati. */
    uint64_t *pat_found;
    pat_found = (uint64_t *)malloc(sizeof(uint64_t) * pat_number);
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

    /* 3. Sincronizza la GPU e fa partire il timer per misurare il tempo di calcolo. */
    CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());
    double ttotal = cp_Wtime();

    /*
     *
     * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
     * DO NOT USE OpenMP IN YOUR CODE
     *
     */

    // OTTIMIZZAZIONE: Non è necessario allocare `sequence_h` o copiare la sequenza dalla GPU.
    // Viene generata sulla GPU e usata direttamente lì.

    /* Lancia il kernel CUDA per generare la sequenza sulla GPU. */
    rng_t random_seq = rng_new(seed);
    rng_t *d_random;
    char *d_sequence;
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_random, sizeof(rng_t)));
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_random, &random_seq, sizeof(rng_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_sequence, sizeof(char) * seq_length)); // Alloca memoria sulla GPU per la sequenza.

    // Configurazione del lancio per il kernel di generazione sequenza
    int threads_per_block = 256;
    int grid_size_seq_gen = (seq_length + threads_per_block - 1) / threads_per_block;

    generate_rng_sequence_kernel<<<grid_size_seq_gen, threads_per_block>>>(d_random, prob_G, prob_C, prob_A, d_sequence, seq_length);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK_FUNCTION(cudaFree(d_random)); // Libera la memoria del RNG sulla GPU.

#ifdef DEBUG
    // La stampa della sequenza richiederebbe una copia su host, che è stata rimossa per ottimizzazione.
    // Per il debug, si dovrebbe decommentare il codice di copia rimosso.
    printf("-----------------\n");
    printf("Patterns: %d\n", pat_number);
    int debug_pat;
    for (debug_pat = 0; debug_pat < pat_number; debug_pat++)
    {
        printf("Pat[%d]: ", debug_pat);
        for (uint64_t lind = 0; lind < pat_length[debug_pat]; lind++)
            printf("%c", pattern[debug_pat][lind]);
        printf("\n");
    }
    printf("-----------------\n\n");
#endif // DEBUG

    /* Dichiarazione dei puntatori per la memoria della GPU (Device). */
    uint64_t *d_pat_length;
    char **d_pattern;
    uint64_t *d_pat_found;
    int *d_seq_matches;

    /* Allocazione della memoria sulla GPU per tutti gli array necessari. */
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_pat_length, sizeof(uint64_t) * pat_number));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_pattern, sizeof(char *) * pat_number));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_pat_found, sizeof(uint64_t) * pat_number));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_seq_matches, sizeof(int) * seq_length));

    /* Trasferimento dei dati di input dalla CPU (Host) alla GPU (Device). */
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_pat_length, pat_length, sizeof(uint64_t) * pat_number, cudaMemcpyHostToDevice));

    /* Trasferimento dell'array di pattern (jagged array). */
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

    /* Configurazione per il lancio dei kernel CUDA. */
    int grid_size_pat = (pat_number + threads_per_block - 1) / threads_per_block;

    /* ESECUZIONE DEI KERNEL */

    // 1. Inizializzazione degli array dei risultati.
    // OTTIMIZZAZIONE: `cudaMemsetAsync` è più veloce di un kernel per azzerare la memoria.
    CUDA_CHECK_FUNCTION(cudaMemsetAsync(d_seq_matches, 0, sizeof(int) * seq_length, 0));
    // Usiamo un kernel solo per d_pat_found che necessita di un valore non-zero.
    initialize_pat_found_kernel<<<grid_size_pat, threads_per_block>>>(d_pat_found, pat_number);
    CUDA_CHECK_KERNEL();

    // OTTIMIZZAZIONE: Rimosse le chiamate `cudaDeviceSynchronize()` intermedie.
    // I kernel nello stesso stream vengono eseguiti in ordine, rendendo la sincronizzazione esplicita superflua e dannosa per le performance.

    // 2. Lancia il kernel per trovare i pattern.
    find_patterns_kernel<<<grid_size_pat, threads_per_block>>>(d_sequence, seq_length, d_pattern, d_pat_length, pat_number, d_pat_found);
    CUDA_CHECK_KERNEL();

    // 3. Lancia il kernel per aggiornare i contatori dei match.
    increment_matches_kernel<<<grid_size_pat, threads_per_block>>>(d_pat_found, d_pat_length, pat_number, d_seq_matches);
    CUDA_CHECK_KERNEL();

    /* Trasferimento dei risultati dalla GPU (Device) alla CPU (Host). */
    CUDA_CHECK_FUNCTION(cudaMemcpy(pat_found, d_pat_found, sizeof(uint64_t) * pat_number, cudaMemcpyDeviceToHost));
    CUDA_CHECK_FUNCTION(cudaMemcpy(seq_matches, d_seq_matches, sizeof(int) * seq_length, cudaMemcpyDeviceToHost));

    /* Calcoli finali sulla CPU. */
    int pat_matches = 0;
    uint64_t checksum_matches = 0;
    uint64_t checksum_found = 0;

    // Calcola il numero totale di pattern trovati e il checksum delle posizioni.
    for (ind = 0; ind < pat_number; ind++)
    {
        if (pat_found[ind] != (uint64_t)NOT_FOUND)
        {
            pat_matches++;
            checksum_found = (checksum_found + pat_found[ind]) % CHECKSUM_MAX;
        }
    }

    // Calcola il checksum dei contatori di copertura.
    for (uint64_t lind = 0; lind < seq_length; lind++)
    {
        checksum_matches = (checksum_matches + seq_matches[lind]) % CHECKSUM_MAX;
    }

#ifdef DEBUG
    /* DEBUG: Stampa i risultati intermedi (posizioni trovate e array dei match). */
    printf("-----------------\n");
    printf("Found start:");
    for (int debug_pat = 0; debug_pat < pat_number; debug_pat++)
    {
        printf(" %lu", pat_found[debug_pat]);
    }
    printf("\n");
    printf("-----------------\n");
    printf("Matches:");
    for (uint64_t lind = 0; lind < seq_length; lind++)
        printf(" %d", seq_matches[lind]);
    printf("\n");
    printf("-----------------\n");
#endif // DEBUG
    /* Liberazione della memoria allocata sulla GPU. */
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

    /* 8. Sincronizza la GPU per assicurarsi che tutti i calcoli siano finiti e ferma il timer. */
    CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());
    ttotal = cp_Wtime() - ttotal;

    /* 9. Stampa i risultati per il leaderboard. */
    printf("\n");
    /* 9.1. Tempo totale di calcolo. */
    printf("Time: %lf\n", ttotal);

    /* 9.2. Risultati: Statistiche e checksum. */
    printf("Result: %d, %lu, %lu\n\n",
           pat_matches,
           checksum_found,
           checksum_matches);

    /* 10. Liberazione finale delle risorse della CPU. */
    int i;
    for (i = 0; i < pat_number; i++)
        free(pattern[i]);
    free(pattern);
    free(pat_length);
    free(pat_found);
    free(seq_matches);

    /* 11. Fine del programma. */
    return 0;
}