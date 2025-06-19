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

__global__ void generate_rng_sequence_kernel(rng_t *d_random, float prob_G, float prob_C, float prob_A, char *d_seq, unsigned long length)
{
    // Calcola l'ID univoco del thread corrente.
    unsigned long ind = blockIdx.x * blockDim.x + threadIdx.x;

    // Ogni thread genera un carattere della sequenza.
    if (ind < length)
    {
        rng_t local_rng = *d_random; // Copia il generatore di numeri casuali dalla memoria della GPU.
        rng_skip(&local_rng, ind); // Salta il generatore di numeri casuali per ottenere un carattere unico.
        double prob = rng_next(&local_rng);
        //__syncthreads(); // Sincronizza i thread per garantire che il generatore di numeri casuali sia aggiornato.
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
 * '__global__' indica che questa funzione viene eseguita sulla GPU e può essere chiamata dalla CPU.
 * Parametri: puntatori a dati che risiedono nella memoria della GPU.
 */
__global__ void find_patterns_kernel(const char *d_sequence, unsigned long seq_length,
                                     char **d_pattern, const unsigned long *d_pat_length,
                                     int pat_number, unsigned long *d_pat_found)
{
    // Calcola l'ID univoco del thread corrente. Questo ID corrisponde all'indice del pattern di cui si occuperà.
    int pat_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // "Guardia": se l'ID del thread è maggiore del numero di pattern, il thread non fa nulla ed esce.
    // Questo è necessario perché il numero totale di thread lanciati potrebbe essere maggiore del numero di pattern.
    if (pat_idx >= pat_number)
    {
        return;
    }

    // Ogni thread ottiene i dati specifici per il suo pattern: la lunghezza e il puntatore al contenuto del pattern.
    unsigned long my_pat_length = d_pat_length[pat_idx];
    char *my_pattern = d_pattern[pat_idx];

    // Loop principale di ricerca (brute-force): scorre tutte le possibili posizioni di partenza nella sequenza.
    for (unsigned long start = 0; start <= seq_length - my_pat_length; start++)
    {
        unsigned long lind;
        // Loop interno: confronta carattere per carattere il pattern con la sottosequenza corrente.
        for (lind = 0; lind < my_pat_length; lind++)
        {
            // Se un carattere non corrisponde, interrompe il confronto per questa posizione di partenza.
            if (d_sequence[start + lind] != my_pattern[lind])
            {
                break;
            }
        }
        // Se il loop interno è terminato perché tutti i caratteri corrispondevano (lind == my_pat_length)...
        if (lind == my_pat_length)
        {
            // ...il pattern è stato trovato. Salva la posizione di inizio nell'array dei risultati.
            d_pat_found[pat_idx] = start;
            // Questo thread ha finito il suo compito (trovare la *prima* corrispondenza), quindi termina.
            return;
        }
    }
}

/*
 * KERNEL CUDA: Aggiorna l'array seq_matches in modo atomico.
 * Viene eseguito dopo find_patterns_kernel.
 */
__global__ void increment_matches_kernel(const unsigned long *d_pat_found, const unsigned long *d_pat_length,
                                         int pat_number, int *d_seq_matches)
{
    // Calcola l'ID univoco del thread, che corrisponde all'indice del pattern.
    int pat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pat_idx >= pat_number)
    {
        return;
    }

    // Il thread procede solo se il suo pattern è stato effettivamente trovato nel kernel precedente.
    if (d_pat_found[pat_idx] != (unsigned long)NOT_FOUND)
    {
        // Ottiene la posizione e la lunghezza del pattern trovato.
        unsigned long start_pos = d_pat_found[pat_idx];
        unsigned long length = d_pat_length[pat_idx];

        // Itera su ogni posizione della sequenza coperta da questo pattern.
        for (unsigned long ind = 0; ind < length; ind++)
        {
            // 'addr' è il puntatore alla cella di memoria in 'd_seq_matches' da aggiornare.
            int *addr = &d_seq_matches[start_pos + ind];

            // Inizia il "CAS loop" per un aggiornamento atomico sicuro.
            // Legge il valore corrente. Questa lettura non è atomica, ma serve solo come primo tentativo.
            int old_val = *addr;

            // Loop che continua finché l'aggiornamento atomico non va a buon fine.
            while (true)
            {
                // Determina il nuovo valore desiderato secondo la logica sequenziale.
                int new_val;
                if (old_val == NOT_FOUND) // Se è il primo pattern a coprire questa posizione
                {
                    new_val = 1;
                }
                else // Se altri pattern hanno già coperto questa posizione
                {
                    new_val = old_val + 1;
                }

                // Tenta di eseguire un'operazione atomica di Compare-And-Swap (CAS).
                // Sostituisce il valore in 'addr' con 'new_val' SOLO SE il valore attuale è ancora 'old_val'.
                // Restituisce il valore che c'era in memoria PRIMA dell'operazione.
                int current_val_in_mem = atomicCAS(addr, old_val, new_val);

                // Se il valore prima del CAS era quello che ci aspettavamo, l'operazione è riuscita.
                if (current_val_in_mem == old_val)
                {
                    break; // Successo! L'aggiornamento è completo, esci dal loop.
                }

                // Se il CAS è fallito, significa che un altro thread ha modificato il valore nel frattempo.
                // Aggiorna 'old_val' con il valore più recente e riprova il loop.
                old_val = current_val_in_mem;
            }
        }
    }
}

/*
 * KERNEL CUDA: Inizializza gli array dei risultati sulla GPU.
 * È più efficiente farlo sulla GPU che sulla CPU e poi trasferire.
 */
__global__ void initialize_arrays_kernel(unsigned long *d_pat_found, int *d_seq_matches,
                                         int pat_number, unsigned long seq_length)
{
    // Calcola l'ID univoco del thread.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Inizializza l'array d_pat_found.
    if (idx < pat_number)
    {
        d_pat_found[idx] = (unsigned long)NOT_FOUND;
    }

    // Inizializza l'array d_seq_matches usando un "grid-stride loop".
    // Questo pattern garantisce che ogni elemento dell'array venga inizializzato
    // anche se il numero di elementi è maggiore del numero di thread.
    for (unsigned long i = idx; i < seq_length; i += gridDim.x * blockDim.x)
    {
        d_seq_matches[i] = NOT_FOUND;
    }
}

/*
 * Funzione CPU (ora vuota): L'aggiornamento dei match.
 * Nella versione CUDA, la sua logica è stata spostata nel kernel 'increment_matches_kernel'.
 * Viene lasciata vuota per mantenere la compatibilità con la struttura del template.
 */
void increment_matches(int pat, unsigned long *pat_found, const unsigned long *pat_length, int *seq_matches)
{
    unsigned long ind;
    for (ind = 0; ind < pat_length[pat]; ind++)
    {
        if (seq_matches[pat_found[pat] + ind] == NOT_FOUND)
            seq_matches[pat_found[pat] + ind] = 0;
        else
            seq_matches[pat_found[pat] + ind]++;
    }
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
    unsigned long *pat_length = (unsigned long *)malloc(sizeof(unsigned long) * pat_number);
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

    /* 3. Sincronizza la GPU e fa partire il timer per misurare il tempo di calcolo. */
    CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());
    double ttotal = cp_Wtime();

    /*
     *
     * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
     * DO NOT USE OpenMP IN YOUR CODE
     *
     */
    /* 2.1. Ora genera la sequenza principale sulla CPU (Host). */
    char *sequence_h = (char *)malloc(sizeof(char) * seq_length);
    if (sequence_h == NULL)
    {
        fprintf(stderr, "\n-- Error allocating the sequence for size: %lu\n", seq_length);
        exit(EXIT_FAILURE);
    }

    //random = rng_new(seed);
    //generate_rng_sequence(&random, prob_G, prob_C, prob_A, sequence, seq_length);

    /* Lancia il kernel CUDA per generare la sequenza sulla GPU. */
    rng_t random_seq = rng_new(seed);
    rng_t *d_random;
    char *d_sequence;
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_random, sizeof(rng_t)));
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_random, &random_seq, sizeof(rng_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_sequence, sizeof(char) * seq_length)); // Alloca memoria sulla GPU per la sequenza.
    unsigned long threads_per_block = 256;
    unsigned long grid_size = (seq_length + threads_per_block - 1) / threads_per_block;
    generate_rng_sequence_kernel<<<grid_size, threads_per_block>>>(d_random, prob_G, prob_C, prob_A, d_sequence, seq_length);
    CUDA_CHECK_KERNEL(); // Controlla errori dopo il lancio
    CUDA_CHECK_FUNCTION(cudaMemcpy(&random_seq, d_random, sizeof(rng_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_FUNCTION(cudaFree(d_random)); // Libera la memoria allocata per il generatore di numeri casuali sulla GPU.
    CUDA_CHECK_FUNCTION(cudaMemcpy(sequence_h, d_sequence, sizeof(char) * seq_length, cudaMemcpyDeviceToHost)); // Copia la sequenza generata dalla GPU alla CPU.
    //sequence_h[seq_length - 1] = '\0';
    //printf("Generated sequence: %s\n", sequence_h); // Stampa la sequenza generata.

#ifdef DEBUG
    /* DEBUG: Stampa la sequenza e i pattern generati, se in modalità DEBUG. */
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

    /* Dichiarazione dei puntatori per la memoria della GPU (Device). */
    //char *d_sequence;
    unsigned long *d_pat_length;
    char **d_pattern;
    unsigned long *d_pat_found;
    int *d_seq_matches;

    /* Allocazione della memoria sulla GPU per tutti gli array necessari. */
    //CUDA_CHECK_FUNCTION(cudaMalloc(&d_sequence, sizeof(char) * seq_length));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_pat_length, sizeof(unsigned long) * pat_number));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_pattern, sizeof(char *) * pat_number));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_pat_found, sizeof(unsigned long) * pat_number));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_seq_matches, sizeof(int) * seq_length));

    /* Trasferimento dei dati di input dalla CPU (Host) alla GPU (Device). */
    //CUDA_CHECK_FUNCTION(cudaMemcpy(d_sequence, sequence, sizeof(char) * seq_length, cudaMemcpyHostToDevice));
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_pat_length, pat_length, sizeof(unsigned long) * pat_number, cudaMemcpyHostToDevice));

    /* Trasferimento dell'array di pattern (jagged array), che richiede una procedura a due passaggi. */
    // 1. Alloca un array temporaneo sulla CPU per contenere i puntatori della GPU.
    char **d_pattern_in_host = (char **)malloc(sizeof(char *) * pat_number);
    if (d_pattern_in_host == NULL)
    {
        fprintf(stderr, "\n-- Error allocating host-side device pointers array for size: %d\n", pat_number);
        exit(EXIT_FAILURE);
    }
    // 2. Per ogni pattern...
    for (int i = 0; i < pat_number; i++)
    {
        // ...alloca memoria sulla GPU per il singolo pattern...
        CUDA_CHECK_FUNCTION(cudaMalloc(&(d_pattern_in_host[i]), sizeof(char) * pat_length[i]));
        // ...e copia il suo contenuto dalla CPU alla GPU.
        CUDA_CHECK_FUNCTION(cudaMemcpy(d_pattern_in_host[i], pattern[i], sizeof(char) * pat_length[i], cudaMemcpyHostToDevice));
    }
    // 3. Infine, copia l'array di puntatori (che ora puntano a locazioni di memoria GPU) dalla CPU alla GPU.
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_pattern, d_pattern_in_host, sizeof(char *) * pat_number, cudaMemcpyHostToDevice));

    /* Configurazione per il lancio dei kernel CUDA: numero di thread per blocco e numero di blocchi nella griglia. */
    //int threads_per_block = 256;
    int grid_size_pat = (pat_number + threads_per_block - 1) / threads_per_block;
    int grid_size_seq = (seq_length + threads_per_block - 1) / threads_per_block;

    /* ESECUZIONE DEI KERNEL */

    // 1. Lancia il kernel per inizializzare gli array dei risultati sulla GPU.
    initialize_arrays_kernel<<<grid_size_seq, threads_per_block>>>(d_pat_found, d_seq_matches, pat_number, seq_length);
    CUDA_CHECK_KERNEL(); // Controlla errori dopo il lancio
    // REMOVE
    CUDA_CHECK_FUNCTION(cudaDeviceSynchronize()); // Sincronizza la GPU per assicurarsi che l'inizializzazione sia completata.
    printf("initialize_arrays_kernel\n");

    // 2. Lancia il kernel per trovare i pattern.
    find_patterns_kernel<<<grid_size_pat, threads_per_block>>>(d_sequence, seq_length, d_pattern, d_pat_length, pat_number, d_pat_found);
    CUDA_CHECK_KERNEL();
    // REMOVE
    CUDA_CHECK_FUNCTION(cudaDeviceSynchronize()); // Sincronizza la GPU per assicurarsi che l'inizializzazione sia completata.
    printf("find_patterns_kernel\n");

    // 3. Lancia il kernel per aggiornare i contatori dei match.
    increment_matches_kernel<<<grid_size_pat, threads_per_block>>>(d_pat_found, d_pat_length, pat_number, d_seq_matches);
    CUDA_CHECK_KERNEL();
    // REMOVE
    CUDA_CHECK_FUNCTION(cudaDeviceSynchronize()); // Sincronizza la GPU per assicurarsi che l'inizializzazione sia completata.
    printf("increment_matches_kernel\n");

    /* Trasferimento dei risultati dalla GPU (Device) alla CPU (Host) per il calcolo finale e la stampa. */
    CUDA_CHECK_FUNCTION(cudaMemcpy(pat_found, d_pat_found, sizeof(unsigned long) * pat_number, cudaMemcpyDeviceToHost));
    CUDA_CHECK_FUNCTION(cudaMemcpy(seq_matches, d_seq_matches, sizeof(int) * seq_length, cudaMemcpyDeviceToHost));

    /* Calcoli finali sulla CPU basati sui risultati ottenuti dalla GPU. */
    int pat_matches = 0;
    unsigned long checksum_matches = 0;
    unsigned long checksum_found = 0;

    // Calcola il numero totale di pattern trovati e il checksum delle posizioni.
    for (ind = 0; ind < pat_number; ind++)
    {
        if (pat_found[ind] != (unsigned long)NOT_FOUND)
        {
            pat_matches++;
            checksum_found = (checksum_found + pat_found[ind]) % CHECKSUM_MAX;
        }
    }

    // Calcola il checksum dei contatori di copertura.
    for (unsigned long lind = 0; lind < seq_length; lind++)
    {
        if (seq_matches[lind] != NOT_FOUND)
        {
            checksum_matches = (checksum_matches + seq_matches[lind]) % CHECKSUM_MAX;
        }
    }

#ifdef DEBUG
    /* DEBUG: Stampa i risultati intermedi (posizioni trovate e array dei match), se in modalità DEBUG. */
    printf("-----------------\n");
    printf("Found start:");
    for (int debug_pat = 0; debug_pat < pat_number; debug_pat++)
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

    /* Liberazione delle risorse della CPU e della GPU. */
    free(sequence_h);
    // seq_matches verrà liberato alla fine del main.

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