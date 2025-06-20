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
// #define NOT_FOUND 0
#define NOT_FOUND ((long long)-1)

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
void increment_matches(int pat, long long *pat_found, unsigned long *pat_length, int *seq_matches)
{
    unsigned long ind;
    for (ind = 0; ind < pat_length[pat]; ind++)
    {
        // if (seq_matches[pat_found[pat] + ind] == NOT_FOUND)
        //     seq_matches[pat_found[pat] + ind] = 1;
        // else
        //     seq_matches[pat_found[pat] + ind]++;
        seq_matches[pat_found[pat] + ind] += 1; // Incrementa il conteggio per questa posizione
    }
}

void generate_rng_sequence_cpu(rng_t *random, float prob_G, float prob_C, float prob_A, char *seq, unsigned long length)
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
 * Function: Fill random sequence or pattern (MPI parallelized version)
 * Each process generates a part of the sequence, ensuring the final result
 * is identical to the sequential version.
 */
void generate_rng_sequence_mpi(
    rng_t *random_seed, // Puntatore allo stato iniziale del RNG (basato sul seme)
    float prob_G,
    float prob_C,
    float prob_A,
    char *local_seq, // Buffer locale per la porzione di sequenza
    unsigned long seq_length,
    int rank,
    int nprocs)
{
    // --- 1. Calcola la divisione del lavoro ---
    unsigned long n_per_proc = seq_length / nprocs;
    unsigned long remainder = seq_length % nprocs;

    // Calcola la dimensione del blocco per questo processo
    unsigned long my_block_size = n_per_proc + (rank < remainder ? 1 : 0);

    // Calcola l'offset di partenza (quanti caratteri sono stati generati dai processi precedenti)
    unsigned long my_start_offset = rank * n_per_proc + (rank < remainder ? rank : remainder);

    // Se questo processo non ha lavoro da fare (es. più processi che caratteri), esce.
    if (my_block_size <= 0)
    {
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
    for (ind = 0; ind < my_block_size; ind++)
    {
        double prob = rng_next(&local_random);
        if (prob < prob_G)
            local_seq[ind] = 'G';
        else if (prob < prob_C)
            local_seq[ind] = 'C';
        else if (prob < prob_A)
            local_seq[ind] = 'A';
        else
            local_seq[ind] = 'T';
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
    generate_rng_sequence_cpu(&local_random, prob_G, prob_C, prob_A, pattern, length);
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
    // Inizializzazione MPI
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    /* 0. Default output and error without buffering, forces to write immediately */
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    if (rank == 0)
        printf("Inizio del programma...\n");

    /* 1. Read scenary arguments */

    /* 1.1. Check minimum number of arguments */
    if (argc < 15)
    {
        // if (rank == 0)
        //{
        fprintf(stderr, "\n-- Error: Not enough arguments when reading configuration from the command line\n\n");
        show_usage(argv[0]);
        //}
        // MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    /* 1.2. Read argument values */
    unsigned long seq_length = atol(argv[1]);
    float prob_G = atof(argv[2]);
    float prob_C = atof(argv[3]);
    float prob_A = atof(argv[4]);
    if (prob_G + prob_C + prob_A > 1)
    {
        // if (rank == 0)
        //{
        fprintf(stderr, "\n-- Error: The sum of G,C,A,T nucleotid probabilities cannot be higher than 1\n\n");
        show_usage(argv[0]);
        //}
        // MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
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
        // if (rank == 0)
        //{
        fprintf(stderr, "\n-- Error: Incorrect first character of pat_samp_mix: %c\n\n", pat_samp_mix);
        show_usage(argv[0]);
        //}
        // MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
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

    /* 2. Initialize data structures IN ALL PROCESSES */
    /* Ogni processo genera tutti i dati per sapere quale parte gli spetta */
    rng_t random = rng_new(seed);
    rng_skip(&random, seq_length);

    int pat_number = pat_rng_num + pat_samp_num;
    if (rank == 0)
        printf("Inizio allocazione delle strutture dati...\n");
    unsigned long *pat_length = (unsigned long *)malloc(sizeof(unsigned long) * pat_number);
    char **pattern = (char **)malloc(sizeof(char *) * pat_number);
    if (pattern == NULL || pat_length == NULL)
    {
        // fprintf(stderr, "\n--[%d] Error allocating the basic patterns structures for size: %d\n", rank, pat_number);
        fprintf(stderr, "\n-- Error allocating the basic patterns structures for size: %d\n", pat_number);
        // MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    int ind;
    unsigned long lind;
#define PAT_TYPE_NONE 0
#define PAT_TYPE_RNG 1
#define PAT_TYPE_SAMP 2
    char *pat_type = (char *)malloc(sizeof(char) * pat_number);
    if (pat_type == NULL)
    {
        // fprintf(stderr, "\n--[%d] Error allocating ancillary structure for pattern of size: %d\n", rank, pat_number);
        fprintf(stderr, "\n-- Error allocating ancillary structure for pattern of size: %d\n", pat_number);
        // MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
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

    if (rank == 0)
        printf("Inizio generazione dei pattern...\n");
    for (ind = 0; ind < pat_number; ind++)
    {
        if (pat_type[ind] == PAT_TYPE_RNG)
        {
            pattern[ind] = pattern_allocate(&random, pat_rng_length_mean, pat_rng_length_dev, seq_length, &pat_length[ind]);
            generate_rng_sequence_cpu(&random, prob_G, prob_C, prob_A, pattern[ind], pat_length[ind]);
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
            // fprintf(stderr, "\n--[%d] Error internal: Paranoic check! A pattern without type at position %d\n", rank, ind);
            fprintf(stderr, "\n-- Error internal: Paranoic check! A pattern without type at position %d\n", ind);
            // MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return EXIT_FAILURE;
        }
    }
    free(pat_type);

    /* 2.3. Allocate result data structures */
    if (rank == 0)
        printf("Inizio allocazione e memset delle strutture di risultato...\n");
    long long *pat_found = (long long *)malloc(sizeof(unsigned long) * pat_number);
    // long long *pat_found = (long long *)calloc(pat_number, sizeof(long long));
    if (pat_found == NULL)
    {
        // fprintf(stderr, "\n--[%d] Error allocating aux pattern structure for size: %d\n", rank, pat_number);
        fprintf(stderr, "\n-- Error allocating aux pattern structure for size: %d\n", pat_number);
        // MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    // Inizializza pat_found a NOT_FOUND
    memset(pat_found, NOT_FOUND, pat_number * sizeof(long long));

    /* 3. Start global timer */
    // MPI_Barrier(MPI_COMM_WORLD);
    double ttotal = cp_Wtime();

    /*
     *
     * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
     *
     */
    /* 2.1. Generate the main sequence */
    rng_t random_base_state = rng_new(seed);
    // char *sequence = malloc(seq_length);
    char *sequence = NULL;
    if (rank == 0)
        printf("Inizio generazione della sequenza principale...\n");

    // generate_rng_sequence_cpu(&random_base_state, prob_G, prob_C, prob_A, sequence, seq_length);
    MPI_Comm shmcomm;
    // Aggiungiamo al nuovo comunicatore tutti i processi locali allo stesso nodo
    MPI_Comm_split_type(MPI_COMM_WORLD,
                        MPI_COMM_TYPE_SHARED,
                        0, MPI_INFO_NULL, &shmcomm);

    int shm_rank, shm_size;
    MPI_Comm_rank(shmcomm, &shm_rank);
    MPI_Comm_size(shmcomm, &shm_size);

    MPI_Win win;
    // Creiamo una finestra condivisa per la sequenza principale
    int err = MPI_Win_allocate_shared(seq_length * sizeof(char), sizeof(char), MPI_INFO_NULL, shmcomm, &sequence, &win);
    if (err != MPI_SUCCESS)
    {
        fprintf(stderr, "\n--[%d] Error allocating main sequence of size: %lu\n", rank, seq_length);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Se non siamo il rank 0, prendiamo il puntatore alla memoria condivisa
    if (shm_rank != 0)
    {
        char *shared_sequence;
        MPI_Win_shared_query(win, 0, NULL, NULL, &shared_sequence);
        sequence = shared_sequence;
    }

    // Generiamo la sequenza principale solo sul rank 0
    if (shm_rank == 0)
    {
        generate_rng_sequence_cpu(&random_base_state, prob_G, prob_C, prob_A, sequence, seq_length);
    }

    // Sincronizziamo tutti i processi per assicurarci che la sequenza sia pronta
    MPI_Barrier(shmcomm);

    if (rank == 0)
        printf("Fine generazione della sequenza principale.\n");

    /* 2.3.2. Allocate local seq_matches array */
    int *seq_matches = (int *)malloc(sizeof(int) * seq_length);
    if (seq_matches == NULL)
    {
        fprintf(stderr, "\n--[%d] Error allocating aux sequence structures for size: %lu\n",
                rank, seq_length);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    // Inizializza seq_matches a NOT_FOUND
    if (rank == 0)
        printf("Inizio memset della struttura di risultato seq_matches...\n");
    memset(seq_matches, NOT_FOUND, seq_length * sizeof(int));
    if (rank == 0)
        printf("Fine memset della struttura di risultato seq_matches.\n");

    /* 4. Initialize local result structures */
    int local_pat_matches = 0;

    /* 5. Search for each pattern using MPI for division of work and OpenMP for internal parallelism */
    int pat;
    unsigned long start;
    // Parallelizza il ciclo esterno sui pattern con OpenMP
    // Ogni thread processa un pattern diverso in parallelo
    if (rank == 0)
        printf("Inizio del for con OpenMP per la ricerca dei pattern...\n");
#pragma omp parallel for schedule(dynamic) private(pat, lind, start) reduction(+ : local_pat_matches)
    for (pat = rank; pat < pat_number; pat += nprocs)
    {
        // Questo processo è responsabile del pattern 'pat'
        // La logica di ricerca per un singolo pattern rimane sequenziale

        /* 5.1  Per ogni possibile posizione iniziale */
        for (start = 0; start <= seq_length - pat_length[pat]; start++)
        {
            /* 5.1.1. Per ogni elemento del pattern */
            for (lind = 0; lind < pat_length[pat]; lind++)
            {
                /* Ferma la ricerca se viene trovata una sequenza diversa */
                if (sequence[start + lind] != pattern[pat][lind])
                    break;
            }
            /* 5.1.2. Controlla se il loop è terminato con un match */
            if (lind == pat_length[pat])
            {
                local_pat_matches++;
                pat_found[pat] = start;
                break; // Smetti di cercare altre posizioni per questo pattern
            }
        }
    } // --- Fine dell'omp parallel for ---

    /* 5.2 Dopo la ricerca parallela, aggiorna seq_matches in modo sequenziale per evitare race conditions.
       Questo viene fatto da ogni processo solo per i pattern che ha trovato. */
    if (rank == 0)
        printf("Fine del for con OpenMP per la ricerca dei pattern. Inizio dell'aggiornamento di seq_matches...\n");
    for (pat = rank; pat < pat_number; pat += nprocs)
    {
        if (pat_found[pat] != NOT_FOUND)
        {
            increment_matches(pat, pat_found, pat_length, seq_matches);
        }
    }

    /* 6. Aggrega i risultati usando MPI */

    /* 6.1. Aggrega il numero di pattern totali trovati */
    int total_pat_matches = 0;
    if (rank == 0)
        printf("Inizio dell'aggregazione dei risultati...\n");
    MPI_Reduce(&local_pat_matches, &total_pat_matches, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    /* 6.2. Aggrega i checksum trovati */
    unsigned long local_checksum_found = 0;
    for (ind = rank; ind < pat_number; ind += nprocs)
    {
        if (pat_found[ind] != NOT_FOUND)
            local_checksum_found = (local_checksum_found + pat_found[ind]);
    }
    // Sommiamo prima senza modulo, poi applichiamo il modulo su rank 0 per mantenere la correttezza
    unsigned long total_checksum_found = 0;
    if (rank == 0)
        printf("Inizio della riduzione dei checksum...\n");
    MPI_Reduce(&local_checksum_found, &total_checksum_found, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        total_checksum_found %= CHECKSUM_MAX;
    }

    /* 6.3. Aggrega seq_matches e calcola il suo checksum */
    // 1. Ogni processo calcola il proprio checksum locale dal suo array seq_matches
    unsigned long local_checksum_matches = 0;

    if (rank == 0)
        printf("Inizio della for OpenMP dei checksum per seq_matches...\n");
#pragma omp parallel for reduction(+ : local_checksum_matches)
    for (lind = 0; lind < seq_length; lind++)
    {
        // Usa la logica originale: somma se la posizione è stata coperta da almeno un pattern
        if (seq_matches[lind] != NOT_FOUND)
        {
            // NOTA: Assicurati che il calcolo qui sia una somma semplice.
            // Il modulo % CHECKSUM_MAX verrà applicato solo alla fine sul rank 0.
            local_checksum_matches = (local_checksum_matches + seq_matches[lind]);
        }
    }

    // 2. Tutti i processi partecipano alla riduzione per sommare i checksum locali
    unsigned long total_checksum_matches = 0;
    if (rank == 0)
        printf("Inizio MPI_Reduce per checksum matches\n");
    MPI_Reduce(&local_checksum_matches, // Buffer di invio (il mio checksum locale)
               &total_checksum_matches, // Buffer di ricezione (solo su rank 0)
               1,                       // Numero di elementi da ridurre (uno solo)
               MPI_UNSIGNED_LONG,       // Tipo di dato
               MPI_SUM,                 // Operazione di riduzione
               0,                       // Rango del processo che riceve
               MPI_COMM_WORLD);

    // 3. Solo il rank 0 applica il modulo finale al risultato globale
    if (rank == 0)
    {
        total_checksum_matches %= CHECKSUM_MAX;
    }

    /* Free local resources */
    // free(sequence);
    free(seq_matches);

    MPI_Win_free(&win);
    MPI_Comm_free(&shmcomm);

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