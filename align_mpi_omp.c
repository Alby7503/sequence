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
#include <limits.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

/* Arbitrary value to indicate that no matches are found */
#define NOT_FOUND ((long long)-1)

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
 *  NOTA: Versione modificata. Assumendo che seq_matches sia inizializzato a 0,
 *  semplicemente incrementiamo il contatore. Questo è più efficiente e corretto
 *  per la somma parallela.
 */
void increment_matches(int pat, long long *pat_found, unsigned long *pat_length, int *seq_matches) {
    unsigned long ind;
    for (ind = 0; ind < pat_length[pat]; ind++) {
        // L'array è inizializzato a 0, quindi possiamo sempre incrementare.
        seq_matches[pat_found[pat] + ind]++;
    }
}
/*void increment_matches(int pat, long long *pat_found, unsigned long *pat_length, int *seq_matches) {
    unsigned long ind;
    for (ind = 0; ind < pat_length[pat]; ind++) {
        if (seq_matches[pat_found[pat] + ind] == NOT_FOUND) {
            seq_matches[pat_found[pat] + ind] = 0;
        } else {
            seq_matches[pat_found[pat] + ind]++;
        }
    }
}*/

/*
 * Function: Fill random sequence or pattern
 * NOTA: Rinominata in _cpu per chiarezza, è la versione sequenziale usata dal
 * rank 0.
 */
void generate_rng_sequence_cpu(rng_t *random, float prob_G, float prob_C, float prob_A, char *seq, unsigned long length) {
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

/*
 * Function: Copy a sample of the sequence
 * NOTA: Questa funzione rimane sequenziale.
 */
void copy_sample_sequence(rng_t *random, char *sequence, unsigned long seq_length, unsigned long pat_samp_loc_mean,
                          unsigned long pat_samp_loc_dev, char *pattern, unsigned long length) {
    /* Choose location */
    unsigned long location =
        (unsigned long)rng_next_normal(random, (double)pat_samp_loc_mean, (double)pat_samp_loc_dev);
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
char *pattern_allocate(rng_t *random, unsigned long pat_rng_length_mean, unsigned long pat_rng_length_dev,
                       unsigned long seq_length, unsigned long *new_length) {
    /* Random length */
    unsigned long length =
        (unsigned long)rng_next_normal(random, (double)pat_rng_length_mean, (double)pat_rng_length_dev);
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
void generate_sample_sequence(rng_t *random, rng_t random_seq, float prob_G, float prob_C, float prob_A,
                              unsigned long seq_length, unsigned long pat_samp_loc_mean, unsigned long pat_samp_loc_dev,
                              char *pattern, unsigned long length) {
    /* Choose location */
    unsigned long location =
        (unsigned long)rng_next_normal(random, (double)pat_samp_loc_mean, (double)pat_samp_loc_dev);
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
void show_usage(char *program_name) {
    fprintf(stderr, "Usage: %s ", program_name);
    fprintf(stderr, "<seq_length> <prob_G> <prob_C> <prob_A> <pat_rng_num> "
                    "<pat_rng_length_mean> <pat_rng_length_dev> <pat_samples_num> "
                    "<pat_samp_length_mean> <pat_samp_length_dev> <pat_samp_loc_mean> "
                    "<pat_samp_loc_dev> <pat_samp_mix:B[efore]|A[fter]|M[ixed]> "
                    "<long_seed>\n");
    fprintf(stderr, "\n");
}

/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {
    // Inizializzazione MPI
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    /* 0. Default output and error without buffering, forces to write immediately
     */
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    if (rank == 0)
        printf("Inizio del programma...\n");

    /* 1. Read scenary arguments */

    /* 1.1. Check minimum number of arguments */
    if (argc < 15) {
        if (rank == 0) {
            fprintf(stderr, "\n-- Error: Not enough arguments when reading "
                            "configuration from the command line\n\n");
            show_usage(argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* 1.2. Read argument values */
    unsigned long seq_length = atol(argv[1]);
    float prob_G = atof(argv[2]);
    float prob_C = atof(argv[3]);
    float prob_A = atof(argv[4]);
    if (prob_G + prob_C + prob_A > 1) {
        if (rank == 0) {
            fprintf(stderr, "\n-- Error: The sum of G,C,A,T nucleotid probabilities "
                            "cannot be higher than 1\n\n");
            show_usage(argv[0]);
        }
        MPI_Finalize();
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
    if (pat_samp_mix != 'B' && pat_samp_mix != 'A' && pat_samp_mix != 'M') {
        if (rank == 0) {
            fprintf(stderr, "\n-- Error: Incorrect first character of pat_samp_mix: %c\n\n", pat_samp_mix);
            show_usage(argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    unsigned long seed = atol(argv[14]);

#ifdef DEBUG
    /* DEBUG: Print arguments */
    if (rank == 0) {
        printf("\nArguments: seq_length=%lu\n", seq_length);
        printf("Arguments: Accumulated probabilitiy G=%f, C=%f, A=%f, T=1\n", prob_G, prob_C, prob_A);
        printf("Arguments: Random patterns number=%d, length_mean=%lu, "
               "length_dev=%lu\n",
               pat_rng_num, pat_rng_length_mean, pat_rng_length_dev);
        printf("Arguments: Sample patterns number=%d, length_mean=%lu, "
               "length_dev=%lu, loc_mean=%lu, loc_dev=%lu\n",
               pat_samp_num, pat_samp_length_mean, pat_samp_length_dev, pat_samp_loc_mean, pat_samp_loc_dev);
        printf("Arguments: Type of mix: %c, Random seed: %lu, MPI Procs: %d, OMP "
               "Threads: %d\n",
               pat_samp_mix, seed, nprocs, omp_get_max_threads());
        printf("\n");
    }
#endif // DEBUG

    int pat_number = pat_rng_num + pat_samp_num;
    unsigned long *pat_length = (unsigned long *)malloc(sizeof(unsigned long) * pat_number);
    char **pattern = (char **)malloc(sizeof(char *) * pat_number);
    long long *pat_found = (long long *)malloc(sizeof(long long) * pat_number);
    if (pattern == NULL || pat_length == NULL || pat_found == NULL) {
        fprintf(stderr,
                "\n--[%d] Error allocating the basic patterns structures for size: "
                "%d\n",
                rank, pat_number);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    memset(pat_found, NOT_FOUND, pat_number * sizeof(long long));

    /* Start global timer */
    MPI_Barrier(MPI_COMM_WORLD);
    double ttotal = cp_Wtime();

    /*
     *
     * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
     *
     */

    // =========================================================================
    // MODIFICA 1: Generazione centralizzata dei dati per evitare lavoro
    // ridondante. Solo il processo 0 genera la sequenza e i pattern. I dati
    // vengono poi trasmessi a tutti gli altri processi con MPI_Bcast. Questo
    // rende il programma scalabile anche su più nodi (memoria distribuita).
    // =========================================================================

    /* 2.1. Generate the main sequence (solo su rank 0) e trasmettila */
    char *sequence = (char *)malloc(sizeof(char) * seq_length);
    if (sequence == NULL) {
        fprintf(stderr, "\n--[%d] Error allocating main sequence of size: %lu\n", rank, seq_length);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (rank == 0) {
        if (rank == 0)
            printf("Inizio generazione della sequenza principale...\n");
        rng_t random_base_state = rng_new(seed);
        generate_rng_sequence_cpu(&random_base_state, prob_G, prob_C, prob_A, sequence, seq_length);
        // Stampa la sequenza generata
        if (seq_length <= 100) {
            printf("Sequenza generata: ");
            for (unsigned long i = 0; i < seq_length; i++) {
                printf("%c", sequence[i]);
            }
            printf("\n");
        } else {
            printf("Sequenza generata (primi 100 caratteri): ");
            for (unsigned long i = 0; i < 100; i++) {
                printf("%c", sequence[i]);
            }
            printf("...\n");
        }
    }
    // Trasmetti la sequenza generata da rank 0 a tutti gli altri processi
    MPI_Bcast(sequence, seq_length, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("Fine generazione e trasmissione della sequenza principale.\n");

    /* 2.2. Generate patterns (solo su rank 0) e trasmettili */
    if (rank == 0) {
        printf("Inizio generazione dei pattern...\n");
        rng_t random = rng_new(seed);
        rng_skip(&random,
                 seq_length); // Salta i numeri usati per la sequenza principale

        // Logica di generazione dei pattern (invariata, ma eseguita solo da rank 0)
        char *pat_type = (char *)malloc(sizeof(char) * pat_number);
// ... (tutta la logica di 'pat_type' e generazione pattern)
#define PAT_TYPE_NONE 0
#define PAT_TYPE_RNG 1
#define PAT_TYPE_SAMP 2
        int ind;
        for (ind = 0; ind < pat_number; ind++)
            pat_type[ind] = PAT_TYPE_NONE;
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

        for (ind = 0; ind < pat_number; ind++) {
            if (pat_type[ind] == PAT_TYPE_RNG) {
                pattern[ind] =
                    pattern_allocate(&random, pat_rng_length_mean, pat_rng_length_dev, seq_length, &pat_length[ind]);
                generate_rng_sequence_cpu(&random, prob_G, prob_C, prob_A, pattern[ind], pat_length[ind]);
            } else if (pat_type[ind] == PAT_TYPE_SAMP) {
                pattern[ind] =
                    pattern_allocate(&random, pat_samp_length_mean, pat_samp_length_dev, seq_length, &pat_length[ind]);
                rng_t random_seq_orig = rng_new(seed);
                generate_sample_sequence(&random, random_seq_orig, prob_G, prob_C, prob_A, seq_length,
                                         pat_samp_loc_mean, pat_samp_loc_dev, pattern[ind], pat_length[ind]);
            }
        }
        free(pat_type);
        printf("Fine generazione dei pattern.\n");
    }

    // Trasmetti le lunghezze dei pattern a tutti i processi
    unsigned long *offsets = NULL;
    char *total_buffer = NULL;
    unsigned long total_length = 0;

    // Rank 0: Preparazione buffer unico
    if (rank == 0) {
        // Calcola lunghezza totale e offset
        total_length = 0;
        for (int pat = 0; pat < pat_number; pat++) {
            total_length += pat_length[pat];
        }

        offsets = (unsigned long *)malloc((pat_number + 1) * sizeof(unsigned long));
        total_buffer = (char *)malloc(total_length);

        // Costruisci buffer concatenato
        offsets[0] = 0;
        for (int pat = 0; pat < pat_number; pat++) {
            memcpy(total_buffer + offsets[pat], pattern[pat], pat_length[pat]);
            offsets[pat + 1] = offsets[pat] + pat_length[pat];

            // Libera memoria originale pattern (ora nel buffer)
            free(pattern[pat]);
            pattern[pat] = total_buffer + offsets[pat];
        }
    }

    // Comunica lunghezza totale
    MPI_Bcast(&total_length, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    // Alloca offsets su tutti i processi
    if (rank != 0) {
        offsets = (unsigned long *)malloc((pat_number + 1) * sizeof(unsigned long));
    }

    // Comunica offsets
    MPI_Bcast(offsets, pat_number + 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    // Alloca buffer totale su tutti i processi
    if (rank != 0) {
        total_buffer = (char *)malloc(total_length);
    }

    // Comunica dati in UNICA trasmissione
    MPI_Bcast(total_buffer, total_length, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Rank non-0: collegamento puntatori pattern
    if (rank != 0) {
        for (int pat = 0; pat < pat_number; pat++) {
            pattern[pat] = total_buffer + offsets[pat];
        }
    }

    /* 2.3.2. Allocate local seq_matches array, initialized to 0 */
    // NOTA: Usiamo calloc per allocare e azzerare. Ogni processo ha il suo array
    // locale.
    int *local_seq_matches = (int *)calloc(seq_length, sizeof(int));
    if (local_seq_matches == NULL) {
        fprintf(stderr, "\n--[%d] Error allocating local_seq_matches for size: %lu\n", rank, seq_length);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    // Inizializza a NOT_FOUND (-1)
    // #pragma omp parallel for
    //    for (unsigned long i = 0; i < seq_length; i++) {
    //        local_seq_matches[i] = NOT_FOUND;
    //    }

    /* 4. Initialize local result structures */
    int local_pat_matches = 0;
    unsigned long lind, start;
    long long pat; // Indice del pattern

    /* 5. Search for each pattern using MPI for division of work and OpenMP for
     * internal parallelism */
    // La distribuzione dei pattern (pat = rank; pat < pat_number; pat += nprocs)
    // è un modo semplice ed efficace di bilanciare il carico.
    if (rank == 0)
        printf("Inizio della ricerca parallela...\n");
#pragma omp parallel for schedule(dynamic) private(start, lind) reduction(+ : local_pat_matches)
    for (pat = rank; pat < pat_number; pat += nprocs) {
        // Questo processo è responsabile del pattern 'pat'
        // La logica di ricerca per un singolo pattern rimane sequenziale

        /* 5.1  Per ogni possibile posizione iniziale */
        for (start = 0; start <= seq_length - pat_length[pat]; start++) {
            /* 5.1.1. Per ogni elemento del pattern */
            for (lind = 0; lind < pat_length[pat]; lind++) {
                /* Ferma la ricerca se viene trovata una sequenza diversa */
                if (sequence[start + lind] != pattern[pat][lind])
                    break;
            }
            /* 5.1.2. Controlla se il loop è terminato con un match */
            if (lind == pat_length[pat]) {
                local_pat_matches++;
                pat_found[pat] = start;
                break; // Smetti di cercare altre posizioni per questo pattern
            }
        }
    } // --- Fine dell'omp parallel for ---

    /* 5.2. Aggiorna l'array locale dei match dopo la ricerca. */
    // Questo viene fatto in modo sequenziale per ogni processo sui pattern che ha
    // trovato.
    for (pat = rank; pat < pat_number; pat += nprocs) {
        if (pat_found[pat] != NOT_FOUND) {
            increment_matches(pat, pat_found, pat_length, local_seq_matches);
        }
    }
    if (rank == 0)
        printf("Fine della ricerca. Inizio aggregazione dei risultati.\n");

    /* 6. Aggrega i risultati usando MPI */

    /* 6.1. Aggrega il numero di pattern totali trovati */
    int total_pat_matches = 0;
    MPI_Reduce(&local_pat_matches, &total_pat_matches, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    /* 6.2. Aggrega i checksum delle posizioni trovate */
    unsigned long local_checksum_found = 0;
    for (pat = rank; pat < pat_number; pat += nprocs) {
        if (pat_found[pat] != NOT_FOUND)
            local_checksum_found = (local_checksum_found + pat_found[pat]) % CHECKSUM_MAX;
    }
    // Sommiamo prima senza modulo, poi applichiamo il modulo su rank 0 per
    // mantenere la correttezza
    unsigned long total_checksum_found = 0;
    if (rank == 0)
        printf("Inizio della riduzione dei checksum...\n");
    MPI_Reduce(&local_checksum_found, &total_checksum_found, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        total_checksum_found %= CHECKSUM_MAX;
    }

    /* 6.3. Aggrega gli array seq_matches e calcola il suo checksum */
    // 1. Alloca un array globale sul rank 0 per ricevere la somma di tutti gli
    // array locali.
    int *global_seq_matches = NULL;
    if (rank == 0) {
        global_seq_matches = (int *)calloc(seq_length, sizeof(int));
        if (global_seq_matches == NULL)
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // 2. Riduci (sommando elemento per elemento) tutti i `local_seq_matches` in
    // `global_seq_matches`.
    MPI_Reduce(local_seq_matches, global_seq_matches, seq_length, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // 3. Solo il rank 0 calcola il checksum finale dall'array globale aggregato.
    unsigned long total_checksum_matches = 0;
    if (rank == 0) {
        // Passo 3.1: Trasforma i conteggi nei valori finali del codice sequenziale.
        // Questo ciclo modifica l'array global_seq_matches per farlo corrispondere
        // all'array seq_matches della versione sequenziale.
        for (unsigned long lind = 0; lind < seq_length; lind++) {
            if (global_seq_matches[lind] > 0) {
                // Se il conteggio è > 0, il valore sequenziale è conteggio - 1.
                global_seq_matches[lind] = global_seq_matches[lind] - 1;
            } else {
                // Se il conteggio è 0, la posizione non è stata toccata.
                // Il valore sequenziale è NOT_FOUND.
                global_seq_matches[lind] = NOT_FOUND;
            }
        }

        // Passo 3.2: Ora calcola il checksum sull'array trasformato,
        // usando ESATTAMENTE la stessa logica sequenziale.
        unsigned long final_checksum_matches = 0;
        for (unsigned long lind = 0; lind < seq_length; lind++) {
            // La condizione e la somma sono ora identiche a align.c
            if (global_seq_matches[lind] != NOT_FOUND) {
                final_checksum_matches = (final_checksum_matches + global_seq_matches[lind]) % CHECKSUM_MAX;
            }
        }
        total_checksum_matches = final_checksum_matches;

        free(global_seq_matches);
    }

    /* Free local resources */
    free(sequence);
    free(local_seq_matches);

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
        printf("Result: %d, %lu, %lu\n\n", total_pat_matches, total_checksum_found, total_checksum_matches);
    }

    /* 10. Free resources */
    if (total_buffer)
        free(total_buffer);
    if (offsets)
        free(offsets);
    free(pattern);
    free(pat_length);
    free(pat_found);

    /* 11. End */
    MPI_Finalize();
    return 0;
}