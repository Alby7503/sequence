#!/usr/bin/env bash
set -euo pipefail

#: "${MPI_NUMBER_PROC:?Devi impostare MPI_NUMBER_PROC}"
: "${MPI_FLAGS:=}"

readonly TESTS=(
  "300 0.1 0.3 0.35 100 5 5 300 150 50 150 80 M 609823"
  "1000 0.35 0.2 0.25 0 0 0 20000 10 0 500 0 M 4353435"
  "10000 0.35 0.2 0.25 0 0 0 10000 9000 9000 50 100 M 4353435"
  "429496730 0.35 0.2 0.25 0 0 0 1 1 0 4294967298 0 M 683224"
  "4294967300 0.35 0.2 0.25 0 0 0 1 1 0 4294967298 0 M 683224"
  "10000000 0.25 0.5 0.25 100000 500 50 0 1 1 1 1 M 12345"
)

# Estrae la riga che inizia con "Time:" dall'output
extract_time() {
  grep '^Time:' <<< "$1" | head -n1
}

# Estrae la riga che inizia con "Result:" dall'output
extract_result() {
  grep '^Result:' <<< "$1" | head -n1
}

# Esegue un comando con parametri e restituisce output completo
run_and_capture_output() {
  local -n cmd=$1
  local args=$2
  "${cmd[@]}" $args
}

make align_mpi_omp_2 align_cuda_copy >/dev/null

echo
for idx in "${!TESTS[@]}"; do
  testnum=$((idx + 1))
  params="${TESTS[$idx]}"
  echo "=== Test #$testnum: $params ==="

  # Definizione comandi
  #mpi_cmd=(mpirun -np "$MPI_NUMBER_PROC" $MPI_FLAGS ./align_mpi_omp_2)
  mpi_cmd=(mpirun $MPI_FLAGS ./align_mpi_omp_2)
  cuda_cmd=(./align_cuda_copy)

  # Inizializza accumulatori per i tempi (come stringhe numeriche)
  sum_mpi=0.0
  sum_cuda=0.0

  # Esegue 10 volte MPI
  echo "-- Esecuzioni MPI (10 run) --"
  for run in $(seq 1 10); do
    out_mpi=$(run_and_capture_output mpi_cmd "$params")
    time_line=$(extract_time "$out_mpi")
    result_line=$(extract_result "$out_mpi")

    # Estrae valore numerico dal time_line (es. "Time: 0.123s" o "Time: 0.123")
    num=$(awk '{print $2}' <<< "$time_line" | tr -d '[:alpha:]')
    if [[ -z "$num" || ! $num =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
      echo "  Run MPI #$run: impossibile estrarre tempo da '$time_line'"
    else
      # Somma con awk
      sum_mpi=$(awk "BEGIN {printf \"%.6f\", $sum_mpi + $num}")
    fi

    echo "  MPI run #$run -> $result_line [$time_line]"
  done

  # Esegue 10 volte CUDA
  echo "-- Esecuzioni CUDA (10 run) --"
  for run in $(seq 1 10); do
    out_cuda=$(run_and_capture_output cuda_cmd "$params")
    time_line=$(extract_time "$out_cuda")
    result_line=$(extract_result "$out_cuda")

    num=$(awk '{print $2}' <<< "$time_line" | tr -d '[:alpha:]')
    if [[ -z "$num" || ! $num =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
      echo "  Run CUDA #$run: impossibile estrarre tempo da '$time_line'"
    else
      sum_cuda=$(awk "BEGIN {printf \"%.6f\", $sum_cuda + $num}")
    fi

    echo "  CUDA run #$run -> $result_line [$time_line]"
  done

  # Calcola medie con awk
  avg_mpi=$(awk "BEGIN {printf \"%.6f\", $sum_mpi / 10}")
  avg_cuda=$(awk "BEGIN {printf \"%.6f\", $sum_cuda / 10}")

  echo "-- Medie dei tempi --"
  echo "  Media MPI  su 10 run: $avg_mpi"
  echo "  Media CUDA su 10 run: $avg_cuda"
  echo
done

echo "🎉 Tutti i test MPI/CUDA sono stati eseguiti (10 run ciascuno)."
