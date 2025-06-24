#!/usr/bin/env bash
set -euo pipefail

: "${MPI_NUMBER_PROC:?Devi impostare MPI_NUMBER_PROC}"
: "${MPI_FLAGS:=}"

readonly TESTS=(
  #"300 0.1 0.3 0.35 100 5 5 300 150 50 150 80 M 609823"
  #"1000 0.35 0.2 0.25 0 0 0 20000 10 0 500 0 M 4353435"
  #"10000 0.35 0.2 0.25 0 0 0 10000 9000 9000 50 100 M 4353435"
  #"429496730 0.35 0.2 0.25 0 0 0 1 1 0 4294967298 0 M 683224"
  #"4294967300 0.35 0.2 0.25 0 0 0 1 1 0 4294967298 0 M 683224"
  #"10000000 0.25 0.5 0.25 100000 500 50 0 1 1 1 1 M 12345"
  #"600000 0.35 0.2 0.25 35000 1500 1000 25000 1500 500 500 100 M 4353435"
  #"800000 0.2 0.35 0.25 40000 2500 200 25000 1800 200 400 50 M 1047392"
  "750000 0.25 0.2 0.35 30000 3500 800 25000 1200 900 350 80 B 9472048"
  "600000 0.2 0.25 0.2 20000 1000 100 25000 1300 100 275 90 A 7305729"
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

make align_seq_new align_mpi_omp_2 align_cuda_copy >/dev/null

echo
for idx in "${!TESTS[@]}"; do
  testnum=$((idx + 1))
  params="${TESTS[$idx]}"
  echo "=== Test #$testnum: $params ==="

  # Definizione comandi
  seq_cmd=(./align_seq_new)
  mpi_cmd=(mpirun -np "$MPI_NUMBER_PROC" $MPI_FLAGS ./align_mpi_omp_2)
  cuda_cmd=(./align_cuda_copy)

  # Esecuzioni
  out_seq=$(run_and_capture_output seq_cmd "$params")
  out_mpi=$(run_and_capture_output mpi_cmd "$params")
  out_cuda=$(run_and_capture_output cuda_cmd "$params")

  # Estrazione tempo e risultato
  time_seq=$(extract_time "$out_seq")
  time_mpi=$(extract_time "$out_mpi")
  time_cuda=$(extract_time "$out_cuda")

  result_seq=$(extract_result "$out_seq")
  result_mpi=$(extract_result "$out_mpi")
  result_cuda=$(extract_result "$out_cuda")

  # Output leggibile
  echo "Sequenziale -> $result_seq [$time_seq]"
  echo "MPI+OMP     -> $result_mpi [$time_mpi]"
  echo "CUDA        -> $result_cuda [$time_cuda]"

  # Confronto
  if [[ "$result_mpi" == "$result_seq" && "$result_cuda" == "$result_seq" ]]; then
    echo "✓ I risultati corrispondono esattamente."
  else
    echo "⚠ Discrepanza nei risultati!"
    [[ "$result_mpi" != "$result_seq" ]] && echo "  MPI+OMP differisce: $result_mpi"
    [[ "$result_cuda" != "$result_seq" ]] && echo "  CUDA    differisce: $result_cuda"
    exit 1
  fi

  echo
done

echo "🎉 Tutti i test sono passati con successo."
