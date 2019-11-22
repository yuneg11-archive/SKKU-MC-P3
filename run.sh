#!/bin/bash

if [ "$1" = "seq" ]; then
    matrix_size=$2
    random_seed=$3

    echo "Build sequential version of LUD (lud_seq)"
    make seq
    echo ""
    echo "Execute process with size $matrix_size and seed $random_seed"
    time ./lud_seq $matrix_size $random_seed
elif [ "$1" = "par" ]; then
    matrix_size=$2
    random_seed=$3
    process_num=$4

    echo "Build parallel version of LUD (lud_par)"
    make par
    echo ""
    echo "Execute $process_num processes with size $matrix_size and seed $random_seed"
    time mpiexec --mca btl self --mca btl_openib_cpc_include rdmacm --machinefile hosts.txt -n $process_num lud_par $matrix_size $random_seed
else
    echo "Build and run sequential version"
    echo "$ . run.sh seq <matrix_size> <random_seed>"
    echo "Build and run parallel version"
    echo "$ . run.sh par <matrix_size> <random_seed> <process_num>"
fi
