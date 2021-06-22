#!/bin/bash

# 'KMEANS' 'ALL-DATA' 
# 'BADGE'
strategies=('ALL' 'CIRAL' 'CORESET' 'DFAL' 'SOFTMAX_HYBRID' 'RANDOM' 'BADGE')

# Run experiment
for proj in "${strategies[@]}"; do

    data_set='PLANKTON10'
    
    num_query=400

    echo $proj
    python3 run.py --dataset $data_set --num_query $num_query --strategy $proj
done

echo 'Finished running'
