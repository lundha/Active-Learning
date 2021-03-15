#!/bin/bash


strategies=("bayesian_spase_set" "coreset" "uncertainty")

for proj in "${strategies[@]}"; do
    python3 ../run.py --strategy $proj
done