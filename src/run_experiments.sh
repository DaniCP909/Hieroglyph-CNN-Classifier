#!/bin/bash

# Ensure script exits on first error
set -e

# Start experiment counter
experiment=5

# Loop through all combinations
for glyphnet in "" "--glyphnet"; do
    for short_font in "" "--short-font"; do
        echo "Running: python3 main.py --epochs=300 --batch-size=32 $glyphnet $short_font --experimen=$experiment" >> LOG.txt 2>&1
        python3 main.py --batch-size=32 --epochs=300 $glyphnet $short_font --experimen=$experiment >> LOG.txt 2>&1
        experiment=$((experiment + 1))
    done
    echo "Running: python3 main.py --epochs=300 --batch-size=32 $glyphnet --fill --experimen=$experiment" >> LOG.txt 2>&1
    python3 main.py --batch-size=32 --epochs=300 $glyphnet --fill --experimen=$experiment >> LOG.txt 2>&1
    experiment=$((experiment + 1))
done
