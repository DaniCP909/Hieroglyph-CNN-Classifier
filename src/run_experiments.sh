#!/bin/bash

# Ensure script exits on first error
set -e

# Loop through all combinations
for glyphnet in "" "--glyphnet"; do
    for short_font in "" "--short-font"; do
        for fill in "" "--fill"; do
            echo "Running: python3 main.py --epochs=1 --batch-size=64 $glyphnet $short_font $fill"
            
            python3 main.py --batch-size=64 --epochs=1 $glyphnet $short_font $fill
        done
    done
done
