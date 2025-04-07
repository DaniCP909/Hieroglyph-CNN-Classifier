#!/bin/bash

# Ensure script exits on first error
set -e

echo "Running: main2.py --batch-size=64 --epochs=300 --experiment=0" >> LOG2.txt 2>&1
python3 main2.py --batch-size=64 --epochs=300 --experiment=0 >> LOG2.txt 2>&1

echo "Running: main2.py --batch-size=64 --epochs=300 --experiment=1 --large-model" >> LOG2.txt 2>&1
python3 main2.py --batch-size=64 --epochs=300 --experiment=1 --large-model >> LOG2.txt 2>&1

echo "Running: main2.py --batch-size=64 --epochs=300 --experiment=2 --fill" >> LOG2.txt 2>&1
python3 main2.py --batch-size=64 --epochs=300 --experiment=2 --fill >> LOG2.txt 2>&1

echo "Running: main2.py --batch-size=64 --epochs=300 --experiment=3 --large-model --fill" >> LOG2.txt 2>&1
python3 main2.py --batch-size=64 --epochs=300 --experiment=3 --large-model --fill >> LOG2.txt 2>&1

echo "Running: main2.py --batch-size=64 --epochs=300 --experiment=4 --large-model --short-font" >> LOG2.txt 2>&1
python3 main2.py --batch-size=64 --epochs=300 --experiment=4 --large-model --short-font >> LOG2.txt 2>&1


