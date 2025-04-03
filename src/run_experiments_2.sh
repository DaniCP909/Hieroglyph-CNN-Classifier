#!/bin/bash

# Ensure script exits on first error
set -e


echo "Running: python3 main.py --epochs=300 --batch-size=64" >> LOG2.txt 2>&1
python3 main2.py --batch-size=64 --epochs=300 >> LOG2.txt 2>&1


echo "Running: python3 main.py --epochs=300 --batch-size=64 --large-mod" >> LOG2.txt 2>&1
python3 main2.py --batch-size=64 --epochs=300 --large-mod >> LOG2.txt 2>&1


echo "Running: python3 main.py --epochs=300 --batch-size=64 --fill" >> LOG2.txt 2>&1
python3 main2.py --batch-size=64 --epochs=300  --fill >> LOG2.txt 2>&1


echo "Running: python3 main.py --epochs=300 --batch-size=64 --large-mod --fill" >> LOG2.txt 2>&1
python3 main2.py --batch-size=64 --epochs=300 --large-mod --fill >> LOG2.txt 2>&1


echo "Running: python3 main.py --epochs=300 --batch-size=64 --short-font" >> LOG2.txt 2>&1
python3 main2.py --batch-size=64 --epochs=300  --short-font >> LOG2.txt 2>&1


echo "Running: python3 main.py --epochs=300 --batch-size=64 --large-mod" >> LOG2.txt 2>&1
python3 main2.py --batch-size=64 --epochs=300 --large-mod --short-font >> LOG2.txt 2>&1


