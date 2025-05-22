# Audio Reinforcement Learning Environment (Proof of concept)

A high-performance audio reinforcement learning environment that follows the principles of PufferLib to achieve maximum training throughput.

## Features

- Pre-allocated memory for maximum performance
- Multi-process parallelization using shared memory
- Direct observation writing to avoid copying
- Simple audio synthesis engine with 4 tracks
- PPO-based reinforcement learning algorithm

## Installation

```bash
# Install dependencies
pip install numpy torch

# Build the C++ extension
python setup.py build_ext --inplace
