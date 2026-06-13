# ⚡ TinyML Inference Engine

Pure C neural network inference engine — no ML framework at runtime.

## How it works
1. Train MLP on MNIST (PyTorch)
2. Export weights to C header
3. Run inference in pure C

## Quick Start
```bash
python models/export_weights.py
python src/load_weights.py
gcc -O2 -o tinyml_v2 src/inference_v2.c -lm -I src/
./tinyml_v2
