# GraphML_GAMLP_v2 — Improved GAMLP Experiments

Duplicate of `GraphML_subject` with three targeted improvements addressing GAMLP's known weaknesses.
Designed to run comfortably within 16 GB GPU memory.

## Improvements over baseline

| # | Name | Weakness addressed | Key args |
|---|------|--------------------|----------|
| 1 | **SDLP** — Smoothed Decay Label Propagation | Overfitting from hard one-hot label propagation | `--label-smooth-eps 0.1 --label-decay 0.8` |
| 2 | **SHA** — Sparse Hop Attention | Model wastes capacity on near-zero attention hops | `--att-sparsity 0.01` |
| 3 | **CAPS** — Confidence-Adaptive Pseudo-Label Selection | Fixed threshold=0.85 is brittle across RLU stages | `--dynamic-threshold --conf-percentile 85` |

Set each improvement's args to their "off" values to isolate individual contributions (see ablation table below).

## Directory layout

```
GraphML_GAMLP_v2/
├── load_dataset.py                  # Dataset loading (Amazon Co-Buy Computer)
├── train_improved_gamlp.py          # Improved GAMLP with all 3 improvements
├── requirements.txt                 # Python dependencies
├── GAMLP_original/
│   └── train_gamlp_products.py      # Unmodified baseline for comparison
└── plans/
    └── architecture.md              # Design rationale for all improvements
```

## Installation

```bash
pip install -r requirements.txt
```

## Running experiments

### Baseline (original GAMLP, for comparison)
```bash
python GAMLP_original/train_gamlp_products.py --mode plain --cache-features
python GAMLP_original/train_gamlp_products.py --mode rlu --cache-features
```

### Improved GAMLP — all improvements active
```bash
# Plain mode: only SHA is active (no label propagation in plain mode)
python train_improved_gamlp.py --mode plain --cache-features

# RLU mode: all three improvements active
python train_improved_gamlp.py --mode rlu --cache-features
```

### Ablation — disable individual improvements
```bash
# Disable SDLP only (revert to original hard one-hot, uniform hops)
python train_improved_gamlp.py --mode rlu --label-smooth-eps 0.0 --label-decay 1.0

# Disable SHA only
python train_improved_gamlp.py --mode rlu --att-sparsity 0.0

# Disable CAPS only (use fixed threshold like original)
python train_improved_gamlp.py --mode rlu --no-dynamic-threshold

# Full baseline parity (all improvements off)
python train_improved_gamlp.py --mode rlu \
    --label-smooth-eps 0.0 --label-decay 1.0 \
    --att-sparsity 0.0 --no-dynamic-threshold
```

### Recommended ablation matrix

| Run | `--label-smooth-eps` | `--label-decay` | `--att-sparsity` | dynamic threshold |
|-----|---------------------|-----------------|------------------|-------------------|
| Baseline (original) | 0.0 | 1.0 | 0.0 | off |
| +SDLP only | 0.1 | 0.8 | 0.0 | off |
| +SHA only | 0.0 | 1.0 | 0.01 | off |
| +CAPS only | 0.0 | 1.0 | 0.0 | on |
| All improvements | 0.1 | 0.8 | 0.01 | on |

## Verify syntax before running
```bash
python -m py_compile train_improved_gamlp.py GAMLP_original/train_gamlp_products.py
```

## Coding conventions
- Python 3, 4-space indentation, hyphenated CLI args (e.g. `--batch-size`)
- Outputs go to `outputs/` (not committed)
- Dataset files go to `data/` (not committed)
- Use `Path` for filesystem paths, JSON/JSONL for results
