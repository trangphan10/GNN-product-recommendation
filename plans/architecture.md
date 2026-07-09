# Improved GAMLP — Design Document

## Background

GAMLP decouples graph propagation from learning: multi-hop node features are precomputed
offline, then a purely MLP-based classifier with learnable hop-attention is trained.
In RLU mode, training labels are propagated through the graph and fed as auxiliary input,
which boosts accuracy but introduces several weaknesses.

---

## Weakness 1 — Overfitting from direct label propagation

### Problem

In RLU mode, hard one-hot labels for training nodes are propagated K times through the graph.
At far hops the signal becomes a diffuse, noisy average of training labels. The model can
overfit to this noisy signal, especially on smaller datasets. The paper addresses this with a
"last residual connection + cosine penalty" that is complex, adds learned parameters, and must
be carefully tuned per dataset.

### Fix: Smoothed Decay Label Propagation (SDLP)

`prepare_smoothed_label_emb` replaces `prepare_label_emb` with two changes:

**1. Label smoothing (ε = 0.1)**

```
y[train] = (1 - ε) * one_hot(label) + ε / C
```

Soft labels reduce over-confidence before propagation. Even if a training label is wrong or
atypical, the propagated signal is no longer a sharp peak — the model cannot fully memorise it.

**2. Hop-decay weighted accumulation (γ = 0.8)**

Instead of returning only the K-th chained propagation (equal implicit weighting of all K hops),
SDLP returns a weighted average of hops 1 through K:

```
label_emb = Σ_{k=1}^{K}  γ^(k-1) · propagate^k(y)
           ─────────────────────────────────────────
                       Σ_{k=1}^{K} γ^(k-1)
```

With γ = 0.8: hop 1 has weight 1.0, hop 2 has 0.8, hop 3 has 0.64, etc.
Near-hop structural context dominates; distant noisy signal is down-weighted automatically —
without any cosine similarity computation or extra learned parameters.

**Hyperparameters**

| Arg | Default | Effect |
|-----|---------|--------|
| `--label-smooth-eps` | 0.1 | ε: 0.0 = original hard labels |
| `--label-decay` | 0.8 | γ: 1.0 = uniform weighting (close to original) |

---

## Weakness 2 — Training complexity from unused hops

### Problem

GAMLP always computes attention over all K+1 hops even when some hops contribute near-zero
weight. The model and its training cost scale with K regardless of how much each hop actually
helps. Simplified GNNs like SGC are up to 6–8× faster precisely because they avoid this.

### Fix: Sparse Hop Attention via Entropy Regularisation (SHA)

Add an entropy penalty on the per-node hop-attention distribution to the training loss:

```
L_total = L_CE + λ · H(α)
```

where `H(α) = -Σ_k α_k · log(α_k)` is the Shannon entropy of the K-hop softmax attention
weights and λ = `--att-sparsity` (default 0.01).

Minimising entropy encourages the model to concentrate attention on the most informative hops
(peaked distribution) rather than spreading weight uniformly (high entropy). Over training,
some hops converge to near-zero attention — these can be pruned for inference speedup.

The attention weights `_last_att` (shape `[N_batch, K]`) are stored as an attribute on both
`RGAMLP` and `JKGAMLP` after each forward pass; the training loop reads them after computing
the cross-entropy loss.

**Why entropy and not L1?**  
The attention weights are already softmax-normalised (they sum to 1.0), so L1 would be
constant. Entropy is the natural measure of concentration for a probability distribution.

**Hyperparameters**

| Arg | Default | Effect |
|-----|---------|--------|
| `--att-sparsity` | 0.01 | λ: 0.0 = original behaviour, no sparsity pressure |

---

## Weakness 3 — Fixed confidence threshold is brittle across RLU stages

### Problem

The original RLU mode selects pseudo-labelled nodes for stage k+1 by thresholding
`max_class_prob > 0.85`. This static value is a heuristic: at stage 1 the model may be
poorly calibrated (few nodes pass), while at later stages it may be over-confident (too many
noisy nodes pass). The threshold needs manual re-tuning for each dataset.

### Fix: Confidence-Adaptive Pseudo-Label Selection (CAPS)

Replace the fixed threshold with the P-th percentile of the confidence distribution:

```python
conf = teacher_probs.max(dim=1).values      # max class probability per node
threshold = max(quantile(conf, P/100), base * 0.9)
```

With `--conf-percentile 85`: always selects the top 15% most confident unlabelled nodes.
As the model improves across stages its confidence distribution shifts upward, and CAPS
naturally raises the bar — no manual intervention needed. The floor `base * 0.9` prevents
the threshold from dropping too low during stage 0 when the model is underfit.

**Hyperparameters**

| Arg | Default | Effect |
|-----|---------|--------|
| `--dynamic-threshold` | True | Enable CAPS; use `--no-dynamic-threshold` for fixed |
| `--conf-percentile` | 85.0 | P: higher = fewer but more confident pseudo-labels |
| `--threshold` | 0.85 | Floor (CAPS) or sole threshold (fixed mode) |

---

## GPU memory footprint (16 GB target)

| Component | Location | Size |
|-----------|----------|------|
| 5-hop features (13K nodes × 767 dim) | CPU RAM | ~300 MB |
| Label embeddings (13K × 10) | CPU RAM | < 1 MB |
| Model parameters (hidden=512) | GPU | ~50 MB |
| One training batch (50K is the full graph) | GPU | ~200 MB |
| Total GPU peak | GPU | < 500 MB |

All three improvements add zero additional memory overhead.

---

## How to isolate each improvement (ablation)

| Run | `--label-smooth-eps` | `--label-decay` | `--att-sparsity` | `--dynamic-threshold` |
|-----|---------------------|-----------------|------------------|-----------------------|
| Full baseline | 0.0 | 1.0 | 0.0 | off |
| +SDLP | 0.1 | 0.8 | 0.0 | off |
| +SHA | 0.0 | 1.0 | 0.01 | off |
| +CAPS | 0.0 | 1.0 | 0.0 | on |
| All three | 0.1 | 0.8 | 0.01 | on |

Compare `best_val` and `best_test` from each run's `results.json` to measure contribution.
