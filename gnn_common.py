"""Shared utilities for the lightweight GNN baseline scripts:
train_mlp.py, train_gcn.py, train_graphsage.py, train_gat.py, train_graph_transformer.py.

Every model here is built from plain PyTorch tensor ops (scatter_add_ /
scatter_reduce) instead of torch-geometric's compiled message-passing kernels
(torch-scatter / torch-sparse). requirements.txt already warns that those
compiled extensions must match the installed torch build; avoiding them keeps
these baselines runnable on any CPU-only or small-GPU box with nothing beyond
`torch` itself, and keeps peak memory to O(E) rather than O(N^2).
"""

import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(gpu):
    if gpu >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def make_output_dir(base_dir, run_name):
    out_dir = Path(base_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def setup_logger(name, out_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for h in [logging.StreamHandler(sys.stdout), logging.FileHandler(out_dir / "train.log")]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------
# Pure-PyTorch graph ops (no torch-scatter / torch-sparse dependency)
# ----------------------------------------------------------------------------

def scatter_add(src, index, dim_size):
    """Sum rows of `src` into `dim_size` buckets given by `index` (grouping on dim 0)."""
    shape = (dim_size,) + src.shape[1:]
    out = src.new_zeros(shape)
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return out.scatter_add_(0, idx, src)


def scatter_mean(src, index, dim_size):
    summed = scatter_add(src, index, dim_size)
    count = scatter_add(src.new_ones(src.size(0), 1), index, dim_size).clamp(min=1)
    return summed / count


def scatter_softmax(src, index, dim_size):
    """Softmax of 1-D `src` within each group defined by `index` (used by GAT / graph
    transformer attention, where the "group" is the set of edges pointing at one target node)."""
    src_max = src.new_full((dim_size,), float("-inf"))
    src_max = src_max.scatter_reduce(0, index, src, reduce="amax", include_self=True)
    src_max = torch.nan_to_num(src_max, neginf=0.0)
    shifted = (src - src_max[index]).exp()
    denom = scatter_add(shifted.unsqueeze(-1), index, dim_size).squeeze(-1).clamp(min=1e-16)
    return shifted / denom[index]


def add_self_loops(edge_index, num_nodes):
    loop = torch.arange(num_nodes, device=edge_index.device)
    return torch.cat([edge_index, torch.stack([loop, loop], dim=0)], dim=1)


def gcn_norm_edge_weight(edge_index, num_nodes):
    """Symmetric D^-1/2 A D^-1/2 edge weights (Kipf & Welling). `edge_index` must
    already include self loops."""
    row, col = edge_index
    deg = scatter_add(row.new_ones(row.size(0), 1, dtype=torch.float32), row, num_nodes).squeeze(-1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    return deg_inv_sqrt[row] * deg_inv_sqrt[col]


def scatter_max_feat(src, index, dim_size):
    """Max-aggregation over feature vectors using scatter_reduce_ (PyTorch 1.12+).
    Nodes with no incoming edges keep the initial value of 0."""
    out = src.new_zeros(dim_size, src.size(-1))
    out.scatter_reduce_(0, index.unsqueeze(-1).expand_as(src), src, reduce="amax", include_self=False)
    return out


def scatter_std_feat(src, index, dim_size, mean=None):
    """Std-dev aggregation: sqrt(E[x²] − E[x]²). Uses a pre-computed mean if supplied."""
    if mean is None:
        mean = scatter_mean(src, index, dim_size)
    sq_mean = scatter_mean(src * src, index, dim_size)
    # eps giữ variance > 0: gradient của sqrt tại 0 là vô cùng → loss NaN
    return ((sq_mean - mean * mean).clamp(min=0) + 1e-6).sqrt()


def build_adjacency_list(edge_index, num_nodes):
    """Python list of 1-D LongTensors: neighbours of each node. Only used by
    train_graphsage.py's mini-batch neighbour-sampling mode."""
    row, col = edge_index.cpu()
    order = torch.argsort(row)
    row, col = row[order], col[order]
    counts = torch.bincount(row, minlength=num_nodes).tolist()
    adj, ptr = [], 0
    for c in counts:
        adj.append(col[ptr:ptr + c])
        ptr += c
    return adj
