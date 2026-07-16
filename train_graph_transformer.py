"""Graph Transformer baseline on Amazon Computers: a Transformer block (multi-head
scaled dot-product attention + residual + LayerNorm + feed-forward) where each
node only attends over its 1-hop neighbourhood (+ itself via a self loop),
instead of the full O(N^2) attention a plain Transformer would use.

Attention: for edge (i -> j), score = (Q_i . K_j) / sqrt(d_head), softmax'd
per target node i over its incoming edges (same `scatter_softmax` trick as
train_gat.py). Cost is O(E * heads * head_dim), i.e. it scales with edges, not
nodes squared — the key resource-saving choice that keeps this runnable on a
small GPU/CPU where a dense N x N attention matrix would not fit
(13.7K nodes would need a 13.7K x 13.7K attention matrix per head — ~1.9GB in
fp32 for a *single* head; the sparse/local version needs a few MB).

This is the same trick used by real "graph transformer" architectures meant
for large graphs (e.g. UniMP, GraphTrans, TransformerConv in PyG) — attention
restricted to the graph's edges rather than a full dense pairwise matrix.
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from load_dataset import AccuracyEvaluator, load_products, load_split_idx_csv
from gnn_common import (
    add_self_loops, append_jsonl, count_params, get_device, make_output_dir,
    plot_training_curves, scatter_add, scatter_softmax, set_seed, setup_logger,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Local (edge-restricted) graph transformer baseline on Amazon Computers")
    parser.add_argument("--dataset-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/graph_transformer")
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--hidden", type=int, default=64, help="model width, must be divisible by --heads")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2, help="number of transformer blocks")
    parser.add_argument("--ffn-mult", type=int, default=2, help="feed-forward hidden size = hidden * ffn-mult")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--att-dropout", type=float, default=0.3)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--split-file", type=str, default="split_idx.csv",
                        help="Pre-saved CSV split for fair cross-model comparison. "
                             "Empty string to regenerate from --seed.")
    return parser.parse_args()


class LocalMultiHeadAttention(nn.Module):
    """Scaled dot-product attention restricted to graph edges."""

    def __init__(self, dim, heads, att_dropout):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"hidden dim {dim} must be divisible by heads {heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.att_dropout = nn.Dropout(att_dropout)

    def forward(self, x, edge_index, num_nodes):
        row, col = edge_index  # row = target (query), col = source (key/value)
        q = self.q_proj(x).view(num_nodes, self.heads, self.head_dim)
        k = self.k_proj(x).view(num_nodes, self.heads, self.head_dim)
        v = self.v_proj(x).view(num_nodes, self.heads, self.head_dim)

        edge_score = (q[row] * k[col]).sum(dim=-1) * self.scale  # [E, heads]
        edge_att = torch.stack([
            scatter_softmax(edge_score[:, h], row, num_nodes) for h in range(self.heads)
        ], dim=-1)  # [E, heads]
        edge_att = self.att_dropout(edge_att)

        msg = v[col] * edge_att.unsqueeze(-1)  # [E, heads, head_dim]
        out = scatter_add(msg.reshape(msg.size(0), -1), row, num_nodes).view(num_nodes, self.heads * self.head_dim)
        return self.out_proj(out)


class GraphTransformerBlock(nn.Module):
    def __init__(self, dim, heads, ffn_mult, dropout, att_dropout):
        super().__init__()
        self.attn = LocalMultiHeadAttention(dim, heads, att_dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, num_nodes):
        x = self.norm1(x + self.dropout(self.attn(x, edge_index, num_nodes)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class GraphTransformer(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers, heads, ffn_mult, dropout, att_dropout):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([
            GraphTransformerBlock(hidden, heads, ffn_mult, dropout, att_dropout) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, num_nodes):
        x = self.dropout(F.relu(self.input_proj(x)))
        for block in self.blocks:
            x = block(x, edge_index, num_nodes)
        return self.output_proj(x)


def train_epoch(model, x, edge_index, num_nodes, labels, train_idx, optimizer, evaluator):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index, num_nodes)
    loss = F.cross_entropy(out[train_idx], labels[train_idx])
    loss.backward()
    optimizer.step()
    pred = out[train_idx].argmax(dim=-1).detach().cpu()
    acc = evaluator.eval({"y_true": labels[train_idx].cpu().view(-1, 1), "y_pred": pred.view(-1, 1)})["acc"]
    return float(loss.item()), acc


@torch.no_grad()
def evaluate(model, x, edge_index, num_nodes, labels, idx, evaluator):
    model.eval()
    out = model(x, edge_index, num_nodes)
    pred = out[idx].argmax(dim=-1).cpu()
    return evaluator.eval({"y_true": labels[idx].cpu().view(-1, 1), "y_pred": pred.view(-1, 1)})["acc"]


def run_once(args, run_id, device):
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_run{run_id}_graph_transformer"
    out_dir = make_output_dir(args.output_dir, run_name)
    logger = setup_logger("graph_transformer", out_dir)
    logger.info("Args: %s", json.dumps(vars(args), sort_keys=True))
    set_seed(args.seed + run_id)

    data, labels, split_idx, num_classes = load_products(args.dataset_root, logger, split_seed=args.seed)
    split_file = Path(args.split_file) if args.split_file else None
    if split_file and split_file.is_file():
        split_idx = load_split_idx_csv(split_file)
        logger.info("Fixed split from %s  train=%d valid=%d test=%d",
                    split_file, split_idx["train"].numel(), split_idx["valid"].numel(), split_idx["test"].numel())
    x = data.x.float().to(device)
    edge_index = add_self_loops(data.edge_index, data.num_nodes).to(device)
    labels = labels.to(device)
    split_idx = {k: v.to(device) for k, v in split_idx.items()}
    evaluator = AccuracyEvaluator()

    model = GraphTransformer(
        x.size(1), args.hidden, num_classes, args.num_layers, args.heads, args.ffn_mult,
        args.dropout, args.att_dropout,
    ).to(device)
    logger.info("Model params: %d", count_params(model))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = best_test = -1.0
    best_epoch = stale = 0
    ckpt_path = out_dir / "best.pt"
    metrics_path = out_dir / "metrics.jsonl"

    for epoch in range(args.epochs):
        t0 = time.time()
        loss, train_acc = train_epoch(model, x, edge_index, data.num_nodes, labels, split_idx["train"], optimizer, evaluator)
        val_acc = test_acc = None
        if epoch % args.eval_every == 0:
            val_acc = evaluate(model, x, edge_index, data.num_nodes, labels, split_idx["valid"], evaluator)
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch
                best_test = evaluate(model, x, edge_index, data.num_nodes, labels, split_idx["test"], evaluator)
                torch.save({"model_state": model.state_dict(), "args": vars(args)}, ckpt_path)
                stale = 0
            else:
                stale += args.eval_every
            test_acc = best_test
        elapsed = time.time() - t0
        append_jsonl(metrics_path, {
            "epoch": epoch, "loss": loss, "train_acc": train_acc, "val_acc": val_acc,
            "best_val": best_val, "best_test": best_test, "best_epoch": best_epoch, "time_sec": elapsed,
        })
        logger.info(
            "epoch=%d loss=%.4f train=%.4f val=%s best_val=%.4f best_test=%.4f time=%.2fs",
            epoch, loss, train_acc, "None" if val_acc is None else f"{val_acc:.4f}",
            best_val, test_acc if test_acc is not None else best_test, elapsed,
        )
        if stale >= args.patience:
            logger.info("Early stopping at epoch=%d", epoch)
            break

    result = {
        "run": run_id, "model": "graph_transformer", "best_val": best_val, "best_test": best_test,
        "best_epoch": best_epoch, "output_dir": str(out_dir),
    }
    write_json(out_dir / "results.json", result)
    plot_path = plot_training_curves(metrics_path, out_dir, title="Graph Transformer")
    if plot_path:
        logger.info("Saved training curves: %s", plot_path)
    logger.info("Final: %s", result)
    return result


def main():
    args = parse_args()
    device = get_device(args.gpu)
    results = [run_once(args, run_id, device) for run_id in range(args.num_runs)]
    if len(results) > 1:
        tests = np.array([r["best_test"] for r in results], dtype=np.float64)
        vals = np.array([r["best_val"] for r in results], dtype=np.float64)
        print(json.dumps({
            "valid_mean": float(vals.mean()), "valid_std": float(vals.std()),
            "test_mean": float(tests.mean()), "test_std": float(tests.std()),
        }, indent=2))


if __name__ == "__main__":
    main()
