"""GAT baseline (Velickovic et al., 2018) on Amazon Computers.

Per-edge attention: e_ij = LeakyReLU(a^T [W h_i || W h_j]), normalised per
target node with a softmax over its incoming edges (`scatter_softmax` in
gnn_common.py — a 4-line replacement for what torch-scatter's
`scatter_softmax` would otherwise provide). Heads are concatenated on the
hidden layer and averaged on the output layer, as in the original paper.

Kept deliberately small for limited-resource runs: few heads (default 4) with
a small per-head dimension (default 16 => 64 total hidden width), since
attention memory scales with the *edge* count, not head width squared.
Full-batch training — see train_graphsage.py for a mini-batch pattern to
reuse here if the graph gets too large to fit in memory.
"""

import argparse
import json
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
    scatter_add, scatter_softmax, set_seed, setup_logger, write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="GAT baseline on Amazon Computers")
    parser.add_argument("--dataset-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/gat")
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--head-dim", type=int, default=16, help="output dim per attention head")
    parser.add_argument("--heads", type=int, default=4, help="attention heads in the hidden layer(s)")
    parser.add_argument("--out-heads", type=int, default=1, help="attention heads in the output layer (averaged)")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--att-dropout", type=float, default=0.6, help="dropout on attention coefficients")
    parser.add_argument("--negative-slope", type=float, default=0.2)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--split-file", type=str, default="split_idx.csv",
                        help="Pre-saved CSV split for fair cross-model comparison. "
                             "Empty string to regenerate from --seed.")
    return parser.parse_args()


class GATLayer(nn.Module):
    def __init__(self, in_dim, head_dim, heads, dropout, att_dropout, negative_slope, concat=True):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.concat = concat
        self.lin = nn.Linear(in_dim, heads * head_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, head_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, head_dim))
        self.bias = nn.Parameter(torch.zeros(heads * head_dim if concat else head_dim))
        self.dropout = nn.Dropout(dropout)
        self.att_dropout = nn.Dropout(att_dropout)
        self.negative_slope = negative_slope
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x, edge_index, num_nodes):
        row, col = edge_index  # row = target (aggregator), col = source (neighbour)
        x = self.dropout(x)
        h = self.lin(x).view(-1, self.heads, self.head_dim)  # [N, heads, head_dim]

        alpha_src = (h * self.att_src).sum(dim=-1)  # [N, heads]
        alpha_dst = (h * self.att_dst).sum(dim=-1)  # [N, heads]
        edge_alpha = F.leaky_relu(alpha_src[col] + alpha_dst[row], self.negative_slope)  # [E, heads]

        edge_alpha = torch.stack([
            scatter_softmax(edge_alpha[:, k], row, num_nodes) for k in range(self.heads)
        ], dim=-1)  # [E, heads]
        edge_alpha = self.att_dropout(edge_alpha)

        msg = h[col] * edge_alpha.unsqueeze(-1)  # [E, heads, head_dim]
        out = scatter_add(msg.reshape(msg.size(0), -1), row, num_nodes).view(num_nodes, self.heads, self.head_dim)

        if self.concat:
            out = out.reshape(num_nodes, self.heads * self.head_dim)
        else:
            out = out.mean(dim=1)
        return out + self.bias


class GAT(nn.Module):
    def __init__(self, in_dim, head_dim, out_dim, num_layers, heads, out_heads, dropout, att_dropout, negative_slope):
        super().__init__()
        self.layers = nn.ModuleList()
        dim = in_dim
        for _ in range(num_layers - 1):
            self.layers.append(GATLayer(dim, head_dim, heads, dropout, att_dropout, negative_slope, concat=True))
            dim = head_dim * heads
        self.layers.append(GATLayer(dim, out_dim, out_heads, dropout, att_dropout, negative_slope, concat=False))
        self.elu = nn.ELU()

    def forward(self, x, edge_index, num_nodes):
        for layer in self.layers[:-1]:
            x = self.elu(layer(x, edge_index, num_nodes))
        return self.layers[-1](x, edge_index, num_nodes)


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
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_run{run_id}_gat"
    out_dir = make_output_dir(args.output_dir, run_name)
    logger = setup_logger("gat", out_dir)
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

    model = GAT(
        x.size(1), args.head_dim, num_classes, args.num_layers, args.heads, args.out_heads,
        args.dropout, args.att_dropout, args.negative_slope,
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
        "run": run_id, "model": "gat", "best_val": best_val, "best_test": best_test,
        "best_epoch": best_epoch, "output_dir": str(out_dir),
    }
    write_json(out_dir / "results.json", result)
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
