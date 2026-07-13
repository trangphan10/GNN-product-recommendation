"""GCN baseline (Kipf & Welling, 2017) on Amazon Computers.

Propagation is H' = D^-1/2 (A + I) D^-1/2 H W, implemented with a single
`scatter_add` per layer over the (self-looped) edge list — no torch-scatter /
torch-sparse extension needed, see gnn_common.py.

Full-batch training: the whole graph (~13.7K nodes, ~490K directed edges after
symmetrising + self loops) and a 2-layer, 64-hidden GCN together take well
under 200MB, so there is no need for neighbour sampling here — the graph
itself is the "limited resource" regime. See train_graphsage.py for a
mini-batch neighbour-sampling example if you point this codebase at a much
larger graph.
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
    add_self_loops, append_jsonl, count_params, gcn_norm_edge_weight, get_device,
    make_output_dir, scatter_add, set_seed, setup_logger, write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="GCN baseline on Amazon Computers")
    parser.add_argument("--dataset-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/gcn")
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--split-file", type=str, default="split_idx.csv",
                        help="Pre-saved CSV split for fair cross-model comparison. "
                             "Empty string to regenerate from --seed.")
    return parser.parse_args()


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, edge_weight, num_nodes):
        row, col = edge_index
        x = self.lin(x)
        msg = x[col] * edge_weight.unsqueeze(-1)
        return scatter_add(msg, row, num_nodes)


class GCN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers, dropout):
        super().__init__()
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList([GCNLayer(dims[i], dims[i + 1]) for i in range(num_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(num_layers - 1)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight, num_nodes):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight, num_nodes)
            x = self.dropout(F.relu(self.bns[i](x)))
        return self.convs[-1](x, edge_index, edge_weight, num_nodes)


def train_epoch(model, x, edge_index, edge_weight, num_nodes, labels, train_idx, optimizer, evaluator):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index, edge_weight, num_nodes)
    loss = F.cross_entropy(out[train_idx], labels[train_idx])
    loss.backward()
    optimizer.step()
    pred = out[train_idx].argmax(dim=-1).detach().cpu()
    acc = evaluator.eval({"y_true": labels[train_idx].cpu().view(-1, 1), "y_pred": pred.view(-1, 1)})["acc"]
    return float(loss.item()), acc


@torch.no_grad()
def evaluate(model, x, edge_index, edge_weight, num_nodes, labels, idx, evaluator):
    model.eval()
    out = model(x, edge_index, edge_weight, num_nodes)
    pred = out[idx].argmax(dim=-1).cpu()
    return evaluator.eval({"y_true": labels[idx].cpu().view(-1, 1), "y_pred": pred.view(-1, 1)})["acc"]


def run_once(args, run_id, device):
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_run{run_id}_gcn"
    out_dir = make_output_dir(args.output_dir, run_name)
    logger = setup_logger("gcn", out_dir)
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
    edge_weight = gcn_norm_edge_weight(edge_index, data.num_nodes)
    labels = labels.to(device)
    split_idx = {k: v.to(device) for k, v in split_idx.items()}
    evaluator = AccuracyEvaluator()

    model = GCN(x.size(1), args.hidden, num_classes, args.num_layers, args.dropout).to(device)
    logger.info("Model params: %d", count_params(model))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = best_test = -1.0
    best_epoch = stale = 0
    ckpt_path = out_dir / "best.pt"
    metrics_path = out_dir / "metrics.jsonl"

    for epoch in range(args.epochs):
        t0 = time.time()
        loss, train_acc = train_epoch(model, x, edge_index, edge_weight, data.num_nodes, labels, split_idx["train"], optimizer, evaluator)
        val_acc = test_acc = None
        if epoch % args.eval_every == 0:
            val_acc = evaluate(model, x, edge_index, edge_weight, data.num_nodes, labels, split_idx["valid"], evaluator)
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch
                best_test = evaluate(model, x, edge_index, edge_weight, data.num_nodes, labels, split_idx["test"], evaluator)
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
        "run": run_id, "model": "gcn", "best_val": best_val, "best_test": best_test,
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
