"""MLP baseline: classifies each node from its own features only, ignoring the
graph entirely. Cheapest possible baseline for the Amazon Computers benchmark —
O(N) compute, no adjacency stored, so it's the resource floor the graph models
(GCN / GraphSAGE / GAT / graph transformer) are expected to beat.

python train_mlp.py --cache-features (cache flag kept for CLI parity, unused here)
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

from load_dataset import AccuracyEvaluator, load_products
from gnn_common import (
    append_jsonl, count_params, get_device, make_output_dir, set_seed,
    setup_logger, write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MLP baseline (no graph) on Amazon Computers")
    parser.add_argument("--dataset-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/mlp")
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2, help="number of hidden layers")
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=1)
    return parser.parse_args()


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers, dropout):
        super().__init__()
        dims = [in_dim] + [hidden] * num_layers + [out_dim]
        self.linears = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for i, lin in enumerate(self.linears[:-1]):
            x = self.dropout(F.relu(self.bns[i](lin(x))))
        return self.linears[-1](x)


def run_batches(indices, batch_size, shuffle):
    return torch.utils.data.DataLoader(indices.cpu(), batch_size=batch_size, shuffle=shuffle)


def train_epoch(model, x, labels, train_idx, optimizer, evaluator, batch_size, device):
    model.train()
    total_loss = total_n = 0
    y_true, y_pred = [], []
    for batch in run_batches(train_idx, batch_size, shuffle=True):
        out = model(x[batch].to(device))
        y = labels[batch].to(device)
        loss = F.cross_entropy(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * batch.numel()
        total_n += batch.numel()
        y_true.append(y.cpu())
        y_pred.append(out.argmax(dim=-1).detach().cpu())
    acc = evaluator.eval({"y_true": torch.cat(y_true).view(-1, 1), "y_pred": torch.cat(y_pred).view(-1, 1)})["acc"]
    return total_loss / max(total_n, 1), acc


@torch.no_grad()
def evaluate(model, x, labels, idx, evaluator, batch_size, device):
    model.eval()
    preds = []
    for batch in run_batches(idx, batch_size, shuffle=False):
        preds.append(model(x[batch].to(device)).argmax(dim=-1).cpu())
    pred = torch.cat(preds)
    return evaluator.eval({"y_true": labels[idx.cpu()].view(-1, 1), "y_pred": pred.view(-1, 1)})["acc"]


def run_once(args, run_id, device):
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_run{run_id}_mlp"
    out_dir = make_output_dir(args.output_dir, run_name)
    logger = setup_logger("mlp", out_dir)
    logger.info("Args: %s", json.dumps(vars(args), sort_keys=True))
    set_seed(args.seed + run_id)

    data, labels, split_idx, num_classes = load_products(args.dataset_root, logger, split_seed=args.seed + run_id)
    x = data.x.float()
    labels = labels.cpu()
    split_idx = {k: v.cpu() for k, v in split_idx.items()}
    evaluator = AccuracyEvaluator()

    model = MLP(x.size(1), args.hidden, num_classes, args.num_layers, args.dropout).to(device)
    logger.info("Model params: %d", count_params(model))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = best_test = -1.0
    best_epoch = stale = 0
    ckpt_path = out_dir / "best.pt"
    metrics_path = out_dir / "metrics.jsonl"

    for epoch in range(args.epochs):
        t0 = time.time()
        loss, train_acc = train_epoch(model, x, labels, split_idx["train"], optimizer, evaluator, args.batch_size, device)
        val_acc = test_acc = None
        if epoch % args.eval_every == 0:
            val_acc = evaluate(model, x, labels, split_idx["valid"], evaluator, args.batch_size, device)
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch
                best_test = evaluate(model, x, labels, split_idx["test"], evaluator, args.batch_size, device)
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
        "run": run_id, "model": "mlp", "best_val": best_val, "best_test": best_test,
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
