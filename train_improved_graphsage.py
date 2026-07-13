"""Improved GraphSAGE — ba cải tiến nhắm đúng điểm yếu của mô hình gốc.

Điểm yếu 1 — Mean aggregator không phân biệt được cấu trúc hàng xóm (non-injective):
  Cải tiến: Multi-aggregator {mean, max, std} nối lại (PNA-style).
  • max  — bắt đặc trưng nổi bật nhất trong vùng lân cận
  • std  — đo mức độ đa dạng / phân tán của hàng xóm
  Bật/tắt bằng --no-multi-aggr (tắt → về mean-only như bản gốc).

Điểm yếu 2 — Over-smoothing khi stack nhiều lớp (thông tin lớp sâu bị làm mượt mất):
  Cải tiến: Jumping Knowledge (JK) — mỗi node kết hợp biểu diễn từ TẤT CẢ các lớp,
  cho phép model tự chọn độ sâu lan truyền phù hợp.
  Chọn chế độ: --jk-mode concat | maxpool | none (none → về bản gốc).

Điểm yếu 3 — Uniform random sampling bỏ qua tầm quan trọng của node (mini-batch mode):
  Cải tiến: Importance sampling tỉ lệ nghịch với bậc hàng xóm (∝ 1/sqrt(degree)).
  Node "hiếm" (bậc thấp) — thường là node chuyên biệt / bridge — được sample với xác suất cao hơn.
  Bật/tắt bằng --no-importance-sampling.

Refs:
  PNA — Corso et al., "Principal Neighbourhood Aggregation", NeurIPS 2020
  JK-Net — Xu et al., "Representation Learning on Graphs with Jumping Knowledge Networks", ICML 2018
"""

import argparse
import json
import random
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
    append_jsonl, build_adjacency_list, count_params, get_device, make_output_dir,
    scatter_add, scatter_max_feat, scatter_mean, scatter_std_feat,
    set_seed, setup_logger, write_json,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Improved GraphSAGE on Amazon Computers")
    p.add_argument("--dataset-root", type=str, default="data")
    p.add_argument("--output-dir",   type=str, default="outputs/improved_graphsage")
    p.add_argument("--num-runs",     type=int, default=1)
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--gpu",          type=int, default=0)

    # Model
    p.add_argument("--hidden",     type=int,   default=128)
    p.add_argument("--num-layers", type=int,   default=3,
                   help="Số lớp SAGE (JK giúp dùng nhiều lớp hơn mà không bị over-smoothing)")
    p.add_argument("--dropout",    type=float, default=0.4)

    # Cải tiến 1 — Multi-aggregator
    p.add_argument("--no-multi-aggr", dest="multi_aggr", action="store_false",
                   help="Tắt multi-aggregator, về mean-only (bản gốc)")
    p.set_defaults(multi_aggr=True)

    # Cải tiến 2 — Jumping Knowledge
    p.add_argument("--jk-mode", type=str, default="concat",
                   choices=["concat", "maxpool", "none"],
                   help="JK aggregation mode: concat (default) | maxpool | none (bản gốc)")

    # Cải tiến 3 — Importance sampling (mini-batch mode)
    p.add_argument("--sampling",   action="store_true",
                   help="Dùng mini-batch neighbour sampling thay vì full-batch")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--fanout",     type=int, nargs="+", default=[15, 10, 5],
                   help="Số hàng xóm lấy mỗi hop (outer → inner), một giá trị/lớp")
    p.add_argument("--no-importance-sampling", dest="importance_sampling",
                   action="store_false",
                   help="Tắt importance sampling, về uniform random (bản gốc)")
    p.set_defaults(importance_sampling=True)

    # Training
    p.add_argument("--epochs",       type=int,   default=300)
    p.add_argument("--lr",           type=float, default=0.005)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--patience",     type=int,   default=40)
    p.add_argument("--eval-every",   type=int,   default=1)
    p.add_argument("--split-file",   type=str,   default="split_idx.csv",
                   help="Pre-saved CSV split. Empty string để tạo mới từ --seed.")
    return p.parse_args()


# ── Model ────────────────────────────────────────────────────────────────────

class MultiAggSAGELayer(nn.Module):
    """
    GraphSAGE layer với multi-aggregator (Cải tiến 1).

    Thay vì chỉ dùng mean:
        h' = W_self·h  +  W_neigh·mean(h_N)

    Dùng mean + max + std nối lại:
        h' = Linear([h ; mean(h_N) ; max(h_N) ; std(h_N)])

    Với use_multi_aggr=False thì chỉ dùng mean → tương đương bản gốc.
    """

    def __init__(self, in_dim, out_dim, use_multi_aggr=True, dropout=0.0):
        super().__init__()
        self.use_multi_aggr = use_multi_aggr
        # Số aggregator: 1 (mean) hoặc 3 (mean + max + std)
        n_aggr = 3 if use_multi_aggr else 1
        self.lin = nn.Linear(in_dim * (1 + n_aggr), out_dim)
        self.bn  = nn.BatchNorm1d(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, num_nodes):
        row, col = edge_index

        neigh_mean = scatter_mean(x[col], row, num_nodes)         # [N, D]

        if self.use_multi_aggr:
            neigh_max = scatter_max_feat(x[col], row, num_nodes)  # [N, D]
            neigh_std = scatter_std_feat(x[col], row, num_nodes, mean=neigh_mean)  # [N, D]
            h = torch.cat([x, neigh_mean, neigh_max, neigh_std], dim=-1)  # [N, 4D]
        else:
            h = torch.cat([x, neigh_mean], dim=-1)                # [N, 2D]

        return self.drop(F.relu(self.bn(self.lin(h))))


class ImprovedGraphSAGE(nn.Module):
    """
    GraphSAGE cải tiến với:
    • Multi-aggregator trong mỗi lớp (Cải tiến 1)
    • Jumping Knowledge kết hợp output tất cả lớp (Cải tiến 2)
    """

    def __init__(self, in_dim, hidden, out_dim, num_layers, dropout,
                 use_multi_aggr=True, jk_mode="concat"):
        super().__init__()
        self.jk_mode   = jk_mode
        self.num_layers = num_layers

        # Lớp đầu: in_dim → hidden; các lớp còn lại: hidden → hidden
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                MultiAggSAGELayer(
                    in_dim  if i == 0 else hidden,
                    hidden,
                    use_multi_aggr=use_multi_aggr,
                    dropout=dropout,
                )
            )

        # Lớp phân loại cuối
        if jk_mode == "concat":
            # Nối output tất cả num_layers lớp → linear
            self.jk_proj = nn.Linear(num_layers * hidden, hidden)
            self.jk_bn   = nn.BatchNorm1d(hidden)
        self.classifier = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, num_nodes):
        layer_outs = []
        for conv in self.convs:
            x = conv(x, edge_index, num_nodes)
            layer_outs.append(x)

        if self.jk_mode == "concat":
            # Concat tất cả lớp → project về hidden
            jk = torch.cat(layer_outs, dim=-1)             # [N, L*H]
            x = F.relu(self.jk_bn(self.jk_proj(jk)))      # [N, H]
        elif self.jk_mode == "maxpool":
            # Element-wise max qua các lớp
            x = torch.stack(layer_outs, dim=0).max(dim=0).values  # [N, H]
        else:
            # Không JK — chỉ dùng lớp cuối (bản gốc)
            x = layer_outs[-1]

        return self.classifier(x)


# ── Importance Sampling (Cải tiến 3) ─────────────────────────────────────────

def compute_node_degree(edge_index, num_nodes):
    """Tính bậc của mỗi node (số cạnh đi vào = số hàng xóm gửi thông điệp)."""
    row = edge_index[0]
    deg = torch.zeros(num_nodes, dtype=torch.float32)
    deg.scatter_add_(0, row, torch.ones(row.size(0)))
    return deg.clamp(min=1.0)


def sample_subgraph(adj_list, seed_nodes, fanouts, deg=None, importance_sampling=True):
    """
    Sample ego-subgraph xung quanh seed_nodes.

    importance_sampling=True (Cải tiến 3):
        Sample hàng xóm với xác suất ∝ 1/sqrt(degree[neighbour]).
        Node ít phổ biến (bậc thấp) được ưu tiên hơn — thường là node chuyên biệt
        hoặc bridge giữa cộng đồng, mang nhiều thông tin cấu trúc.

    importance_sampling=False:
        Uniform random (hành vi mặc định của bản gốc).
    """
    frontier = set(seed_nodes.tolist())
    visited  = set(frontier)

    for fanout in fanouts:
        next_frontier = set()
        for n in frontier:
            nbrs = adj_list[n]
            if nbrs.numel() == 0:
                continue
            if nbrs.numel() <= fanout:
                sampled = nbrs
            elif importance_sampling and deg is not None:
                # Xác suất tỉ lệ nghịch với sqrt(bậc)
                w = 1.0 / deg[nbrs].sqrt()
                probs = w / w.sum()
                idx = torch.multinomial(probs, fanout, replacement=False)
                sampled = nbrs[idx]
            else:
                # Uniform random
                idx = torch.randperm(nbrs.numel())[:fanout]
                sampled = nbrs[idx]
            next_frontier.update(sampled.tolist())
        visited.update(next_frontier)
        frontier = next_frontier

    node_ids = torch.tensor(sorted(visited), dtype=torch.long)
    g2l = {int(g): i for i, g in enumerate(node_ids.tolist())}
    node_set = visited

    rows, cols = [], []
    for n in node_ids.tolist():
        for nb in adj_list[n].tolist():
            if nb in node_set:
                rows.append(g2l[n])
                cols.append(g2l[nb])
    if not rows:
        rows = cols = list(g2l.values())
    return node_ids, torch.tensor([rows, cols], dtype=torch.long)


# ── Training / Evaluation ─────────────────────────────────────────────────────

def run_batches(indices, batch_size, shuffle):
    return torch.utils.data.DataLoader(
        indices.cpu(), batch_size=batch_size, shuffle=shuffle
    )


def train_epoch_full(model, x, edge_index, num_nodes, labels, train_idx, optimizer, evaluator):
    model.train()
    optimizer.zero_grad()
    out  = model(x, edge_index, num_nodes)
    loss = F.cross_entropy(out[train_idx], labels[train_idx])
    loss.backward()
    optimizer.step()
    pred = out[train_idx].argmax(-1).detach().cpu()
    acc  = evaluator.eval({"y_true": labels[train_idx].cpu().view(-1, 1),
                           "y_pred": pred.view(-1, 1)})["acc"]
    return float(loss.item()), acc


@torch.no_grad()
def evaluate_full(model, x, edge_index, num_nodes, labels, idx, evaluator):
    model.eval()
    out  = model(x, edge_index, num_nodes)
    pred = out[idx].argmax(-1).cpu()
    return evaluator.eval({"y_true": labels[idx].cpu().view(-1, 1),
                           "y_pred": pred.view(-1, 1)})["acc"]


def train_epoch_sampled(model, x, adj_list, labels, train_idx, fanouts,
                        batch_size, optimizer, evaluator, device, deg, importance_sampling):
    model.train()
    total_loss = total_n = 0
    y_true, y_pred = [], []
    for seed in run_batches(train_idx, batch_size, shuffle=True):
        node_ids, sub_ei = sample_subgraph(adj_list, seed, fanouts,
                                           deg=deg, importance_sampling=importance_sampling)
        sub_x  = x[node_ids].to(device)
        sub_ei = sub_ei.to(device)
        out    = model(sub_x, sub_ei, node_ids.size(0))
        local  = torch.searchsorted(node_ids, seed)
        y      = labels[seed].to(device)
        loss   = F.cross_entropy(out[local], y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * seed.numel()
        total_n    += seed.numel()
        y_true.append(y.cpu())
        y_pred.append(out[local].argmax(-1).detach().cpu())
    acc = evaluator.eval({"y_true": torch.cat(y_true).view(-1, 1),
                          "y_pred": torch.cat(y_pred).view(-1, 1)})["acc"]
    return total_loss / max(total_n, 1), acc


@torch.no_grad()
def evaluate_sampled(model, x, edge_index, num_nodes, labels, idx, evaluator, device):
    model.eval()
    out  = model(x.to(device), edge_index.to(device), num_nodes)
    pred = out[idx].argmax(-1).cpu()
    return evaluator.eval({"y_true": labels[idx].cpu().view(-1, 1),
                           "y_pred": pred.view(-1, 1)})["acc"]


# ── Main run ──────────────────────────────────────────────────────────────────

def run_once(args, run_id, device):
    suffix = "_sampled" if args.sampling else "_full"
    run_name = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_run{run_id}"
        f"_improved_sage{suffix}"
        f"_jk{args.jk_mode}"
        f"{'_multiag' if args.multi_aggr else ''}"
    )
    out_dir = make_output_dir(args.output_dir, run_name)
    logger  = setup_logger("improved_sage", out_dir)
    logger.info("Args: %s", json.dumps(vars(args), sort_keys=True))
    set_seed(args.seed + run_id)

    data, labels, split_idx, num_classes = load_products(
        args.dataset_root, logger, split_seed=args.seed
    )
    split_file = Path(args.split_file) if args.split_file else None
    if split_file and split_file.is_file():
        split_idx = load_split_idx_csv(split_file)
        logger.info("Fixed split from %s  train=%d valid=%d test=%d",
                    split_file, split_idx["train"].numel(),
                    split_idx["valid"].numel(), split_idx["test"].numel())

    x          = data.x.float()
    edge_index = data.edge_index
    labels_cpu = labels.cpu()
    split_cpu  = {k: v.cpu() for k, v in split_idx.items()}
    evaluator  = AccuracyEvaluator()

    model = ImprovedGraphSAGE(
        in_dim       = x.size(1),
        hidden       = args.hidden,
        out_dim      = num_classes,
        num_layers   = args.num_layers,
        dropout      = args.dropout,
        use_multi_aggr = args.multi_aggr,
        jk_mode      = args.jk_mode,
    ).to(device)
    logger.info("Model params: %d", count_params(model))
    logger.info(
        "Improvements — multi_aggr=%s  jk_mode=%s  importance_sampling=%s",
        args.multi_aggr, args.jk_mode,
        args.importance_sampling if args.sampling else "N/A (full-batch)",
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    # Chuẩn bị cho từng chế độ training
    if args.sampling:
        fanouts  = (args.fanout * args.num_layers)[:args.num_layers]
        adj_list = build_adjacency_list(edge_index, data.num_nodes)
        deg      = compute_node_degree(edge_index, data.num_nodes) if args.importance_sampling else None
        logger.info("Mini-batch sampling  fanouts=%s  importance=%s",
                    fanouts, args.importance_sampling)
    else:
        x_dev  = x.to(device)
        ei_dev = edge_index.to(device)
        lbl_dev = labels.to(device)
        split_dev = {k: v.to(device) for k, v in split_idx.items()}

    best_val = best_test = -1.0
    best_epoch = stale = 0
    ckpt_path    = out_dir / "best.pt"
    metrics_path = out_dir / "metrics.jsonl"

    for epoch in range(args.epochs):
        t0 = time.time()
        if args.sampling:
            loss, train_acc = train_epoch_sampled(
                model, x, adj_list, labels_cpu, split_cpu["train"],
                fanouts, args.batch_size, optimizer, evaluator, device,
                deg=deg, importance_sampling=args.importance_sampling,
            )
        else:
            loss, train_acc = train_epoch_full(
                model, x_dev, ei_dev, data.num_nodes, lbl_dev,
                split_dev["train"], optimizer, evaluator,
            )

        val_acc = test_acc = None
        if epoch % args.eval_every == 0:
            if args.sampling:
                val_acc = evaluate_sampled(model, x, edge_index, data.num_nodes,
                                           labels_cpu, split_cpu["valid"], evaluator, device)
            else:
                val_acc = evaluate_full(model, x_dev, ei_dev, data.num_nodes,
                                        lbl_dev, split_dev["valid"], evaluator)

            if val_acc > best_val:
                best_val   = val_acc
                best_epoch = epoch
                if args.sampling:
                    best_test = evaluate_sampled(model, x, edge_index, data.num_nodes,
                                                 labels_cpu, split_cpu["test"], evaluator, device)
                else:
                    best_test = evaluate_full(model, x_dev, ei_dev, data.num_nodes,
                                              lbl_dev, split_dev["test"], evaluator)
                torch.save({"model_state": model.state_dict(), "args": vars(args)}, ckpt_path)
                stale = 0
            else:
                stale += args.eval_every
            test_acc = best_test

        elapsed = time.time() - t0
        append_jsonl(metrics_path, {
            "epoch": epoch, "loss": loss, "train_acc": train_acc,
            "val_acc": val_acc, "best_val": best_val,
            "best_test": best_test, "best_epoch": best_epoch, "time_sec": elapsed,
        })
        logger.info(
            "epoch=%d loss=%.4f train=%.4f val=%s best_val=%.4f best_test=%.4f time=%.2fs",
            epoch, loss, train_acc,
            "—" if val_acc is None else f"{val_acc:.4f}",
            best_val, best_test, elapsed,
        )
        if stale >= args.patience:
            logger.info("Early stopping at epoch=%d", epoch)
            break

    result = {
        "model":         "improved_graphsage",
        "run":           run_id,
        "multi_aggr":    args.multi_aggr,
        "jk_mode":       args.jk_mode,
        "importance_sampling": args.importance_sampling,
        "sampling_mode": "sampled" if args.sampling else "full",
        "best_val":      best_val,
        "best_test":     best_test,
        "best_epoch":    best_epoch,
        "output_dir":    str(out_dir),
    }
    write_json(out_dir / "results.json", result)
    logger.info("Final: %s", result)
    return result


def main():
    args    = parse_args()
    device  = get_device(args.gpu)
    results = [run_once(args, run_id, device) for run_id in range(args.num_runs)]
    if len(results) > 1:
        tests = np.array([r["best_test"] for r in results], dtype=np.float64)
        vals  = np.array([r["best_val"]  for r in results], dtype=np.float64)
        print(json.dumps({
            "valid_mean": float(vals.mean()),  "valid_std": float(vals.std()),
            "test_mean":  float(tests.mean()), "test_std":  float(tests.std()),
        }, indent=2))


if __name__ == "__main__":
    main()
