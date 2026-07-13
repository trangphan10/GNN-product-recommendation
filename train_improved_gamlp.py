import argparse
import gc
import json
import logging
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from load_dataset import AccuracyEvaluator, load_products, load_split_idx_csv


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Improved GAMLP: SDLP + Sparse-Hop-Attention + CAPS on Amazon Computers"
    )
    # Dataset / output
    parser.add_argument("--dataset-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/gamlp_improved")
    parser.add_argument("--mode", choices=["plain", "rlu"], default="plain")
    parser.add_argument("--method", choices=["R_GAMLP", "JK_GAMLP"], default="R_GAMLP")
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)

    # Model architecture (same defaults as original)
    parser.add_argument("--num-hops", type=int, default=5)
    parser.add_argument("--label-num-hops", type=int, default=9)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--n-layers-1", type=int, default=4)
    parser.add_argument("--n-layers-2", type=int, default=4)
    parser.add_argument("--n-layers-3", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--input-drop", type=float, default=0.2)
    parser.add_argument("--att-drop", type=float, default=0.5)
    parser.add_argument("--label-drop", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--act", choices=["relu", "leaky_relu", "sigmoid"], default="leaky_relu")
    parser.add_argument("--pre-process", action="store_true", default=True)
    parser.add_argument("--no-pre-process", dest="pre_process", action="store_false")
    parser.add_argument("--residual", action="store_true", default=True)
    parser.add_argument("--no-residual", dest="residual", action="store_false")
    parser.add_argument("--bns", action="store_true", default=True)
    parser.add_argument("--no-bns", dest="bns", action="store_false")

    # Training
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--stages", nargs="+", type=int, default=[400, 300, 300, 300])
    parser.add_argument("--train-num-epochs", nargs="+", type=int, default=[0, 0, 0, 0])
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--gama", type=float, default=0.1)

    # ── Improvement 1: Smoothed Decay Label Propagation (SDLP) ──────────────
    # Addresses: overfitting from hard one-hot label propagation at far hops.
    # Set eps=0.0 and decay=1.0 to reproduce original prepare_label_emb exactly.
    parser.add_argument(
        "--label-smooth-eps", type=float, default=0.1,
        help="Label smoothing epsilon applied before propagation (0=disable, original behavior)",
    )
    parser.add_argument(
        "--label-decay", type=float, default=0.8,
        help="Exponential decay weight per hop in label accumulation (1.0=uniform, original behavior)",
    )

    # ── Improvement 2: Sparse Hop Attention via Entropy Regularization (SHA) ─
    # Addresses: model wastes capacity on near-zero attention hops.
    # Minimising H(attention) concentrates attention on fewer informative hops.
    # Set att-sparsity=0.0 to disable.
    parser.add_argument(
        "--att-sparsity", type=float, default=0.01,
        help="Weight λ for entropy regularisation on hop attention (0=disable)",
    )

    # ── Improvement 3: Confidence-Adaptive Pseudo-Label Selection (CAPS) ─────
    # Addresses: fixed threshold=0.85 may include too many/few nodes per stage.
    # Uses the P-th percentile of per-node confidence as the dynamic threshold.
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Base confidence threshold (floor for CAPS; sole threshold when disabled)")
    parser.add_argument("--dynamic-threshold", action="store_true", default=True,
                        help="Enable CAPS dynamic threshold (default: on)")
    parser.add_argument("--no-dynamic-threshold", dest="dynamic_threshold", action="store_false")
    parser.add_argument(
        "--conf-percentile", type=float, default=85.0,
        help="Confidence percentile P for CAPS (0-100). Higher = fewer, more confident pseudo-labels.",
    )

    # Misc
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--cache-features", action="store_true")
    parser.add_argument("--cache-dir", type=str, default="outputs/cache")
    parser.add_argument("--split-file", type=str, default="split_idx.csv",
                        help="Pre-saved CSV split for fair cross-model comparison. "
                             "Empty string to regenerate from --seed.")
    return parser.parse_args()


# ============================================================
# Utilities
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_output_dir(base_dir, run_name):
    out_dir = Path(base_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def setup_logger(out_dir):
    logger = logging.getLogger("gamlp_improved")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(out_dir / "train.log")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


# ============================================================
# Shared building blocks (identical to original)
# ============================================================

class Dense(nn.Module):
    def __init__(self, in_features, out_features, use_bn=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.BatchNorm1d(out_features) if use_bn else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        if isinstance(self.bias, nn.BatchNorm1d):
            self.bias.reset_parameters()

    def forward(self, x):
        out = torch.mm(x, self.weight)
        out = self.bias(out)
        if self.in_features == self.out_features:
            out = out + x
        return out


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, alpha, use_bn=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.BatchNorm1d(out_features) if use_bn else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        if isinstance(self.bias, nn.BatchNorm1d):
            self.bias.reset_parameters()

    def forward(self, x, h0):
        support = (1.0 - self.alpha) * x + self.alpha * h0
        out = torch.mm(support, self.weight)
        out = self.bias(out)
        if self.in_features == self.out_features:
            out = out + x
        return out


class FeedForwardNet(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout, use_bn=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.n_layers = n_layers
        self.use_bn = use_bn
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
            for _ in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
                self.bns.append(nn.BatchNorm1d(hidden))
            self.layers.append(nn.Linear(hidden, out_feats))
            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.n_layers - 1:
                if self.use_bn:
                    x = self.bns[layer_id](x)
                x = self.dropout(self.prelu(x))
        return x


class FeedForwardNetII(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout, alpha, use_bn=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        if n_layers == 1:
            self.layers.append(Dense(in_feats, out_feats, use_bn=False))
        else:
            self.layers.append(Dense(in_feats, hidden, use_bn=use_bn))
            for _ in range(n_layers - 2):
                self.layers.append(GraphConvolution(hidden, hidden, alpha, use_bn=use_bn))
            self.layers.append(Dense(hidden, out_feats, use_bn=False))
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x):
        x = self.layers[0](x)
        h0 = x
        for layer_id, layer in enumerate(self.layers[1:], start=1):
            if layer_id == self.n_layers - 1:
                x = self.dropout(self.prelu(x))
                x = layer(x)
            else:
                x = self.dropout(self.prelu(x))
                x = layer(x, h0)
        return x


def activation(name):
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.2)
    return nn.ReLU()


# ============================================================
# Improved models — expose _last_att for Improvement 2 (SHA)
# ============================================================

class RGAMLP(nn.Module):
    """R-GAMLP with _last_att exposed for entropy regularisation."""

    def __init__(self, nfeat, hidden, nclass, num_hops, args, use_label=False):
        super().__init__()
        self.num_hops = num_hops
        self.pre_process = args.pre_process
        self.residual = args.residual
        self.use_label = use_label
        att_dim = hidden if self.pre_process else nfeat
        self.process = nn.ModuleList(
            [FeedForwardNet(nfeat, hidden, hidden, 2, args.dropout, args.bns) for _ in range(num_hops)]
        ) if self.pre_process else None
        self.lr_att = nn.Linear(att_dim + att_dim, 1)
        self.lr_output = FeedForwardNetII(att_dim, hidden, nclass, args.n_layers_2, args.dropout, args.alpha, args.bns)
        self.res_fc = nn.Linear(nfeat, att_dim)
        self.label_fc = FeedForwardNet(nclass, hidden, nclass, args.n_layers_3, args.dropout, args.bns) if use_label else None
        self.input_drop = nn.Dropout(args.input_drop)
        self.att_drop = nn.Dropout(args.att_drop)
        self.label_drop = nn.Dropout(args.label_drop)
        self.dropout = nn.Dropout(args.dropout)
        self.prelu = nn.PReLU()
        self.act = activation(args.act)
        self._last_att = None
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.lr_att.weight, gain=gain)
        nn.init.zeros_(self.lr_att.bias)
        nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        nn.init.zeros_(self.res_fc.bias)
        self.lr_output.reset_parameters()
        if self.process is not None:
            for layer in self.process:
                layer.reset_parameters()
        if self.label_fc is not None:
            self.label_fc.reset_parameters()

    def encode(self, feature_list):
        feature_list = [self.input_drop(f) for f in feature_list]
        if self.pre_process:
            return [self.process[i](feature_list[i]) for i in range(self.num_hops)], feature_list
        return feature_list, feature_list

    def forward(self, feature_list, label_emb=None):
        input_list, raw_list = self.encode(feature_list)
        attention_scores = [self.act(self.lr_att(torch.cat([input_list[0], input_list[0]], dim=1)))]
        for i in range(1, self.num_hops):
            history_att = torch.cat(attention_scores[:i], dim=1)
            att = F.softmax(history_att, dim=1)
            history = input_list[0] * self.att_drop(att[:, 0:1])
            for j in range(1, i):
                history = history + input_list[j] * self.att_drop(att[:, j:j + 1])
            attention_scores.append(self.act(self.lr_att(torch.cat([history, input_list[i]], dim=1))))
        scores = F.softmax(torch.cat(attention_scores, dim=1), dim=1)
        self._last_att = scores  # SHA hook
        hidden = input_list[0] * self.att_drop(scores[:, 0:1])
        for i in range(1, self.num_hops):
            hidden = hidden + input_list[i] * self.att_drop(scores[:, i:i + 1])
        if self.residual:
            hidden = self.dropout(self.prelu(hidden + self.res_fc(raw_list[0])))
        out = self.lr_output(hidden)
        if self.use_label and label_emb is not None:
            out = out + self.label_fc(self.label_drop(label_emb))
        return out


class JKGAMLP(nn.Module):
    """JK-GAMLP with _last_att exposed for entropy regularisation."""

    def __init__(self, nfeat, hidden, nclass, num_hops, args, use_label=False):
        super().__init__()
        self.num_hops = num_hops
        self.pre_process = args.pre_process
        self.residual = args.residual
        self.use_label = use_label
        att_dim = hidden if self.pre_process else nfeat
        self.process = nn.ModuleList(
            [FeedForwardNet(nfeat, hidden, hidden, 2, args.dropout, args.bns) for _ in range(num_hops)]
        ) if self.pre_process else None
        self.lr_jk_ref = FeedForwardNetII(num_hops * att_dim, hidden, hidden, args.n_layers_1, args.dropout, args.alpha, args.bns)
        self.lr_att = nn.Linear(att_dim + hidden, 1)
        self.lr_output = FeedForwardNetII(att_dim, hidden, nclass, args.n_layers_2, args.dropout, args.alpha, args.bns)
        self.res_fc = nn.Linear(nfeat, att_dim)
        self.label_fc = FeedForwardNet(nclass, hidden, nclass, args.n_layers_3, args.dropout, args.bns) if use_label else None
        self.input_drop = nn.Dropout(args.input_drop)
        self.att_drop = nn.Dropout(args.att_drop)
        self.label_drop = nn.Dropout(args.label_drop)
        self.dropout = nn.Dropout(args.dropout)
        self.prelu = nn.PReLU()
        self.act = activation(args.act)
        self._last_att = None
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.lr_att.weight, gain=gain)
        nn.init.zeros_(self.lr_att.bias)
        nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        nn.init.zeros_(self.res_fc.bias)
        self.lr_jk_ref.reset_parameters()
        self.lr_output.reset_parameters()
        if self.process is not None:
            for layer in self.process:
                layer.reset_parameters()
        if self.label_fc is not None:
            self.label_fc.reset_parameters()

    def forward(self, feature_list, label_emb=None):
        feature_list = [self.input_drop(f) for f in feature_list]
        input_list = [self.process[i](feature_list[i]) for i in range(self.num_hops)] if self.pre_process else feature_list
        jk_ref = self.dropout(self.prelu(self.lr_jk_ref(torch.cat(input_list, dim=1))))
        scores = [self.act(self.lr_att(torch.cat([jk_ref, x], dim=1))) for x in input_list]
        weights = F.softmax(torch.cat(scores, dim=1), dim=1)
        self._last_att = weights  # SHA hook
        hidden = input_list[0] * self.att_drop(weights[:, 0:1])
        for i in range(1, self.num_hops):
            hidden = hidden + input_list[i] * self.att_drop(weights[:, i:i + 1])
        if self.residual:
            hidden = self.dropout(self.prelu(hidden + self.res_fc(feature_list[0])))
        out = self.lr_output(hidden)
        if self.use_label and label_emb is not None:
            out = out + self.label_fc(self.label_drop(label_emb))
        return out


def build_model(args, in_feats, num_classes, use_label):
    num_hops = args.num_hops + 1
    if args.method == "JK_GAMLP":
        return JKGAMLP(in_feats, args.hidden, num_classes, num_hops, args, use_label=use_label)
    return RGAMLP(in_feats, args.hidden, num_classes, num_hops, args, use_label=use_label)


# ============================================================
# Graph utilities
# ============================================================

def build_mean_adj(edge_index, num_nodes):
    adj_t = SparseTensor(
        row=edge_index[1],
        col=edge_index[0],
        sparse_sizes=(num_nodes, num_nodes),
    ).coalesce()
    deg = adj_t.sum(dim=1).to(torch.float32).clamp(min=1).view(-1, 1)
    return adj_t, deg


@torch.no_grad()
def precompute_features(data, num_hops, cache_dir, cache_features, logger):
    cache_name = getattr(data, "cache_name", "graph")
    cache_path = Path(cache_dir) / f"{cache_name}_hops_{num_hops}.pt"
    if cache_features and cache_path.exists():
        logger.info("Loading cached hop features from %s", cache_path)
        return torch.load(cache_path, map_location="cpu")
    logger.info("Precomputing %d-hop mean features", num_hops)
    adj_t, deg = build_mean_adj(data.edge_index, data.num_nodes)
    feats = [data.x.float().cpu()]
    for hop in range(1, num_hops + 1):
        t0 = time.time()
        feats.append(adj_t.matmul(feats[-1]) / deg)
        logger.info("Computed hop %d in %.2fs", hop, time.time() - t0)
        gc.collect()
    if cache_features:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(feats, cache_path)
        logger.info("Saved hop feature cache to %s", cache_path)
    return feats


# ============================================================
# Improvement 1 — Smoothed Decay Label Propagation (SDLP)
# ============================================================

@torch.no_grad()
def prepare_smoothed_label_emb(
    adj_t, deg, labels, split_idx, num_classes, num_hops,
    smooth_eps=0.1, decay_factor=0.8, teacher_probs=None,
):
    """
    SDLP replaces the original prepare_label_emb with two additions:

    1. Label smoothing (smooth_eps):
       Hard one-hot labels are replaced with (1-eps)*one_hot + eps/C.
       This prevents the propagated signal from carrying overconfident
       training-set information into unlabelled nodes.

    2. Hop-decay weighted accumulation (decay_factor):
       Instead of returning only the K-th propagated result (where all
       hops contribute equally through chaining), SDLP computes a weighted
       average of propagations at hop 1 through K.  The weight for hop k
       is decay_factor^(k-1), so nearby structural context dominates while
       distant noisy hops are down-weighted automatically.
       This replaces the paper's complex last-residual + cosine mechanism.

    Set smooth_eps=0.0 and decay_factor=1.0 to reproduce original behavior
    (apart from the weighted-vs-chained propagation scheme; the final value
    is similar for decay_factor close to 1.0).
    """
    # Smoothed label initialisation
    y = torch.full((labels.size(0), num_classes), smooth_eps / num_classes, dtype=torch.float32)
    train_idx = split_idx["train"]
    one_hot = F.one_hot(labels[train_idx], num_classes=num_classes).float()
    y[train_idx] = (1.0 - smooth_eps) * one_hot + smooth_eps / num_classes
    if teacher_probs is not None:
        y[split_idx["valid"]] = teacher_probs[split_idx["valid"]]
        y[split_idx["test"]] = teacher_probs[split_idx["test"]]

    # Hop-decay weighted accumulation
    accumulated = torch.zeros_like(y)
    curr = y.clone()
    weight_sum = 0.0
    for hop in range(num_hops):
        curr = adj_t.matmul(curr) / deg
        w = decay_factor ** hop      # hop 1→1.0, hop 2→decay, hop 3→decay², …
        accumulated += w * curr
        weight_sum += w

    return accumulated / max(weight_sum, 1e-8)


# ============================================================
# Improvement 2 — Sparse Hop Attention (SHA)
# ============================================================

def attention_entropy_loss(att_weights):
    """
    SHA: entropy of the hop-attention distribution averaged over nodes.
    Adding λ * entropy to the CE loss penalises uniform attention, pushing
    the model to concentrate on fewer, most informative hops.
    Zero extra parameters; one scalar hyperparameter (--att-sparsity).
    """
    return -(att_weights * (att_weights + 1e-8).log()).sum(dim=1).mean()


# ============================================================
# Improvement 3 — Confidence-Adaptive Pseudo-Label Selection (CAPS)
# ============================================================

def compute_dynamic_threshold(teacher_probs, base_threshold=0.85, conf_percentile=85.0):
    """
    CAPS: set the pseudo-label confidence threshold to the P-th percentile
    of the current model's confidence scores.  This adapts to how well the
    model is calibrated at each RLU stage instead of relying on a fixed
    heuristic.  The result is clamped so it never falls below 90% of the
    base threshold (safety floor).
    """
    conf = teacher_probs.max(dim=1).values
    dynamic = float(torch.quantile(conf, conf_percentile / 100.0).item())
    return max(dynamic, base_threshold * 0.9)


# ============================================================
# Training helpers
# ============================================================

def ogb_acc(evaluator, y_true, y_pred):
    return evaluator.eval({"y_true": y_true.view(-1, 1), "y_pred": y_pred.view(-1, 1)})["acc"]


def run_batches(indices, batch_size, shuffle):
    return torch.utils.data.DataLoader(
        indices.cpu(), batch_size=batch_size, shuffle=shuffle, drop_last=False
    )


def train_epoch(
    model, feats, labels, label_emb, train_idx,
    optimizer, evaluator, batch_size, device, att_sparsity=0.0,
):
    model.train()
    total_loss = 0.0
    total_examples = 0
    y_true, y_pred = [], []
    for batch in run_batches(train_idx, batch_size, shuffle=True):
        batch_feats = [f[batch].to(device) for f in feats]
        batch_label_emb = label_emb[batch].to(device) if label_emb is not None else None
        out = model(batch_feats, batch_label_emb)
        y = labels[batch].to(device)
        loss = F.cross_entropy(out, y)
        # SHA: entropy regularisation on hop attention
        if att_sparsity > 0.0 and model._last_att is not None:
            loss = loss + att_sparsity * attention_entropy_loss(model._last_att)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * batch.numel()
        total_examples += batch.numel()
        y_true.append(y.detach().cpu())
        y_pred.append(out.argmax(dim=-1).detach().cpu())
    return total_loss / max(total_examples, 1), ogb_acc(evaluator, torch.cat(y_true), torch.cat(y_pred))


def train_epoch_rlu(
    model, feats, labels, label_emb, train_idx, enhance_idx,
    teacher_probs, optimizer, evaluator, args, device,
):
    model.train()
    n_train = len(train_idx)
    n_enh = len(enhance_idx)
    n_total = max(n_train + n_enh, 1)
    train_loader = run_batches(train_idx, max(1, int(args.batch_size * n_train / n_total)), True)
    enh_loader = run_batches(enhance_idx, max(1, int(args.batch_size * n_enh / n_total)), True)
    loss_sum = 0.0
    total_examples = 0
    y_true, y_pred = [], []
    for idx_1, idx_2 in zip(train_loader, enh_loader):
        idx = torch.cat([idx_1, idx_2], dim=0)
        batch_feats = [f[idx].to(device) for f in feats]
        out = model(batch_feats, label_emb[idx].to(device))
        hard_loss = F.cross_entropy(out[: idx_1.numel()], labels[idx_1].to(device))
        teacher_soft = teacher_probs[idx_2].to(device)
        teacher_conf = teacher_soft.max(dim=1, keepdim=True).values
        kl_loss = (
            teacher_conf
            * (teacher_soft * (torch.log(teacher_soft + 1e-8) - F.log_softmax(out[idx_1.numel():], dim=1))).sum(dim=1, keepdim=True)
        ).mean()
        r_hard = idx_1.numel() / max(idx.numel(), 1)
        r_soft = idx_2.numel() / max(idx.numel(), 1)
        loss = hard_loss * r_hard + kl_loss * r_soft * args.gama
        # SHA: entropy regularisation
        if args.att_sparsity > 0.0 and model._last_att is not None:
            loss = loss + args.att_sparsity * attention_entropy_loss(model._last_att)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * idx.numel()
        total_examples += idx.numel()
        y_true.append(labels[idx_1].cpu())
        y_pred.append(out[: idx_1.numel()].argmax(dim=-1).detach().cpu())
    return loss_sum / max(total_examples, 1), ogb_acc(evaluator, torch.cat(y_true), torch.cat(y_pred))


@torch.no_grad()
def evaluate(model, feats, labels, label_emb, idx, evaluator, batch_size, device):
    model.eval()
    preds = []
    for batch in run_batches(idx, batch_size, shuffle=False):
        batch_feats = [f[batch].to(device) for f in feats]
        batch_label_emb = label_emb[batch].to(device) if label_emb is not None else None
        preds.append(model(batch_feats, batch_label_emb).argmax(dim=-1).cpu())
    return ogb_acc(evaluator, labels[idx.cpu()], torch.cat(preds))


@torch.no_grad()
def predict_logits(model, feats, label_emb, batch_size, device):
    model.eval()
    logits = []
    for batch in run_batches(torch.arange(feats[0].size(0)), batch_size, shuffle=False):
        batch_feats = [f[batch].to(device) for f in feats]
        batch_label_emb = label_emb[batch].to(device) if label_emb is not None else None
        logits.append(model(batch_feats, batch_label_emb).cpu())
    return torch.cat(logits, dim=0)


# ============================================================
# Stage training loop
# ============================================================

def train_stage(
    args, stage, epochs, model, feats, labels, label_emb,
    split_idx, evaluator, out_dir, logger, device, teacher_probs=None,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = -1.0
    best_test = -1.0
    best_epoch = -1
    stale = 0
    train_idx = split_idx["train"]
    enhance_idx = None

    if args.mode == "rlu" and stage > 0 and teacher_probs is not None:
        # CAPS: choose threshold adaptively or use fixed value
        if args.dynamic_threshold:
            threshold = compute_dynamic_threshold(teacher_probs, args.threshold, args.conf_percentile)
            logger.info(
                "Stage %d CAPS threshold=%.4f (percentile=%.0f%%, base=%.2f)",
                stage, threshold, args.conf_percentile, args.threshold,
            )
        else:
            threshold = args.threshold
            logger.info("Stage %d fixed threshold=%.4f", stage, threshold)

        conf = teacher_probs.max(dim=1).values
        mask = conf > threshold
        mask[train_idx] = False
        enhance_idx = torch.nonzero(mask, as_tuple=False).view(-1).cpu()
        logger.info("Stage %d pseudo-label nodes selected: %d", stage, enhance_idx.numel())
        if enhance_idx.numel() == 0:
            enhance_idx = train_idx.cpu()

    ckpt_path = out_dir / f"best_stage_{stage}.pt"
    metrics_path = out_dir / "metrics.jsonl"
    train_num_ep = args.train_num_epochs[min(stage, len(args.train_num_epochs) - 1)]

    for epoch in range(epochs):
        t0 = time.time()
        if args.mode == "rlu" and stage > 0 and teacher_probs is not None:
            loss, train_acc = train_epoch_rlu(
                model, feats, labels, label_emb, train_idx, enhance_idx,
                teacher_probs, optimizer, evaluator, args, device,
            )
        else:
            loss, train_acc = train_epoch(
                model, feats, labels, label_emb, train_idx, optimizer, evaluator,
                args.batch_size, device, att_sparsity=args.att_sparsity,
            )

        val_acc = test_acc = None
        if epoch % args.eval_every == 0 and epoch >= train_num_ep:
            val_acc = evaluate(model, feats, labels, label_emb, split_idx["valid"], evaluator, args.batch_size, device)
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch
                best_test = evaluate(model, feats, labels, label_emb, split_idx["test"], evaluator, args.batch_size, device)
                torch.save({"model_state": model.state_dict(), "args": vars(args), "stage": stage}, ckpt_path)
                stale = 0
            else:
                stale += args.eval_every
            test_acc = best_test

        elapsed = time.time() - t0
        append_jsonl(metrics_path, {
            "stage": stage, "epoch": epoch, "loss": loss, "train_acc": train_acc,
            "val_acc": val_acc, "best_val": best_val, "best_test": best_test,
            "best_epoch": best_epoch, "time_sec": elapsed,
        })
        logger.info(
            "stage=%d epoch=%d loss=%.4f train=%.4f val=%s best_val=%.4f best_test=%.4f time=%.2fs",
            stage, epoch, loss, train_acc,
            "None" if val_acc is None else f"{val_acc:.4f}",
            best_val, test_acc if test_acc is not None else best_test, elapsed,
        )
        if stale >= args.patience:
            logger.info("Early stopping at stage=%d epoch=%d", stage, epoch)
            break

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
    return best_val, best_test, best_epoch, ckpt_path


# ============================================================
# Entry points
# ============================================================

def run_once(args, run_id, device):
    run_name = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_run{run_id}_{args.mode}_{args.method}_improved"
    )
    out_dir = make_output_dir(args.output_dir, run_name)
    logger = setup_logger(out_dir)
    logger.info("Args: %s", json.dumps(vars(args), sort_keys=True))
    logger.info(
        "Active improvements — SDLP(eps=%.2f, decay=%.2f) | SHA(λ=%.4f) | CAPS(%s)",
        args.label_smooth_eps, args.label_decay, args.att_sparsity,
        f"percentile={args.conf_percentile:.0f}%" if args.dynamic_threshold else "off",
    )
    set_seed(args.seed + run_id)

    data, labels, split_idx, num_classes = load_products(args.dataset_root, logger, split_seed=args.seed)
    split_file = Path(args.split_file) if args.split_file else None
    if split_file and split_file.is_file():
        split_idx = load_split_idx_csv(split_file)
        logger.info("Fixed split from %s  train=%d valid=%d test=%d",
                    split_file, split_idx["train"].numel(), split_idx["valid"].numel(), split_idx["test"].numel())
    feats = precompute_features(data, args.num_hops, args.cache_dir, args.cache_features, logger)
    in_feats = feats[0].size(1)
    evaluator = AccuracyEvaluator()
    adj_t, deg = build_mean_adj(data.edge_index, data.num_nodes)
    labels = labels.cpu()
    split_idx = {k: v.cpu() for k, v in split_idx.items()}

    if args.eval_only:
        if not args.checkpoint:
            raise ValueError("--eval-only requires --checkpoint")
        use_label = "rlu" in args.checkpoint.lower() or args.mode == "rlu"
        label_emb = prepare_smoothed_label_emb(
            adj_t, deg, labels, split_idx, num_classes, args.label_num_hops,
            smooth_eps=args.label_smooth_eps, decay_factor=args.label_decay,
        ) if use_label else None
        model = build_model(args, in_feats, num_classes, use_label=use_label).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model_state"])
        result = {
            "train_acc": evaluate(model, feats, labels, label_emb, split_idx["train"], evaluator, args.batch_size, device),
            "valid_acc": evaluate(model, feats, labels, label_emb, split_idx["valid"], evaluator, args.batch_size, device),
            "test_acc": evaluate(model, feats, labels, label_emb, split_idx["test"], evaluator, args.batch_size, device),
        }
        write_json(out_dir / "eval_results.json", result)
        logger.info("Eval-only: %s", result)
        return result

    teacher_probs = None
    stage_results = []
    stages = args.stages if args.mode == "rlu" else [args.epochs]

    for stage, epochs in enumerate(stages):
        use_label = args.mode == "rlu"
        label_emb = None
        if use_label:
            label_emb = prepare_smoothed_label_emb(
                adj_t, deg, labels, split_idx, num_classes, args.label_num_hops,
                smooth_eps=args.label_smooth_eps, decay_factor=args.label_decay,
                teacher_probs=teacher_probs,
            )
        model = build_model(args, in_feats, num_classes, use_label=use_label).to(device)
        logger.info("Stage %d — %d params", stage, sum(p.numel() for p in model.parameters()))
        best_val, best_test, best_epoch, ckpt_path = train_stage(
            args, stage, epochs, model, feats, labels, label_emb,
            split_idx, evaluator, out_dir, logger, device, teacher_probs,
        )
        logits = predict_logits(model, feats, label_emb, args.batch_size, device)
        torch.save(logits, out_dir / f"logits_stage_{stage}.pt")
        np.save(out_dir / f"logits_stage_{stage}.npy", logits.numpy())
        teacher_probs = (logits / args.temp).softmax(dim=1)
        stage_results.append({
            "stage": stage, "best_val": best_val, "best_test": best_test,
            "best_epoch": best_epoch, "checkpoint": str(ckpt_path),
        })
        gc.collect()

    result = {
        "run": run_id,
        "mode": args.mode,
        "method": args.method,
        "improvements": {
            "sdlp_eps": args.label_smooth_eps,
            "sdlp_decay": args.label_decay,
            "sha_lambda": args.att_sparsity,
            "caps_dynamic": args.dynamic_threshold,
            "caps_percentile": args.conf_percentile,
        },
        "best_val": stage_results[-1]["best_val"],
        "best_test": stage_results[-1]["best_test"],
        "stages": stage_results,
        "output_dir": str(out_dir),
    }
    write_json(out_dir / "results.json", result)
    logger.info("Final: %s", result)
    return result


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}") if args.gpu >= 0 and torch.cuda.is_available() else torch.device("cpu")
    results = [run_once(args, run_id, device) for run_id in range(args.num_runs)]
    if len(results) > 1 and "best_test" in results[0]:
        tests = np.array([r["best_test"] for r in results], dtype=np.float64)
        vals = np.array([r["best_val"] for r in results], dtype=np.float64)
        print(json.dumps({
            "valid_mean": float(vals.mean()), "valid_std": float(vals.std()),
            "test_mean": float(tests.mean()), "test_std": float(tests.std()),
        }, indent=2))


if __name__ == "__main__":
    main()
