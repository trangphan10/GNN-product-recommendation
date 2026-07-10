# GraphML_GAMLP_v2 — Thực nghiệm GNN trên Amazon Computers

Bộ thực nghiệm so sánh nhiều kiến trúc Graph Neural Network trên tập
**Amazon Co-Buy Computer** (~13.7K sản phẩm, 767 chiều đặc trưng, 10 lớp,
đồ thị đồng mua hàng). Gồm hai nhóm:

1. **GAMLP và các biến thể cải tiến** — decouple truyền lan đồ thị khỏi huấn
   luyện: đặc trưng K-hop được tính trước, sau đó một MLP với attention theo
   hop được huấn luyện thuần túy trên đặc trưng đó.
2. **Baseline kiến trúc kinh điển** (mới thêm) — MLP, GCN, GraphSAGE, GAT,
   Graph Transformer — cài đặt gọn nhẹ, không phụ thuộc các extension biên
   dịch (torch-scatter/torch-sparse), để chạy được trên tài nguyên hạn chế
   (CPU, GPU nhỏ).

## Cài đặt

```bash
pip install -r requirements.txt
```

`requirements.txt` ghim theo wheel CPU của PyG 2.2.2; nếu chạy trên GPU, đổi
dòng `-f https://data.pyg.org/whl/torch-2.2.2+cpu.html` thành index CUDA
tương ứng (xem comment trong file).

Dữ liệu (`amazon_co_buy_computer.npz`) được tự tải về thư mục `data/` qua
`torch_geometric.datasets.Amazon` nếu chưa có sẵn — xem `load_dataset.py`.

## Cấu trúc thư mục

```
GraphML_GAMLP_v2/
├── load_dataset.py                  # Load + split Amazon Co-Buy Computer
├── gnn_common.py                    # Tiện ích dùng chung cho các baseline (seed, logger, scatter ops thuần PyTorch)
│
├── GAMLP_original/
│   └── train_gamlp_products.py      # GAMLP gốc, không sửa đổi (baseline so sánh)
├── train_improved_gamlp.py          # GAMLP + SDLP + SHA + CAPS
├── train_resnext_gamlp.py           # GAMLP + ResNeXt-FFN/SwiGLU backbone + 3 cải tiến trên
│
├── train_mlp.py                     # MLP thuần (không dùng đồ thị)
├── train_gcn.py                     # GCN (Kipf & Welling)
├── train_graphsage.py               # GraphSAGE mean-aggregator (full-batch hoặc mini-batch neighbor sampling)
├── train_gat.py                     # GAT (multi-head attention)
├── train_graph_transformer.py       # Graph Transformer (attention giới hạn trên cạnh đồ thị, không phải O(N²))
│
├── requirements.txt
├── IMPROVEMENT_METHODS.md           # Giải thích chi tiết SDLP / SHA / CAPS / ResNeXt-FFN (tiếng Việt)
└── plans/
    └── architecture.md              # Design rationale (tiếng Anh)
```

## Chạy các baseline kiến trúc kinh điển

Tất cả script dưới đây dùng chung format: log ra `outputs/<model>/<timestamp>_run.../`,
gồm `train.log`, `metrics.jsonl` (theo epoch) và `results.json` (kết quả cuối).

```bash
# MLP — không dùng đồ thị, baseline rẻ nhất
python train_mlp.py

# GCN
python train_gcn.py

# GraphSAGE — full-batch (mặc định, phù hợp vì đồ thị nhỏ)
python train_graphsage.py

# GraphSAGE — mini-batch neighbor sampling (bộ nhớ huấn luyện O(batch_size × fanout^layers),
# minh hoạ cách scale khi đồ thị lớn hơn nhiều so với RAM/VRAM)
python train_graphsage.py --sampling --batch-size 512 --fanout 10 10

# GAT — multi-head attention, heads/head-dim nhỏ theo mặc định
python train_gat.py

# Graph Transformer — attention chỉ tính trên các cạnh của đồ thị (O(E)), không phải
# attention toàn cục O(N²) như Transformer thường
python train_graph_transformer.py
```

Mỗi script có `--hidden`, `--num-layers`, `--dropout`, `--lr`, `--epochs`,
`--patience`, `--seed`, `--num-runs`, `--gpu` (đặt `--gpu -1` để ép chạy CPU).
Chạy `python train_gcn.py --help` (tương tự cho các file khác) để xem đầy đủ.

Vì sao nhẹ tài nguyên: cả 5 kiến trúc được cài bằng tensor op thuần PyTorch
(`scatter_add_`, `scatter_reduce` trong `gnn_common.py`) thay vì
`torch-geometric.nn`/`torch-scatter`/`torch-sparse` — tránh rủi ro "extension
biên dịch không khớp bản torch" mà `requirements.txt` đã cảnh báo, đồng thời
giữ bộ nhớ đỉnh dưới ~200MB cho toàn bộ đồ thị + mô hình 2 lớp, hidden=64.

## Chạy các thực nghiệm GAMLP

```bash
# Baseline GAMLP gốc
python GAMLP_original/train_gamlp_products.py --mode plain --cache-features
python GAMLP_original/train_gamlp_products.py --mode rlu --cache-features

# GAMLP cải tiến (SDLP + SHA + CAPS)
python train_improved_gamlp.py --mode rlu --cache-features

# GAMLP + ResNeXt-FFN/SwiGLU backbone + 3 cải tiến trên
python train_resnext_gamlp.py --mode rlu --cache-features
```

Xem `IMPROVEMENT_METHODS.md` (giải thích chi tiết bằng tiếng Việt) và
`plans/architecture.md` để hiểu rationale + bảng ablation cho từng cải tiến.

## Kiểm tra cú pháp trước khi chạy

```bash
python -m py_compile train_mlp.py train_gcn.py train_graphsage.py train_gat.py \
    train_graph_transformer.py train_improved_gamlp.py \
    GAMLP_original/train_gamlp_products.py
```

## Quy ước code

- Python 3, thụt lề 4 space, CLI arg dạng gạch nối (`--batch-size`)
- Output ghi vào `outputs/` (không commit)
- Dữ liệu ghi vào `data/` (không commit)
- Dùng `Path` cho đường dẫn file; JSON/JSONL cho kết quả
