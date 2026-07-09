# Phương Pháp Cải Tiến Mô Hình GAMLP

## Tổng Quan

Mô hình GAMLP (Graph Attention Multi-Layer Perceptron) đạt hiệu năng cao nhờ tách biệt bước tiền xử lý đồ thị khỏi quá trình huấn luyện: đặc trưng K-hop được tính trước offline, sau đó một MLP với cơ chế attention theo hop được huấn luyện thuần túy trên dữ liệu đó. Tuy nhiên, mô hình gốc tồn tại ba điểm yếu chính. Tài liệu này mô tả bốn phương pháp cải tiến được triển khai nhằm khắc phục các điểm yếu đó.

---

## 1. Smoothed Decay Label Propagation (SDLP)

### Điểm Yếu Gốc

Trong chế độ RLU (Reliable Label Utilization), GAMLP lan truyền nhãn one-hot cứng của tập huấn luyện qua đồ thị K lần liên tiếp rồi lấy kết quả cuối cùng làm đặc trưng bổ sung. Nhãn ở các hop xa (hop 4, 5, ...) mang tín hiệu nhiễu và loãng, nhưng mô hình không có cơ chế nào để tự động giảm trọng số chúng. Nhóm tác giả gốc phải thiết kế thêm "last residual connection" kết hợp hàm cosine để phạt các nhãn lan truyền xa — một cơ chế phức tạp và khó tune.

### Phương Pháp Cải Tiến

SDLP thay thế hàm `prepare_label_emb` gốc bằng hai thay đổi đơn giản:

**Làm mịn nhãn (Label Smoothing, ε = 0.1)**

Thay vì dùng nhãn one-hot cứng, áp dụng label smoothing trước khi lan truyền:

$$\tilde{y}_i = (1 - \varepsilon) \cdot \mathbf{1}_{y_i} + \frac{\varepsilon}{C}$$

trong đó $C$ là số class, $\varepsilon = 0.1$. Nhãn mềm ngăn mô hình học quá mức tự tin vào từng nhãn riêng lẻ trước khi truyền qua đồ thị.

**Tổng hợp có trọng số theo hop (Hop-Decay Weighted Sum, γ = 0.8)**

Thay vì chỉ lấy kết quả của hop K cuối cùng (tất cả K hop đóng góp ngang nhau qua chuỗi nhân), SDLP tính tổng có trọng số của hop 1 đến K với trọng số giảm dần theo cấp số nhân:

$$\text{label\_emb} = \frac{\sum_{k=1}^{K} \gamma^{k-1} \cdot A^k \tilde{y}}{\sum_{k=1}^{K} \gamma^{k-1}}$$

Với γ = 0.8: hop 1 có trọng số 1.0, hop 2 có 0.8, hop 3 có 0.64, ... Cấu trúc cục bộ (hop gần) được ưu tiên tự nhiên mà không cần cơ chế cosine phức tạp.

### So Sánh với Gốc

| Tiêu chí | GAMLP gốc | SDLP |
|----------|-----------|------|
| Loại nhãn | One-hot cứng | Làm mịn (1−ε)·one_hot + ε/C |
| Tổng hợp hop | Chỉ lấy hop K | Trung bình có trọng số γ^(k−1) |
| Phạt hop xa | Last residual + cosine (phức tạp) | Tự nhiên qua trọng số decay |
| Tham số thêm | Có (learned residual) | Không |

### Hyperparameter

| Tham số | Mặc định | Tắt cải tiến |
|---------|---------|-------------|
| `--label-smooth-eps` | 0.1 | 0.0 |
| `--label-decay` | 0.8 | 1.0 |

---

## 2. Sparse Hop Attention via Entropy Regularization (SHA)

### Điểm Yếu Gốc

GAMLP luôn tính attention cho đủ K+1 hop trong mỗi bước huấn luyện và suy luận, kể cả những hop có trọng số attention gần bằng 0. Điều này dẫn đến: (a) mô hình lãng phí năng lực tính toán vào các hop không cần thiết; (b) tốc độ huấn luyện chậm hơn các mô hình đơn giản (SGC, SIGN) từ 6–8 lần theo báo cáo gốc.

### Phương Pháp Cải Tiến

Thêm một hạng phạt entropy vào hàm loss trong quá trình huấn luyện:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \cdot H(\boldsymbol{\alpha})$$

trong đó $H(\boldsymbol{\alpha}) = -\sum_{k} \alpha_k \log(\alpha_k)$ là entropy Shannon của phân phối attention theo hop và $\lambda$ là hệ số điều chỉnh (`--att-sparsity`, mặc định 0.01).

**Cơ chế hoạt động:**

- Entropy cao → attention trải đều trên nhiều hop → phức tạp
- Entropy thấp → attention tập trung vào ít hop → đơn giản, hiệu quả
- Tối thiểu hóa $\lambda H(\boldsymbol{\alpha})$ khuyến khích mô hình tự học bỏ qua hop không cần thiết

**Ví dụ kết quả mong đợi:**

```
Trước khi có SHA:   [0.17, 0.16, 0.18, 0.17, 0.16, 0.16]  ← đều
Sau khi có SHA:     [0.04, 0.58, 0.29, 0.06, 0.02, 0.01]  ← tập trung
```

Trọng số attention `_last_att` được lưu lại sau mỗi lần forward pass để tính hạng phạt này mà không cần thay đổi kiến trúc hay thêm tham số.

### So Sánh với Gốc

| Tiêu chí | GAMLP gốc | SHA |
|----------|-----------|-----|
| Tất cả K+1 hop | Luôn được sử dụng | Tự động giảm trọng số hop yếu |
| Điều chỉnh độ phức tạp | Không có | Qua hệ số λ |
| Tham số thêm | Không | Không (chỉ 1 scalar λ) |

### Hyperparameter

| Tham số | Mặc định | Tắt cải tiến |
|---------|---------|-------------|
| `--att-sparsity` | 0.01 | 0.0 |

---

## 3. Confidence-Adaptive Pseudo-Label Selection (CAPS)

### Điểm Yếu Gốc

Trong chế độ RLU, GAMLP chọn các node unlabeled có độ tin cậy `max(softmax) > 0.85` để dùng làm pseudo-label ở stage tiếp theo. Ngưỡng cố định 0.85 là heuristic thủ công:

- Ở stage đầu, mô hình chưa tốt → rất ít node vượt ngưỡng → thiếu dữ liệu bổ sung
- Ở stage sau, mô hình overconfident → quá nhiều node nhiễu vượt ngưỡng → pseudo-label kém chất lượng
- Phải tune lại cho từng dataset khác nhau

### Phương Pháp Cải Tiến

Thay ngưỡng cố định bằng ngưỡng động dựa trên phân vị (percentile) của phân phối độ tin cậy hiện tại:

$$\tau_{\text{dynamic}} = \max\left(\text{Quantile}(\text{conf}, P\%), \ 0.9 \times \tau_{\text{base}}\right)$$

trong đó $\text{conf}_i = \max_c p_i(c)$ là độ tin cậy của node $i$ và $P$ là percentile (`--conf-percentile`, mặc định 85).

**Cơ chế hoạt động:**

- Luôn chọn top (100−P)% node tự tin nhất trong số unlabeled
- Khi mô hình tốt hơn qua các stage, phân phối confidence dịch lên cao → ngưỡng tự động tăng → chất lượng pseudo-label được duy trì
- Không cần tune tay cho từng dataset

**Ví dụ qua các stage:**

```
Stage 1: conf phân phối thấp  → threshold = 0.76 (25% top)
Stage 2: conf phân phối tăng  → threshold = 0.83 (25% top)
Stage 3: conf phân phối cao   → threshold = 0.91 (25% top)
```

### So Sánh với Gốc

| Tiêu chí | GAMLP gốc | CAPS |
|----------|-----------|------|
| Ngưỡng | Cố định 0.85 | Động theo percentile |
| Thích nghi | Không | Tự động qua các stage |
| Phụ thuộc dataset | Cao | Thấp |

### Hyperparameter

| Tham số | Mặc định | Tắt cải tiến |
|---------|---------|-------------|
| `--dynamic-threshold` | True | `--no-dynamic-threshold` |
| `--conf-percentile` | 85.0 | — |
| `--threshold` | 0.85 | Dùng làm ngưỡng cố định khi tắt |

---

## 4. ResNeXt-FFN + SwiGLU Backbone

### Điểm Yếu Gốc

Ba cải tiến trên (SDLP, SHA, CAPS) chủ yếu là regularization và cải tiến tiền xử lý — chúng không thay đổi **năng lực biểu diễn** của mô hình. MLP gốc trong GAMLP xử lý từng feature độc lập qua một đường đơn tuyến tính với activation cố định (PReLU), hạn chế khả năng capture các tương tác phi tuyến giữa các chiều đặc trưng.

### Phương Pháp Cải Tiến

Thay toàn bộ backbone MLP (`FeedForwardNet` và `FeedForwardNetII`) bằng **ResNeXt-FFN** với **SwiGLU activation** — kiến trúc được sử dụng trong LLaMA, Mistral và các mô hình ngôn ngữ lớn hiện đại.

#### 4.1 SwiGLU Activation

Thay activation cố định PReLU bằng Gated Linear Unit với cổng SiLU:

$$\text{SwiGLU}(x) = \text{SiLU}(W_{\text{gate}} \cdot x) \otimes (W_{\text{val}} \cdot x)$$

trong đó $\otimes$ là phép nhân theo phần tử. Cổng $\text{SiLU}(W_{\text{gate}} \cdot x)$ học cách chọn lọc **khi nào** nên kích hoạt từng chiều feature, thay vì áp dụng một hàm kích hoạt cố định cho tất cả. Điều này cho phép mô hình suppres các chiều không liên quan một cách có học.

#### 4.2 ResNeXt Block (Cardinality G = 4)

Thay một đường xử lý đơn bằng G nhánh song song, mỗi nhánh chuyên về một subspace đặc trưng khác nhau:

```
x [N, in_dim]
│
├── Branch-1: Linear(in, h/G) → BN → SwiGLU → Dropout
├── Branch-2: Linear(in, h/G) → BN → SwiGLU → Dropout
├── Branch-3: Linear(in, h/G) → BN → SwiGLU → Dropout
└── Branch-4: Linear(in, h/G) → BN → SwiGLU → Dropout
                    │
                    └── cat → Linear(h, out) → BN
                                    │
                        + shortcut(x) ← residual
                                    │
                                 output
```

Kiến trúc này lấy cảm hứng từ ResNeXt (CVPR 2017): thay vì một transformation rộng, dùng G transformation hẹp song song rồi tổng hợp lại. Mỗi nhánh học một "cách nhìn" khác nhau về cùng input feature.

#### 4.3 ResNeXtFFN — Stack Nhiều Block

Nhiều ResNeXt block được xếp chồng, block cuối là một Linear layer đơn giản để project về số chiều output mong muốn:

```
Input
  → ResNeXtBlock(in, h, h)     ← block 1
  → ResNeXtBlock(h, h, h)      ← blocks 2..n-1
  → Linear(h, out)              ← output layer
```

`ResNeXtFFN` thay thế cả `FeedForwardNet` (encoder per-hop) lẫn `FeedForwardNetII` (output MLP), loại bỏ các lớp graph convolution với alpha-mixing trong `FeedForwardNetII` — vai trò regularization được đảm nhận bởi shortcut connection trong mỗi ResNeXt block.

### So Sánh Kiến Trúc

| Thành phần | GAMLP gốc | ResNeXt-GAMLP |
|-----------|-----------|--------------|
| Per-hop encoder | FeedForwardNet (1 đường, PReLU) | ResNeXtFFN (G nhánh, SwiGLU) |
| Output MLP | FeedForwardNetII (graph conv + alpha-mix) | ResNeXtFFN (ResNeXt blocks + residual) |
| Activation | PReLU (fixed) | SwiGLU (learned gate) |
| Residual | Chỉ trong FeedForwardNetII | Trong mỗi ResNeXt block |
| Hop attention | Scalar attention (unchanged) | Scalar attention (unchanged) |

### Hyperparameter

| Tham số | Mặc định | Ý nghĩa |
|---------|---------|---------|
| `--num-groups` | 4 | Cardinality G (số nhánh song song) |
| `--hidden` | 512 | Phải chia hết cho G |

---

## Tổng Hợp: Ablation Study

Để đánh giá đóng góp riêng lẻ của từng cải tiến, chạy theo bảng sau:

| Thực nghiệm | `--label-smooth-eps` | `--label-decay` | `--att-sparsity` | `--dynamic-threshold` | File |
|-------------|---------------------|-----------------|------------------|-----------------------|------|
| Baseline gốc | — | — | — | — | `GAMLP_original/train_gamlp_products.py` |
| +SDLP | 0.1 | 0.8 | 0.0 | off | `train_improved_gamlp.py` |
| +SHA | 0.0 | 1.0 | 0.01 | off | `train_improved_gamlp.py` |
| +CAPS | 0.0 | 1.0 | 0.0 | on | `train_improved_gamlp.py` |
| +SDLP+SHA+CAPS | 0.1 | 0.8 | 0.01 | on | `train_improved_gamlp.py` |
| +ResNeXt+tất cả | 0.1 | 0.8 | 0.01 | on | `train_resnext_gamlp.py` |

### Lệnh Chạy

```bash
# Baseline
python GAMLP_original/train_gamlp_products.py --mode plain --cache-features

# Chỉ SDLP
python train_improved_gamlp.py --mode rlu --cache-features \
    --label-smooth-eps 0.1 --label-decay 0.8 \
    --att-sparsity 0.0 --no-dynamic-threshold

# Tất cả cải tiến (không đổi backbone)
python train_improved_gamlp.py --mode rlu --cache-features

# ResNeXt backbone + tất cả cải tiến
python train_resnext_gamlp.py --mode rlu --cache-features

# ResNeXt với cardinality G=8
python train_resnext_gamlp.py --mode rlu --cache-features --num-groups 8
```

---

## Bộ Nhớ GPU (16 GB)

| Thành phần | Vị trí | Kích thước ước tính |
|-----------|--------|---------------------|
| Hop features (5 hop × 13K node × 767 dim) | CPU RAM | ~300 MB |
| Label embeddings | CPU RAM | < 1 MB |
| Model (GAMLP gốc, hidden=512) | GPU | ~50 MB |
| Model (ResNeXt-GAMLP, G=4, hidden=512) | GPU | ~120 MB |
| Một batch train (50K node) | GPU | ~200 MB |
| **Tổng GPU peak** | GPU | **< 400 MB** |

Tất cả bốn cải tiến đều không tăng đáng kể yêu cầu bộ nhớ GPU, phù hợp với môi trường 16 GB.
