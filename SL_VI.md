# Giải Thích Toàn Bộ SCALING_LAWS.md — Từ First Principles

## Bối Cảnh: NanoSeek Là Gì?

NanoSeek là một model MoE (Mixture of Experts) tái hiện DeepSeek V3.2 ở quy mô nhỏ: 1.08B active params / 4.75B total params. Câu hỏi cốt lõi mà file này trả lời:

> "Với một ngân sách compute cố định C, nên phân bổ compute thế nào cho model MoE ở quy mô nano?"

Chưa ai nghiên cứu hệ thống câu hỏi này dưới 1B active params với đầy đủ routing diagnostics.

---

## Nguyên Lý Nền Tảng #1: "Kiến trúc đã được chứng minh — không cần re-ablate"

### WHY: Tại Sao Không Ablate Kiến Trúc?

DeepSeek V3 đã chứng minh kiến trúc này hoạt động ở 671B params. DeepSeekMoE đã chứng minh shared experts quan trọng ở 16B params (+33% loss khi bỏ chúng). NanoSeek là **reimplementation** (tái hiện), không phải thiết kế mới.

Re-ablate feature đã proven = đốt compute để khám phá lại kết quả đã biết.

**Câu hỏi khoa học đúng** không phải "kiến trúc này có đúng không?" mà là "kiến trúc này scale down thế nào?"

### Vấn Đề Với Architecture Ablation

1. **HP chỉ valid cho kiến trúc đã test.** Nếu ablation thay đổi kiến trúc (bỏ shared experts), HP search phải làm lại — combinatorial explosion.
2. **Ablation ở nano scale cho tín hiệu sai.** Feature có vẻ marginal ở 400M có thể critical ở 1B. DeepSeek đã test ở 16B và 671B — bracket cả hai bên của chúng ta.
3. **Mỗi dollar cho ablation → kết quả về model sai** (variant đã ablate). Mỗi dollar cho HP search → trực tiếp inform model cuối cùng.

### Nguyên Lý Đúng

Calibrate HPs trước → IsoFLOP với tuned HPs → kết quả đáng tin cậy. Không "bet" nào, không rủi ro re-run $200.

---

## Plan Đúng

```
Phase 0 ──> Phase 1 ──> Phase 2 ──> Phase 3 ──> Phase 4
Gate 1      HP Search    IsoFLOP     Miniseries   1B Grad.
(verify)    (calibrate)  (discover)  (validate)   (produce)
$0.50       ~$30         ~$200       ~$120        ~$350
5 min       4 giờ        2-3 ngày    1-2 ngày     12 giờ
```

Kiến trúc được **khóa từ Phase 0**: Full DeepSeek V3 — MLA + MoE (64 experts top-8, bias routing, 2 shared) + MTP. Mọi phase chạy trên cùng một kiến trúc.

---

## Phase 0: Gate 1 — Smoke Test ($0.50, 5 phút)

**WHAT**: Chạy 100 steps ở scale d16 ablation
**WHY**: 7 bugs đã được phát hiện trong quá trình phát triển. 3 bugs CRITICAL — model hoàn toàn không học (loss stuck ở ln(32768) ≈ 10.39). Không có smoke test, bạn có thể đốt toàn bộ ngân sách với code hỏng.

**HOW**: Checklist pass/fail

| Check | Pass nghĩa là | Fail nghĩa là |
|-------|---------------|---------------|
| Loss step 0 ≈ 10.9 | Init đúng | RMSNorm/RoPE bị zero |
| Loss step 100 < 8.0 | Model đang học | Gradient flow hỏng |
| H_load > 4.0 bits | Routing cân bằng | Tất cả token đổ vào 1 expert |
| MTP loss giảm | MTP head hoạt động | MTP zero-init bug |

**First principle**: Trước khi đầu tư, verify rằng hệ thống cơ bản hoạt động. Chi phí verify rẻ hơn chi phí sai lầm hàng trăm lần.

---

## Phase 1: HP Search + muP Validation — Calibrate Hệ Thống (~$30, 4 giờ)

### Sub-phase 1a: LR Sweep ở Reference Scale (~$18)

**WHAT**: 6 runs = 3 matrix_lr × 2 embedding_lr, 500 steps mỗi run ở d16

**WHY chỉ 2 hyperparameters?**
NanoSeek dùng MuonAdamW — optimizer hybrid với các nhóm tham số khác nhau:

```
hidden_lr  = matrix_lr × √(B/B_ref) × (w_ref/w)   ← Muon, auto-scale
embed_lr   = embedding_lr × √(B/B_ref)              ← AdamW, auto-scale
router_lr  = 3e-4                                    ← CONSTANT (muP prescription)
norm_lr    = 3e-4                                    ← CONSTANT
```

Router và norm LRs là constant theo muP. Chỉ cần tune **2 base rates**: `matrix_lr` (Muon) và `embedding_lr` (AdamW).

**WHY 6 runs thay vì 320 (như nanochat)?**
nanochat không dùng muP — Karpathy phát hiện "What works at d12 doesn't transfer to d20". Phải brute-force sweep 320 experiments.

NanoSeek dùng muP (Tensor Programs V):
```
η_d24 = η_d16 × (1280/1920)  ← tự tính, không cần re-tune
```
Tune ở d16 → transfer sang mọi depth bằng công thức. **50x rẻ hơn**.

**WHY HP search không cần ratio\*?**
Plan cũ nói HP search phải sau IsoFLOP vì weight decay phụ thuộc vào tổng tokens D. Nhưng coupling này bị overestimate:

1. **HP search là so sánh tương đối ở fixed budget.** 6 runs cùng 500 steps, cùng WD. Ranking giữa các (matrix_lr, embedding_lr) pairs stable bất kể D.
2. **WD sensitivity thấp.** WD sai 2x do D sai → ảnh hưởng bậc hai so với LR.
3. **Chinchilla 20x là estimate hợp lý cho D.** Ratio* thực có thể 15x hoặc 25x — không đổi LR ranking.

### Sub-phase 1b: muP Transfer Validation (~$9)

**WHAT**: 3 runs ở d12, d18, d20 với best HPs từ 1a (muP auto-scale)

**WHY đây là câu hỏi khoa học novel nhất?**
muP (Tensor Programs V) được thiết kế cho dense transformers. MoE thêm routing dynamics mà muP không account: router weights, expert load balancing, bias updates — tất cả nằm ngoài lý thuyết muP.

**First principle**: Nếu muP không transfer cho MoE, toàn bộ "tune at d16 → deploy at 1B" strategy sụp đổ. Phải verify trước khi đầu tư $200 vào IsoFLOP.

**Tiêu chí kiểm tra:**
- Loss curves ở d12, d18, d20 phải **stable** (không diverge, không NaN)
- Thứ tự loss: d20 < d18 < d12 (nhiều params hơn = loss thấp hơn)
- H_load > 4.0 bits ở mọi depth
- Không depth nào gradient instability mà d16 không thấy

**Nếu muP fail:**
- Một depth diverge hoặc loss/step tệ hơn kỳ vọng
- H_load sụp ở depth lớn hoặc nhỏ
- **Mitigation**: Sweep nhỏ ở depth bị fail. Nếu correction factor nhất quán → incorporate. Nếu random → muP không work cho MoE, phải per-depth sweep (đắt hơn nhưng vẫn rẻ hơn 320 runs).

### Sub-phase 1c (optional): MTP Cost-Benefit (~$3)

**WHAT**: 2 runs × 2000 steps — với MTP và không MTP, cùng compute budget

**WHY đây KHÔNG phải architecture ablation:**
Không hỏi "có nên dùng MTP?" — DeepSeek V3 đã trả lời. Hỏi **compute allocation question**: "Ở nano scale, MTP's ~15% compute overhead có pay for itself không?"

Ở 671B / 14.8T tokens, 15% overhead negligible. Ở 410M / 8.2B tokens, mỗi token quan trọng hơn.

**Decision rule:**
- `delta_bpb > 0.02`: MTP rõ ràng giúp → KEEP (expected)
- `delta_bpb < 0.005`: MTP overhead > benefit ở nano scale → DROP, reclaim compute
- `0.005-0.02`: Borderline → KEEP (trust V3 paper)

### Phase 1 Output
- Best (matrix_lr, embedding_lr). Ví dụ: `(0.01, 0.3)`
- muP transfer: WORKS / NEEDS_CORRECTION / FAILS
- MTP: KEEP / DROP (nếu chạy 1c)

---

## Phase 2: IsoFLOP Sweep — Khám Phá Scaling Law (~$200, 2-3 ngày)

**WHAT**: 20 runs = 5 depths × 4 FLOPs budgets, dùng **tuned HPs từ Phase 1**

```
           |  1e18    3e18    1e19    3e19
-----------+--------------------------------
  d12      |  run_1   run_2   run_3   run_4
  d14      |  run_5   run_6   run_7   run_8
  d16      |  run_9   run_10  run_11  run_12
  d18      |  run_13  run_14  run_15  run_16
  d20      |  run_17  run_18  run_19  run_20
```

### WHY đáng tin cậy hơn plan cũ (không "bet")

Plan cũ chạy IsoFLOP với default HPs và hy vọng chúng đủ tốt — đặt cược $200. Bây giờ Phase 1 đã calibrate HPs:
- **Tuned LRs** → không rủi ro instability ở một số depth
- **muP validated** → confident LR scaling across depth ladder đúng
- **Mọi data point đáng tin** → không cần re-run nếu HPs sai

### HOW — Quy trình phân tích

**Bước 1: Vẽ IsoFLOP curves.** Với mỗi budget FLOPs cố định, plot N_scaling vs val_bpb. Mỗi curve hình chữ U — đáy là model size tối ưu N* cho budget đó.

**Bước 2: Fit Chinchilla power law.**

```
L(N, D) = A/N^α + B/D^β + E
```

Đây là phương trình Chinchilla — nền tảng của mọi scaling law hiện đại:
- **A/N^α**: Loss do model quá nhỏ (underfitting). Model lớn hơn → term này nhỏ hơn.
- **B/D^β**: Loss do data quá ít. Data nhiều hơn → term này nhỏ hơn.
- **E**: Irreducible loss — entropy tối thiểu của ngôn ngữ, không model nào giảm được.
- **α, β**: Tốc độ returns diminish khi tăng N hoặc D.

20 data points (N, D, L) → fit 5 tham số A, B, α, β, E.

**Bước 3: Tìm ratio tối ưu.** Với constraint C = 6ND:

```
N* ∝ C^(β/(α+β))
D* ∝ C^(α/(α+β))
ratio* = D*/N*
```

Nếu α ≈ β → nên tăng N và D đều nhau.
Chinchilla (dense): α ≈ 0.34, β ≈ 0.28 → ratio ≈ 20.
MoE có thể khác vì N_active ≪ N_total.

**Bước 4: MoE-specific analysis.** Kiểm tra H_load theo depth và FLOPs — routing có stable khi scale lên không?

---

## Phase 3: Miniseries — Validate Toàn Hệ Thống (~$120, 1-2 ngày)

**WHAT**: 6 runs ở depths 12, 14, 16, 18, 20, 24. Mỗi model train ở compute-optimal (ratio* từ Phase 2, best HPs từ Phase 1, kiến trúc V3 locked).

### WHY: Validate 4 thứ cùng lúc

**1. muP transfer hoạt động ở full training horizon?**
Phase 1 validated muP ở 500 steps. Đây test ở thousands of steps. Nếu muP break ở horizon dài — ví dụ d24 diverge ở step 5000 nhưng fine ở step 500 — sẽ bắt được ở đây.

**2. Power Lines batch scaling đúng?**
B_opt ∝ D^0.383 (Bergsma et al.). Nếu đúng, mỗi model ở auto-computed batch → comparable loss/step efficiency.

**3. Weight decay scaling đúng?**
T_epoch framework: λ = λ_ref × √(B/B_ref) × (D_ref/D). Giữ regularization constant across scales.

**4. MoE routing stable ở scale?**
- H_load > 4.0 bits ở mọi depth
- Dead experts ≤ 2-3 ở mọi depth
- MTP lambda annealing (0.3→0.1 ở 60% training) lần đầu validate (Phase 1 chỉ 2000 steps, không cover transition)

### Output: MoE Compute-Optimal Frontier
Plot val_bpb vs N_scaling → so sánh với IsoFLOP predictions. Match → hệ thống hoạt động end-to-end.

---

## Phase 4: NanoSeek-1B Graduation (~$350, 12 giờ)

**WHAT**: Train model 1B thực sự trên 8xH100
- hidden=2048, 16 heads, 16 layers
- ~1.08B active / ~4.75B total
- 22B tokens

### WHY chạy với confidence

Mọi quyết định đã được validate:

| Quyết định | Nguồn | Phase |
|------------|--------|-------|
| Kiến trúc | DeepSeek V3 paper (proven ở 671B) | Locked từ đầu |
| Learning rates | HP Search | Phase 1 |
| muP transfer | Validated ở d12, d18, d20 | Phase 1 |
| Tỷ lệ param-data | IsoFLOP → ratio* | Phase 2 |
| Auto-compute cascade | Miniseries → verified ở 6 depths | Phase 3 |

muP tự chuyển HPs từ d16 sang 1B:
```
matrix_lr_1b = matrix_lr_d16 × √(B_1b/B_ref) × (1280/2048)
```

---

## Nguyên Lý Nền Tảng #2: Auto-Compute Cascade — "Single Dial" Philosophy

**WHAT**: Chỉ cần truyền `--depth=14`, hệ thống tự tính mọi thứ:

```
depth=14 → hidden_size = ceil(14×80/128)×128 = 1152
         → num_heads = 1152/128 = 9
         → build model → N_active, N_scaling
         → tokens = ratio* × N_scaling
         → batch = B_REF × (tokens/D_REF)^0.383
         → iterations = tokens / batch
         → muP LR = matrix_lr × √(B/B_ref) × (1280/1152)
         → WD = 0.1 × √(B/B_ref) × (D_ref/tokens)
```

**WHY**: 3 công thức từ 3 papers compose lại:
- **Chinchilla** (Hoffmann 2022): D* = ratio* × N — bao nhiêu data cho model size này
- **Power Lines** (Bergsma 2025): B ∝ D^0.383 — batch size tối ưu
- **muP** (Yang 2022) + T_epoch (2024): LR và WD scale theo width và batch

**First principle**: Derive HPs từ physical laws thay vì brute-force sweep. Giống vật lý — biết hằng số cơ bản → suy ra mọi thứ.

---

## Nguyên Lý Nền Tảng #3: Parameter Counting cho MoE — "Active" vs "Total"

**WHY exclude embeddings?**
Embeddings là bảng tra cứu — O(1) FLOPs/token bất kể vocab size. Transformer weights làm O(1) FLOPs per parameter per token. Chỉ cái sau đóng góp vào L(N, D).

**WHY "active" không phải "total"?**
Mỗi token chỉ route qua 8/64 experts. 56 experts còn lại không đóng góp compute. N_active = total - inactive expert params.

Đây là lý do MoE scaling law khác dense: cùng compute C, MoE có N_total lớn hơn nhiều → nhiều "knowledge capacity" → scaling behavior khác.

---

## Nguyên Lý Nền Tảng #4: muP — Tại Sao Nó Hoạt Động?

muP dựa trên 2 yếu tố scaling composable:

**Factor 1: √(B/B_ref) — Batch scaling (CompletedP)**
- Batch lớn hơn → gradient sạch hơn → có thể bước lớn hơn
- Gradient noise ∝ 1/√B, nên LR ∝ √B

**Factor 2: w_ref/w — Width scaling (Tensor Programs V)**
- Network rộng hơn → mỗi weight đóng góp ít hơn vào activation
- LR ∝ 1/width để giữ ‖Δh‖ = Θ(1) across widths
- Ý nghĩa: cập nhật mỗi hidden state có magnitude không đổi bất kể width

**First principle**: muP đảm bảo "effect of each gradient step" is constant across scales. Tune ở scale nhỏ → transfer sang scale lớn miễn phí. Nhưng MoE routing nằm ngoài lý thuyết muP → cần empirical validation (Phase 1b).

---

## Dependency Graph

```
Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4
  code     tuned      ratio*    validated   trained
  works    HPs +               cascade     model
           muP OK
```

Kiến trúc không phải phase output — nó là **precondition** (locked từ đầu).

**Cải tiến so với plan cũ:** Không "HP/IsoFLOP bet". Phase 1 calibrate HPs rẻ ($30), Phase 2 dùng tuned HPs. Không rủi ro $200 re-run.

---

## Scientific Output — Đóng Góp Khoa Học

### 1. MoE Scaling Laws ở Nano Scale (Phase 2 + 3)
- L(N, D) power law cho MoE — nghiên cứu hệ thống đầu tiên dưới 1B active params với routing diagnostics
- So sánh α_MoE vs α_dense (từ nanochat) → quantify MoE scale hiệu quả hơn bao nhiêu

### 2. muP Transfer cho MoE — Đóng Góp Novel (Phase 1 + 3 + 4)
- muP work cho MoE không? Routing dynamics nằm ngoài lý thuyết muP
- Nếu work: first published evidence. Nếu fail: first published negative result — cũng valuable

### 3. MoE Routing Science (Phase 2 + 3 + 4)
- H_load trajectory qua 22B tokens → routing dynamics trong training
- I_spec: I(Expert; Domain) → expert nào specialize vào domain nào?
- Dead expert analysis → bao nhiêu trong 64 experts thực sự useful?

### 4. MTP Cost-Benefit ở Nano Scale (Phase 1c, optional)
- MTP's ~15% compute overhead pay for itself ở 400M active params không?
- First compute-allocation analysis of MTP ở sub-1B scale

---

## Tổng Chi Phí (Tính Từ FLOPs)

Tổng compute: **7.37e20 FLOPs** → **591 GPU-hours** trên H100 (35% MFU).

| Phase | FLOPs | GPU-hrs | @ $1/hr | @ $2.50/hr |
|-------|-------|---------|---------|------------|
| 0. Gate 1 | 1.4e17 | 0.1 | $0.10 | $0.30 |
| 1. HP Search + muP | 1.2e19 | 10 | $10 | $25 |
| 2. IsoFLOP | 2.2e20 | 176 | $176 | $440 |
| **Research total** | **2.3e20** | **186** | **$186** | **$465** |
| 3. Miniseries | 3.6e20 | 290 | $290 | $725 |
| 4. 1B Graduation | 1.4e20 | 114 | $114 | $286 |
| **Grand total** | **7.4e20** | **591** | **$591** | **$1,476** |

### Tại Sao Chọn H100 80GB?

- H100 bf16 peak: 989 TFLOPS vs A100: 312 TFLOPS (3.2× nhanh hơn)
- MoE cần memory bandwidth cao (64 experts trong memory, chỉ dùng 8) → H100 HBM3 3.35 TB/s thắng lớn
- Cost-efficiency: H100 $2.50/hr ÷ A100 $1.50/hr = 1.7× đắt hơn, nhưng 3.2× nhanh hơn → **1.9× tiết kiệm hơn**

### Memory: Mấy GPU Cho Mỗi Depth?

```
d12-d20: 1× H100 80GB    (N_total 965M → 4.17B, training memory 12-53 GB)
d24:     2× H100 80GB    (N_total 6.65B, training memory 84 GB > 80 GB)
1b:      8× H100 80GB    (fit 1 GPU nhưng cần 8 cho speed: 14 hrs thay vì 114 hrs)
```

### Chiến Lược Thuê GPU

**Tiết kiệm nhất (~$591):** Spot H100 @ $1/hr
```
Phase 0-1:  1× H100    10 hrs     $10      ← Ngày 1
Phase 2:    1× H100    176 hrs    $176     ← 7 ngày (hoặc 2× GPU = 3.5 ngày)
Phase 3:    2× H100    145 hrs    $290     ← 6 ngày
Phase 4:    8× H100    14 hrs     $114     ← Nửa ngày
```

**Ghi chú:** d24 trong Phase 3 chiếm 155 GPU-hrs (53% Phase 3). Nếu scaling law fit tốt từ d12-d20, có thể skip d24 → tiết kiệm ~$155.

---

## Tóm Lại — 4 Nguyên Lý First Principles

1. **Kiến trúc đã proven → không re-ablate** → Lock từ đầu, mọi compute đều inform model cuối cùng
2. **Calibrate trước, đo sau** → HP search trước IsoFLOP → mọi data point đáng tin, không bet
3. **Active vs Total params** → MoE cần parameter counting riêng vì compute per token ≠ total params
4. **Derive từ physical laws** → muP + Power Lines + Chinchilla compose thành auto-compute cascade

Toàn bộ pipeline thiết kế để **eliminate risk**: phase rẻ (Gate 1, HP search) đi trước để catch errors sớm, phase đắt (IsoFLOP, 1B) đi sau khi mọi thứ đã calibrated.
