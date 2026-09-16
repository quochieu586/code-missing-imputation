# Pipeline điền khuyết hai tầng cho dữ liệu Compositional

## Tầng A: kNN-Aitchison (khởi tạo) → Tầng B: Vòng lặp điền khuyết với lõi Diffusion (CSDI)

> **Trạng thái:** Đã có Tầng A, cầu nối CLR, CSDI thích ứng chạy **một lượt** và runner đánh giá pilot theo địa điểm. Các cổng tái lập bài báo, ablation E1 và đánh giá đầy đủ vẫn chưa hoàn tất.

## Chạy CLR + diffusion một lượt

```powershell
& 'C:/Users/admin/miniconda3/python.exe' -m pytest tests/unit/core tests/unit/stage_b -q
& 'C:/Users/admin/miniconda3/python.exe' scripts/run_stage_b.py --epochs 100 --samples 5
```

Runner dùng `data/covariants.csv` và tiếp nối `data/processed/X1_knn_imputed.csv` cho đầu ra đầy đủ. Nhánh đánh giá dựng lại KNN sau khi che validation/test, chỉ dùng láng giềng và fallback thuộc train. Không dùng artifact toàn bộ dữ liệu để tính metric held-out.

Kết quả ở `artifacts/stage_b_single_pass_final/`: `X2_diffusion_imputed.csv`, `Z1_clr.csv`, `Z2_clr.csv`, checkpoint, masks và dự đoán audit, `metrics.csv`, `results.json`. Báo cáo: `reports/stage_b_single_pass_report.md`. Đây là một fold tại MCAR 50%, không phải bảng mean (SD) của 5-fold. Không kết luận hội tụ (`n_iter=1`, `converged=False`). 50 bước khử nhiễu DDPM và 100 epochs huấn luyện nằm **trong một lượt**, không phải vòng lặp điền lại dữ liệu.

`artifacts/stage_b_single_pass/` lưu lượt thử bị loại vì lỗi triển khai objective: chiếu tensor noisy bị mask về CLR rồi quy đổi ngược epsilon làm sai loss, dù denoiser dự đoán epsilon chính xác. Regression test `test_perfect_epsilon_has_zero_training_loss` kiểm tra lỗi này. Loss cuối dùng đúng epsilon MSE; epsilon Gaussian không phải CLR nên không được trung tâm hóa. Đầu ra CLR của sampling và đầu ra dùng chấm metric luôn được trung tâm hóa.

Lõi tham khảo `metag_time_impute_phylo/CSDI_phylo/{main_model,diff_models}.py`: epsilon prediction, lịch beta quadratic, DDPM reverse, transformer thời gian/đặc trưng. Model pilot nhỏ hơn cấu hình bài báo; không tuyên bố tái lập CSDI gốc. Dữ liệu biến thể virus không có phylum map, nên lượt chạy không bật phylum CNN. `Denoiser` có nhánh CNN tùy chọn cho nhóm do người dùng cung cấp; `feature_order` cung cấp thứ tự Spearman train-only trước khi dùng nhánh đó.

R5/T12 yêu cầu giữ nguyên counts quan sát, trong khi softmax ở §4.3 chỉ trả về tỷ lệ. Code biến đổi ngược, khôi phục scale theo median log-ratio quan sát, rồi khóa counts quan sát; CLR dùng đánh giá được tính lại từ kết quả cuối. Đây là cách giải quyết mâu thuẫn đó, không khẳng định rằng softmax tự giữ nguyên counts.

---

## Kiến trúc Pipeline

```
X₀ (thô, thưa theo ô, có zero, có NaN)
 │
 ├── TIỀN XỬ LÝ (src/preprocessing)
 │     lọc <0.01% → 0; pseudo-count; mask M₀
 │
 ├── TẦNG A (src/stage_a)
 │     kNN-Aitchison, chiến lược 2a → ma trận đầy đủ sơ cấp
 │
 ├── CẦU NỐI (src/bridge)
 │     clr transform
 │
 ├── TẦNG B (src/stage_b)
 │     Vòng lặp ngoài: split M₀ → train ε_θ (CSDI + phylum CNN) → sample → điền
 │
 ├── ĐÁNH GIÁ (src/evaluation)
 │     M1 (MAE clr), M2 (compositional error variance), M3 (covariance diff)
 │
 └── X̂ (đầy đủ)
```

## Cấu trúc thư mục

```
code-missing-imputation/
├── directive/              # Tài liệu hướng dẫn & papers gốc (READ-ONLY)
├── src/
│   ├── core/               # Nền tảng: clr, d_A, ilr, subcomposition (§1)
│   ├── preprocessing/      # Tiền xử lý: pseudo-count, mask M₀, sắp OTU (§2)
│   ├── stage_a/            # Tầng A: kNN-Aitchison, chiến lược 2a (§3)
│   ├── bridge/             # Cầu nối A→B: clr transform/inverse (§4)
│   ├── stage_b/
│   │   ├── csdi/           # Lõi: CSDI conditional diffusion (§5.1)
│   │   ├── phylum_cnn/     # Phylogenetic CNN branches
│   │   └── loop/           # Vòng lặp ngoài + 5 quy tắc (§5.2)
│   ├── evaluation/         # Thước đo: M1, M2, M3 + bio checks (§6)
│   ├── baselines/          # Mean, LOCF, Linear, kNN+LTS(ilr) (§7.2)
│   └── pipeline/           # Orchestrator tổng thể
├── configs/                # YAML configs cho data, model, training
├── data/
│   ├── raw/                # Dữ liệu gốc
│   ├── processed/          # Dữ liệu đã tiền xử lý
│   └── splits/             # Train/val/test splits theo subject
├── tests/
│   ├── unit/               # Unit tests theo module
│   │   ├── core/           # T1-T6: clr invariance, d_A, roundtrip
│   │   ├── preprocessing/  # Mask contract tests
│   │   ├── stage_a/        # T7-T10: scale invariance, Hron replication
│   │   ├── stage_b/        # T11-T17: loss mask, no-overwrite, drift
│   │   └── evaluation/     # Metric correctness
│   ├── integration/        # End-to-end pipeline tests
│   ├── acceptance/         # T18-T19: reproduce [S] Table 1, baseline #5
│   └── fixtures/           # Test data (Aitchison 1986, synthetic)
├── experiments/            # E1 (Tầng A value), E2 (Loop value), full grid
├── notebooks/              # Exploration & visualization
├── reports/                # Kết quả thí nghiệm
├── artifacts/              # Model checkpoints, predictions
└── scripts/                # Entry points: train, evaluate, run_pipeline
```

## Thứ tự triển khai (từ directive)

| GĐ | Nội dung | Tests |
|---|---|---|
| 0 | Nền tảng: `src/core/` — clr, d_A, subcomposition | T1–T6 |
| 1 | Tiền xử lý: `src/preprocessing/` — mask M₀, pseudo-count | T13, T15 |
| 2 | Tầng A: `src/stage_a/` — kNN-Aitchison | T7–T10 |
| 3 | Tái lập [S]: CSDI một lượt | T18 |
| 4 | **E1** — Tầng A có đáng không? | Quyết định GIỮ/BỎ |
| 5 | Khung vòng lặp: `src/stage_b/loop/` | T11, T12, T16, T17 |
| 6 | **E2** — Vòng lặp có đáng không? (H1) | Quyết định GIỮ/BỎ |
| 7 | Lưới đầy đủ, 5-fold CV | Bảng so sánh |
| 8 | Downstream evaluation | ROC-AUC, PR-AUC |
| 9 | Metadata FT (chỉ nếu GĐ 6 = GIỮ) | |

## Nguồn tham chiếu

- **[H]** Hron, Templ, Filzmoser (2009) — Hình học Aitchison, Tầng A, khung vòng lặp
- **[S]** Seki, Zhang, Imoto (2025) — Lõi CSDI + phylum CNN, baseline, mốc số liệu
- **[D]** Sơ đồ nội bộ — Cấu trúc vòng lặp ngoài Tầng B

Xem chi tiết trong `directive/`.

Lượt trung gian rtifacts/stage_b_single_pass_corrected/ có checkpoint epsilon-MSE hợp lệ nhưng output sampling bị loại: tọa độ quan sát phải khôi phục từ conditioning trước CLR inverse. Lượt stage_b_single_pass_final lấy mẫu lại từ chính checkpoint đó, không train thêm và không dùng X2 làm khởi tạo mới.
