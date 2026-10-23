# Synthetic panel v1 — bàn giao dữ liệu

Dataset đã sinh: `datasets/synthetic/wave_panel_v1/{easy,medium,hard}` tính từ thư mục repo. Mỗi bộ có 60 thực thể × 120 thời điểm × 12 component, thêm residual riêng tư. Module nằm tại `longitudinal_impute.synthetic`; không thay đổi implementation ba baseline.


## Contract hiện hành sau FIX_TICKET (2026-10-23)

`wave_panel_v1` giữ nguyên như snapshot đối chiếu. `datasets/synthetic/wave_panel_v2/{easy,medium,hard}` là bản xuất mới cùng seed/config/arrays/masks. Format mới `synthetic-panel-v2`, suite `synthetic-suite-v2`; không đổi sampler hoặc missingness.

- NPZ v2 có `row_ids` Unicode shape `(entity,time)`, mỗi key JSON `[entity_id,date]` trỏ trực tiếp đến phần tử cùng index của arrays. Evaluator căn CSV và NPZ bằng key riêng. V1 dùng grid entity-major/date theo `entities.csv`, bắt đầu 2000-01-01; private CSV v1 đổi thứ tự bị từ chối.
- Public loader kiểm format/status, checksum schema/split/conditioning/manifest/hidden mask, dimensions, row IDs, mask identity và schema; chạy được khi không có `private/`. Evaluator kiểm thêm hash hai file private, exact keys/shapes, finite values, conservation, strata, split và consistency CSV/NPZ/public trước khi tạo labels. Manifest là nguồn hash đáng tin trong bundle; hash không phải chữ ký chống sửa đồng thời manifest và dữ liệu.
- Export qua staging cùng filesystem rồi rename nguyên bundle/suite. Spec sai hoặc backend lỗi không để final path nửa chừng; sửa và retry được. Không overwrite output tồn tại.
- `synthetic.scoring.score` báo `count_mae/rmse` so X, `sample_ratio_mae/rmse` so X/N, `latent_p_mae/rmse` so p bằng prediction/N. Chỉ hidden cells, overall và ba strata; nhóm rỗng trả n=0, metric=null. `ratio_*` lịch sử có nghĩa sample ratio.
- Objective chọn candidate: **validation hidden-cell sample_ratio_mae so X/N**, tie theo candidate index; refit train+validation, test chỉ báo cáo. Không retune bằng test/latent p. Sáu prediction gốc được rescore bổ sung tại `docs/synthetic_review_2026-10-23/dual_truth_metrics_v2.json`.
- Residual contract: `sum(named counts) <= N`, `sum(named p) < 1`, `sum(named counts)+residual_counts=N`. Residual p dương không bảo đảm residual count dương. Strict `sum(named counts)<N` mỗi sample **chưa đáp ứng và không được áp dụng**: hard có 178/7200 sample residual count=0. Nếu vẫn cần strict count phải chọn protocol mới: rejection sampling điều kiện residual>=1 đổi phân phối/RNG; reserve một count đổi multinomial moments; lọc sample đổi population/splits. Không chọn phương án nào trong ticket này.

Xem `docs/synthetic_review_2026-10-23/FIX_COMPLETION_VI.md`. Nội dung dưới mô tả snapshot v1 và generator giữ nguyên; số test cũ là lịch sử.

## Hợp đồng dữ liệu

Mỗi thư mục profile là một bundle `synthetic-panel-v1`:

| Artifact | Vai trò |
|---|---|
| `manifest.json` | Phiên bản, cấu hình đầy đủ, seed, hash code/files, diagnostics và capabilities |
| `schema.toml` | Dùng trực tiếp với `baselines.csv_io.schema_from_config` |
| `entities.csv` | Entity-disjoint train/validation/test; dùng chung cả ba profile |
| `conditioning/<mask_name>/conditioning.csv` | Input model: entity_id, date, N, component_01..12, entity_trait_01; ô thiếu để trống |
| `conditioning/<mask_name>/hidden_mask.csv` | Boolean mask 0/1, thứ tự hàng/cột giống CSV; 1 là bị che |
| `conditioning/<mask_name>/manifest.json` | Provenance Schouten, seed, pattern, mask ID và tỷ lệ thực tế |
| `mask_specs/*.json` | Spec dùng để tái tạo mask |
| `private/complete_counts.csv` | Counts đầy đủ, evaluator-only; không dùng làm input model |
| `private/ground_truth.npz` | Latent p, residual p/counts, N, counts, zero strata, timing và split; evaluator-only |

Thứ tự CSV là entity-major rồi thời gian tăng dần. NPZ có axes `(entity,time,component)` cho p/counts/strata; N/residual có axes `(entity,time)`. `p` chỉ gồm 12 component và có tổng bằng `1-residual_p`, không đóng lại thành tổng 1. Counts cộng residual_counts bằng đúng N. Ngày ISO bắt đầu 2000-01-01 là chỉ số thời gian nhân tạo, không có nghĩa case study. Tên mẫu được xác định duy nhất bởi entity/date; loader giữ row_ids theo hợp đồng CSV hiện tại.

Hai mask đã bàn giao:

- `01_mcar_declared_patterns`
- `02_mar_declared_patterns`

Cả hai dùng cùng ba pattern cố định che 4/12 component, frequencies=1/3, case_probability=.3. Tỷ lệ target-cell danh nghĩa là .1, không phải .3. MCAR cùng seed cho cùng mask giữa các profile; MAR có thể khác vì N khác. Không có hiệu chỉnh quota hay lưới rate mới.

## Nạp trực tiếp cho baseline hoặc model mới

```python
from longitudinal_impute.synthetic import load_conditioning, load_evaluation
from longitudinal_impute.contracts import FitScope
from longitudinal_impute.baselines.adapters import CountBaselineAdapter
from longitudinal_impute.baselines.hron import HronKNN

bundle = "datasets/synthetic/wave_panel_v1/easy"
mask = "01_mcar_declared_patterns"
train = load_conditioning(bundle, mask, split="train")
test = load_conditioning(bundle, mask, split="test")
scope = FitScope("inductive", train.row_ids, "wave_panel_v1_entities", train.mask_id)
model = CountBaselineAdapter(HronKNN(k=3)).fit(train, scope)
prediction = model.predict(test)

# Chỉ evaluator chạy phần dưới; không truyền latent/labels cho model.
labels, latent = load_evaluation(bundle, mask, split="test")
assert labels.row_ids == test.row_ids
hidden = labels.target_mask
# So counts: prediction[hidden] với labels.values[hidden].
# So latent: (prediction / test.totals[:, None])[hidden] với latent["p"][hidden].
# Phân tầng theo latent["strata"]: 0=true zero, 1=sampling zero, 2=positive count.
```

`load_conditioning` trả đúng `ConditioningPanel` hiện có: counts là NaN tại ô thiếu, mask boolean, N/entity/time và trait luôn quan sát. Loader này kiểm hash file public và không mở `private/`; test đã xác nhận vẫn chạy khi chuyển toàn bộ private đi nơi khác. Đây là ranh giới API, không phải cơ chế phân quyền hệ điều hành. Model mới có thể dùng cùng panel hoặc group theo entity/time để tạo tensor. Fit preprocessing chỉ trên train; validation/test không thành donor train.

`load_evaluation` trả `EvaluationLabels` của counts bị che cùng dictionary `p`, `strata`, `residual_p`, `residual_counts`, `N`, căn hàng theo panel của split được chọn. Dùng mask khi tính metric; không trộn ô quan sát vào score điền khuyết. Manifest ghi support mỗi stratum tổng thể và trong ô bị che, theo split. Scorer dual-truth hiện có như contract trên; chưa chạy so sánh đầy đủ mọi profile/scale/seed.

## Các thông số đã chốt

| Profile | SD lệch thời điểm | SD log dominance | SD nhiễu thời gian | E[N] | CV(N) mục tiêu |
|---|---:|---:|---:|---:|---:|
| easy | 1 | .10 | .05 | 2000 | .30 |
| medium | 4 | .35 | .15 | 500 | .60 |
| hard | 8 | .70 | .35 | 100 | 1.00 |

Scale small/standard/large lần lượt 30/60/120 thực thể; luôn 120 thời điểm và 12 component. Residual có latent mean .10, SD .03; counts dùng multinomial trên 13 phần. Ground truth phân biệt chính xác p=0 và p>0 nhưng count=0.

## Giả định triển khai v1 — không gán cho paper

Các phương trình dưới là thiết kế generator benchmark để cụ thể hóa yêu cầu; Schouten chỉ là nguồn cho missingness MCAR/MAR. Đây không phải một mô hình sinh dữ liệu đã được paper đó chứng minh.

Cho e,t,j là thực thể, thời gian và component. Peak chung `c_j=linspace(0,119,12)`. Dịch thời gian thực thể `a_e ~ Normal(0, offset_sd²)`, peak `c_ej=c_j+a_e`; onset `o_ej=c_ej-16`. Onset component đầu được chặn trên tại 0 để luôn có ít nhất một phần hoạt động. Profile dùng chung schedule.

Trọng số trước khi normalize:

`w_etj = I(t >= o_ej) * [exp(-0.5*((t-c_ej)/8)²) + 1e-4*I(t>=c_ej)] * exp(b_ej + u_etj)`.

`b_ej ~ Normal(0, dominance_sd²)`. `u` là AR(1) dừng với correlation .8, marginal SD bằng noise_sd của profile; khởi tạo từ phân phối dừng. Trước onset, p bằng đúng 0; sau peak có tail dương. Các peak nối tiếp tạo thay thế ưu thế nhưng không ép một component luôn thắng.

Residual `r_et ~ Beta(m*k,(1-m)*k)` độc lập qua entity/time, với `k=m*(1-m)/s²-1`, m=.10, s=.03. `p_etj=(1-r_et)*w_etj/sum_j(w_etj)`. Count residual có thể bằng 0 do lấy mẫu; không ép nó dương để làm sai sampler.

Cho `sigma²=log(1+CV²)`, `z_e ~ Normal(0,1)` và `v_et` là AR(1) dừng correlation .8, variance 1, độc lập. `N_et=max(1, round(mean_N*exp(sigma*(z_e+v_et)/sqrt(2)-sigma²/2)))`. Variance log được chia đều cho mức thực thể và thời gian, không áp hai lần CV. Mean/CV là moment mục tiêu của phân phối trước làm tròn, không phải quota của một realization hữu hạn.

Trait là một giá trị Normal(0,1) độc lập cho mỗi thực thể, giữ nguyên theo thời gian. Split v1 dùng 60/20/20 theo hoán vị entity từ stream seed riêng (36/12/12 ở scale chuẩn). Đây là lựa chọn triển khai cho tỷ lệ chia trước đó chưa chốt. Không triển khai protocol secondary trong thực thể đã biết.

`SeedSequence(seed)` tách 9 stream cho timing, dominance, noise, residual, N-entity, N-time, multinomial, trait, split. Dùng cùng seed giữa các profile để ghép các nguồn ngẫu nhiên nền; residual/trait/split giống hệt nhau. Nhãn easy/medium/hard là cấu hình, chưa phải kết luận về thứ hạng sai số. Không dùng kết quả baseline để chọn seed/profile.

Các giá trị chưa chốt trước đây (AR .8, width 8, onset lead 16, tail 1e-4, beta residual, variance split và split fraction) đều được công khai ở đây; các giá trị cấu hình số nằm trong `GeneratorConfig`/manifest. Có thể override qua Python API rồi xuất thành bundle mới, không sửa dataset đã bàn giao.

## Tái tạo và sensitivity

Chạy từ repo với PYTHONPATH/RSCRIPT như `SCHOUTEN_VI.md`:

```powershell
& '.\.venv-e3-revised\Scripts\python.exe' -m longitudinal_impute.synthetic generate --output datasets\synthetic\wave_panel_reproduction --profile all --scale standard --seed 2026 --mask-spec nextgen\configs\schouten2018_mcar_patterns.json --mask-spec nextgen\configs\schouten2018_mar_patterns.json
```

Đổi `--scale small` hoặc `large` để kiểm độ nhạy; dùng output mới. API `export_dataset(path, GeneratorConfig(...), mask_specs=[...])` cho phép cấu hình chi tiết. Fixed-wave dùng spec đã hỗ trợ trong `SCHOUTEN_VI.md`; ví dụ wave_time="2000-03-01" trên lưới này. Không có spec không có mask: bundle chỉ chứa truth/schema/split để tác vụ tạo mask tách riêng; bản bàn giao hiện tại có đủ hai mask.

## Kiểm tra và diagnostics bản chuẩn seed 2026

| Profile | N mean thực tế | N CV thực tế | True zero cells | Sampling zero cells | Positive cells |
|---|---:|---:|---:|---:|---:|
| easy | 2059.97 | .304 | 32883 | 22218 | 31299 |
| medium | 529.60 | .619 | 32880 | 25696 | 27824 |
| hard | 109.60 | 1.049 | 32964 | 29543 | 23893 |

Residual mean thực tế .09942, SD .03006, giống nhau giữa ba profile. Mỗi hàng bảng có 86400 target cells. Không sửa mẫu để ép moment/zero rate về mục tiêu.

Test mới kiểm tính tái lập và khác seed, exact zero trước onset, tail dương sau onset, tổng p/counts, ba zero strata, scale, split, hash bundle, nạp không cần private và lời gọi fit/predict của cả Hron/MICoDa/Tsagris. Smoke test model chỉ kiểm hợp đồng input/output; MICoDa dùng 3 iterations để kiểm tương thích, không phải kết quả hội tụ hoặc so sánh hiệu quả.

Kết quả cuối: **128 tests passed trong 19.49 giây**. CLI generate đã kiểm thêm trên hard/small; wheel build thành công và có đủ module generator/loaders cùng nguồn R. Audit bản bàn giao kiểm mọi file hash, bảo toàn tổng và căn hàng model/evaluator trên cả ba profile và hai mask; kết quả lưu tại `datasets/synthetic/wave_panel_v1/verification.json`.

MNAR latent-p, random temporal blocks, whole-component trajectories, beta strength grid, full independent cell-rate grid, exact realized rates và paper inference tables tiếp tục NOT-READY. Không có phương pháp missingness khác được thêm.
