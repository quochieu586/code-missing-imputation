# Ticket 03 — nguồn, metric và phạm vi

## Source map

| Trang | Claim / phép tính | Nguồn kiểm |
|---|---|---|
|21|Complete fit donors + IDs; query riêng; TsagrisKNN; validation k/α|nextgen/src/longitudinal_impute/baselines/tsagris.py, interfaces.py, registry.py; saved run manifests/selection|
|22|Closure cùng anchors, donor mass>0, top-k; natural log, 2×JSD chuẩn, không sqrt|tsagris.py: jsd, TsagrisKNN.predict_result; paper Eq.(5) DOI10.1080/02664763.2026.2677908|
|23|Row-normalized powers → average → inverse/closure; α=1 arithmetic; α=0 support dương chung khác rỗng|tsagris.py: alpha_mean; paper Eq.(6). Source lưu ý chỉ số mẫu số paper không nhất quán, không dùng nhận xét này làm kết luận slide.|
|24|Core R=1−visible mass; missing weights; zero R/zero weights policies; counts lock/budget|tsagris.py: predict_result; adapters.py: predict_result; panels.py: extended_proportions; constraints.py: CountBudgetConstraint|
|25–26|Pooled hidden-target ratio MAE×100; 12 estimate/low/high; 5folds/3maskseeds/150locations|data/primary_metrics_source.csv byte-identical docs/benchmark_offline_2026-10-15/primary_metrics.csv; accepted report_vi.md; accepted run offline_baselines_2026-10-15_v2|
|27|Location-macro paired MAE difference×100; all3 whole-feature contrasts|data/paired_location_differences_source.csv byte-identical accepted paired_location_differences.csv; not pooled subtraction|
|28|18 saved results, X/N vs p MAE×100; 3profiles×2masks×3models|experiments/synthetic_baselines_v2_2026-10-23/plots/test_metrics.csv, plot_eda.R; per-profile summary.json/run_manifest.json|
|29|Fixed illustrative subset hard/MCAR; latent-p MAE×100; n637/576/467; sum1680|experiments/synthetic_hard_baselines_v2_2026-10-23/summary.json: results[MCAR].results[*].test.by_stratum|
|30|Variants, validity vs accuracy, limitations|Implementation sources, accepted report, synthetic v2 manifests/README_vi.md|

## Công thức và giới hạn

D_code=KL(q||m)+KL(v||m)=2JSD chuẩn, log tự nhiên. Terms zero dùng0; midpoint zero chỉ khi cả hai zero. Zero-anchor global dùng toàn donor bank khi policy bật, không lấy top-k ngẫu nhiên. α<0 cần donor dương; α>0 cho phép zero. α=0 chỉ geometric trên common-positive support, báo lỗi nếu support rỗng.

Core proposal trở lại adapter trước chấm. Other là phần tổng N ngoài named components. Core residual R khác count budget B=N−sum observed counts và residual truth riêng của evaluator. Zero-missing-mass uniform và fallback train_mean là adaptation đã khai báo. Không khẳng định full-paper reproduction hoặc dùng TsagrisAdaptive làm primary.

COVID CI là location-bootstrap, fixed fits, 150locations, chưa hiệu chỉnh đa so sánh. Không p-value/stars. Targets tính qua mask repetitions, không là quan sát độc lập. Native missing không có nhãn. MICoDA COVID5/5folds chạm50iterations; không gọi là hội tụ.

Synthetic standard, seed2026, metric dual-truth-v2, declared MCAR/MAR, 1seed/profile. Không MNAR/repeated-seed inference trong artifacts đã kiểm. Chỉ mô tả run. Không suy nguyên nhân lỗi MICoDA hoặc hội tụ từ MAE. Count MAE phụ thuộc N (preset mean2000/500/100), không đọc count thấp hơn là hard dễ hơn.

## Plot provenance / lệnh

Chạy từ repo:
```powershell
py presentation/implementation_report/scripts/build_figures.py
py presentation/implementation_report/scripts/build_ticket03.py
```
Script thứ hai bố trí lại scenario panels, xuất synthetic từ saved CSV/summaries và kiểm18 endpoint triples với sai số<2e-9pp. x-plus và x-minus là hai cột sai số riêng theo cú pháp PGFPlots `+= (...) -= (...)`. Giữ estimate/low/high accepted. Jitter ngang±0.1 chỉ tách marker theo model ở trục categorical synthetic; không thay giá trị y. Không nối line qua profile/khôngCI synthetic.

Manifest/hash: data/ticket03_figure_manifest.json; CI comparison: .harness/ticket-03-frames-21-30-2026-10-23/ci_endpoint_comparison.json. Counts/RMSE và selected parameters đầy đủ: data/ticket03_synthetic_full_metrics.json. Toàn bộ54 support/metric strata: data/ticket03_synthetic_support.json. Empty support giữ null và không vẽ thành0.

## Metric synthetic đầy đủ (hidden test cells)

|Profile|Mask|Model|n|Count MAE|Count RMSE|X/N MAE pp|p MAE pp|p RMSE pp|Selected params|
|---|---|---|---:|---:|---:|---:|---:|---:|---|
|easy|MCAR|hron_knn|1680|12.493349|31.055709|0.626880|0.609235|1.504878|{'k': 10}|
|easy|MCAR|micoda_v2|1680|151.135992|362.259890|7.420998|7.405202|16.777093|{'W': 1e-06, 'max_iter': 50}|
|easy|MCAR|tsagris_alpha|1680|12.169494|29.673174|0.601851|0.599159|1.438447|{'k': 10, 'alpha': 1.0}|
|easy|MAR|hron_knn|1568|11.660371|28.737237|0.687208|0.659672|1.573232|{'k': 10}|
|easy|MAR|micoda_v2|1568|127.151228|291.366945|7.497759|7.472772|16.576500|{'W': 1e-06, 'max_iter': 50}|
|easy|MAR|tsagris_alpha|1568|10.695893|25.954442|0.631817|0.603028|1.443032|{'k': 10, 'alpha': 1.0}|
|medium|MCAR|hron_knn|1680|4.397744|12.960000|0.899414|0.882018|2.171632|{'k': 10}|
|medium|MCAR|micoda_v2|1680|39.817648|111.914575|7.492763|7.501310|17.261337|{'W': 1e-06, 'max_iter': 50}|
|medium|MCAR|tsagris_alpha|1680|5.563630|15.323255|1.074295|1.001647|2.345845|{'k': 10, 'alpha': 1.0}|
|medium|MAR|hron_knn|1576|3.824821|9.813390|1.080180|1.077970|2.702574|{'k': 10}|
|medium|MAR|micoda_v2|1576|27.603852|70.396670|7.412921|7.410873|16.768612|{'W': 1e-06, 'max_iter': 50}|
|medium|MAR|tsagris_alpha|1576|4.636294|12.029083|1.254149|1.226216|3.007018|{'k': 5, 'alpha': 1.0}|
|hard|MCAR|hron_knn|1680|1.542539|5.146161|1.654746|1.886646|4.709779|{'k': 10}|
|hard|MCAR|micoda_v2|1680|8.681676|31.996618|7.730843|7.640323|18.324432|{'W': 1e-06, 'max_iter': 50}|
|hard|MCAR|tsagris_alpha|1680|2.488940|9.868701|2.384386|2.469475|6.341107|{'k': 10, 'alpha': 0.5}|
|hard|MAR|hron_knn|1472|1.165079|3.400299|2.054596|2.306618|5.894364|{'k': 10}|
|hard|MAR|micoda_v2|1472|5.148170|15.501325|7.623635|7.599942|18.002806|{'W': 1e-06, 'max_iter': 50}|
|hard|MAR|tsagris_alpha|1472|1.466781|4.829605|2.458606|2.440791|6.067959|{'k': 10, 'alpha': 0.5}|

## Strata support (giống nhau giữa các model trong mỗi profile/mask)

|Profile|Mask|Stratum|n|
|---|---|---|---:|
|easy|MCAR|true_zero|629|
|easy|MCAR|sampling_zero|457|
|easy|MCAR|positive_count|594|
|easy|MAR|true_zero|588|
|easy|MAR|sampling_zero|402|
|easy|MAR|positive_count|578|
|medium|MCAR|true_zero|634|
|medium|MCAR|sampling_zero|502|
|medium|MCAR|positive_count|544|
|medium|MAR|true_zero|599|
|medium|MAR|sampling_zero|475|
|medium|MAR|positive_count|502|
|hard|MCAR|true_zero|637|
|hard|MCAR|sampling_zero|576|
|hard|MCAR|positive_count|467|
|hard|MAR|true_zero|557|
|hard|MAR|sampling_zero|512|
|hard|MAR|positive_count|403|
