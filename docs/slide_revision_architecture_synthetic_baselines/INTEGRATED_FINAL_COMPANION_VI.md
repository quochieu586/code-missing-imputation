# Companion tích hợp — 30 trang chính + 2 trang tham khảo

## Quy ước trích dẫn

Ký hiệu [1]–[4] nhảy tới tài liệu phương pháp ở trang31. [5]–[10] nhảy tới hàng tương ứng trong bảng source/file–vai trò trang32. Mỗi nguồn dự án gom các file cùng trách nhiệm. Không đặt tên file/đường dẫn trong30slide chính. Đơn vị, CI và giới hạn khoa học vẫn hiện trên slide liên quan.

## Tài liệu phương pháp (IEEE-style)

[1] K. Hron, M. Templ, and P. Filzmoser, “Imputation of missing values for compositional data using classical and robust methods,” Computational Statistics & Data Analysis, vol.54,no.12,pp.3095–3107,2010. [DOI](https://doi.org/10.1016/j.csda.2009.11.023). Metadata kiểm từ [nguồn gốc](https://is.muni.cz/do/rect/habilitace/1431/Hron/habilitace/8_Hron_et_al__2010_.pdf).

[2] A. Saha, D. L. Putnick, H. Lin, E. Yeung, R. Sundaram, and S. D. Peddada, “Multiple Imputation for Compositional Data (MICoDa) Adjusting for Covariates,” in Statistical Methods at the Forefront of Biomedical Advances, Y.Larriba,Ed. Springer,2023,pp.157–184. [DOI](https://doi.org/10.1007/978-3-031-32729-2_8). Metadata kiểm từ [nguồn gốc](https://link.springer.com/chapter/10.1007/978-3-031-32729-2_8).

[3] M. Tsagris, C. Stewart, and A. Alenazi, “A Jensen-Shannon divergence based k–NN algorithm for missing value imputation in compositional data,” Journal of Applied Statistics,2026. [DOI](https://doi.org/10.1080/02664763.2026.2677908). Metadata kiểm từ [nguồn gốc](https://arxiv.org/abs/2605.29702).

[4] R. M. Schouten, P. Lugtig, and G. Vink, “Generating missing values for simulation purposes: a multivariate amputation procedure,” Journal of Statistical Computation and Simulation,vol.88,no.15,pp.2909–2930,2018. [DOI](https://doi.org/10.1080/00949655.2018.1491577). Metadata kiểm từ [nguồn gốc](https://research.tue.nl/en/publications/generating-missing-values-for-simulation-purposes-a-multivariate-/).

Thứ tự tác giả Tsagris đã đối chiếu trang arXiv và DOI liên quan, sửa thứ tự không nhất quán của section legacy. Không tự điền volume/pages chưa xác minh. Wave generator là thiết kế dự án [6], Schouten [4] là nguồn missingness. Không có tuyên bố Q1/Q2 hoặc tái lập toàn bộ paper.

## Danh mục file đầy đủ và vai trò

|Ref|File|Vai trò|
|---|---|---|
|[5]|[nextgen/src/longitudinal_impute/contracts.py](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/contracts.py)|Dữ liệu conditioning, labels và FitScope|
|[5]|[nextgen/src/longitudinal_impute/benchmark/engine.py](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/benchmark/engine.py)|Orchestration COVID, fit/select/test|
|[5]|[nextgen/src/longitudinal_impute/benchmark/protocol.py](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/benchmark/protocol.py)|Split theo location|
|[5]|[nextgen/src/longitudinal_impute/baselines/interfaces.py](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/baselines/interfaces.py)|Hợp đồng NumericalCore|
|[5]|[nextgen/src/longitudinal_impute/baselines/registry.py](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/baselines/registry.py)|9model keys và factory/policies|
|[5]|[nextgen/src/longitudinal_impute/baselines/adapters.py](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/baselines/adapters.py)|Đổi counts/tỷ lệ, fallback, constraints|
|[5]|[nextgen/src/longitudinal_impute/baselines/panels.py](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/baselines/panels.py)|Counts/N, Other và fit subset|
|[5]|[nextgen/src/longitudinal_impute/constraints.py](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/constraints.py)|Observed lock và named-count budget|
|[5]|[nextgen/src/longitudinal_impute/metadata.py](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/metadata.py)|Fit/transform metadata|
|[6]|[nextgen/src/longitudinal_impute/synthetic/generator.py](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/synthetic/generator.py)|Wave weights, latent p, multinomial counts|
|[6]|[nextgen/src/longitudinal_impute/synthetic/cli.py](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/synthetic/cli.py)|CLI profile/scale/seed|
|[6]|[nextgen/src/longitudinal_impute/synthetic/service.py](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/synthetic/service.py)|MaskSpec và export/apply service|
|[6]|[nextgen/src/longitudinal_impute/synthetic/masking.py](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/synthetic/masking.py)|Declared mask/fixed-wave capability|
|[6]|[nextgen/src/longitudinal_impute/synthetic/capabilities.py](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/synthetic/capabilities.py)|Khả năng và nhánh chưa ready|
|[6]|[nextgen/src/longitudinal_impute/synthetic/datasets.py](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/synthetic/datasets.py)|Loader public conditioning/private evaluation|
|[6]|[nextgen/src/longitudinal_impute/synthetic/scoring.py](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/synthetic/scoring.py)|Count, sample-ratio, latent-p metrics|
|[6]|[nextgen/SYNTHETIC_DATA_VI.md](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/SYNTHETIC_DATA_VI.md)|Contract synthetic v2|
|[6]|[nextgen/SCHOUTEN_VI.md](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/SCHOUTEN_VI.md)|Scope missingness từ Schouten|
|[6]|[docs/synthetic_missingness_spec_vi.md](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/docs/synthetic_missingness_spec_vi.md)|Thiết kế benchmark dự án|
|[6]|[datasets/synthetic/wave_panel_v2/easy/manifest.json](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/datasets/synthetic/wave_panel_v2/easy/manifest.json)|Bundleeasy: cấu hình/entity split/truth đã lưu|
|[6]|[datasets/synthetic/wave_panel_v2/easy/entities.csv](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/datasets/synthetic/wave_panel_v2/easy/entities.csv)|Bundleeasy: cấu hình/entity split/truth đã lưu|
|[6]|[datasets/synthetic/wave_panel_v2/easy/private/ground_truth.npz](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/datasets/synthetic/wave_panel_v2/easy/private/ground_truth.npz)|Bundleeasy: cấu hình/entity split/truth đã lưu|
|[6]|[datasets/synthetic/wave_panel_v2/medium/manifest.json](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/datasets/synthetic/wave_panel_v2/medium/manifest.json)|Bundlemedium: cấu hình/entity split/truth đã lưu|
|[6]|[datasets/synthetic/wave_panel_v2/medium/entities.csv](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/datasets/synthetic/wave_panel_v2/medium/entities.csv)|Bundlemedium: cấu hình/entity split/truth đã lưu|
|[6]|[datasets/synthetic/wave_panel_v2/medium/private/ground_truth.npz](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/datasets/synthetic/wave_panel_v2/medium/private/ground_truth.npz)|Bundlemedium: cấu hình/entity split/truth đã lưu|
|[6]|[datasets/synthetic/wave_panel_v2/hard/manifest.json](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/datasets/synthetic/wave_panel_v2/hard/manifest.json)|Bundlehard: cấu hình/entity split/truth đã lưu|
|[6]|[datasets/synthetic/wave_panel_v2/hard/entities.csv](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/datasets/synthetic/wave_panel_v2/hard/entities.csv)|Bundlehard: cấu hình/entity split/truth đã lưu|
|[6]|[datasets/synthetic/wave_panel_v2/hard/private/ground_truth.npz](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/datasets/synthetic/wave_panel_v2/hard/private/ground_truth.npz)|Bundlehard: cấu hình/entity split/truth đã lưu|
|[7]|[nextgen/src/longitudinal_impute/baselines/hron.py](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/baselines/hron.py)|Available-donor CLR, median-scale/median proposal|
|[7]|[nextgen/src/longitudinal_impute/baselines/micoda.py](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/baselines/micoda.py)|ALR/OLS, missing allocation, fit loop|
|[7]|[nextgen/src/longitudinal_impute/baselines/tsagris.py](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/baselines/tsagris.py)|JSD doubled, alpha_mean, allocation, donor state|
|[7]|[nextgen/src/longitudinal_impute/baselines/common.py](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/baselines/common.py)|Closure/CLR/design matrix và helpers|
|[8]|[experiments/offline_baselines_2026-10-15_v2/manifest.json](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/experiments/offline_baselines_2026-10-15_v2/manifest.json)|Accepted protocol,5folds/3maskseeds/150locations|
|[8]|[docs/benchmark_offline_2026-10-15/primary_metrics.csv](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/docs/benchmark_offline_2026-10-15/primary_metrics.csv)|Pooled ratio estimates và CI endpoints|
|[8]|[docs/benchmark_offline_2026-10-15/paired_location_differences.csv](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/docs/benchmark_offline_2026-10-15/paired_location_differences.csv)|Paired location-macro estimand và CI|
|[8]|[docs/benchmark_offline_2026-10-15/selected_parameters.csv](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/docs/benchmark_offline_2026-10-15/selected_parameters.csv)|Tham số đã chọn theo folds|
|[8]|[docs/benchmark_offline_2026-10-15/report_vi.md](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/docs/benchmark_offline_2026-10-15/report_vi.md)|Tổng hợp accepted, native-missing/convergence caveats|
|[9]|[scripts/run_synthetic_easy_baselines.py](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/scripts/run_synthetic_easy_baselines.py)|Runner dùng chung profiles; validation selection/refit/test|
|[9]|[experiments/synthetic_baselines_v2_2026-10-23/plots/test_metrics.csv](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/experiments/synthetic_baselines_v2_2026-10-23/plots/test_metrics.csv)|18result rows dùng vẽ|
|[9]|[experiments/synthetic_baselines_v2_2026-10-23/plots/README_vi.md](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/experiments/synthetic_baselines_v2_2026-10-23/plots/README_vi.md)|Run/metric scope và giới hạn|
|[9]|[experiments/synthetic_easy_baselines_v2_2026-10-23/run_manifest.json](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/experiments/synthetic_easy_baselines_v2_2026-10-23/run_manifest.json)|Runeasy: seed/protocol,selected parameters,hidden support và metrics|
|[9]|[experiments/synthetic_easy_baselines_v2_2026-10-23/summary.json](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/experiments/synthetic_easy_baselines_v2_2026-10-23/summary.json)|Runeasy: seed/protocol,selected parameters,hidden support và metrics|
|[9]|[experiments/synthetic_medium_baselines_v2_2026-10-23/run_manifest.json](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/experiments/synthetic_medium_baselines_v2_2026-10-23/run_manifest.json)|Runmedium: seed/protocol,selected parameters,hidden support và metrics|
|[9]|[experiments/synthetic_medium_baselines_v2_2026-10-23/summary.json](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/experiments/synthetic_medium_baselines_v2_2026-10-23/summary.json)|Runmedium: seed/protocol,selected parameters,hidden support và metrics|
|[9]|[experiments/synthetic_hard_baselines_v2_2026-10-23/run_manifest.json](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/experiments/synthetic_hard_baselines_v2_2026-10-23/run_manifest.json)|Runhard: seed/protocol,selected parameters,hidden support và metrics|
|[9]|[experiments/synthetic_hard_baselines_v2_2026-10-23/summary.json](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/experiments/synthetic_hard_baselines_v2_2026-10-23/summary.json)|Runhard: seed/protocol,selected parameters,hidden support và metrics|
|[10]|[presentation/implementation_report/scripts/extract_profile_plot_data.py](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/presentation/implementation_report/scripts/extract_profile_plot_data.py)|Trích entity_001/component_06,120times từ saved NPZ|
|[10]|[presentation/implementation_report/scripts/extract_ticket02_missingness.py](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/presentation/implementation_report/scripts/extract_ticket02_missingness.py)|Trích mask easy,entity_001,t52–67|
|[10]|[presentation/implementation_report/scripts/build_figures.py](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/presentation/implementation_report/scripts/build_figures.py)|Xuất acceptedCOVID, x-plus/x-minus CI|
|[10]|[presentation/implementation_report/scripts/build_ticket03.py](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/presentation/implementation_report/scripts/build_ticket03.py)|Bố cục panel và kiểm saved metric/support|
|[10]|[experiments/synthetic_baselines_v2_2026-10-23/plots/plot_eda.R](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/experiments/synthetic_baselines_v2_2026-10-23/plots/plot_eda.R)|Nguồn biểu đồ EDA gốc|

SHA256 đầy đủ: `.harness/integrated-deck-32-2026-10-23/file_role_inventory.json`. Bảng32 rút gọn tên theo nhóm; bảng này chỉ rõ thư mục để phân biệt các file trùng tên. Các doc đặc tả thuộc [6].

## Ký hiệu và phạm vi

- C/H/M: quan sát để conditioning / che để chấm / thiếu gốc không nhãn. Missing mask khác true-zero/sampling-zero/positive-count strata.
- X: sampled counts, N: tổng counts mỗi mẫu, p: latent proportions. MAE ratio×100=percentage points (pp), không phải relative error%.
- Other: phần N ngoài named components, không phải missing rate hoặc private residual truth. Core width D_c=D+1, synthetic12named+Other=13.
- A_i (Hron) và O (Tsagris) đều chỉ anchors quan sát, được định nghĩa tại từng cơ chế. Donor bank chỉ từ fit scope, không nhận private labels.
- Hron available-donor median-scale/median aggregation; anchors gốc không được bổ sung từ giá trị mới điền.
- MICoDA-V2: W positive shift, least-missing reference, complete-case init, OLS/update/refit/delta. Core shifted/rescaled values khác counts cuối khóa bởi adapter. Synthetic intercept-only Q_i=1; không suy sang COVID, nơi manifest khai báo metadata. COVID5/5folds chạm50iterations, không gọi là hội tụ.
- TsagrisKNN là primary variant. D_code=KL(q||m)+KL(v||m)=2JSD chuẩn, natural log, không sqrt. alpha powers chuẩn hóa từng donor row; alpha=0 cần common positive support; không coi alpha nhỏ hơn là tốt hơn.
- COVID refit dùng train trong mỗifold; synthetic refit dùng train+validation. Test không chọn tham số.
- COVID25–26 pooled hidden-target ratio MAE. Trang27 paired location-macro difference, không trừ pooled rồi gán CI macro. CI theo location, fixed fits, chưa hiệu chỉnh đa so sánh.
- Synthetic: standard v2,seed2026,một seed/profile,declared MCAR/MAR,hidden test cells. Chưa có CI giữa seeds/MNAR/repeated-seed inference trong artifacts đã xem. Count MAE phụ thuộcN. n=0 giữnull,không đổi thành0.

## Nguồn số liệu và kiểm lại

[Kiểm số liệu](../../.harness/integrated-deck-32-2026-10-23/numerical_audit.json):18COVIDestimate/low/high triples;120times×3profiles từarrays;36profileplot yvalues;9strataplot yvalues;18overall và54stratarecords;maskexport/input hashes; presets/support đã đối chiếu. CSV/arrays/scientific data không đổi. Chỉ manifesthash plot cập nhật cho đường mốc0 của pairedplot.

Bảng metric/count/RMSE/selectedparams/support đầy đủ tiếp tục dùng [Ticket03 companion](TICKET_03_COMPANION_VI.md), data/ticket03_synthetic_full_metrics.json và ticket03_synthetic_support.json. Tài liệu Ticket03 là lịch sử30trang, không thay thế báo cáo tích hợp32trang này.

## Lệnh

```powershell
# Tại presentation/implementation_report, chạy2lần
xelatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
# Từ repo
pdftoppm -r 150 -png presentation/implementation_report/main.pdf presentation/implementation_report/qa_integrated/final/slide
py .harness/integrated-deck-32-2026-10-23/audit_numbers.py
py .harness/integrated-deck-32-2026-10-23/audit_pdf.py
```

Plot builder khi cần tái xuất (không cần chạy lại cho bàn giao này): chạy build_figures.py rồi build_ticket03.py theo manifest. Không chạy benchmark, generator, retuning, model suite hay tạo seed mới trong lần tích hợp.
