# Protocol benchmark offline — đăng ký trước test

Dataset original_long_format.csv:11.671 rows,150 locations,17 targets,
112.357 observed cells và86.050 native-missing cells. Hash dataset/schema,
package source, split và masks nằm ở experiments/offline_baselines_2026-10-15_v2/
manifest.json. Snapshot config được lưu trước khi validation/test chạy.

Mỗi fold có90 train locations,30 validation và30 test; số rows train~7.000,
validation/test~2.330. Năm complete-data locations được phân đều: một ở mỗi
fold. Đây là stratification theo **observation availability**, không theo
giá trị target. Mỗi location/row xuất hiện trong test đúng một fold.

Mask seeds42/43/44; nominal rate30%:

- Whole-feature: chọn ceil(30% số feature được quan sát tại location), che
  feature đó suốt chuỗi; luôn giữ ít nhất một feature nếu có thể. Do feature
  count rời rạc, tỷ lệ realised là34,911% observed cells,39.225 targets/seed.
- Random-cell: Bernoulli30% trên observed cells; realised29,8–30,0%.
- Time-block: chọn một đoạn rows liên tiếp sau sort timestamp cho mỗi location,
  che toàn bộ observed features của đoạn. Đây là30% timepoints, không phải
  một calendar duration giống nhau giữa countries có lịch sampling khác.
- Whole-row: chọn rows ngẫu nhiên theo location rồi che toàn bộ observed parts;
  realised30,50% cells do ceil theo panel length,34.274 targets/seed.

Native missing không được biến thành nhãn0. Target H được che ở raw boundary;
C=O-H. Other chỉ biết nếu tất cả17 raw parts vẫn visible. Train dùng native
observed của riêng train locations; donor/scaler/zero policy/metadata fit chỉ
train. Không triangular temporal mask. Các general core hiện tại vẫn mang
tính cross-sectional; quyền dùng tương lai không tự làm chúng thành temporal model.

Ba họ primary: Hron-kNN; MICoDa-V2; Tsagris-alpha (alpha1 chính là basic).
Chọn k3/5/10, W1e-6/1e-4/1e-3, alpha.25/.5/1 bằng mean validation ratio MAE
trên3 seeds của whole-feature; cấu hình được khóa trong selection.json rồi
mới chạy test của fold. Lưới đầy đủ75 candidate fits qua5 folds,225 validation
prediction jobs;180 primary test jobs (5x3x4x3). Không chọn theo test.

Metadata approved cho benchmark chính: median_age và new_cases_smoothed_per_million.
Encoder dùng train-only scaling/missing flags. MICoDa dùng covariate regression;
Hron/Tsagris gốc không dùng metadata đó trong distance/model. Đây là so sánh
thuật toán với capabilities gốc và cùng provenance, không claim mọi model dùng
chính xác cùng biểu diễn. Các metadata score/group chưa rõ provenance bị loại.

Các phương pháp supplementary: Hron-LS/LTS, MICoDa-V1, Tsagris basic/globalCV/
adaptive. Chạy5fold x3seeds, whole-feature; kế thừa k/W của primary tương ứng,
CV Tsagris chỉ sử dụng train donors (leave16out,3 repetitions, fixed seed42).
Giới hạn180s **wall time tổng mỗi fold/method** được đăng ký trước test vì
local runtime có8 logical CPUs,15,65GiB RAM và khoảng2,1GiB available khi đo.
Hai workers chạy các fold riêng, mỗi worker1 BLAS thread. Timeout, rank/domain
not-applicable và numerical failure được báo riêng, không gán score0 hoặc
thay phương pháp. Không ranking supplementary trên các tập fold khác nhau.

Primary metrics: count MAE/RMSE, ratio/N MAE/RMSE, location/feature macro,
zero/nonzero và rare-positive<1%. Coverage và conditional17 JSD/LR chỉ trên
complete truth rows có mass phù hợp. LR diagnostics thêm fixed1e-6 vào
conditional proportions **chỉ trong scorer**, không làm input imputer.
Temporal first difference/slope/lag và feature covariance/correlation báo
support; pooled relationships có thể bị confounding country/time.

Bootstrap2000 resamples theo location sau gộp3 mask repetitions; paired
differences trên cùng locations. Seeds của primary là mask repetitions,
không phải ba mô hình neural độc lập. Runtime, donor policies, repair magnitude,
fallback và convergence flags phải đọc cùng accuracy.

Sau OOS test, mỗi họ chọn final config bằng mean validation scores across
folds, refit toàn bộ native observations rồi xuất CSV11.671rows. Giữ nguyên
text observed counts/metadata, không âm, sum17<=N. Không dùng final-refit CSV
để báo accuracy native missing hoặc tính lại test error như held-out.

Nguồn chuẩn: confirmed_design_vi.md, offline_masking_review_vi.md và kế hoạch
directive/plans/research_plan_longitudinal_2026-10-15_vi.md. Full execution/progress/checks nằm
trong .harness/full-baseline-benchmark-2026-10-15/. Không xóa file nào.

Lượt đầu ở offline_baselines_2026-10-15 được giữ nguyên làm evidence profiling.
Lượt v2 chỉ thay cách giải tie ở prefix k: predictions và donor IDs bitwise
bằng nhau trên proof200rows cho k3/5/10, test/R parity đạt. Config, schema,
dataset, split và toàn bộ masks có hash giống hệt; không xóa file.
