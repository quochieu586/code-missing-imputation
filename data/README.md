# Dữ liệu dùng chung

- `covariants.csv`: bảng count COVID hiện dùng cho ba baseline.
- `original_long_format.csv`: dữ liệu nguồn ở dạng long.

Migration giữ nguyên vị trí và nội dung hai CSV. Schema tương ứng nằm ở `../configs/baselines_covariants.toml`. Output mới được ghi vào thư mục người chạy truyền bằng `--output`, theo ví dụ tại `../outputs/`; không ghi vào CSV nguồn.

Output KNN trước đây và placeholder splits được lưu tại `../archive/legacy_knn_csdi_2026-11-02/data/`.
