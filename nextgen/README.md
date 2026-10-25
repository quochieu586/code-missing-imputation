# Chạy imputation trên COVID-19

Chạy lệnh từ thư mục gốc của repo với Python >= 3.11.

```powershell
python -m pip install -e "./nextgen[test]"
python -m longitudinal_impute.cli --data data/covariants.csv --schema-config nextgen/configs/baselines_covariants.toml --output experiments/covid_run
```

Chọn một thư mục output chưa tồn tại. Runner gọi lần lượt Hron, MICoDa và
Tsagris bằng các constructor mặc định hiện có. Cấu hình model cuối cùng còn
chờ duyệt riêng, xem [BASELINES_VI.md](BASELINES_VI.md).

Để chạy thêm MSE trên một phần observed được che, thêm `--mask-rate 0.05 --seed 7`.
Lượt MSE chạy riêng. Lượt điền NaN thật luôn dùng toàn bộ observed.
Mask đánh giá giữ ít nhất một ô observed trên mỗi hàng có dữ kiện.
Số ô thực tế được che nằm trong `evaluation_cells`.

Đọc các file output sau khi chạy.

- `hron.csv`, `micoda.csv`, `tsagris.csv` chứa kết quả của model chạy thành công.
- `summary.json` ghi Wasserstein theo từng cột, MSE nếu có, và lỗi của từng model.
- `evaluation_mask.npy` lưu mask dùng chung giữa ba model để tính MSE.

Exit code 1 báo ít nhất một model hoặc lượt MSE bị lỗi. Runner vẫn thử các
model tiếp theo và giữ kết quả thành công. Không dùng CSV của model này để
thay kết quả model khác.

Kiểm thử bằng lệnh sau.

```powershell
python -m pytest nextgen/tests -q --basetemp nextgen/.test-tmp
```

Xem [báo cáo review](review_20261025/REVIEW_VI.md) để đọc cấu trúc mới,
bằng chứng kiểm chứng và các quyết định baseline còn chờ duyệt.
