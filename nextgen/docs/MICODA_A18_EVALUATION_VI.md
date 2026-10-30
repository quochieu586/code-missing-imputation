# Lưu trữ đánh giá A18: prototype `MICoDa18` (fit/predict) — KHÔNG phải mã runtime

> **Trạng thái:** Tài liệu lưu trữ mục đích đánh giá. Class `MICoDa18` dưới đây **không nằm trong `src/`**, không được registry `MODELS` chạy, và không có trong bất kỳ đường chạy production. Mặc định chính thức vẫn là **A17/R1** (`MICoDaCovid` trong [`baselines/micoda.py`](../src/longitudinal_impute/baselines/micoda.py), Decision D07/D08).

## Vì sao lưu ở đây

Đáp án cho phản biện "theo interface fit/predict chung" cần một bằng chứng chạy được, không chỉ lý thuyết. Prototype dưới đây chứng minh: MICoDa **có thể** viết theo interface `fit/predict` dùng chung `model_input`/`restore_counts` với Hron/Tsagris — nhưng `predict` của nó chỉ trả về **bảng đã fit** (transductive), không phải dự đoán ngoài mẫu. Sau khi chạy đánh giá đầy đủ (Wasserstein, MSE, MAE, bias, bootstrap), A18 **không đạt gate** thăng hạng, nên prototype bị rút khỏi `src/` để giữ codebase gọn và lưu lại ở đây làm tư liệu.

## Prototype (nguyên trạng từ nhánh audit `micoda-a18-review`, commit `c9dd2bc`)

```python
from .common import CoreResult, matrix
from ..errors import (BaselineInputError, InsufficientDataError,
                      NotFittedError, NumericalError)


class MICoDa18:
    """Transductive MICoDa with Other; predict returns the fitted table only.

    Input is the shared D+1 proportion matrix: Other is unknown on partial rows.
    Raw counts are restored by the runner, not by this model.
    """
    input_domain = 'nonnegative'

    def __init__(self, *, max_iter=120, tol=1e-8, reference=None):
        self.core = MICoDa(max_iter=max_iter, tol=tol, reference=reference)

    def fit(self, values):
        x = matrix(values)
        if x.shape[1] < 3:
            raise BaselineInputError('Expected at least two targets and Other')
        known = np.isfinite(x)
        complete = known.all(axis=1)
        if (np.isfinite(x[:, -1]) != np.isfinite(x[:, :-1]).all(axis=1)).any():
            raise BaselineInputError('Other must be known exactly on complete target rows')
        mass = np.nansum(x, axis=1)
        if (mass > 1 + 1e-10).any() or not np.allclose(mass[complete], 1., atol=1e-10, rtol=0):
            raise BaselineInputError('Invalid composition total')
        positives = x[:, :-1][x[:, :-1] > 0]
        if not positives.size:
            raise InsufficientDataError('No positive visible target for min/2')
        delta = float(positives.min() / 2)
        if delta <= 0:
            raise NumericalError('min/2 is not representable in float64')
        excluded = complete & (x[:, :-1] == 0).all(axis=1)
        prepared = x.copy()
        prepared[known & (prepared == 0)] = delta
        for i in np.flatnonzero(~excluded):
            if complete[i]:
                prepared[i] /= prepared[i].sum()
            else:
                headroom = max(1e-12, int((~known[i]).sum()) * delta)
                if not 0 < headroom < 1:
                    raise BaselineInputError(f'row {i}: invalid model headroom {headroom}')
                visible_mass = np.nansum(prepared[i])
                if visible_mass > 1 - headroom:
                    prepared[i, known[i]] *= (1 - headroom) / visible_mass
        core = MICoDa(max_iter=self.core.max_iter, tol=self.core.tol, reference=self.core.reference)
        core.fit(prepared[~excluded], require_positive_state=True)
        output = x.copy()
        output[~excluded] = core.fit_result_.values
        info = dict(core.fit_result_.diagnostics)
        info.update(delta=delta, other_column=True, budget_policy='R2',
                    excluded_all_zero_rows=int(excluded.sum()),
                    initial_complete_rows=int((complete & ~excluded).sum()),
                    prediction_mode='fitted_table_only')
        # Publish all fitted state only after successful fitting.
        self.core, self.input_, self.info_ = core, x, info
        self.fit_result_ = CoreResult(output, info)
        return self

    def predict(self, values):
        if not hasattr(self, 'fit_result_'):
            raise NotFittedError('Fit MICoDa18 before prediction')
        x = matrix(values, width=self.input_.shape[1])
        if not np.array_equal(x, self.input_, equal_nan=True):
            raise BaselineInputError('MICoDa18 predicts only the fitted table')
        return self.fit_result_.values.copy()
```

Test kèm theo (cũng lưu trữ, chạy được khi dán vào `nextgen/tests/`): `nextgen/tests/test_micoda_a18_interface.py` trên nhánh audit — 3 test: fit/predict dùng fitted state, firewall observed counts + không lộ hidden truth, Other=NaN trên dòng bị che.

## Kết quả đánh giá tóm tắt

Chi tiết đầy đủ: `reports/DANH_GIA_THONG_KE_MICODA_A17_A18_2026-10-30.md` và `reports/MICODA_NATIVE_COVID_FULL_2026-10-30.md` (cục bộ; thư mục `reports/` không nằm trong Git).

COVID, mask 5%, seeds 7+42, trung bình hai seed:

| Mask | Metric | A17/R1 | A18/R2 | A18 tốt hơn? |
|---|---|---:|---:|---|
| whole_feature | MSE count | 1.090.263 | **757.559** | Có |
| random_cell | MSE count | **734.706** | 3.341.057 | **Không (~×4,5)** |
| whole_feature | MSE tỷ lệ | 0,04239 | **0,03705** | Có |
| random_cell | MSE tỷ lệ | 0,03729 | **0,03325** | Có |
| whole_feature | W1 tỷ lệ macro | 0,06067 | **0,04103** | Có |
| random_cell | W1 tỷ lệ macro | 0,04406 | **0,03129** | Có |

- Gate đã chốt (D08): **cả MSE count và MSE tỷ lệ phải cải thiện trên cả hai mask** → A18 **không đạt** vì MSE count random-cell.
- MSE count bị chi phối bởi ít ô quy mô lớn: random_cell, 10 ô chiếm ~99% tổng squared error của A18 — cần điều tra nguyên nhân trước mọi quyết định.
- Native (điền 86.050 NaN thật, không có truth): A18 giữ Other ngầm (mean 15,1%/dòng) trong khi A17 đẩy Other về ~0 (100% dòng); W1 tỷ lệ observed-vs-imputed macro: A17 0,04492, A18 0,03594. Chỉ là mô tả phân phối, không phải độ chính xác.
- Bootstrap ghép cặp 150 địa điểm (500 lượt/mask-seed, prediction cố định): W1 tỷ lệ A18−A17 dưới 0 ở cả 4 lượt — bằng chứng có điều kiện, không phải kiểm định khẳng định.

## Nếu sau này muốn thúc A18

1. Chẩn đoán các ô outlier random-cell (N, residual, proposal) trên máy nội bộ.
2. Thiết kế đánh giá xác nhận: primary metric tỷ lệ `count/N` + count phân tầng theo N, ≥10 seed, tách khám phá/xác nhận.
3. Đưa `MICoDa18` (hoặc nâng cấp `other=True` thành mặc định) qua PR riêng với đầy đủ diagnostics/failed/partial như `MICoDaCovid` — suite cũ từng fail 4 test khi ép prototype vào registry, phải giải quyết triệt để.
4. Cập nhật D08 và archive A17 theo đúng quy trình.
