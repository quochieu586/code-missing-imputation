# Ba module baseline hiện tại

Đợt cleanup giữ nguyên nội dung ba file thuật toán và các hàm toán chúng dùng.
Đây chưa phải kết luận rằng ba baseline đã phù hợp với COVID-19.

| Module | Constructor runner đang gọi | Trạng thái lựa chọn |
|---|---|---|
| `baselines/hron.py` | `HronKNN()` | Giữ mặc định hiện có để kiểm tra luồng. Chưa chọn giữa kNN, LS và LTS. |
| `baselines/micoda.py` | `MICoDa()` | Mặc định V2, W=0. COVID-19 có zero nên hiện báo lỗi. |
| `baselines/tsagris.py` | `TsagrisKNN()` | Mặc định k=3, alpha=1, báo lỗi khi observed có tổng bằng zero. |

Các biến thể trong ba file được giữ nguyên để review sau. Runner gọi trực tiếp
`fit(values)` rồi `predict_result(values)` trên toàn bộ bảng. Không có
`NumericalCore` protocol, `CountBaselineAdapter`, FitScope hay tập train riêng.

Phần `data.py` giữ phép chuyển đổi cần thiết của luồng cũ.
Counts được chia cho `total_sequence`. Cột Other chỉ biết khi toàn bộ target
của hàng còn quan sát được. Với Hron, zero trong input được thay bằng một nửa
giá trị dương nhỏ nhất, theo hành vi adapter cũ. Sau prediction, observed được
giữ nguyên và ngân sách count còn lại được chia giữa missing và Other.
Đây là xử lý đầu vào và đầu ra, không phải thuật toán baseline mới.

Các quyết định cần thống nhất trước khi sửa baseline gồm chọn biến thể Hron,
chọn cơ chế zero của MICoDa, và chọn chính sách cho các hàng không có anchor
dương của Tsagris. `zero_anchor='global'` và `zero_missing_mass='uniform'` là
các lựa chọn đã tồn tại trong Tsagris, chưa bật trong runner mới.
Với MICoDa, cần nghiên cứu và chốt W trên miền proportion trước khi áp dụng.
Cleanup không tự gán W hoặc đổi sang một biến thể khác.

`baselines/registry.py` chỉ còn helper `configure_core`, được hai phương thức
`with_degenerate_policies()` trong Tsagris gọi nội bộ. Runner không gọi helper
này. Xóa nốt hai phương thức đó sẽ sửa file baseline nên cần duyệt riêng.

Đối chiếu số học trong `tests/test_covid_flow.py` dùng kết quả adapter cũ đã lưu
ở `review_20261025/adapter_reference.json`. Kiểm chứng này chứng minh phép
chuyển đổi trên fixture đó, không chứng minh paper parity hoặc chất lượng model.
