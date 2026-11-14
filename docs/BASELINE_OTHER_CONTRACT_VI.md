# Contract Other và benchmark ba baseline

Hiệu lực: 2026-11-12, theo yêu cầu trực tiếp của người dùng. Thay thế mặc định A17/R1 và gate thăng hạng D08 cũ. Không triển khai main model.

## Quyết định

- Hron, Tsagris và MICoDa mặc định đều dùng D target + Other nội bộ. MICoDa constructor và cấu hình phát hành: `other=true`.
- Other là phần ngoài D target. Dòng visible-complete: `Other = total - sum(targets)` (đổi sang proportion khi vào model). Dòng partial, kể cả do evaluation mask: Other unknown, không được dùng residual chưa phân bổ làm Other observed.
- Trên dòng partial, thuật toán dự đoán trọng số missing targets và Other. Với `r = total - sum(visible targets)`, phân bổ `r*w_j/sum(w_free)` cho từng phần tự do; observed gồm zero giữ nguyên. Residual zero thì missing bằng zero. Không có trọng số hợp lệ thì giữ status lỗi/partial hiện hành, không fallback ngầm.
- Output vẫn chỉ D target. Other cuối là `total - sum(completed targets)`; tổng target không bắt buộc bằng total. Không coi Other dự đoán ở dòng partial là observed truth.
- MICoDa dùng restore riêng để giữ partial/status/roundoff handling; công thức phân bổ R2 phải khớp `data.restore_counts` khi cùng proposal. Không refactor thuật toán để ép chung interface.
- MICoDa `other=false` chỉ còn là legacy A17/R1 opt-in, gắn nhãn riêng. Helper `prepare(..., other=False)` vẫn mặc định legacy để giữ API cấp thấp hiện có; adapter luôn truyền mode tường minh. Không thay đổi core MICoDa.

## Benchmark có cùng điều kiện ngân sách

1. Cùng CSV/schema/ordered targets và total hợp lệ. Total là đầu vào được phép biết; không suy Other/total từ hidden truth. Với total suy từ bảng complete trước masking, công bố đây là known-total reconstruction.
2. Dùng cấu hình phát hành với MICoDa Other bật. Nếu thử R1, báo thành sensitivity riêng, không gộp với common-budget table. Trường `space` 18_with_other/17_targets là nhãn COVID lịch sử, không phải D tổng quát.
3. Cùng evaluation mask từng seed, fresh instance mỗi model/lượt; natural NaN không phải label. Không dùng native fitted state để giải masked run.
4. Khóa config, mask kind/rate/seeds và metric trước chạy. Điểm khởi đầu là MAE proportion cell-weighted, kèm MSE/MAE count/proportion và nhóm zero/positive/forced/model hiện có; không chọn metric sau khi thấy winner. Thay aggregation phải công bố trước.
5. Accuracy paired dùng giao seed mọi comparator completed và coverage=1, cùng scored cells; báo danh sách seed giữ/loại. Runner xuất `summary.json.paired` và bảng console trên common-success seeds; từ chối bảng paired nếu trộn model space R1/R2 hoặc giao rỗng. `aggregate` cũ vẫn là mô tả success riêng, **không phải bảng paired ranking**. Khi báo chênh lệch dùng các dòng per-seed cùng tập này, không lấy hai mean từ hai tập success khác nhau.
6. Báo toàn bộ partial/failed, convergence và thời gian riêng. Không bỏ failure khỏi reliability hoặc tính lỗi riêng trên các ô partial dễ giải.
7. Native Wasserstein chỉ diagnostic; không xác nhận accuracy ở NaN thật. Cùng ngân sách không đồng nghĩa cùng support/zero replacement hoặc cùng thuật toán: Hron positive replacement, Tsagris zero-aware và MICoDa min/2+closure/complete-case vẫn được giữ nguyên và công bố.

Đây là chuẩn hóa domain so sánh, không phải bằng chứng R2 tốt hơn, không bảo đảm mọi seed đủ donors, không tự giải quyết exhaustive-target/unknown-total datasets. Không đổi `has_other` schema trong task này. Một benchmark xác nhận cần chạy theo protocol đã đóng băng; task này chỉ kiểm chứng regression và CLI trên fixture hai dataset.
