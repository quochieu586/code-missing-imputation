# Đối chiếu CROSSCHECK_vi và cập nhật tích hợp — 2026-10-30

## Kết luận

Đồng ý phần lớn cross-check, đặc biệt X1/X2/X5/X6. Không coi phản biện là việc cần bác bỏ để giữ kết luận cũ. Giữ contract MICoDa đã duyệt để tái lập, nhưng **chặn kết luận benchmark/main-model trên A17/R1** cho đến khi nhóm quyết định ngân sách Other. PR #2 và #3 vẫn chưa merge main tại thời điểm kiểm tra.

PR #2 cập nhật từ `edfb0a2` lên `4b3e97177241b4e100853371372fb53d65944122`: 6 commit F1, F2, F3, F4, E1, docs. Tích hợp bằng merge trong nhánh audit riêng, không rewrite nhánh tác giả. Hron/Tsagris giữ đúng bytes của PR #2 mới.

| Finding | Đối chiếu / hành động |
|---|---|
| X1 R1 phình khối lượng | **Đồng ý, trọng yếu về phương pháp.** `space` không loại bỏ thiên lệch, cũng không khóa người dùng khỏi xếp hạng. Giữ R1 để không đổi quyết định khoa học âm thầm; bổ sung cảnh báo và gate rõ ràng. Test ví dụ 30 → 80 khi Other=50. |
| X2 failed mất lỗi gốc | **Đồng ý, đã sửa.** Failed không gọi writer; giữ error code/message/context và unassigned, ghi log, exit 1. Partial xuất `_partial.csv` native và không chấm masked. Thêm real insufficient-fit và simulated failed/partial tests. Legacy impute cũng từ chối trả raw failed values. |
| X3 F1/F2/E1 còn ở TV1 cũ | **Đồng ý tại eb8c0ef.** Đã nhận nguyên sửa tác giả từ PR #2 mới; chạy lại lifecycle/numerics/full suite. |
| X4 merge conflict | **Đồng ý.** Giữ loader thông báo chi tiết bên TV2, bỏ kiểm tra width trùng; writer TV1 hỗ trợ partial; regex test dùng `field`. Một validate_mask dùng chung; rate/seed ValueError theo quyết định trước, không thêm cơ chế CLI exit 2 riêng cho các tùy chọn này. Sửa test kỳ vọng tương ứng, giữ test không side effect. |
| X5 thiếu spec trong Git | **Đồng ý.** Thêm `nextgen/docs/THREE_BASELINE_CONTRACT_VI.md`, sửa link README. Không dùng file reports bị ignore làm contract chia sẻ. |
| X6 seed | **Đồng ý, đã nhận kiểm tra >=0 và distinct**, chạy trước output/fit. |
| X7 bảng terminal | **Đồng ý, thêm cột space**; đây chỉ là hiển thị, không phải scientific safeguard tự động. |
| X8 thừa code | Bỏ import MICoDa không dùng, khối width lặp, preflight rate lặp và coverage reassignment. Adapter vẫn nhánh riêng có chủ đích, không dựng framework. |
| X9 commit quá rộng | Ghi nhận. Không rewrite commit đã được người khác dùng; cập nhật bằng merge có dấu vết nguồn và commit reviewable. |

## Những điểm cần nói chính xác hơn

- Interface adapter khác fit/predict là thực tế, không tự nó là bug: MICoDa cần counts/N/mask để thực thi A17/R1. Dispatch tường minh là giải pháp nhỏ nhất hiện tại.
- P1 ở X1 là mức độ **rủi ro hợp lệ khoa học** khi dùng benchmark, không phải lỗ hổng bảo mật hay bug vi phạm contract R1 đã chọn. Không dùng phân biệt này để giảm tầm quan trọng.
- Hai không gian vẫn có thể tính metric trên cùng target hidden; điều chưa hợp lệ là suy luận cùng điều kiện/ngân sách rồi kết luận ưu thế thuật toán. Cần thống nhất budget trước khi so sánh xác nhận.
- Không gọi toàn bộ repo runtime là nextgen: chỉ workflow ba baseline mới ở đó. Commit eb8c0ef chưa merge main. Câu trả lời cũ nói “đã merge” là sai.
- “57 tests nguyên vẹn” không chính xác: test_run đã đổi kỳ vọng MICoDa. G4 cũ có 12 checks trong JSON, không phải 19; không dùng các con số khác phạm vi thay thế.
- Các tỉ lệ ×1.186/×2.243 trong CROSSCHECK chưa được tái tạo đúng cùng mẫu bằng probe gốc (không có script đó trong checkout này). Audit mới dùng **chỉ originally-complete rows bị ảnh hưởng**, định nghĩa khác, không nhận là trùng bằng chứng cũ.

## Kết quả kiểm tra độc lập tại nhánh tích hợp

- Full suite: **111 passed**, RuntimeWarning coi là lỗi. Bao gồm các tests tác giả PR #2 và 5 cases mới (partial/failed parametrized).
- COVID whole_feature 5%, seeds 7/42: **9/9 run ok**, cả native và masked; chạy với RuntimeWarning là lỗi. MSE count mean Hron=42953.6, Tsagris=50844.8, MICoDa=1090263 xấp xỉ, khớp quy mô bảng cross-check. Đây không phải bảng xếp hạng có ngân sách tương đương.
- Audit ngân sách R1, originally-complete affected rows: random_cell seeds 7/42 = **1.42480 / 1.64448**, whole_feature = **1.27068 / 2.37433**. Là `sum(N-visible)/sum(hidden truth)` trước model, không phải tỷ lệ output của fit đã hoàn tất trên mọi dòng.
- Script kiểm chứng trong Git: `nextgen/scripts/verify_three_baselines.py`; bằng chứng cục bộ `.harness/baseline-crosscheck-update/` ở workspace người thực hiện. Không đưa raw dữ liệu nội bộ vào tài liệu/PR.

## Quyết định còn cần nhóm chốt

Khuyến nghị: thống nhất model-space + budget có Other cho benchmark xác nhận cả baseline/main model; nếu đổi adapter MICoDa phải version contract và chạy lại đối chiếu, không gọi bản mới là A17/R1 cũ. Chưa thực hiện thay đổi khoa học đó trong cập nhật này. Tests pass chỉ chứng minh hành vi đã kiểm, không mở gate khoa học.
