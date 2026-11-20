# Bản đồ bằng chứng

Paths code dưới đây tính từ `src/longitudinal_impute/` trong repository gốc.
Code không phải dependency để build deck. Tám slide hiện tại dùng năm references.

| Slide | Hành vi/ký hiệu và wording hiện tại | Nguồn hiện tại |
|---|---|---|
| 1 | Ước lượng missing counts trong repeated measurements; giữ known totals/observed counts; source line trên bìa | `main_model.py:26–114,149–163`; `data.py:162–217` |
| 2 | Ví dụ country/case counts; input counts, total, entity, time; D targets + Other; Figure 1 pipeline 0.9 linewidth; caption nhắc mạng không initialize/restore | `main_model.py:26–114,149–163,263–294`; `data.py:15–24,43–93,162–217` |
| 2 | CLR dùng proportions sau zero replacement; features không phải counts; [5] | `main_model.py:149–163`; Martín-Fernández2003 PDF1–4 |
| 3 | Table 1 có ba cột, bốn hàng; không còn ba dòng tổng kết cũ; caption dẫn [1]–[4], nhắc giới hạn memory/computation khác nhau | Literature locators bên dưới |
| 4 | H/Q không giao nhau; loại Q trước initialization; Q cố định và H bị loại khỏi cả hai views; Figure 2 rộng 0.9; caption nói shared weights, separate completions | `main_model.py:93–146,196–260,368–401`; `data.py:119–159` |
| 4 | Batch-Q proportion MSE sau restoration; y=c/T và restored training counts | `main_model.py:196–208,235–260` |
| 5 | Figure 3 rộng 0.8; linear Z,m; additive learned parts/fixed sine time codes; time rồi part attention; normalization và residual updates; caption chỉ trục trao đổi thông tin | `network.py:16–26,39–60` |
| 5 | Gate tanh nhân sigmoid; shared C→C→1 head; softmax K weights dương tổng 1; LN rescale; C=32; không nêu heads/dropout | `network.py:8,16–26,39–60` |
| 6 | Giữ visible counts; residual allocation trên missing targets + Other; Figure 4 rộng 0.9; caption nhắc giữ known count | `data.py:162–217`; `main_model.py:166–208` |
| 6 | T=100; T1=30; remaining70; weights .18/.12/.30 → .30/.20/.50; T2=21,T3=14,Other=35; exported65 | Số học minh họa do tác giả chọn, không phải dataset/model output |
| 7 | Figure 5 rộng 0.8; fit training, restore hai views riêng và share theta; giữ parameters, reset Adam, masks/labels cố định; Delta trên inference missing targets | `main_model.py:211–260,297–339,409–421` |
| 7 | Bốn câu stage handoff: initialization lưu state, refinement cập nhật counts, prediction tạo outputs, evaluation không train | `run.py:288–431,508–513`; `main_model.py:431–480` |
| 7 | Tuning/component tests còn phía trước; không guarantee convergence; masks share random streams; restoration có thể reject nhầm decimal counts | `docs/MAIN_MODEL_AUDIT_20261113_VI.md:54–98,100–120,188–206` |
| 8 | Năm IEEE entries với DOI/publisher/arXiv links | `slides/08_references.tex` |

## Slide3 và slide8: literature + suy luận có điều kiện

Slide3 so sánh thiết kế, không trình bày experimental results. Citation [1]–[4]
nằm trong bảng; [5] nằm trên slide2. Table 1 có source caption và một câu comment.

| Citation | Slide wording và giới hạn | Primary locator / verified claim |
|---|---|---|
| [1] Bengio1994 | RNN đọc tuần tự, nhớ bước trước; tín hiệu học nhỏ từ bước xa hạn chế memory | Author manuscript9/13, §4 Eq2/Theorem4; C01/C02 |
| [2] Hochreiter1997 | LSTM dùng gates giữ/quên/xuất thông tin, giữ memory qua nhiều bước; steps tuần tự hạn chế parallel training | PDF1–2/9–10/33–34; C04–C07 |
| [3] Bai2018 | CNN/TCN lọc nhóm bước, xử lý parallel; spaced inputs mở rộng khoảng nhìn; finite range, past-only không dùng future | PDF3–5 §§3.1–3.5; C08–C11; preprint |
| [4] Vaswani2017 | Attention liên hệ allowed/distant steps, parallel; full pairwise attention có chi phí quadratic theo sequence length; cần time codes/link rules | PDF3–6 §§3.2–4/Table1; C13–C15 |
| [5] Martín-Fernández2003 | Zero replacement và log-ratio proportions; project dùng CLR feature-only | PDF1–4; C16–C18/U04; implementation theo module project ở bảng đầu |

Slide8 dùng metadata IEEE của năm nguồn [1]–[5], giữ named anchors và links.

## Style/illustration lineage

- Theme 4:3/navy được tái dựng từ `docs/2026_10_09-Revised proposal.pdf`.
- Nguồn hình nằm trong `figures/`; các hình dùng font nhãn 11pt trước scaling.
- Figure 1–5 là sơ đồ thiết kế hoặc số học minh họa, không phải measured results.
- Stage handoff trên slide7 là các câu hoàn chỉnh, không còn arrow-fragment chain.
- Captions dẫn tên module gốc và nói điều người đọc cần chú ý.
