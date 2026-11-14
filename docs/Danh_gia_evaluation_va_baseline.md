# Đề xuất thiết kế evaluation và rà soát baseline

2026-11-11 · Tuanh

Đề xuất: đánh giá theo 5 tầng, trong đó **che giả có đáp án là thước đo chính** và chỉ số không cần đáp án chỉ được dùng sau khi đã kiểm định là xếp hạng giống thước đo chính. Kèm theo là 7 thay đổi cho baseline hiện tại, ưu tiên cao nhất là thêm một baseline có dùng thông tin thời gian và ba "đối thủ giả".

## Bối cảnh và mục tiêu

Bài toán: điền 86.050 ô thiếu (43%) trong bảng đếm biến thể SARS-CoV-2 (11.671 dòng, 150 quốc gia, 17 biến thể). Dữ liệu có ba đặc điểm cùng lúc: **compositional** (các phần cộng lại bằng tổng N), **longitudinal** (nhiều mốc thời gian mỗi nước) và **zero-inflated** (82% ô đã biết bằng 0).

Ô thiếu thật **không có đáp án**, nên không thể đo độ đúng trực tiếp. Thêm vào đó, dữ liệu thiếu theo khối: 1.252 cặp (quốc gia, biến thể) trống cả chuỗi, và nhiều khả năng thiếu vì biến thể hiếm (không ngẫu nhiên, MNAR).

Mục tiêu của module evaluation:

1. Xếp hạng được các phương pháp điền (baseline và model chính) một cách công bằng, có khoảng tin cậy.
2. Phát hiện kết quả phi lý trên ô thiếu thật dù không có đáp án.
3. Đo đúng cả ba đặc điểm của dữ liệu, không chỉ sai số từng ô.
4. Không bị "gian lận" bởi phương pháp chỉ làm khớp phân phối mà sai từng điểm.

Ví dụ dùng xuyên suốt tài liệu: Pháp, ngày 05/07/2021, N = 100; giá trị thật Alpha = 10, Delta = 70, Omicron = 0, Other = 20; ô Delta bị che và model điền 55.

## Bài học từ paper tham khảo

Farjallah et al. (ICC 2025), *Evaluation of Missing Data Imputation for Time Series Without Ground Truth*, đề xuất đo chất lượng điền bằng **độ khớp phân phối** (Wasserstein distance và Jensen–Shannon divergence) giữa giá trị đã điền và giá trị ngay trước khoảng trống ("pre-gap"). Họ kiểm chứng bằng cách tạo 100 khoảng trống nhân tạo trên hai bộ dữ liệu giao thông gần đầy đủ, điền bằng 5 phương pháp, và cho thấy thứ hạng theo WD/JSD trùng thứ hạng theo RMSE/MAE.

**Điều nhóm nên học:** quy trình kiểm định. Một chỉ số không cần đáp án chỉ đáng tin khi đã chứng minh nó xếp hạng giống chỉ số có đáp án trên dữ liệu che giả.

**Không áp dụng nguyên văn được**, vì các giả định của paper không đúng với dữ liệu của nhóm:

| Giả định của paper | Dữ liệu của nhóm | Hệ quả |
|---|---|---|
| Khoảng trống ngắn, ngay trước đó còn dữ liệu | Thiếu trọn chuỗi (quốc gia, biến thể) | Không có pre-gap để làm tham chiếu |
| Thiếu gần như ngẫu nhiên | Nhiều khả năng MNAR (biến thể hiếm thì không báo) | Ép phân phối điền giống phân phối quan sát có thể phạt nhầm model tốt |
| Một biến, liên tục | Composition 17 biến thể + Other | WD/JSD từng biến bỏ qua ràng buộc tổng |
| Ít số 0 | 82% số 0 | WD bị khối số 0 chi phối; JSD phụ thuộc cách chia bin |
| So thứ hạng bằng mắt qua biểu đồ | — | Cần hệ số tương quan thứ hạng và khoảng tin cậy |

**Lỗ hổng lớn nhất:** WD/JSD có thể bị gian lận. Một phương pháp chỉ rút ngẫu nhiên một giá trị quan sát để điền sẽ có WD gần 0 nhưng sai từng điểm. Paper không kiểm tra trường hợp này; tài liệu này xử lý nó ở Tầng 3.

## Kiến trúc 5 tầng

Mỗi tầng trả lời một câu hỏi: có hợp lệ không (Tầng 0), model nào chính xác hơn (Tầng 1), ô thiếu thật có hợp lý không (Tầng 2), thước đo Tầng 2 có đáng tin không (Tầng 3), và bảng đã điền có dùng được cho phân tích không (Tầng 4).

```
Bảng đầu vào + mask
        │
   Tầng 0: Kiểm tra hợp lệ
        │
   ┌────┴─────────────────────┐
   │                          │
Tầng 1 (lượt che giả)     Tầng 2 (lượt native, ô thiếu thật)
 có đáp án, thước đo chính  không đáp án, bổ trợ
   │                          │
   └──────► Tầng 3: kiểm định Tầng 2 bằng Tầng 1 + đối thủ giả

Tầng 0 ──(bảng đã điền)──► Tầng 4 (tùy chọn): tác vụ downstream
```

Kết luận chính lấy từ Tầng 1; Tầng 2 là bằng chứng bổ trợ và chỉ được dùng để xếp hạng khi đã qua Tầng 3.

## Tầng 0: Kiểm tra hợp lệ

Mỗi lượt chạy phải qua đủ 4 kiểm tra dưới đây trước khi được chấm điểm; không qua thì ghi lỗi và bỏ qua lượt đó.

1. Ô đã quan sát giữ nguyên tuyệt đối (ví dụ: Alpha = 10, Omicron = 0 không đổi).
2. Không có ô âm, NaN hoặc vô cực.
3. Tổng mỗi dòng ≤ N (ví dụ: 10 + 55 + 0 = 65 ≤ 100).
4. Đếm số ô "bị ép về 0": ô thiếu nằm ở dòng có phần dư N − Σ quan sát = 0, nên model nào cũng điền 0. Trên COVID là 60.595/86.050 ô (70%). Con số này không dùng để chấm mà để biết bao nhiêu ô thật sự do model quyết định.

Trạng thái: đã có (`check_completed` trong `data.py`). Chỉ cần ghi gọn kết quả vào summary.

## Tầng 1: Che giả có đáp án (thước đo chính)

Tầng này quyết định model nào tốt hơn: che một phần ô đã biết, cho model điền lại mà không thấy giá trị thật, rồi chấm trên các ô bị che.

### 1.1 Cách che: đề thi phải giống tình huống thật

| Kiểu che | Cách làm | Mô phỏng | Trạng thái |
|---|---|---|---|
| `random_cell` | Mỗi ô đã biết bị che với xác suất 5% | Thiếu lẻ tẻ (dễ) | Đã có, chỉ tham khảo |
| `whole_feature` | Che trọn chuỗi của các cặp (quốc gia, biến thể) ngẫu nhiên tới ≈ 5% ô đã biết | Thiếu theo khối | Đã có, **dùng làm chính** |
| `pattern_transplant` | Lấy "hình dạng thiếu" của một nước có dữ liệu thiếu thật, áp lên một nước gần đầy đủ | Thiếu y như thực tế | Cần thêm |
| `mnar` | Ưu tiên che các chuỗi có tỷ lệ biến thể thấp | Thiếu vì biến thể hiếm | Cần thêm |

Ví dụ `pattern_transplant`: ở Aruba thực tế thiếu `20C`, `Kappa`, `Iota`; áp đúng kiểu thiếu đó lên Đan Mạch (có đáp án) để chấm. Nếu model tốt ở `whole_feature` nhưng tệ ở `mnar`, model đang giả định sai về lý do dữ liệu thiếu.

**Chia seed thành hai bộ:** validation (ví dụ seed 0–4) để chọn tham số, test (ví dụ seed 100–109) chỉ để báo cáo cuối. Chọn tham số và báo cáo trên cùng một mask làm điểm số đẹp giả.

### 1.2 Chỉ số: mỗi chỉ số trả lời một câu hỏi

| Câu hỏi | Chỉ số | Ví dụ | Trạng thái |
|---|---|---|---|
| Đúng từng ô? | MAE, RMSE trên **tỷ lệ c/N**, nhóm ô do model quyết định | Thật 0,70, điền 0,55 → sai 0,15 | Đã có |
| Đúng cơ cấu cả dòng? (compositional) | Tổng biến thiên TV = ½·Σ\|p̂ − p\| hoặc JSD theo dòng, trên 17 biến thể + Other | Phần dư 15 rơi vào Other → TV = 0,15 | Cần thêm |
| Đoán đúng có mặt / vắng mặt? (zero-inflated) | Precision, recall, F1 cho câu "ô này > 0?" (ngưỡng 0,5 mẫu); MAE riêng nhóm đáp án = 0 và > 0 | Báo cáo audit ghi các model điền số dương vào 80–85% ô có đáp án bằng 0 | Tách nhóm đã có; F1 cần thêm |
| Đúng hình dạng theo thời gian? (longitudinal) | Sai số trên sai phân Δ giữa hai mốc liền kề; tương quan quỹ đạo từng chuỗi bị che | Thật 0,10 → 0,40 → 0,70; điền 0,52 → 0,52 → 0,53: MAE vừa phải nhưng sai hình dạng | Cần thêm |
| Độ bất định (khi có diffusion) | Độ phủ của khoảng 90%, CRPS | — | Để sau |

TV và JSD theo dòng chỉ tính được trên dòng mà đáp án biết đủ (cả Other); phải báo kèm số dòng dùng để tính. Chỉ số thời gian cần cột `time` được đọc như ngày thật.

### 1.3 Thống kê

- **So ghép cặp:** cùng seed, cùng mask, tính hiệu model A − model B; báo "A thắng 4/5 seed, hiệu trung bình −0,006".
- **Bootstrap theo quốc gia:** rút có hoàn lại 150 nước, 1.000 lần, lấy khoảng 2,5%–97,5% của hiệu. Lấy mẫu theo nước vì các ô cùng nước không độc lập.
- **Phân tầng theo N** (nhỏ < 50, vừa, lớn), vì ô có N nhỏ vốn nhiễu hơn.

## Tầng 2: Chỉ số không cần đáp án

Trên 86.050 ô thiếu thật chỉ hỏi được "kết quả có hợp lý không". Bốn phép kiểm tra dưới đây sửa ba điểm yếu của cách làm trong paper và của `evaluate.py` hiện tại.

### 2.1 Tham chiếu cùng bối cảnh (thay cho pre-gap)

`evaluate.py` hiện so với **toàn bộ** quan sát, trộn năm 2020 (chưa có Delta) với năm 2021 (Delta chiếm ưu thế). Thay bằng: với ô thiếu Pháp–Delta ngày 05/07/2021, tham chiếu là giá trị quan sát của Delta ở các nước **cùng châu lục, trong 05/07/2021 ± 4 tuần**. Gộp theo (biến thể, châu lục, cửa sổ thời gian), lấy trung bình có trọng số theo số ô; cửa sổ có dưới 20 quan sát thì bỏ qua và đếm riêng.

### 2.2 Tách phần số 0 và phần dương (two-part)

Với 82% số 0, Wasserstein chủ yếu chỉ đo khối số 0. Tách thành hai số:

1. **Chênh lệch tỷ lệ số 0:** tham chiếu 30% số 0, ô điền 5% số 0 → 0,25 (model điền dương quá nhiều).
2. **Wasserstein trên phần dương:** khi có mặt thì độ lớn có hợp lý không.

### 2.3 So trên composition

Mỗi tuần tính composition trung bình toàn cầu của bảng đã điền và của dữ liệu quan sát, đo JSD giữa hai composition, rồi lấy trung bình qua các tuần. Ví dụ tuần 27/2021: quan sát Alpha 0,15 | Delta 0,75; sau điền Alpha 0,30 | Delta 0,55 → bức tranh toàn cầu bị méo.

### 2.4 Kiểm tra hợp lý về dịch tễ

- **Tỷ lệ "điền lệch thời đại":** từ dữ liệu quan sát lấy ngày xuất hiện đầu tiên toàn cầu của mỗi biến thể (ví dụ Omicron cuối tháng 11/2021); đếm ô điền dương trước ngày đó, hoặc rất lâu sau khi biến thể đã biến mất. Mỗi ô như vậy là sai chắc chắn.
- **Độ mượt:** tự tương quan bậc 1 của chuỗi đã điền so với chuỗi quan sát.
- **Other ngầm:** phân phối phần dư rơi vào Other sau khi điền so với Other quan sát.

Tầng 2 là bằng chứng bổ trợ, chỉ dùng để xếp hạng khi đã qua kiểm định ở Tầng 3.

## Tầng 3: Kiểm định chỉ số không đáp án

Một chỉ số Tầng 2 chỉ được dùng để xếp hạng khi nó xếp hạng giống Tầng 1 và không bị đối thủ giả qua mặt.

1. **Chạy trên dữ liệu che giả.** Ở các lượt Tầng 1, tính cả chỉ số có đáp án (MAE) lẫn chỉ số Tầng 2 (giả vờ không có đáp án, chỉ dùng tham chiếu).
2. **Đo mức đồng thuận.** Xếp hạng các model theo từng chỉ số, tính tương quan thứ hạng Spearman trên nhiều model × seed × kiểu che, kèm khoảng tin cậy.
3. **Thêm đối thủ giả** (xem bảng). Chỉ số nào xếp marginal sampler lên đầu thì bị loại.
4. **Lặp lại với che `mnar`.** Nếu tương quan sụp đổ, Tầng 2 trên ô thiếu thật chỉ được dùng để mô tả, không dùng để kết luận.

| Đối thủ giả | Cách điền | Kỳ vọng |
|---|---|---|
| Marginal sampler | Rút ngẫu nhiên một giá trị quan sát cùng biến thể | Phân phối khớp (Wasserstein ≈ 0) nhưng MAE cao |
| Zero-fill | Điền toàn 0 | Đúng ở ô bằng 0, sai mọi ô dương |
| Mean-fill | Điền tỷ lệ trung bình của biến thể | Phẳng lì, sai hình dạng theo thời gian |

Các đối thủ giả cũng là "mức sàn": baseline thật phải tốt hơn hẳn chúng.

## Tầng 4 (tùy chọn): Đánh giá qua tác vụ downstream

Dùng bảng đã điền để trả lời một câu hỏi thực tế, ví dụ thời điểm Delta vượt 50% ở từng nước hoặc đường cong cơ cấu biến thể toàn cầu. So câu trả lời giữa các phương pháp điền và với số liệu tổng hợp công bố bên ngoài.

## Rà soát baseline hiện tại

Về kỹ thuật, ba baseline (Hron, Tsagris, MICoDa) đã ổn: chạy toàn bộ COVID, giữ nguyên ô quan sát, 115 test pass. Về mặt làm mốc so sánh khoa học thì chưa đủ; có 7 điểm cần sửa hoặc bổ sung, xếp theo mức quan trọng.

| # | Vấn đề | Vì sao quan trọng | Đề xuất |
|---|---|---|---|
| 1 | Thiếu baseline có dùng thời gian | Cả ba baseline xử lý từng dòng độc lập; model chính thắng có thể chỉ nhờ dùng thông tin thời gian, không nhờ kiến trúc | Thêm baseline "trung bình cùng ngày": tỷ lệ biến thể ở các nước cùng châu lục trong cùng tuần, rồi chia phần dư theo tỷ lệ đó; thêm nội suy theo thời gian khi chỉ thiếu vài mốc |
| 2 | Chưa có đối thủ giả | Cần cho Tầng 3 và làm mức sàn | Thêm marginal sampler, zero-fill, mean-fill |
| 3 | Quy tắc chia phần dư khác nhau | Hron/Tsagris chia cho ô thiếu và Other; MICoDa mặc định dồn hết vào ô thiếu (R1), điền thừa 1,3–2,4 lần | Chốt R2/Other làm quy tắc chung (MICoDa đã có `other = true`) |
| 4 | Tsagris thiếu donor | Chỉ 516 dòng đầy đủ từ 5 nước; còn 80–298 khi che `whole_feature`; 1.274 dòng phải dùng trung bình mọi donor | Báo rõ hạn chế; thêm biến thể "donor mở rộng" (nhận donor không đầy đủ nhưng đủ cột cần thiết, giống chiến lược 2a của Hron) |
| 5 | Hron mất thông tin do thay số 0 | Thay 0 bằng số nhỏ cố định khiến 58% ô hòa khoảng cách; hệ số chỉnh quy mô dùng median các ô quan sát, thường chính là số nhỏ đó | Đo số ô bị ảnh hưởng; thử biến thể `adjustment = "sum"` (Eq. 5) |
| 6 | MICoDa chỉ có hệ số chặn | Chia phần dư theo tỷ lệ gần như cố định, không phân biệt giai đoạn dịch | Giữ ngoài bảng xếp hạng; muốn là đối thủ thật thì thêm covariate (tháng, châu lục) |
| 7 | Cách chạy và báo cáo chưa thống nhất | Hron/Tsagris dùng 5 seed, MICoDa dùng 2 seed khác; mỗi model chỉ một cấu hình trong một lượt chạy | Chạy cả ba trên cùng bộ seed; cho phép đặt tên biến thể trong `models.toml` (ví dụ `[tsagris_tuned]`, `[hron_sum]`); chọn tham số trên seed validation, báo cáo trên seed test |

Nếu model chính không thắng nổi baseline "trung bình cùng ngày", kiến trúc chưa đáng giá.

## Kế hoạch triển khai và điểm cần chốt

Thứ tự đề xuất, việc rẻ và cần cho các bước sau được làm trước:

1. Thêm 3 đối thủ giả và baseline "trung bình cùng ngày" vào `run.py`; cho cột `time` được đọc như ngày thật trong `data.py`.
2. Thống nhất quy tắc phần dư R2/Other cho cả ba baseline; cho phép đặt tên biến thể trong `models.toml`; chia seed validation/test.
3. Tầng 1 trong `evaluate.py`: sai số theo dòng (TV/JSD), F1 có/không, chỉ số sai phân thời gian, so ghép cặp và bootstrap theo quốc gia.
4. Tầng 2: tham chiếu cùng bối cảnh, two-part, JSD composition theo tuần, tỷ lệ điền lệch thời đại.
5. Tầng 3: script kiểm định tính tương quan thứ hạng giữa Tầng 1 và Tầng 2.
6. Thêm kiểu che `pattern_transplant` và `mnar` trong `data.py`.
7. Biến thể baseline mạnh hơn: Tsagris donor mở rộng, Hron `sum`, MICoDa có covariate.

**Nhóm cần chốt trước khi làm:**

- Quy tắc chia phần dư chung (đề xuất R2/Other).
- Chỉ số chính (đề xuất: MAE và RMSE trên tỷ lệ, nhóm ô do model quyết định) và kiểu che chính (đề xuất `whole_feature`).
- Tỷ lệ che, số seed validation và test.
- Ngưỡng "có mặt" cho F1 (đề xuất 0,5 mẫu) và độ rộng cửa sổ tham chiếu thời gian (đề xuất ± 4 tuần, theo châu lục).
- Có làm Tầng 4 hay không, và nguồn số liệu bên ngoài nếu làm.
