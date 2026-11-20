# Tier 3 â€” exploratory

Sai lá»‡ch giao thá»©c: seed 300â€“309 thay 0â€“4; thĂªm main (metadata 4 trÆ°á»ng) vĂ  main_nometa. KhĂ´ng Ă¡p dá»¥ng cá»•ng a17/tĂªn run/git sáº¡ch. Scorer vĂ  ngÆ°á»¡ng giá»¯ nguyĂªn.

U1: thá»© háº¡ng cháº©n Ä‘oĂ¡n cĂ³ khá»›p sai sá»‘ tháº­t? U2: chá»n theo cháº©n Ä‘oĂ¡n cĂ³ gáº§n model tá»‘t nháº¥t? U3: tĂ­n hiá»‡u ngoĂ i khoáº£ng thá»i gian cĂ³ phĂ¢n biá»‡t Ă´ dÆ°Æ¡ng? U4: decoy cĂ³ bá»‹ Ä‘Ă¡nh giĂ¡ tá»‡ hÆ¡n model tháº­t?

TIE=0.001; PASS_RATE=0.8; MIN_CASE_RATE=0.5; REGRET_MAX=0.005; U3_GAP=0.2; GOOD_CASES=9

## whole_feature â€” exploratory

| Mask | Seed | thiếu: khớp (khớp/quyết định) | thừa: khớp |
|---|---:|---|---|
| whole_feature | 300 | 0.57 (8/14, hòa 1) | 0.80 (12/15) |
| whole_feature | 301 | 0.67 (10/15) | 0.73 (11/15) |
| whole_feature | 302 | 0.53 (8/15) | 0.80 (12/15) |
| whole_feature | 303 | 0.43 (6/14, hòa 1) | 0.80 (12/15) |
| whole_feature | 304 | 0.40 (6/15) | 0.87 (13/15) |
| whole_feature | 305 | 0.67 (10/15) | 0.71 (10/14, hòa 1) |
| whole_feature | 306 | 0.53 (8/15) | 0.53 (8/15) |
| whole_feature | 307 | 0.47 (7/15) | 0.73 (11/15) |
| whole_feature | 308 | 0.67 (10/15) | 0.64 (9/14, hòa 1) |
| whole_feature | 309 | 0.71 (10/14, hòa 1) | 0.60 (9/15) |
- U1 thiếu (D1 cao ↔ MAE ô đáp án > 0): whole_feature: TB 0.56, min 0.40 (10 ca) → **không đạt**.
- U1 thừa (D1 thấp ↔ MAE ô đáp án = 0): whole_feature: TB 0.72, min 0.53 (10 ca) → **không đạt**.

| Mask | Seed | khớp \|D2 − D2t\| | chọn → regret | khớp D2 thô | regret thô | khớp, MAE toàn ô model | regret, MAE toàn ô model |
|---|---:|---|---|---|---:|---|---:|
| whole_feature | 300 | 0.73 (11/15) | main_nometa → 0.0260 | 0.73 (11/15) | 0.0260 | 0.73 (11/15) | 0.0267 |
| whole_feature | 301 | 0.93 (14/15) | same_day_mean → 0.0000 | 0.93 (14/15) | 0.0000 | 0.93 (14/15) | 0.0000 |
| whole_feature | 302 | 0.87 (13/15) | main_nometa → 0.0081 | 0.87 (13/15) | 0.0081 | 0.86 (12/14, hòa 1) | 0.0101 |
| whole_feature | 303 | 0.80 (12/15) | main → 0.0145 | 0.80 (12/15) | 0.0145 | 0.80 (12/15) | 0.0166 |
| whole_feature | 304 | 0.93 (14/15) | same_day_mean → 0.0000 | 0.93 (14/15) | 0.0000 | 1.00 (15/15) | 0.0000 |
| whole_feature | 305 | 0.93 (13/14, hòa 1) | same_day_mean → 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 | 0.92 (12/13, hòa 2) | 0.0000 |
| whole_feature | 306 | 0.93 (14/15) | same_day_mean → 0.0000 | 0.93 (14/15) | 0.0000 | 0.93 (14/15) | 0.0000 |
| whole_feature | 307 | 0.67 (10/15) | same_day_mean → 0.0000 | 0.67 (10/15) | 0.0000 | 0.67 (10/15) | 0.0000 |
| whole_feature | 308 | 0.93 (14/15) | same_day_mean → 0.0000 | 0.93 (14/15) | 0.0000 | 0.93 (14/15) | 0.0000 |
| whole_feature | 309 | 1.00 (14/14, hòa 1) | same_day_mean → 0.0000 | 1.00 (14/14, hòa 1) | 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 |
- U2: whole_feature: TB 0.87, min 0.67 (10 ca); regret ≤ 0.005 ở 7/10 ca → **không đạt**.

| Mask | Seed | ô dương bị cờ | tỷ lệ đáp án 0 khi bị cờ | ô dương không cờ | khi không cờ | chênh |
|---|---:|---:|---:|---:|---:|---:|
| whole_feature | 300 | 686 | 1.000 | 6643 | 0.241 | +0.759 |
| whole_feature | 301 | 674 | 1.000 | 6028 | 0.310 | +0.690 |
| whole_feature | 302 | 1142 | 0.954 | 7349 | 0.231 | +0.723 |
| whole_feature | 303 | 1029 | 0.998 | 6561 | 0.286 | +0.712 |
| whole_feature | 304 | 1007 | 0.991 | 6334 | 0.351 | +0.640 |
| whole_feature | 305 | 929 | 0.981 | 5948 | 0.279 | +0.702 |
| whole_feature | 306 | 984 | 0.988 | 6632 | 0.305 | +0.683 |
| whole_feature | 307 | 1003 | 0.995 | 6378 | 0.286 | +0.709 |
| whole_feature | 308 | 799 | 0.994 | 6986 | 0.241 | +0.753 |
| whole_feature | 309 | 1047 | 0.982 | 6961 | 0.310 | +0.672 |
- U3: chênh ≥ 0.2 ở 10/10 ca → **đạt**.

**D1 với ngưỡng zero c/N < 0.001 (R-local)**

| Mask | Seed | thiếu: khớp (khớp/quyết định) | thừa: khớp |
|---|---:|---|---|
| whole_feature | 300 | 0.50 (7/14, hòa 1) | 0.87 (13/15) |
| whole_feature | 301 | 0.47 (7/15) | 0.80 (12/15) |
| whole_feature | 302 | 0.47 (7/15) | 0.73 (11/15) |
| whole_feature | 303 | 0.36 (5/14, hòa 1) | 0.73 (11/15) |
| whole_feature | 304 | 0.40 (6/15) | 0.87 (13/15) |
| whole_feature | 305 | 0.53 (8/15) | 0.86 (12/14, hòa 1) |
| whole_feature | 306 | 0.47 (7/15) | 0.60 (9/15) |
| whole_feature | 307 | 0.40 (6/15) | 0.80 (12/15) |
| whole_feature | 308 | 0.47 (7/15) | 0.71 (10/14, hòa 1) |
| whole_feature | 309 | 0.57 (8/14, hòa 1) | 0.73 (11/15) |
- U1 thiếu (D1 cao ↔ MAE ô đáp án > 0): whole_feature: TB 0.46, min 0.36 (10 ca) → **không đạt**.
- U1 thừa (D1 thấp ↔ MAE ô đáp án = 0): whole_feature: TB 0.77, min 0.60 (10 ca) → **không đạt**.

**D1 với R-pooled**

| Mask | Seed | thiếu: khớp (khớp/quyết định) | thừa: khớp |
|---|---:|---|---|
| whole_feature | 300 | 0.60 (9/15) | 0.80 (12/15) |
| whole_feature | 301 | 0.60 (9/15) | 0.73 (11/15) |
| whole_feature | 302 | 0.53 (8/15) | 0.80 (12/15) |
| whole_feature | 303 | 0.42 (5/12, hòa 3) | 0.73 (11/15) |
| whole_feature | 304 | 0.40 (6/15) | 0.87 (13/15) |
| whole_feature | 305 | 0.60 (9/15) | 0.86 (12/14, hòa 1) |
| whole_feature | 306 | 0.53 (8/15) | 0.53 (8/15) |
| whole_feature | 307 | 0.47 (7/15) | 0.73 (11/15) |
| whole_feature | 308 | 0.67 (10/15) | 0.60 (9/15) |
| whole_feature | 309 | 0.71 (10/14, hòa 1) | 0.60 (9/15) |
- U1 thiếu (D1 cao ↔ MAE ô đáp án > 0): whole_feature: TB 0.55, min 0.40 (10 ca) → **không đạt**.
- U1 thừa (D1 thấp ↔ MAE ô đáp án = 0): whole_feature: TB 0.73, min 0.53 (10 ca) → **không đạt**.

**D2 với R-pooled**

| Mask | Seed | khớp \|D2 − D2t\| | chọn → regret | khớp D2 thô | regret thô | khớp, MAE toàn ô model | regret, MAE toàn ô model |
|---|---:|---|---|---|---:|---|---:|
| whole_feature | 300 | 0.80 (12/15) | same_day_mean → 0.0000 | 0.80 (12/15) | 0.0000 | 0.80 (12/15) | 0.0000 |
| whole_feature | 301 | 0.93 (14/15) | same_day_mean → 0.0000 | 0.93 (14/15) | 0.0000 | 0.93 (14/15) | 0.0000 |
| whole_feature | 302 | 0.93 (14/15) | same_day_mean → 0.0000 | 0.87 (13/15) | 0.0082 | 0.93 (13/14, hòa 1) | 0.0000 |
| whole_feature | 303 | 0.73 (11/15) | main → 0.0191 | 0.73 (11/15) | 0.0191 | 0.73 (11/15) | 0.0166 |
| whole_feature | 304 | 1.00 (14/14, hòa 1) | same_day_mean → 0.0000 | 1.00 (14/14, hòa 1) | 0.0000 | 1.00 (15/15) | 0.0000 |
| whole_feature | 305 | 0.92 (12/13, hòa 2) | same_day_mean → 0.0000 | 0.92 (12/13, hòa 2) | 0.0000 | 0.92 (12/13, hòa 2) | 0.0000 |
| whole_feature | 306 | 0.93 (14/15) | same_day_mean → 0.0000 | 0.93 (14/15) | 0.0000 | 0.93 (14/15) | 0.0000 |
| whole_feature | 307 | 0.87 (13/15) | same_day_mean → 0.0000 | 0.87 (13/15) | 0.0000 | 0.87 (13/15) | 0.0000 |
| whole_feature | 308 | 0.93 (13/14, hòa 1) | same_day_mean → 0.0008 | 0.93 (13/14, hòa 1) | 0.0008 | 0.93 (14/15) | 0.0000 |
| whole_feature | 309 | 1.00 (14/14, hòa 1) | same_day_mean → 0.0000 | 1.00 (14/14, hòa 1) | 0.0000 | 1.00 (14/14, hòa 1) | 0.0000 |
- U2: whole_feature: TB 0.91, min 0.73 (10 ca); regret ≤ 0.005 ở 9/10 ca → **đạt**.

Số ca diagnostic xếp đối thủ giả kém hơn **mọi** model thật (trên số ca có giá trị), R-local.

| Đối thủ giả | Mask | \|D1 − D1_truth\| | \|D1\| thô | \|D2 − D2_truth\| | D2 thô | D3: ô cờ / ô E | *tham chiếu: MAE Tầng 1, toàn ô model* |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| mean_fill | whole_feature | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| zero_fill | whole_feature | 10/10 | 9/10 | NA | NA | 0/10 | 10/10 |
| marginal_sampler | whole_feature | 7/10 | 0/10 | 0/9 | 0/9 | 0/10 | 10/10 |

Giá trị trung bình qua 10 seed (R-local, trên target chung của model thật):

| Model | Mask | \|D1 − D1_truth\| | \|D1\| thô | \|D2 − D2_truth\| | D2 thô | D3: ô cờ / ô E | *tham chiếu: MAE Tầng 1, toàn ô model* |
|---|---|---:|---:|---:|---:|---:|---:|
| hron | whole_feature | 0.0248 ± 0.0170 | 0.1005 ± 0.0128 | 0.0681 ± 0.0262 | 0.1256 ± 0.0213 | 0.0250 ± 0.0117 | 0.0897 ± 0.0130 |
| tsagris | whole_feature | 0.1167 ± 0.0371 | 0.2004 ± 0.0301 | 0.0856 ± 0.0309 | 0.1431 ± 0.0216 | 0.0784 ± 0.0101 | 0.1009 ± 0.0104 |
| micoda | whole_feature | 0.1445 ± 0.0379 | 0.2120 ± 0.0384 | 0.1579 ± 0.0350 | 0.2154 ± 0.0382 | 0.1281 ± 0.0192 | 0.1043 ± 0.0091 |
| same_day_mean | whole_feature | 0.0406 ± 0.0166 | 0.0997 ± 0.0139 | 0.0234 ± 0.0159 | 0.0809 ± 0.0148 | 0.0027 ± 0.0023 | 0.0311 ± 0.0057 |
| main | whole_feature | 0.1397 ± 0.0252 | 0.2222 ± 0.0247 | 0.0438 ± 0.0298 | 0.1012 ± 0.0294 | 0.0557 ± 0.0154 | 0.0471 ± 0.0101 |
| main_nometa | whole_feature | 0.0916 ± 0.0613 | 0.1957 ± 0.0535 | 0.0417 ± 0.0337 | 0.0990 ± 0.0329 | 0.0448 ± 0.0175 | 0.0516 ± 0.0140 |
| mean_fill | whole_feature | 0.3296 ± 0.0424 | 0.3864 ± 0.0428 | 0.2043 ± 0.0275 | 0.2618 ± 0.0295 | 0.1449 ± 0.0186 | 0.1339 ± 0.0114 |
| zero_fill | whole_feature | 0.3736 ± 0.0240 | 0.3204 ± 0.0227 | NA | NA | 0.0000 ± 0.0000 | 0.1717 ± 0.0222 |
| marginal_sampler | whole_feature | 0.1840 ± 0.0123 | 0.1320 ± 0.0112 | 0.0603 ± 0.0268 (n = 9) | 0.1176 ± 0.0189 (n = 9) | 0.0447 ± 0.0059 | 0.1352 ± 0.0113 |
## random_cell â€” exploratory

| Mask | Seed | thiếu: khớp (khớp/quyết định) | thừa: khớp |
|---|---:|---|---|
| random_cell | 300 | 0.47 (7/15) | 0.86 (12/14, hòa 1) |
| random_cell | 301 | 0.53 (8/15) | 0.64 (9/14, hòa 1) |
| random_cell | 302 | 0.53 (8/15) | 0.71 (10/14, hòa 1) |
| random_cell | 303 | 0.40 (6/15) | 0.80 (12/15) |
| random_cell | 304 | 0.40 (6/15) | 0.86 (12/14, hòa 1) |
| random_cell | 305 | 0.40 (6/15) | 0.87 (13/15) |
| random_cell | 306 | 0.40 (6/15) | 0.64 (9/14, hòa 1) |
| random_cell | 307 | 0.40 (6/15) | 0.86 (12/14, hòa 1) |
| random_cell | 308 | 0.40 (6/15) | 0.71 (10/14, hòa 1) |
| random_cell | 309 | 0.46 (6/13, hòa 2) | 0.62 (8/13, hòa 2) |
- U1 thiếu (D1 cao ↔ MAE ô đáp án > 0): random_cell: TB 0.44, min 0.40 (10 ca) → **không đạt**.
- U1 thừa (D1 thấp ↔ MAE ô đáp án = 0): random_cell: TB 0.76, min 0.62 (10 ca) → **không đạt**.

| Mask | Seed | khớp \|D2 − D2t\| | chọn → regret | khớp D2 thô | regret thô | khớp, MAE toàn ô model | regret, MAE toàn ô model |
|---|---:|---|---|---|---:|---|---:|
| random_cell | 300 | 0.87 (13/15) | same_day_mean → 0.0078 | 0.87 (13/15) | 0.0078 | 0.87 (13/15) | 0.0075 |
| random_cell | 301 | 0.93 (14/15) | main → 0.0013 | 0.93 (14/15) | 0.0013 | 0.93 (14/15) | 0.0016 |
| random_cell | 302 | 0.87 (13/15) | main → 0.0012 | 0.87 (13/15) | 0.0012 | 0.87 (13/15) | 0.0012 |
| random_cell | 303 | 0.93 (14/15) | main → 0.0053 | 0.93 (14/15) | 0.0053 | 0.93 (14/15) | 0.0032 |
| random_cell | 304 | 1.00 (15/15) | main_nometa → 0.0000 | 1.00 (15/15) | 0.0000 | 1.00 (15/15) | 0.0000 |
| random_cell | 305 | 1.00 (15/15) | main_nometa → 0.0000 | 1.00 (15/15) | 0.0000 | 1.00 (15/15) | 0.0000 |
| random_cell | 306 | 0.80 (12/15) | same_day_mean → 0.0086 | 0.80 (12/15) | 0.0086 | 0.80 (12/15) | 0.0082 |
| random_cell | 307 | 1.00 (14/14, hòa 1) | main → 0.0000 | 1.00 (14/14, hòa 1) | 0.0000 | 1.00 (15/15) | 0.0000 |
| random_cell | 308 | 1.00 (15/15) | main → 0.0000 | 1.00 (15/15) | 0.0000 | 1.00 (15/15) | 0.0000 |
| random_cell | 309 | 1.00 (14/14, hòa 1) | main_nometa → 0.0000 | 1.00 (14/14, hòa 1) | 0.0000 | 1.00 (14/14, hòa 1) | 0.0000 |
- U2: random_cell: TB 0.94, min 0.80 (10 ca); regret ≤ 0.005 ở 7/10 ca → **không đạt**.

| Mask | Seed | ô dương bị cờ | tỷ lệ đáp án 0 khi bị cờ | ô dương không cờ | khi không cờ | chênh |
|---|---:|---:|---:|---:|---:|---:|
| random_cell | 300 | 773 | 1.000 | 6607 | 0.232 | +0.768 |
| random_cell | 301 | 744 | 0.997 | 6424 | 0.237 | +0.760 |
| random_cell | 302 | 796 | 0.996 | 6722 | 0.252 | +0.745 |
| random_cell | 303 | 727 | 0.993 | 6336 | 0.226 | +0.767 |
| random_cell | 304 | 677 | 0.999 | 6558 | 0.266 | +0.732 |
| random_cell | 305 | 741 | 0.992 | 6323 | 0.234 | +0.758 |
| random_cell | 306 | 779 | 1.000 | 6400 | 0.240 | +0.760 |
| random_cell | 307 | 724 | 0.999 | 6303 | 0.242 | +0.757 |
| random_cell | 308 | 685 | 0.997 | 6605 | 0.238 | +0.759 |
| random_cell | 309 | 738 | 1.000 | 6422 | 0.237 | +0.763 |
- U3: chênh ≥ 0.2 ở 10/10 ca → **đạt**.

**D1 với ngưỡng zero c/N < 0.001 (R-local)**

| Mask | Seed | thiếu: khớp (khớp/quyết định) | thừa: khớp |
|---|---:|---|---|
| random_cell | 300 | 0.47 (7/15) | 0.71 (10/14, hòa 1) |
| random_cell | 301 | 0.40 (6/15) | 0.71 (10/14, hòa 1) |
| random_cell | 302 | 0.33 (5/15) | 0.57 (8/14, hòa 1) |
| random_cell | 303 | 0.40 (6/15) | 0.67 (10/15) |
| random_cell | 304 | 0.40 (6/15) | 0.71 (10/14, hòa 1) |
| random_cell | 305 | 0.33 (5/15) | 0.67 (10/15) |
| random_cell | 306 | 0.40 (6/15) | 0.64 (9/14, hòa 1) |
| random_cell | 307 | 0.40 (6/15) | 0.71 (10/14, hòa 1) |
| random_cell | 308 | 0.40 (6/15) | 0.71 (10/14, hòa 1) |
| random_cell | 309 | 0.31 (4/13, hòa 2) | 0.62 (8/13, hòa 2) |
- U1 thiếu (D1 cao ↔ MAE ô đáp án > 0): random_cell: TB 0.38, min 0.31 (10 ca) → **không đạt**.
- U1 thừa (D1 thấp ↔ MAE ô đáp án = 0): random_cell: TB 0.67, min 0.57 (10 ca) → **không đạt**.

**D1 với R-pooled**

| Mask | Seed | thiếu: khớp (khớp/quyết định) | thừa: khớp |
|---|---:|---|---|
| random_cell | 300 | 0.47 (7/15) | 0.86 (12/14, hòa 1) |
| random_cell | 301 | 0.50 (7/14, hòa 1) | 0.57 (8/14, hòa 1) |
| random_cell | 302 | 0.53 (8/15) | 0.71 (10/14, hòa 1) |
| random_cell | 303 | 0.33 (5/15) | 0.87 (13/15) |
| random_cell | 304 | 0.40 (6/15) | 0.86 (12/14, hòa 1) |
| random_cell | 305 | 0.33 (5/15) | 0.80 (12/15) |
| random_cell | 306 | 0.40 (6/15) | 0.62 (8/13, hòa 2) |
| random_cell | 307 | 0.40 (6/15) | 0.86 (12/14, hòa 1) |
| random_cell | 308 | 0.40 (6/15) | 0.71 (10/14, hòa 1) |
| random_cell | 309 | 0.46 (6/13, hòa 2) | 0.62 (8/13, hòa 2) |
- U1 thiếu (D1 cao ↔ MAE ô đáp án > 0): random_cell: TB 0.42, min 0.33 (10 ca) → **không đạt**.
- U1 thừa (D1 thấp ↔ MAE ô đáp án = 0): random_cell: TB 0.75, min 0.57 (10 ca) → **không đạt**.

**D2 với R-pooled**

| Mask | Seed | khớp \|D2 − D2t\| | chọn → regret | khớp D2 thô | regret thô | khớp, MAE toàn ô model | regret, MAE toàn ô model |
|---|---:|---|---|---|---:|---|---:|
| random_cell | 300 | 0.87 (13/15) | same_day_mean → 0.0076 | 0.87 (13/15) | 0.0076 | 0.87 (13/15) | 0.0075 |
| random_cell | 301 | 0.93 (14/15) | main → 0.0016 | 0.93 (14/15) | 0.0016 | 0.93 (14/15) | 0.0016 |
| random_cell | 302 | 0.87 (13/15) | main → 0.0012 | 0.87 (13/15) | 0.0012 | 0.87 (13/15) | 0.0012 |
| random_cell | 303 | 0.93 (14/15) | main → 0.0050 | 0.93 (14/15) | 0.0050 | 0.93 (14/15) | 0.0032 |
| random_cell | 304 | 1.00 (15/15) | main_nometa → 0.0000 | 1.00 (15/15) | 0.0000 | 1.00 (15/15) | 0.0000 |
| random_cell | 305 | 1.00 (15/15) | main_nometa → 0.0000 | 1.00 (15/15) | 0.0000 | 1.00 (15/15) | 0.0000 |
| random_cell | 306 | 0.87 (13/15) | main_nometa → 0.0013 | 0.87 (13/15) | 0.0013 | 0.87 (13/15) | 0.0012 |
| random_cell | 307 | 1.00 (14/14, hòa 1) | main → 0.0000 | 1.00 (14/14, hòa 1) | 0.0000 | 1.00 (15/15) | 0.0000 |
| random_cell | 308 | 1.00 (15/15) | main → 0.0000 | 1.00 (15/15) | 0.0000 | 1.00 (15/15) | 0.0000 |
| random_cell | 309 | 1.00 (14/14, hòa 1) | main_nometa → 0.0000 | 1.00 (14/14, hòa 1) | 0.0000 | 1.00 (14/14, hòa 1) | 0.0000 |
- U2: random_cell: TB 0.95, min 0.87 (10 ca); regret ≤ 0.005 ở 8/10 ca → **không đạt**.

Số ca diagnostic xếp đối thủ giả kém hơn **mọi** model thật (trên số ca có giá trị), R-local.

| Đối thủ giả | Mask | \|D1 − D1_truth\| | \|D1\| thô | \|D2 − D2_truth\| | D2 thô | D3: ô cờ / ô E | *tham chiếu: MAE Tầng 1, toàn ô model* |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| mean_fill | random_cell | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| zero_fill | random_cell | 10/10 | 10/10 | NA | NA | 0/10 | 10/10 |
| marginal_sampler | random_cell | 9/10 | 0/10 | 0/3 | 0/3 | 0/10 | 10/10 |

Giá trị trung bình qua 10 seed (R-local, trên target chung của model thật):

| Model | Mask | \|D1 − D1_truth\| | \|D1\| thô | \|D2 − D2_truth\| | D2 thô | D3: ô cờ / ô E | *tham chiếu: MAE Tầng 1, toàn ô model* |
|---|---|---:|---:|---:|---:|---:|---:|
| hron | random_cell | 0.0113 ± 0.0080 | 0.0821 ± 0.0075 | 0.0698 ± 0.0084 | 0.1040 ± 0.0075 | 0.0253 ± 0.0030 | 0.0906 ± 0.0034 |
| tsagris | random_cell | 0.1515 ± 0.0115 | 0.2067 ± 0.0077 | 0.0850 ± 0.0093 | 0.1192 ± 0.0082 | 0.0789 ± 0.0039 | 0.0993 ± 0.0030 |
| micoda | random_cell | 0.1433 ± 0.0075 | 0.1969 ± 0.0076 | 0.1668 ± 0.0058 | 0.2010 ± 0.0059 | 0.1300 ± 0.0043 | 0.1069 ± 0.0025 |
| same_day_mean | random_cell | 0.0425 ± 0.0080 | 0.0963 ± 0.0060 | 0.0375 ± 0.0049 | 0.0717 ± 0.0043 | 0.0017 ± 0.0007 | 0.0304 ± 0.0020 |
| main | random_cell | 0.0338 ± 0.0145 | 0.1047 ± 0.0131 | 0.0283 ± 0.0076 | 0.0625 ± 0.0074 | 0.0152 ± 0.0046 | 0.0237 ± 0.0018 |
| main_nometa | random_cell | 0.0331 ± 0.0276 | 0.1171 ± 0.0194 | 0.0327 ± 0.0116 | 0.0669 ± 0.0141 | 0.0131 ± 0.0047 | 0.0233 ± 0.0029 |
| mean_fill | random_cell | 0.3407 ± 0.0086 | 0.3938 ± 0.0059 | 0.2102 ± 0.0079 | 0.2444 ± 0.0073 | 0.1503 ± 0.0048 | 0.1373 ± 0.0029 |
| zero_fill | random_cell | 0.3757 ± 0.0067 | 0.3225 ± 0.0057 | NA | NA | 0.0000 ± 0.0000 | 0.1731 ± 0.0053 |
| marginal_sampler | random_cell | 0.1808 ± 0.0120 | 0.1277 ± 0.0121 | 0.0628 ± 0.0067 (n = 3) | 0.0954 ± 0.0058 (n = 3) | 0.0484 ± 0.0024 | 0.1384 ± 0.0057 |
## whole_feature_matched â€” exploratory

| Mask | Seed | thiếu: khớp (khớp/quyết định) | thừa: khớp |
|---|---:|---|---|
| whole_feature_matched | 300 | 0.60 (9/15) | 0.87 (13/15) |
| whole_feature_matched | 301 | 0.67 (10/15) | 0.73 (11/15) |
| whole_feature_matched | 302 | 0.47 (7/15) | 0.73 (11/15) |
| whole_feature_matched | 303 | 0.40 (6/15) | 0.73 (11/15) |
| whole_feature_matched | 304 | 0.53 (8/15) | 0.80 (12/15) |
| whole_feature_matched | 305 | 0.60 (9/15) | 0.73 (11/15) |
| whole_feature_matched | 306 | 0.60 (9/15) | 0.80 (12/15) |
| whole_feature_matched | 307 | 0.60 (9/15) | 0.67 (10/15) |
| whole_feature_matched | 308 | 0.47 (7/15) | 0.79 (11/14, hòa 1) |
| whole_feature_matched | 309 | 0.60 (9/15) | 0.80 (12/15) |
- U1 thiếu (D1 cao ↔ MAE ô đáp án > 0): whole_feature_matched: TB 0.55, min 0.40 (10 ca) → **không đạt**.
- U1 thừa (D1 thấp ↔ MAE ô đáp án = 0): whole_feature_matched: TB 0.77, min 0.67 (10 ca) → **không đạt**.

| Mask | Seed | khớp \|D2 − D2t\| | chọn → regret | khớp D2 thô | regret thô | khớp, MAE toàn ô model | regret, MAE toàn ô model |
|---|---:|---|---|---|---:|---|---:|
| whole_feature_matched | 300 | 0.80 (12/15) | same_day_mean → 0.0000 | 0.80 (12/15) | 0.0000 | 0.73 (11/15) | 0.0000 |
| whole_feature_matched | 301 | 0.73 (11/15) | main → 0.0225 | 0.73 (11/15) | 0.0225 | 0.73 (11/15) | 0.0191 |
| whole_feature_matched | 302 | 0.87 (13/15) | same_day_mean → 0.0000 | 0.87 (13/15) | 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 |
| whole_feature_matched | 303 | 1.00 (15/15) | same_day_mean → 0.0000 | 1.00 (15/15) | 0.0000 | 1.00 (15/15) | 0.0000 |
| whole_feature_matched | 304 | 0.87 (13/15) | same_day_mean → 0.0000 | 0.87 (13/15) | 0.0000 | 0.87 (13/15) | 0.0000 |
| whole_feature_matched | 305 | 0.80 (12/15) | same_day_mean → 0.0000 | 0.80 (12/15) | 0.0000 | 0.80 (12/15) | 0.0000 |
| whole_feature_matched | 306 | 1.00 (13/13, hòa 2) | same_day_mean → 0.0000 | 1.00 (13/13, hòa 2) | 0.0000 | 1.00 (14/14, hòa 1) | 0.0002 |
| whole_feature_matched | 307 | 0.87 (13/15) | same_day_mean → 0.0000 | 0.87 (13/15) | 0.0000 | 0.87 (13/15) | 0.0000 |
| whole_feature_matched | 308 | 0.79 (11/14, hòa 1) | main → 0.0186 | 0.79 (11/14, hòa 1) | 0.0186 | 0.73 (11/15) | 0.0166 |
| whole_feature_matched | 309 | 0.93 (13/14, hòa 1) | same_day_mean → 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 |
- U2: whole_feature_matched: TB 0.86, min 0.73 (10 ca); regret ≤ 0.005 ở 8/10 ca → **không đạt**.

| Mask | Seed | ô dương bị cờ | tỷ lệ đáp án 0 khi bị cờ | ô dương không cờ | khi không cờ | chênh |
|---|---:|---:|---:|---:|---:|---:|
| whole_feature_matched | 300 | 783 | 0.991 | 6260 | 0.335 | +0.657 |
| whole_feature_matched | 301 | 783 | 0.996 | 5809 | 0.308 | +0.688 |
| whole_feature_matched | 302 | 1367 | 0.990 | 7234 | 0.246 | +0.743 |
| whole_feature_matched | 303 | 615 | 0.995 | 4786 | 0.378 | +0.617 |
| whole_feature_matched | 304 | 963 | 1.000 | 6439 | 0.253 | +0.747 |
| whole_feature_matched | 305 | 869 | 0.993 | 6471 | 0.259 | +0.734 |
| whole_feature_matched | 306 | 1023 | 0.996 | 6332 | 0.297 | +0.699 |
| whole_feature_matched | 307 | 1105 | 0.996 | 7017 | 0.219 | +0.777 |
| whole_feature_matched | 308 | 829 | 1.000 | 7196 | 0.249 | +0.751 |
| whole_feature_matched | 309 | 636 | 1.000 | 5652 | 0.309 | +0.691 |
- U3: chênh ≥ 0.2 ở 10/10 ca → **đạt**.

**D1 với ngưỡng zero c/N < 0.001 (R-local)**

| Mask | Seed | thiếu: khớp (khớp/quyết định) | thừa: khớp |
|---|---:|---|---|
| whole_feature_matched | 300 | 0.53 (8/15) | 0.80 (12/15) |
| whole_feature_matched | 301 | 0.47 (7/15) | 0.80 (12/15) |
| whole_feature_matched | 302 | 0.47 (7/15) | 0.73 (11/15) |
| whole_feature_matched | 303 | 0.33 (5/15) | 0.80 (12/15) |
| whole_feature_matched | 304 | 0.60 (9/15) | 0.73 (11/15) |
| whole_feature_matched | 305 | 0.53 (8/15) | 0.80 (12/15) |
| whole_feature_matched | 306 | 0.47 (7/15) | 0.73 (11/15) |
| whole_feature_matched | 307 | 0.47 (7/15) | 0.67 (10/15) |
| whole_feature_matched | 308 | 0.60 (9/15) | 0.64 (9/14, hòa 1) |
| whole_feature_matched | 309 | 0.53 (8/15) | 0.73 (11/15) |
- U1 thiếu (D1 cao ↔ MAE ô đáp án > 0): whole_feature_matched: TB 0.50, min 0.33 (10 ca) → **không đạt**.
- U1 thừa (D1 thấp ↔ MAE ô đáp án = 0): whole_feature_matched: TB 0.74, min 0.64 (10 ca) → **không đạt**.

**D1 với R-pooled**

| Mask | Seed | thiếu: khớp (khớp/quyết định) | thừa: khớp |
|---|---:|---|---|
| whole_feature_matched | 300 | 0.67 (10/15) | 0.80 (12/15) |
| whole_feature_matched | 301 | 0.67 (10/15) | 0.73 (11/15) |
| whole_feature_matched | 302 | 0.40 (6/15) | 0.80 (12/15) |
| whole_feature_matched | 303 | 0.47 (7/15) | 0.80 (12/15) |
| whole_feature_matched | 304 | 0.53 (8/15) | 0.80 (12/15) |
| whole_feature_matched | 305 | 0.60 (9/15) | 0.73 (11/15) |
| whole_feature_matched | 306 | 0.67 (10/15) | 0.80 (12/15) |
| whole_feature_matched | 307 | 0.60 (9/15) | 0.67 (10/15) |
| whole_feature_matched | 308 | 0.47 (7/15) | 0.79 (11/14, hòa 1) |
| whole_feature_matched | 309 | 0.60 (9/15) | 0.80 (12/15) |
- U1 thiếu (D1 cao ↔ MAE ô đáp án > 0): whole_feature_matched: TB 0.57, min 0.40 (10 ca) → **không đạt**.
- U1 thừa (D1 thấp ↔ MAE ô đáp án = 0): whole_feature_matched: TB 0.77, min 0.67 (10 ca) → **không đạt**.

**D2 với R-pooled**

| Mask | Seed | khớp \|D2 − D2t\| | chọn → regret | khớp D2 thô | regret thô | khớp, MAE toàn ô model | regret, MAE toàn ô model |
|---|---:|---|---|---|---:|---|---:|
| whole_feature_matched | 300 | 0.67 (10/15) | hron → 0.0696 | 0.67 (10/15) | 0.0696 | 0.67 (10/15) | 0.0634 |
| whole_feature_matched | 301 | 0.67 (10/15) | same_day_mean → 0.0000 | 0.67 (10/15) | 0.0110 | 0.67 (10/15) | 0.0000 |
| whole_feature_matched | 302 | 0.80 (12/15) | same_day_mean → 0.0000 | 0.80 (12/15) | 0.0000 | 0.86 (12/14, hòa 1) | 0.0000 |
| whole_feature_matched | 303 | 0.87 (13/15) | main_nometa → 0.0378 | 0.87 (13/15) | 0.0378 | 0.87 (13/15) | 0.0356 |
| whole_feature_matched | 304 | 0.93 (14/15) | same_day_mean → 0.0000 | 0.93 (14/15) | 0.0000 | 0.93 (14/15) | 0.0000 |
| whole_feature_matched | 305 | 0.73 (11/15) | same_day_mean → 0.0000 | 0.73 (11/15) | 0.0000 | 0.73 (11/15) | 0.0000 |
| whole_feature_matched | 306 | 0.93 (14/15) | same_day_mean → 0.0017 | 0.93 (14/15) | 0.0017 | 1.00 (14/14, hòa 1) | 0.0002 |
| whole_feature_matched | 307 | 0.80 (12/15) | main → 0.0027 | 0.87 (13/15) | 0.0000 | 0.80 (12/15) | 0.0029 |
| whole_feature_matched | 308 | 0.80 (12/15) | same_day_mean → 0.0000 | 0.80 (12/15) | 0.0000 | 0.80 (12/15) | 0.0000 |
| whole_feature_matched | 309 | 0.86 (12/14, hòa 1) | main_nometa → 0.0038 | 0.86 (12/14, hòa 1) | 0.0038 | 0.86 (12/14, hòa 1) | 0.0029 |
- U2: whole_feature_matched: TB 0.81, min 0.67 (10 ca); regret ≤ 0.005 ở 8/10 ca → **không đạt**.

Số ca diagnostic xếp đối thủ giả kém hơn **mọi** model thật (trên số ca có giá trị), R-local.

| Đối thủ giả | Mask | \|D1 − D1_truth\| | \|D1\| thô | \|D2 − D2_truth\| | D2 thô | D3: ô cờ / ô E | *tham chiếu: MAE Tầng 1, toàn ô model* |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| mean_fill | whole_feature_matched | 10/10 | 10/10 | 10/10 | 10/10 | 9/10 | 10/10 |
| zero_fill | whole_feature_matched | 10/10 | 10/10 | NA | NA | 0/10 | 10/10 |
| marginal_sampler | whole_feature_matched | 4/10 | 0/10 | 0/9 | 0/9 | 0/10 | 10/10 |

Giá trị trung bình qua 10 seed (R-local, trên target chung của model thật):

| Model | Mask | \|D1 − D1_truth\| | \|D1\| thô | \|D2 − D2_truth\| | D2 thô | D3: ô cờ / ô E | *tham chiếu: MAE Tầng 1, toàn ô model* |
|---|---|---:|---:|---:|---:|---:|---:|
| hron | whole_feature_matched | 0.0321 ± 0.0243 | 0.0964 ± 0.0191 | 0.0796 ± 0.0325 | 0.1392 ± 0.0270 | 0.0274 ± 0.0114 | 0.1147 ± 0.0172 |
| tsagris | whole_feature_matched | 0.1575 ± 0.0263 | 0.2241 ± 0.0217 | 0.1154 ± 0.0291 | 0.1749 ± 0.0256 | 0.0883 ± 0.0151 | 0.1219 ± 0.0147 |
| micoda | whole_feature_matched | 0.1357 ± 0.0256 | 0.2048 ± 0.0309 | 0.1801 ± 0.0324 | 0.2396 ± 0.0261 | 0.1264 ± 0.0249 | 0.1185 ± 0.0143 |
| same_day_mean | whole_feature_matched | 0.0322 ± 0.0157 | 0.0828 ± 0.0140 | 0.0355 ± 0.0176 | 0.0951 ± 0.0158 | 0.0027 ± 0.0041 | 0.0364 ± 0.0058 |
| main | whole_feature_matched | 0.1008 ± 0.0414 | 0.1836 ± 0.0472 | 0.0517 ± 0.0225 | 0.1113 ± 0.0163 | 0.0459 ± 0.0153 | 0.0581 ± 0.0136 |
| main_nometa | whole_feature_matched | 0.0541 ± 0.0395 | 0.1599 ± 0.0337 | 0.0574 ± 0.0266 | 0.1170 ± 0.0254 | 0.0207 ± 0.0098 | 0.0644 ± 0.0202 |
| mean_fill | whole_feature_matched | 0.2993 ± 0.0301 | 0.3500 ± 0.0376 | 0.2355 ± 0.0387 | 0.2950 ± 0.0308 | 0.1383 ± 0.0302 | 0.1488 ± 0.0188 |
| zero_fill | whole_feature_matched | 0.3538 ± 0.0252 | 0.3157 ± 0.0200 | NA | NA | 0.0000 ± 0.0000 | 0.1802 ± 0.0294 |
| marginal_sampler | whole_feature_matched | 0.1558 ± 0.0183 | 0.1185 ± 0.0110 | 0.0665 ± 0.0174 (n = 9) | 0.1258 ± 0.0139 (n = 9) | 0.0486 ± 0.0101 | 0.1486 ± 0.0207 |

## Kết luận chính — exploratory

U4 chỉ mô tả trong scorer gốc, không có ngưỡng pass/fail đóng băng; không tự đặt thêm ngưỡng. Zero-fill có D3 bằng 0 nhưng sai số lớn: D3 không dùng riêng để chọn model.

| Mask | U-case | Verdict |
|---|---|---|
| whole_feature | u1_under | fail |
| whole_feature | u1_over | fail |
| whole_feature | u2 | fail |
| whole_feature | u3 | pass |
| whole_feature | u4 | descriptive_no_frozen_threshold |
| random_cell | u1_under | fail |
| random_cell | u1_over | fail |
| random_cell | u2 | fail |
| random_cell | u3 | pass |
| random_cell | u4 | descriptive_no_frozen_threshold |
| whole_feature_matched | u1_under | fail |
| whole_feature_matched | u1_over | fail |
| whole_feature_matched | u2 | fail |
| whole_feature_matched | u3 | pass |
| whole_feature_matched | u4 | descriptive_no_frozen_threshold |

MAE prop toàn ô model (mean qua 10 seed; thấp hơn tốt hơn):

| Mask | main | main_nometa | Baseline tốt nhất | MAE baseline |
|---|---:|---:|---|---:|
| whole_feature | 0.047065 | 0.051560 | same_day_mean | 0.031086 |
| random_cell | 0.023656 | 0.023310 | same_day_mean | 0.030388 |
| whole_feature_matched | 0.058129 | 0.064352 | same_day_mean | 0.036368 |

CSV: diagnostic_vs_error chứa giá trị tổng quan D1/D2/D3 và scalar R-local/R-pooled dùng cho U1/U2; kind=scored_U2 là |D2−D2_truth| đối chiếu MAE prop trên target chung. rates chứa U1/U3; rank_agreement_regret chứa khớp hạng và regret theo seed; u4_decoy_ranks: hạng 1 là tốt nhất, so decoy với sáu model thật; NA được giữ trống. Decoy sinh bằng baseline gốc trên dữ liệu đã che, seed tương ứng, restore_counts như runner.

