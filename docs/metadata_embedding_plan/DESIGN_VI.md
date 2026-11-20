# Thiết kế: thêm metadata embedding (5 trường tĩnh) vào MainModel

Trạng thái: bản thiết kế. Chưa sửa code, chưa train. Spec/AC: `docs/metadata_embedding_plan/plan.md`.
Số dòng code dẫn chiếu theo working tree ngày 2026-11-20 (HEAD `59a7777`).

## 0. Tóm tắt

- Thêm **một vector metadata cho mỗi location**, cộng (broadcast) vào hidden state `h` của `CompositionNet` ngay sau dòng cộng feature/time embedding (`network.py:46`). Làm như side information của CSDI: thông tin tĩnh, lặp theo mọi thời điểm.
- Mã hoá: 4 trường phân loại bằng `nn.Embedding` (continent 6, classification 4, group 4, location 150, cộng 1 slot "unknown" cho mỗi trường). `median_age` chuẩn hoá z-score, rồi đưa `[z, is_missing]` qua `nn.Linear(2, channels)`. Các thành phần được cộng lại, không nối thêm kênh.
- Dữ liệu lấy từ `data/original_long_format.csv`, join theo `location` (đã kiểm: khớp 150/150 location và 11671/11671 dòng với `data/covariants.csv`).
- Cờ `metadata` trong bảng `[main]` mặc định tắt. Khi tắt thì code path, số tham số, thứ tự RNG khởi tạo và output **giống hệt từng byte** bản hiện tại.
- Đánh giá: chạy `[main]` và `[main_meta]` (`model = "main"`) trên cùng mask, seed 300–309, rồi so cặp bằng `evaluation.tier1`. Thêm ablation **không có location** vì location embedding thực chất là ID theo entity.

## 1. Tài liệu: side information / embedding (AC1)

Text trích bằng `pdftotext -layout` ra `%TEMP%\mdemb\*.txt`. "p" là số trang PDF (thứ tự form feed), không phải số trang in.

### 1.1 CSDI 2021 (Tashiro et al., NeurIPS; `2021_CSDI -- ...pdf`)
- p7, §4.3 *Side information*: "we provide some side information as additional inputs to the model. First, we use time embedding of s = {s1:L } to learn the temporal dependency. Following previous studies [29, 30], we use 128-dimensions temporal embedding. Second, we exploit categorical feature embedding for K features, where the dimension is 16."
- p16 (phụ lục, kiến trúc): sinusoidal time embedding `s_embedding(s_l)` với τ = 10000. Toàn bộ side information được nối với nhau: "we expand each side information and concatenate all the side information".
- Hệ quả cho repo: `CompositionNet` **đã có** đúng hai loại side information của CSDI: time embedding sinusoidal (`network.py:18-19, 43-44`) và feature embedding (`network.py:17`). Điểm khác là ở đây chúng được **cộng** vào `h` (`network.py:46`), không nối kênh. CSDI 2021 không có metadata theo từng chuỗi (subject/entity).

### 1.2 Seki et al. 2025, Bioinformatics Advances (`csdi.pdf`, "Diffusion model for imputing time-series gut microbiome profiles using phylogenetic information and metadata integration")
- p1, abstract: "To further improve the performance, we embedded host metadata ..."
- p3, §2.4: "Metadata variables were included during model training and excluded during evaluation. Categorical metadata variables ..." [được đưa qua một embedding]. Công thức (7) và giá trị d được **dựng lại (paraphrase)**, không phải trích nguyên văn, vì pdftotext làm hỏng các ký tự toán (text thô là "Embedding ¼ bðcatÞ þ eT W ðcatÞ 2 Rd" và "set to d ¼ 8"). Dạng dựng lại: Embedding = b(cat) + eᵀW(cat) ∈ R^d (phương trình 7), với eᵀ là vector one-hot của giá trị phân loại; embedding dimension d = 8. Phần chữ trích nguyên văn: "where eT is a one-hot vector representing the categorical val­ues".
- p3, hai cách tích hợp: "In the spatial concatenation approach, embeddings were appended along the feature dimension, producing input tensors of shape (C, L, K + nd)". "In the channel-wise concatenation approach, embeddings were concatenated along the channel dimension, resulting in tensors of shape (C + nd, L, K)."
- Cụm "inspired by CSDI_T (Zheng and ..." cũng xuất hiện ở p2 (phần phương pháp, cạnh Fig. 1), không chỉ ở p6.
- p6, §3.3: "Inspired by CSDI_T ... we modified the CSDI model with CNN to embed and concatenate categorical metadata with abundance profiles". Metadata dùng: country và allergy (milk, egg, peanut). Kết quả: "two embedding concatenation directions, although channel-wise concatenation of allergy metadata tended to produce more stable results". Lợi ích bị giới hạn, "potentially due to the metadata's moderate discriminative power and the constraints of the embedding method".
- p8, kết luận: "This led to modest improve[ment]".
- Hệ quả cho repo: metadata tĩnh theo subject, phân loại, embedding nhỏ (d = 8), lặp theo thời gian, nối theo kênh. Lợi ích chỉ ở mức vừa phải. Bài **không** dùng ID của từng subject làm metadata.

### 1.3 Seki et al. 2023, IEEE BIBM (`csdi_2023.pdf`)
- `csdi_2023.pdf` và `2023_Imputing time-series microbiome abundance profiles with diffusion model.pdf` là **cùng một file** (SHA256 `287466AA…A76A6` giống nhau; text giống nhau, MD5 `5F6CE42B…`).
- p3: dữ liệu DIABIMMUNE gồm subject có đủ mẫu "and metadata information of allergies", nhưng metadata này chỉ dùng để chọn và mô tả dữ liệu. p2: mô hình là CSDI cộng CNN theo nhóm phylum. Bản 2023 **không có metadata embedding** trong mô hình. Metadata embedding chỉ xuất hiện ở bản 2025 (`csdi.pdf`).

### 1.4 Proposal "2026_10_09-Revised proposal.pdf" (slide, 12 trang PDF)
- p2: bài toán có ba đặc tính "Compositional", "Longitudinal: Multiple collected time points per subject", "Sparsity". Ví dụ "Covid-19 variant data".
- p3: dòng nghiên cứu gần nhất là "Choi et al. [2], Seki et al. [8, 9]".
- p7: luồng lặp "Initialize imputed data X0 ... Apply LR transformation ... Fit model f using Zt to predict Xt+1 ... Check stopping condition". MainModel hiện tại đã làm đúng luồng này (`main_model.py:93-114, 297-339`).
- p8: "Proposed Network" chỉ là hình, không có text (pdftotext không trích được nội dung). Phần text **không nhắc đến metadata hay covariate embedding**. Chỉ có tiêu đề tài liệu tham khảo [9] ở p11 ("... phylogenetic information and metadata integration"). Vì vậy metadata embedding là phần **mở rộng** dựa trên [9], không phải yêu cầu đã có trong proposal. Cần người dùng hoặc tác giả xác nhận hình p8 có ô metadata hay không (chưa kiểm được bằng text).

## 2. Kiểm tra dữ liệu 5 trường (AC2)

Lệnh (stdlib, chỉ đọc):
```
.venv\Scripts\python.exe -I %TEMP%\mdemb\audit.py data\original_long_format.csv data\covariants.csv
```

| Trường | Kiểu | Số giá trị khác nhau | Dòng thiếu | Tĩnh theo location? |
|---|---|---|---|---|
| continent | chuỗi | 6 (Europe 44, Africa 36, Asia 34, North America 19, South America 11, Oceania 6 location) | 0 | có (0 location đổi giá trị) |
| location | chuỗi (khoá entity) | 150 | 0 | là khoá |
| median_age | số thực | 114 giá trị trên 150 location, khoảng 16.4–54.4; theo location: mean 32.468, sd 9.123 (population sd, ddof=0; sample sd ddof=1 = 9.153) | 0 | có |
| classification | chuỗi | 4: HIGH-INCOME, UPPER-MIDDLE-INCOME, LOWER-MIDDLE INCOME, LOW-INCOME | 0 | có |
| group | chuỗi | 4: Full democracy, Flawed democracy, Hybrid regime, Authoritarian | 0 | có |

Join với `data/covariants.csv`:
- Cả hai file có 11671 dòng. Khoá `(location, date)` duy nhất trong mỗi file. Hai file có **cùng tập khoá và cùng thứ tự dòng**. 150 location giống nhau. Các cột target và `total_sequence` khớp giá trị 100% (0 dòng lệch).
- **Khoá join: `location`.** Vì 5 trường đều tĩnh, nên lập một bảng 150 dòng (một dòng mỗi location), không cần join theo ngày.
- `continent` khớp 0 lệch với `configs/covid_location_continent.csv`, file đang dùng làm `group_map` trong `configs/baselines_covariants.toml`.
- Mỗi location có 1 `iso_code`, không có `OWID_*` aggregate. Mỗi location có 13–109 dòng (trung vị 83).

Điểm không nhất quán / rủi ro dữ liệu:
- Nhãn `classification` viết không đồng nhất: `LOWER-MIDDLE INCOME` (dấu cách) so với `UPPER-MIDDLE-INCOME` (gạch nối). Không ảnh hưởng nếu chỉ map nhãn thành chỉ số, nhưng **không được chuẩn hoá bằng split/regex**.
- Một số ô của bảng chéo classification × group rất thưa: (HIGH-INCOME, Hybrid regime) = 1, (LOW-INCOME, Full democracy) = 1. Oceania chỉ có 6 location. Embedding cho các nhóm nhỏ sẽ dựa trên rất ít entity.
- `classification`, `group`, `median_age` là ảnh chụp một thời điểm, không theo thời gian. Nguồn và năm của chúng chưa ghi trong repo (cần xác nhận trước khi viết báo cáo).
- Không trường nào thiếu, nên slot "unknown" chỉ để phòng hờ. Trên dữ liệu này nó không bao giờ được dùng.

## 3. Hiện trạng code (trace)

- `data.py:16` `SCHEMA_KEYS` chỉ có `{'entity','time','total','targets','time_format','group_col','group_map'}`. `read_schema` (`data.py:44-61`) từ chối mọi key lạ. `Dataset` (`data.py:20-41`) có `rows` (dict của mọi cột CSV) và `groups`.
- Run chính dùng `data/covariants.csv` với `configs/baselines_covariants.toml`. File này **không có** cột metadata. `continent` được lấy qua `group_map` (`data.py:160-191`).
- `run.py:505` `load_csv`, `run.py:507` `read_models`. `run.py:178-180` `build` = `MODELS[model_kind](**constructor_params)`. `run.py:138-140` cho phép bảng có tên khác với `model = "main"`. `run.py:315-317` `code_hashes` băm `data.py, main_model.py, network.py, baselines/tsagris.py`. `run.py:552` dựng `MainModel(**models['main'])` cho stage refine.
- `main_model.py:342-366` `MainModel.__init__` kiểm tra tham số. Tham số không biết sẽ gây `TypeError`.
- `main_model.py:410-412` khởi tạo mạng trong `fork_rng` với `torch.manual_seed(self.seed)`.
- `main_model.py:52-75` `pack_sequences` xếp entity theo `sorted(set(data.entities))` (dòng 64). Batch b ↔ entity thứ b theo thứ tự đã sort.
- `main_model.py:284-294` `_network_inputs` tạo 4 tensor. `train_one_round` gọi `network(...)` ở dòng 242, `predict_counts` gọi ở dòng 275, cả hai cắt `[start:end]` theo entity.
- `network.py:7-26` `CompositionNet.__init__(components, channels, heads)`. `network.py:28-60` `forward(features, observed_mask, times, valid_steps)`. Chỗ hợp lý để thêm metadata là `network.py:46`: `h = h + self.feature.weight[None, None] + time_embedding[:, :, None]`, với `h` có shape `(B, L, K, C)`.
- `configs/models_all.toml` `[main]`: `outer_iters=5, epochs_per_iter=10, batch_size=8, learning_rate=1e-3, tol=1e-4, train_mask_rate=0.1, seed=0, device="cpu"` (channels=32, heads=4 mặc định).

## 4. Thiết kế tối thiểu (AC3)

### 4.1 Mã hoá từng trường

| Trường | Mã hoá | Kích thước |
|---|---|---|
| continent | `nn.Embedding(6 + 1, C)` | 7 × C |
| classification | `nn.Embedding(4 + 1, C)` | 5 × C |
| group | `nn.Embedding(4 + 1, C)` | 5 × C |
| location | `nn.Embedding(150 + 1, C)`, **tuỳ chọn** (xem 4.6) | 151 × C |
| median_age | `(x − mean) / sd`, mean/sd tính trên **150 location** (không tính theo số dòng), rồi đưa `[z, is_missing]` qua `nn.Linear(2, C)` | 3C tham số (2C trọng số + C bias = 96 với C = 32) |

- Index 0 = unknown. Danh mục = các nhãn đã sort, map sang 1..n (sort để kết quả xác định, không phụ thuộc thứ tự dòng). Giá trị thiếu hoặc nhãn không có trong danh mục → 0. `median_age` thiếu → z = 0 cộng thêm một cờ: dùng `nn.Linear(2, C)` với input `[z, is_missing]`. Với dữ liệu hiện tại `is_missing` luôn bằng 0, nhưng code không cần nhánh riêng.
- C = `channels` (32) để **cộng** thẳng vào `h`. Cách này đơn giản hơn nối kênh (CSDI nối kênh với d = 8). Cộng tương đương với nối kênh rồi chiếu tuyến tính, và không phải sửa `self.input`, attention hay `gate`. Có thể ghi chú lựa chọn này khi so với Seki 2025.
- Vector của một entity: `m = Σ_f E_f[idx_f] + W_age·[z, miss] + b`, shape `(B, C)`.

### 4.2 Chỗ thêm trong CompositionNet

- `__init__(components, channels=32, heads=4, metadata=None)`. `metadata` là dict `{'categorical': {'continent': 6, 'classification': 4, 'group': 4, ['location': 150]}, 'numeric': ['median_age']}` hoặc `None`.
- Chỉ khi `metadata is not None` mới tạo các module mới. Phải tạo chúng **sau mọi module hiện có** (sau `network.py:26`), để thứ tự khởi tạo tham số cũ không đổi khi bật cờ. Khi tắt cờ thì không tạo gì: `state_dict` và RNG giống hệt bản hiện tại.
- `forward(features, observed_mask, times, valid_steps, metadata=None)`. Thêm vào ngay sau `network.py:46`:
  `if self.meta is not None: h = h + self.meta(cat_idx, num)[:, None, None, :]`, broadcast qua L và K.
- Kiểm tra: cờ bật mà thiếu `metadata`, hoặc cờ tắt mà lại truyền `metadata` → `ValueError`. Shape phải là `(B, n_cat)` long và `(B, n_num)` float, cùng device. Phần padding không cần xử lý thêm vì metadata là theo entity, và output đã được `masked_fill` ở `network.py:60`.

### 4.3 Dữ liệu chảy từ CSV vào mạng

Cách được khuyến nghị (ít động vào hạ tầng chung nhất):
1. **Schema**: thêm key tuỳ chọn `[dataset] metadata = "covid_location_metadata.csv"` (đường dẫn tương đối theo file schema, giống `group_map`) và `metadata_cols = ["continent","location","median_age","classification","group"]`. Thêm hai key này vào `SCHEMA_KEYS` (`data.py:16`), kiểm tra trong `read_schema`.
   - File bảng location (150 dòng) được **sinh một lần** từ `data/original_long_format.csv` bằng script nhỏ, rồi commit kèm sha256. Có thể cho đọc trực tiếp `original_long_format.csv` (lấy giá trị duy nhất theo location, báo lỗi nếu không tĩnh). Cách này tránh thêm file, nhưng file 11671 dòng có cả covariate không dùng. Khuyến nghị **bảng 150 dòng**, để provenance rõ và để không ai vô tình đưa covariate theo thời gian vào.
   - Tạo **file schema mới** `configs/baselines_covariants_meta.toml` (bản sao cộng thêm hai key). Không sửa schema cũ, để hash của các run cũ không đổi.
2. **`data.py`**: thêm `read_metadata(rows, lines, entity, path, cols, base)` theo khuôn `read_groups` (`data.py:160-191`): đọc bytes, băm sha256, kiểm cột, mỗi entity đúng một giá trị, entity trong dữ liệu phải có trong bảng. Thêm field `Dataset.metadata: dict | None = None`, dạng `{entity: {col: str}}`, cùng `metadata_info` (provenance). Field mới có mặc định `None`, nên `Dataset(...)` cũ vẫn chạy. `_visible_data` dùng `replace` (`main_model.py:125`) nên metadata đi theo tự động. Metadata không phải target, nên không có rủi ro lộ dữ liệu bị che.
3. **`main_model.py`**:
   - `MainModel.__init__(…, metadata=False, metadata_location=True)`: kiểm tra bool. `metadata=False` là mặc định.
   - Hàm mới `metadata_inputs(data, entities_sorted)` → `(cat_idx int64 (B, n_cat), num float (B, 2), spec)`. Danh mục và mean/sd tính từ `data.metadata` trên **toàn bộ entity của dataset**. Đây là thông tin tĩnh, không phải target, nên không gây leakage giữa ô bị che và ô thấy được.
   - `_network_inputs` (`main_model.py:284-294`): nếu `network.meta` tồn tại thì trả thêm tensor metadata, xếp theo **cùng thứ tự `sorted(set(data.entities))`** như `pack_sequences` (`main_model.py:64`). Nên cho `pack_sequences` trả thêm danh sách entity, hoặc tính lại cùng biểu thức, kèm test kiểm tra căn hàng.
   - `train_one_round` (dòng 242) và `predict_counts` (dòng 275) truyền `metadata=meta[start:end]` khi có.
   - `impute` (dòng 412): `CompositionNet(len(data.targets)+1, self.channels, self.heads, metadata=spec if self.metadata else None)`. Nếu `self.metadata` mà `data.metadata is None` → `BaselineInputError('main: metadata=true requires schema metadata')`. Lỗi được báo trước khi train.
   - `self.info_` thêm `'metadata': {'fields': [...], 'cardinality': {...}, 'age_mean':…, 'age_sd':…}` khi bật cờ. Khi tắt cờ thì **không thêm key**, để `summary.json` giống hệt.
4. **`run.py`**: không cần sửa logic. `build`/`read_models` truyền tham số toml vào constructor. `resolve_models` sẽ tự liệt kê `metadata` trong `effective_params`. Đây là **thay đổi duy nhất trong `config.json`** của run cũ khi chạy lại bằng code mới (thêm key mặc định). Nếu cần config.json cũng giống từng byte, phải giữ chữ ký constructor không đổi khi tắt cờ. Cách đơn giản: thêm hai tham số vào chữ ký và chấp nhận `effective_params` có thêm `metadata: false`. Coordinator cần quyết định (xem §6, R6). `code_hashes` (`run.py:315-317`) đã băm `data.py/main_model.py/network.py`, nên artifact init cũ sẽ bị từ chối khi refine. Đây là hành vi đúng.
   - `config['schema']` đã ghi `read_schema` (`run.py:558`). Nên thêm `'metadata': data.metadata_info` cạnh `group_map` (`run.py:563`) chỉ khi khác `None`.
5. **Config**: `configs/models_main_meta.toml` (file mới) gồm `[main]` y hệt `models_all.toml` và `[main_meta] model = "main"` cùng tham số cộng `metadata = true`. Có thể thêm `[main_meta_noloc]` với `metadata_location = false`. **Không sửa** `configs/models_all.toml`.

### 4.4 Cờ mặc định tắt và tính tương đương từng byte

- `metadata=False` và schema cũ: không tạo module mới, `forward` không có nhánh mới (chỉ một `if self.meta is not None`), `_network_inputs` trả đúng 4 tensor, `info_` không đổi. Vì vậy predictions `.npy`, `history`, `summary.csv` phải trùng hash với run tham chiếu trên cùng seed. `config.json` thì khác ở `code_sha256`/`git` (luôn khác khi code đổi) và có thể ở `effective_params` (R6).

### 4.5 Danh sách thay đổi file/hàm

| File | Thay đổi |
|---|---|
| `src/longitudinal_impute/network.py` | `CompositionNet.__init__(…, metadata=None)` tạo `self.meta` (module con `MetadataEncoder` cùng file, ~25 dòng) **sau** dòng 26. `forward(…, metadata=None)` cộng sau dòng 46. Kiểm tra shape. |
| `src/longitudinal_impute/data.py` | `SCHEMA_KEYS` + `metadata`, `metadata_cols`. `read_schema` kiểm tra. `read_metadata(...)`. `Dataset.metadata`, `Dataset.metadata_info`. `load_csv` gọi `read_metadata`. |
| `src/longitudinal_impute/main_model.py` | `MainModel.__init__` + `metadata`, `metadata_location`. `metadata_inputs(...)`. `_network_inputs`, `train_one_round`, `predict_counts`, `impute` (dòng 412) truyền metadata. `info_['metadata']`. |
| `src/longitudinal_impute/run.py` | (tuỳ chọn) ghi `metadata_info` vào `config.json` khi có. |
| `configs/covid_location_metadata.csv` (mới) | 150 dòng: location, continent, median_age, classification, group. |
| `configs/baselines_covariants_meta.toml` (mới) | bản sao schema + `metadata`, `metadata_cols`. |
| `configs/models_main_meta.toml` (mới) | `[main]`, `[main_meta]`, `[main_meta_noloc]`. |
| (script một lần) `scripts/make_location_metadata.py` (mới) | sinh CSV từ `original_long_format.csv`, báo lỗi nếu trường nào không tĩnh. |

### 4.6 Location embedding: giữ hay bỏ?

`location` có 150 giá trị, đúng bằng số entity, nên embedding của nó là **ID theo từng entity** (một bias học riêng cho mỗi nước). Với mask `whole_feature`, cả chuỗi (location, biến thể) bị che, và mạng chỉ thấy entity đó qua các biến thể khác. Bias theo nước có thể giúp, nhưng cũng dễ học thuộc phân bố các ô thấy được (overfit), nhất là với nước có 13 dòng. Location embedding **không mang được sang** một nước mới. Seki 2025 chỉ dùng thuộc tính nhóm (country, allergy), không dùng ID subject. Khuyến nghị:
- Cấu hình chính `main_meta` = **4 trường** (continent, classification, group, median_age) cộng location, theo đúng yêu cầu 5 trường. Đồng thời **bắt buộc** chạy ablation `main_meta_noloc`, rồi báo cả hai.
- Nếu muốn giảm overfit: khởi tạo location embedding bằng 0 (`nn.init.zeros_`) và/hoặc dùng dropout cho nó. Không thêm trong bản tối thiểu. Ghi lại để thử sau.

## 5. Test cần thêm

Đặt trong `tests/test_main_model.py` / `tests/test_main_run.py` (hoặc `tests/test_metadata.py` mới):
1. **Off = byte-identical**: cùng dataset nhỏ và seed, `MainModel()` sau khi sửa so với output đã lưu từ code cũ (hoặc so `CompositionNet(k)` cũ/mới): `state_dict` cùng khoá và cùng giá trị, `impute` cho mảng bằng nhau từng phần tử (`np.array_equal`). `info_` không có key `metadata`.
2. **Thứ tự init**: `CompositionNet(k, metadata=spec)` có các tham số cũ bằng đúng `CompositionNet(k)` dưới cùng `manual_seed`. Điều này chứng tỏ module mới được tạo sau.
3. **Căn hàng entity**: metadata tensor dòng b ứng với entity `sorted(set(entities))[b]`. Test bằng dataset 3 entity được đặt nhãn đảo thứ tự trong CSV.
4. **Hiệu lực**: với metadata khác nhau giữa hai entity có feature giống hệt, output khác nhau khi bật cờ và giống nhau khi tắt cờ.
5. **Unknown**: nhãn lạ → index 0, `median_age` thiếu → z = 0, cờ = 1. Không ra NaN.
6. **Lỗi rõ ràng**: `metadata=true` với schema không có metadata → `BaselineInputError` trước khi train. Bảng metadata thiếu entity, hoặc một entity có hai giá trị khác nhau → `ValueError` trong `load_csv`. Key schema lạ vẫn bị từ chối.
7. **Không lộ dữ liệu che**: `Dataset.metadata` sau `_visible_data` vẫn như cũ. Metadata không chứa cột target (kiểm `metadata_cols ∩ targets = ∅` trong `read_schema`).
7b. **Metadata theo từng view**: tensor metadata phải được dựng từ `view.data` của **chính view đang dùng**, theo cùng thứ tự `sorted(set(view.data.entities))` như `pack_sequences`, cho cả view training (`train_one_round`) lẫn view inference (`predict_counts`). Test: dựng hai view có thứ tự dòng CSV khác nhau, rồi kiểm dòng b của metadata ứng với entity b của packed features trong cả hai lời gọi. Kiểm thêm rằng lát `[start:end]` của metadata khớp lát của features.
8. **Runner**: `[main_meta] model="main", metadata=true` chạy qua `read_models`/`build`. `resolve_models` ghi đúng tham số. `--stage init/refine` vẫn chỉ nhận đúng `main`. Lưu ý `run.py:508` yêu cầu `set(models) == {'main'}`, nên `main_meta` không chạy được 2 stage. Dùng `--stage full`.
9. Chạy lại toàn bộ test hiện có (`pytest tests`), phải xanh.

## 6. Giao thức đánh giá

1. **Tuning / phát triển** chỉ dùng seed 0–4 (theo `src/longitudinal_impute/evaluation/README.md`, mục "Thêm một model mới"). Không chỉnh siêu tham số trên seed xác nhận.
2. **Lượt xác nhận**: seed **300–309** (seed 200–209 đã dùng). Trước khi chạy phải kiểm 300–309 chưa có trong `used_seeds.json` và tạo config đánh giá mới (`eval_v1_main_meta.toml`), commit **trước** khi chạy, theo quy trình README.
3. Chạy một run gồm cả `[main]`, `[main_meta]`, `[main_meta_noloc]` (`--models configs/models_main_meta.toml --schema configs/baselines_covariants_meta.toml --data data/covariants.csv --mask whole_feature --seeds 300 … 309 --pass both`). Cùng một lần chạy dùng chung một mask cho mỗi seed (`run.py:646-651`), nên cặp so sánh được ghép đúng. Lặp lại cho `random_cell` và `whole_feature_matched` nếu cần đủ ba mask như v1.
   - `[main]` trong run này dùng schema mới. Kiểm tra lại rằng predictions của nó trùng hash với `[main]` chạy bằng schema cũ (đây là kiểm tra cho §4.4).
4. So sánh bằng `longitudinal_impute.evaluation.tier1`: MAE/MSE tỷ lệ trên ô bị che, ghép cặp theo seed chung, sign test, tách theo châu lục và nhóm N. Giả thuyết đặt trước, ví dụ: H_meta1 "main_meta có MAE thấp hơn main trên ≥ 8/10 seed (mask chính whole_feature)". H_meta2 tương tự cho `main_meta_noloc`. Báo thêm Tầng 2 (ô thiếu thật) chỉ để mô tả.
5. Ghi lại: thời gian chạy, số tham số thêm vào (khoảng (7+5+5+151)·32 + 96 (Linear(2,32)) = 5472 với location, 17·32 + 96 = 640 không có location), loss history. Nếu `main_meta` chỉ hơn ở các nước có ít dòng hoặc chỉ khi có location, phải coi đó là tín hiệu **ghi nhớ theo entity**, không phải metadata thực sự giúp.

## 7. Rủi ro / leakage

| ID | Rủi ro | Xử lý |
|---|---|---|
| R1 | Location embedding = ID theo entity, có thể overfit hoặc ghi nhớ (§4.6) | ablation `noloc` bắt buộc. Diễn giải kết quả theo cặp. |
| R2 | Nhóm nhỏ (Oceania 6, ô chéo có 1 nước) dễ overfit | embedding chung C nhỏ, không tăng `channels`. Báo kết quả theo châu lục (tier1 đã có). |
| R3 | Lộ dữ liệu qua metadata | 5 trường là đặc tính tĩnh, không suy ra từ target hay count. Phải kiểm không trường nào trùng target/total. **Không** thêm covariate theo thời gian (`total_sequence`, cases…), vì đó là non-goal và `total_sequence` là mẫu số. |
| R4 | median_age chuẩn hoá theo số dòng sẽ lệch về nước có nhiều dòng | tính mean/sd trên 150 location. |
| R5 | Lệch thứ tự entity giữa metadata và `pack_sequences` | dùng chung `sorted(set(entities))`, có test 3. |
| R6 | "byte-identical" của `config.json` | `effective_params` có thêm `metadata: false`, `code_sha256` đổi. Predictions và summary vẫn phải giống từng byte. Cần coordinator chấp nhận định nghĩa này. |
| R7 | Stage init/refine chỉ cho `{'main'}` (`run.py:508`) | đánh giá bằng `--stage full`. Nếu cần chia stage cho `main_meta` thì là việc riêng. |
| R8 | Nguồn và năm của classification/group/median_age chưa ghi trong repo. Nhãn classification không đồng nhất | ghi provenance trong file metadata. Map nhãn nguyên văn. |
| R9 | Lợi ích có thể nhỏ (Seki 2025 p6/p8: "modest") | đặt giả thuyết trước. Chấp nhận kết quả âm tính. |
| R10 | Proposal p8 chỉ là hình. Chưa xác nhận metadata có trong thiết kế mạng gốc | hỏi tác giả proposal. |

## 8. Lệnh đã chạy (bằng chứng)

- `pdftotext -layout <pdf> %TEMP%\mdemb\{csdi,c23,c21,m23,prop}.txt`: thành công. Với `csdi.pdf` có cảnh báo "Bad bounding box / Bad block header in flate stream", nhưng text các đoạn trích dẫn vẫn đầy đủ. Số trang: csdi 10, c21 31, c23 7, prop 12.
- `Get-FileHash` cho thấy `csdi_2023.pdf` giống hệt `2023_Imputing ... diffusion model.pdf`.
- `.venv\Scripts\python.exe -I %TEMP%\mdemb\audit.py data\original_long_format.csv data\covariants.csv`: kết quả ở §2.
- Đã đọc code: `network.py` 1–60, `main_model.py` 1–480, `data.py` 1–419, `run.py` 138–220, 315–317, 429–728, `configs/models_all.toml`, `configs/baselines_covariants.toml`.
- Không train, không cài đặt, không sửa file nào khác. File tạm chỉ nằm trong `%TEMP%\mdemb\`.
