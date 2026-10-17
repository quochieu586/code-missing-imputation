# Package cho flow offline mới

Đây là package song song với code cũ. Đã hiện thực general baseline modules
Hron, MICoDa/Saha, Tsagris và count adapters/CLI. Xem [BASELINES_VI.md](BASELINES_VI.md)
cho variants, cách chạy, kiểm chứng và giới hạn paper parity. Chưa chuyển logic
từ `src/` hoặc triển khai neural training/outer loop. Các Protocol còn lại là
hợp đồng dự kiến, không tự bảo đảm runtime validation hay chống leakage.

Benchmark offline đã chạy trên toàn bộ CoVariants với 5 location folds,
3 mask seeds và 4 masking scenarios. Xem [BENCHMARK_VI.md](BENCHMARK_VI.md)
cho module, cách tái chạy, bằng chứng kiểm toán và giới hạn kết quả.

Nguồn chính là `.py`; notebook Kaggle chỉ làm launcher. Python >=3.11 đủ để
import package và đọc TOML. Baseline implementations dùng optional extra
`baselines` (NumPy), không cần torch. Lock CPU/GPU full research vẫn cần được
chốt trong giai đoạn benchmark.

Sau khi chủ động cài vào môi trường phù hợp: `python -m pip install -e ./nextgen`.
Lệnh kiểm tra cấu hình (chỉ parse/in, chưa validate dữ liệu):

```text
longitudinal-impute --inspect-config nextgen/configs/covariants_offline.toml
```

`pipeline.run()` cố ý báo `NotImplementedError`. Không gọi Protocol làm model.
Không tự phát hiện 17 targets từ vị trí cột. Không import code từ artifacts
hoặc từ scripts legacy. Không xóa/di chuyển source hay outputs cũ.

Xem kế hoạch nghiên cứu tổng thể ở
`directive/plans/research_plan_longitudinal_2026-10-15_vi.md` tính từ repo root.
