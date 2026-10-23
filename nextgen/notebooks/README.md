# Notebook adapter (planned)

Keep methods, masking, training, metrics and configuration in `.py` / `.toml`.
The Kaggle notebook will only install the package, resolve input/output roots,
select a tested GPU environment, invoke the CLI, and display/export run files.
Do not copy source into notebook cells. `/kaggle/input` is read-only; write
results to `/kaggle/working/<run-id>` without overwriting previous runs.
No notebook or GPU training job has been created/launched at this stage.
