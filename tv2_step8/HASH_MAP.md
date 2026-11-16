# Bảng ánh xạ hash sau khi rebase

Lịch sử của step 1–8 đã được viết lại vì hai lý do:

1. Rebase lên `main` sau PR #5 (`53dbf7b`), khi `nextgen/` được đưa ra gốc repo.
2. Bỏ dòng trailer đồng tác giả trong message, và thống nhất email tác giả thành `tuananhngo2909@gmail.com`.

Thứ tự 77 commit gốc được giữ 1:1: không gộp, không tách, không bỏ commit nào. Message (trừ trailer), ngày author và ngày committer giữ nguyên. Ngoài ra có thêm các commit mới (cột hash cũ ghi "mới"):

- một commit `chore` sửa đường dẫn ở cuối mỗi PR;
- một commit sửa golden test ở PR 2;
- một commit docs ở cuối mỗi PR.

**Vì sao "đóng băng trước khi sinh mask" vẫn kiểm chứng được.** Hash đổi, nhưng thứ tự cha–con giữ nguyên. Ví dụ, `eval_v1.toml` được đóng băng ở `613612b` (cũ `996feba`) trước khi sinh mask xác nhận ở `696f217` (cũ `c13308b`):

```
git merge-base --is-ancestor 613612b 696f217 && echo "đóng băng trước khi sinh mask"
git log --format='%h %ad %s' --date=iso 613612b^..696f217
```

Sau rebase, file `eval_v1.toml` có thêm khối amendment (đường dẫn mới và ghi chú MICoDa). `v1.frozen_bytes()` bỏ khối này và đổi đường dẫn về như cũ, cho ra đúng từng byte bản đã đóng băng. sha256 của bản đó vẫn bằng `config_sha256` trong `masks_manifest.json` (có test).

**Thư mục run và manifest ghi hash cũ.** Các trường `git.commit` trong `config.json` của run, và các hash trong khối `auto:*` của báo cáo, là commit lúc chạy. Chúng giữ nguyên vì là dữ kiện lịch sử; dùng bảng dưới để tra commit mới tương ứng. Commit cũ còn trong các tag cục bộ `backup/step1-orig`, `backup/step4-orig`, `backup/step8-orig` (không push).

| # | PR | Hash mới | Hash cũ | Commit |
|---:|---|---|---|---|
| 1 | PR #4 (step 1) | `2fa6d71` | `a6c6bba` | data: read the time column as real dates and add an optional group column |
| 2 | PR #4 (step 1) | `05eb2f4` | `69a48f2` | baselines: shared helper to cap per-cell proposals at the row residual |
| 3 | PR #4 (step 1) | `0b4e737` | `5729a88` | baselines: zero_fill floor (every missing cell 0, residual to Other) |
| 4 | PR #4 (step 1) | `ebaf0ae` | `9372e72` | baselines: mean_fill floor (global mean c/N of the target, capped by the residual) |
| 5 | PR #4 (step 1) | `5cbbd7c` | `0497315` | baselines: marginal_sampler floor (one seeded draw of c/N from the observed cells of the target) |
| 6 | PR #4 (step 1) | `18bcaf8` | `00e3646` | baselines: same_day_mean (N-weighted c/N in the same group within +-window_days) |
| 7 | PR #4 (step 1) | `68ba1ba` | `9902bfc` | run: register the four new models, enabled = false and --only |
| 8 | PR #4 (step 1) | `601f02e` | `3fe8b0b` | tests: date parsing, group map, the four new models and model selection |
| 9 | PR #4 (step 1) | `fab3b84` | `7b0ccbe` | docs: step 1 report (files, open design decisions, COVID results, step 2 work) |
| 10 | PR #4 (step 1) | `659437a` | `81c7368` | config: run same_day_mean by default as an official baseline |
| 11 | PR #4 (step 1) | `5b5ac87` | `1b17d0a` | docs: same_day_mean and the three floors in the baseline contract |
| 12 | PR #4 (step 1) | `1860c17` | `12fd81d` | config: separate MICoDa other = true (A18/R2) pass for the step 1 runs |
| 13 | PR #4 (step 1) | `63115c4` | `2b4a39d` | docs: correct the share of hidden cells on the global fallback |
| 14 | PR #4 (step 1) | `e38f0c6` | `91c5853` | step1: multi-seed summary script and tables (COVID, 5%, seeds 0-4, two masks) |
| 15 | PR #4 (step 1) | `d650331` | `c7cbfdd` | step1: report with 5-seed results, micoda_other and conclusions |
| 16 | PR #4 (step 1) | `a80357b` | `0800e35` | step1: PR description |
| 17 | PR #4 (step 1) | `a68db71` | `e698e77` | F1: same_day_mean reads only the two window endpoints per query row |
| 18 | PR #4 (step 1) | `5a1eed1` | `c3ea968` | P3: same_day_mean rejects totals outside its exact-sum domain, never fills a silent 0 |
| 19 | PR #4 (step 1) | `097694a` | `a5807df` | F3: record the group map actually used in config.json and check it before recomputing |
| 20 | PR #4 (step 1) | `ed597e7` | `9857243` | F2: refuse to pair main and micoda_other runs that differ; write a results manifest |
| 21 | PR #4 (step 1) | `43daf99` | `bf14cc7` | F4: sort the model table with explicit None checks, so 0.0 comes before positive errors |
| 22 | PR #4 (step 1) | `eb32efe` | `d1e391b` | F5: generate every paired win/loss number from results_multiseed.csv |
| 23 | PR #4 (step 1) | `5761b7c` | `2a2c24f` | F2: regenerate the 5-seed tables from validated runs and commit results_manifest.json |
| 24 | PR #4 (step 1) | `ca11724` | `efa456c` | F5: report win/loss from the generated block; fix the Tsagris vs Hron count claim |
| 25 | PR #4 (step 1) | `dc868c1` | `d8cdb39` | docs: interpretation per PR #4 review (masked pilot, sample SD, four factors) |
| 26 | PR #4 (step 1) | `5bbe67f` | `14818f3` | docs: PR #4 fix log (F1-F5, P3) and reply to the review |
| 27 | PR #4 (step 1) | `03b91e1` | mới | chore: paths after repository restructure (nextgen → root) |
| 28 | PR #4 (step 1) | `1ad69f8` | mới | docs: PR #4 update after the rebase on main with PR #5 (step 1, F1-F5 fixed) |
| 29 | PR 2 (step 2–4) | `ed21434` | `8c8f43c` | evaluate: share cell_errors between masked_errors and long-file scoring; add series-macro MAE |
| 30 | PR 2 (step 2–4) | `4157533` | `7ef9146` | run: opt-in long prediction file per hidden cell, error codes and forced/model cell counts |
| 31 | PR 2 (step 2–4) | `3fe3143` | `ed88dbb` | step1 summarize: validate_pair takes the seed list (default seeds 0-4 unchanged) |
| 32 | PR 2 (step 2–4) | `2904b8e` | `8978d72` | tv2_step2: tier 0-1 tables with common-seed pairing (F07) and a regression check on the long files |
| 33 | PR 2 (step 2–4) | `2836726` | `93507be` | tv2_step2: seed-109 block shows complete rows after masking and the primary metric |
| 34 | PR 2 (step 2–4) | `34d2bbd` | `5310676` | tv2_step2: COVID tier 0-1 results (whole_feature seeds 0-4 + 109, random_cell 0-4) and manifest |
| 35 | PR 2 (step 2–4) | `55322da` | `ab19356` | docs: step 2 tier 0-1 report (exploratory), tables filled by tier01.py |
| 36 | PR 2 (step 2–4) | `c85d79d` | `a76059b` | tv2_step2: metric and pair tables use only the pre-declared seeds 0-4; seed 109 in status tables only |
| 37 | PR 2 (step 2–4) | `389d75d` | `c982e25` | run/README: the run.py summary table averages each model's completed seeds; rank with tier01.py |
| 38 | PR 2 (step 2–4) | `4841a80` | `2d229f7` | same_day_mean: ablation switches use_time, use_group, weighting, residual_rule |
| 39 | PR 2 (step 2–4) | `4e031ff` | `39d78a0` | run: optional model key lets a [table] run a registered model under its own name |
| 40 | PR 2 (step 2–4) | `eef0d3e` | `5ca70d8` | run: resolved_models also records the effective constructor parameters |
| 41 | PR 2 (step 2–4) | `0ebfb8b` | `c41d559` | tier01: factor load_run (one run directory) out of load_mask for reuse by step 3 |
| 42 | PR 2 (step 2–4) | `ad9aa59` | `7ed8921` | tv2_step3: ablation tables on top of tier01 (no second scorer) |
| 43 | PR 2 (step 2–4) | `e001cc9` | `6948635` | tv2_step3: COVID ablation results (whole_feature + random_cell, seeds 0-4) and manifest |
| 44 | PR 2 (step 2–4) | `593dc69` | `32edf86` | docs: step 3 report (same_day_mean ablation, exploratory), tables filled by ablation.py |
| 45 | PR 2 (step 2–4) | `37c4fc0` | `e137c72` | evaluate: implied_other_errors, the error of the implied Other share on rows complete before masking |
| 46 | PR 2 (step 2–4) | `aa1b94f` | `896673d` | run: optional residual_rule (prop_other, cap, rescale) applied at the restore layer |
| 47 | PR 2 (step 2–4) | `32756ca` | `fefd4d0` | configs: frozen residual-rule matrix (six new variants) and the prop_other consistency runs |
| 48 | PR 2 (step 2–4) | `26a60ed` | `b7632a4` | tv2_step4: residual-rule tables on top of tier01 and ablation (no second scorer) |
| 49 | PR 2 (step 2–4) | `94ff49e` | `6a558fa` | data: apply_residual_rule prop_other does not read rows without a missing target |
| 50 | PR 2 (step 2–4) | `d370115` | `050ee17` | tv2_step4: COVID residual-rule results (whole_feature + random_cell, seeds 0-4) and manifest |
| 51 | PR 2 (step 2–4) | `d71e7e0` | `c1f0625` | docs: step 4 report (residual rule, exploratory), tables filled by residual.py |
| 52 | PR 2 (step 2–4) | `b709b82` | mới | chore: paths after repository restructure (nextgen → root) |
| 53 | PR 2 (step 2–4) | `4934862` | mới | test: MICoDa golden of the residual-rule test within rtol 1e-9, not byte for byte |
| 54 | PR 2 (step 2–4) | `48dc681` | mới | docs: PR description of steps 2-4 (tier 0-1, same_day_mean ablation, residual rule) |
| 55 | PR 3 (step 5–8) | `9604e8d` | `b284c3b` | evaluate: tier 2 diagnostics D1/D2/D3 with R-pooled and R-local references, frozen before any output is scored |
| 56 | PR 3 (step 5–8) | `321e52d` | `580f15e` | tv2_step5: tier 2 native tables on top of the frozen diagnostics, with reuse checks and table logic tests |
| 57 | PR 3 (step 5–8) | `33830c2` | `922fafe` | tv2_step5: read completed CSVs without load_csv (rounding above N), decoy lines count only targets with a D1 |
| 58 | PR 3 (step 5–8) | `6253451` | `6e70a8a` | tv2_step5: COVID tier 2 native results (11 baselines), tables and manifest; native CSVs rerun byte-identical |
| 59 | PR 3 (step 5–8) | `9bb16b5` | `3551e13` | README: tier 2 diagnostics (descriptive, explicit eval / reference masks) |
| 60 | PR 3 (step 5–8) | `faa11bd` | `326dfd5` | docs: step 5 report (tier 2 native, descriptive, exploratory), tables filled by tier2.py |
| 61 | PR 3 (step 5–8) | `55507a0` | `bf4f096` | evaluate: tier 3 calibration use cases U1-U4, pass thresholds and sensitivities, frozen before any output is scored |
| 62 | PR 3 (step 5–8) | `3a37d07` | `26ec4a8` | tv2_step6: tier 3 calibration tables on top of tier01 and the step 5 diagnostics, with table logic tests |
| 63 | PR 3 (step 5–8) | `fcac0f3` | `11bfcc9` | tv2_step6: U4 table escapes its headers, shows n when a case has no value, and adds the tier 1 MAE as a reference row |
| 64 | PR 3 (step 5–8) | `b003882` | `b6f8799` | tv2_step6: COVID tier 3 calibration results (10 cases, 6 real models, 3 decoys) and manifest |
| 65 | PR 3 (step 5–8) | `08b029a` | `006e04c` | README: tier 3 calibration (within-case comparisons, scope limited to the masks checked) |
| 66 | PR 3 (step 5–8) | `847c8b2` | `397d3ad` | docs: step 6 report (tier 3 calibration, exploratory), tables filled by tier3.py |
| 67 | PR 3 (step 5–8) | `157983d` | `3bc59a9` | data: whole_feature_matched mask (strata continent x N group, weighted draw towards the native M_m distribution), frozen before any mask is drawn |
| 68 | PR 3 (step 5–8) | `e3643dc` | `a575ad6` | tv2_step6: load_mask takes run directories, seeds and a mask target; tables loop over the masks they are given |
| 69 | PR 3 (step 5–8) | `ebf2521` | `2c21c8c` | tv2_step7: matched-mask script (target, tables 1-6) and run script, declared before any mask is drawn |
| 70 | PR 3 (step 5–8) | `7da9a1d` | `a9d7e90` | configs: frozen strata target pi of the native M_m cells (step 5 N cuts 17 / 149), before any matched mask is drawn |
| 71 | PR 3 (step 5–8) | `f00f1e2` | `53e6a8a` | tv2_step7: stratum labels without '\|' in table 1; run provenance (config / long-file sha256, commit) in the manifest |
| 72 | PR 3 (step 5–8) | `34db2f7` | `0f495b7` | tv2_step7: COVID whole_feature_matched results (11 models, seeds 0-4): match, tiers 0, 1, 3 and transfer |
| 73 | PR 3 (step 5–8) | `c0bc009` | `1d9f831` | tv2_step7: report keeps the tier 0 reliability table and one line per seed; per-seed run tables go to matched_results.md |
| 74 | PR 3 (step 5–8) | `78d038a` | `2335663` | docs: step 7 report (matched mask, exploratory), tables filled by matched.py |
| 75 | PR 3 (step 5–8) | `fd8688e` | `88caf34` | tv2_step6: load_mask reads the case seeds from its seeds argument (default 0-4) |
| 76 | PR 3 (step 5–8) | `613612b` | `996feba` | tv2_step8: evaluation v1 frozen (eval_v1.toml) with hypotheses H1-H9, before any confirmation mask is drawn |
| 77 | PR 3 (step 5–8) | `696f217` | `c13308b` | tv2_step8: confirmation masks drawn (3 masks x seeds 200-209), hashes in masks_manifest.json |
| 78 | PR 3 (step 5–8) | `90e162f` | `9954fd2` | tv2_step6: one_case scores a case where a real model failed (its pairs NA) instead of a KeyError |
| 79 | PR 3 (step 5–8) | `4a59fd6` | `6303a6d` | tv2_step8: keep the first run directories (runs_attempt1) out of git |
| 80 | PR 3 (step 5–8) | `35afaf0` | `4b8c065` | tv2_step8: table 8 lists each hypothesis pair once and averages model MAE over the seeds common to all models (F07) |
| 81 | PR 3 (step 5–8) | `581f1d1` | `4535924` | tv2_step8: confirmation results, seeds 200-209 on three masks (second run after the tier 3 fix) |
| 82 | PR 3 (step 5–8) | `1ddfa27` | `1fbf2f3` | docs: step 8 report (evaluation v1, confirmation on seeds 200-209), tables filled by evaluate_v1.py |
| 83 | PR 3 (step 5–8) | `6d9ce21` | mới | chore: paths after repository restructure (nextgen → root) |
| 84 | PR 3 (step 5–8) | commit chứa file này | mới | docs: hash map, rebase report and PR description (steps 5-8) |
