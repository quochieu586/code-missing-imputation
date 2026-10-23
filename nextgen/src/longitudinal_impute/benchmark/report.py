"""Aggregate held-out jobs, rescore serialized predictions, audit final CSVs."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from .engine import (load_run, get_mask, write_json, file_hash,
                     prediction_artifact_path, final_csv_artifact_path)
from .metrics import score, bootstrap_locations
from .completion import RunArtifacts
from .candidates import final_candidate


def aggregate(records, config):
    locations = {}
    for record in records:
        for row in record['metrics']['by_location']:
            entry = locations.setdefault(row['location'], dict(location=row['location'], n=0,
                absolute_sum=0., squared_sum=0., ratio_absolute_sum=0., ratio_squared_sum=0.))
            for key in entry:
                if key != 'location': entry[key] += row[key]
    values = list(locations.values())
    n = sum(x['n'] for x in values)
    if n == 0:
        return {'n': 0, 'jobs': len(records)}
    result = {'n': n, 'jobs': len(records),
              'count_mae': sum(x['absolute_sum'] for x in values) / n,
              'count_rmse': float(np.sqrt(sum(x['squared_sum'] for x in values) / n)),
              'ratio_mae': sum(x['ratio_absolute_sum'] for x in values) / n,
              'ratio_rmse': float(np.sqrt(sum(x['ratio_squared_sum'] for x in values) / n)),
              'location_macro_count_mae': float(np.mean([x['absolute_sum'] / x['n'] for x in values if x['n']])),
              'location_macro_ratio_mae': float(np.mean([x['ratio_absolute_sum'] / x['n'] for x in values if x['n']])),
              'location_stats': values, 'ci': bootstrap_locations(values, config['bootstrap_seed'], config['bootstrap_draws']),
              'fallback_rows': sum(x['diagnostics']['fallback_rows'] for x in records),
              'evaluated_rows': sum(x['rows'] for x in records),
              'seconds': sum(x['seconds'] for x in records),
              'validity': {k: sum(x['metrics']['validity'][k] for x in records)
                           for k in ['observed_changed', 'negative', 'budget_violations']}}
    for stratum in ['zero', 'positive', 'rare_positive', 'common_positive']:
        denominator = sum(r['metrics'][stratum]['n'] for r in records)
        result[stratum] = {'n': denominator, 'ratio_mae': sum(
            r['metrics'][stratum]['ratio_mae'] * r['metrics'][stratum]['n']
            for r in records if r['metrics'][stratum]['n']) / denominator if denominator else None}
    result['feature_scores'] = []
    for j in range(len(records[0]['metrics']['by_feature'])):
        entries = [r['metrics']['by_feature'][j] for r in records]
        denom = sum(x['n'] for x in entries)
        result['feature_scores'].append({'feature': entries[0]['feature'], 'n': denom,
            'ratio_mae': sum(x['ratio_mae'] * x['n'] for x in entries if x['n']) / denom if denom else None})
    result['feature_macro_ratio_mae'] = float(np.mean([x['ratio_mae'] for x in result['feature_scores'] if x['n']]))
    result['relationship_by_job'] = [{'coverage': r['metrics']['coverage'], 'temporal': r['metrics']['temporal'],
                                       'feature_relationship': r['metrics']['feature_relationship']} for r in records]
    return result


def paired_ci(a, b, config):
    left = {x['location']: x for x in a['location_stats'] if x['n']}
    right = {x['location']: x for x in b['location_stats'] if x['n']}
    keys = sorted(set(left) & set(right))
    delta = np.array([left[k]['ratio_absolute_sum'] / left[k]['n'] -
                      right[k]['ratio_absolute_sum'] / right[k]['n'] for k in keys])
    rng = np.random.default_rng(config['bootstrap_seed'])
    samples = delta[rng.integers(0, len(delta), (config['bootstrap_draws'], len(delta)))].mean(axis=1)
    return {'locations': len(keys), 'location_macro_ratio_mae_difference': float(delta.mean()),
            'ci95': np.quantile(samples, [.025, .975]).tolist(), 'negative_favors': 'first_method'}


def build_report(root, output, destination):
    output, destination = Path(output), Path(destination)
    manifest, raw, _, rows = load_run(root, output)
    artifacts = RunArtifacts(output, manifest, raw)
    config = manifest['config']
    primary, supplementary, failures, rescored = defaultdict(list), defaultdict(list), [], []
    for fold in range(config['folds']):
        artifacts.fold(fold)
        base = output / f'fold{fold}'
        for family in config['candidates']:
            for scenario in config['scenarios']:
                for seed in config['mask_seeds']:
                    path = base / 'test' / family / f'{scenario}_{seed}'
                    record = json.loads((path / 'result.json').read_text(encoding='utf-8'))
                    predictions = prediction_artifact_path(path, record)
                    if file_hash(predictions) != record['predictions_hash']:
                        raise AssertionError('Prediction hash mismatch')
                    with np.load(predictions, allow_pickle=False) as saved:
                        metrics = score(raw, saved['row_indices'], saved['counts'], get_mask(output, scenario, seed))
                    if json.dumps(metrics, sort_keys=True) != json.dumps(record['metrics'], sort_keys=True):
                        raise AssertionError('Serialized prediction rescore mismatch')
                    primary[(family, scenario)].append(record)
                    rescored.append({'fold': fold, 'family': family, 'scenario': scenario, 'seed': seed, 'pass': True})
        for method in config['supplementary_methods']:
            base_method = base / 'supplementary' / method
            finish = json.loads((base_method / 'finished.json').read_text(encoding='utf-8'))
            if finish['status'] != 'ok':
                failures.append({'fold': fold, 'method': method, **finish})
                continue
            for scenario in config['supplementary_scenarios']:
                for seed in config['mask_seeds']:
                    supplementary[(method, scenario)].append(json.loads(
                        (base_method / f'{scenario}_{seed}' / 'result.json').read_text(encoding='utf-8')))
    grouped = {f'{family}/{scenario}': aggregate(records, config) for (family, scenario), records in primary.items()}
    extra = {f'{method}/{scenario}': aggregate(records, config) for (method, scenario), records in supplementary.items()}
    paired = {}
    families = list(config['candidates'])
    for scenario in config['scenarios']:
        for i, a in enumerate(families):
            for b in families[i + 1:]:
                paired[f'{a}-{b}/{scenario}'] = paired_ci(grouped[f'{a}/{scenario}'], grouped[f'{b}/{scenario}'], config)
    final_audit, final_csv_paths = [], {}
    selections = [artifacts.selection(fold) for fold in range(config['folds'])]
    for family in families:
        directory = output / 'final' / family
        final_manifest = artifacts.final(family, final_candidate(config, selections, family))
        if final_manifest is None:
            raise ValueError(f'Missing final-refit completion: {family}')
        path = final_csv_artifact_path(directory, final_manifest)
        if file_hash(path) != final_manifest['csv_hash']:
            raise AssertionError('Final CSV hash mismatch')
        with path.open(encoding='utf-8', newline='') as stream:
            completed = list(csv.DictReader(stream))
        if len(completed) != len(rows): raise AssertionError('Final CSV row count')
        filled = 0
        for i, (original, done) in enumerate(zip(rows, completed)):
            for key, value in original.items():
                if key not in raw.schema.target_columns or value.strip().lower() not in ('', 'nan', 'na', 'none'):
                    if done[key] != value: raise AssertionError('Final source text changed')
                else: filled += 1
            values = np.array([float(done[name]) for name in raw.schema.target_columns])
            if not np.isfinite(values).all() or (values < 0).any() or values.sum() > raw.totals[i] + 1e-8:
                raise AssertionError('Final CSV infeasible')
        final_audit.append({'family': family, 'rows': len(completed), 'native_filled': filled,
                            'observed_changed': 0, 'invalid_rows': 0, 'sha256': file_hash(path)})
        final_csv_paths[family] = path
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / 'summary.json', {'manifest': manifest, 'primary': grouped, 'supplementary': extra,
               'not_applicable': failures, 'paired_differences': paired, 'final_csv_audit': final_audit})
    write_json(destination / 'rescoring.json', {'jobs': len(rescored), 'all_pass': True, 'details': rescored})
    lines = ['# Benchmark offline: Hron, MICoDa, Tsagris', '',
             f"Dữ liệu: {manifest['rows']} dòng, {manifest['locations']} địa điểm; "
             f"{config['folds']}-fold group theo location, {len(config['mask_seeds'])} mask seeds, "
             f"rate{config['mask_rate']:.0%}. Whole-feature là kịch bản chính. "
             'Validation chọn tham số trước test. Không có causal mask.', '',
             '| Phương pháp | Kịch bản | Targets | MAE counts | MAE tỷ lệ/N | CI95 tỷ lệ (location bootstrap) | Fallback rows |',
             '|---|---|---:|---:|---:|---|---:|']
    for key, value in grouped.items():
        family, scenario = key.split('/')
        ci = value['ci']['ratio_mae_pooled_ci95']
        lines.append(f"| {family} | {scenario} | {value['n']} | {value['count_mae']:.4f} | {value['ratio_mae']:.6f} | [{ci[0]:.6f}, {ci[1]:.6f}] | {value['fallback_rows']}/{value['evaluated_rows']} |")
    lines += ['', 'Targets là tổng các lần che qua3 seeds; không phải số ô độc lập. '
              'CI resample theo location, đã gộp các mask repetitions trong location. '
              'Ba phương pháp dùng đúng cùng split/mask/labels.', '',
              '## Biến thể bổ sung (whole-feature)', '',
              '| Method | Successful jobs | MAE tỷ lệ | Fallback |', '|---|---:|---:|---:|']
    for key, value in extra.items():
        expected = config['folds'] * len(config['mask_seeds']) * len(config['supplementary_scenarios'])
        lines.append(f"| {key} | {value['jobs']}/{expected} | {value['ratio_mae']:.6f} | {value['fallback_rows']} |")
    for failure in failures:
        lines.append(f"\n- Fold {failure['fold']}, {failure['method']}: không áp dụng — {failure['error']}.")
    lines += ['', '## Diễn giải và giới hạn', '',
              '- Baselines là general/cross-sectional; offline cho phép ngữ cảnh tương lai, '
              'nhưng ba core này không tự học attention hoặc dynamics dọc chuỗi. '
              'Row/time-block che tất cả features khiến Hron dùng train-mean fallback; '
              'Tsagris dùng global donor policy; MICoDa dùng covariates.',
              '- Metadata chính: median_age và new_cases_smoothed_per_million; fit encoder chỉ train. '
              'Hron/Tsagris core không dùng covariate regression; MICoDa có dùng. '
              'classification/Overall_score/group chưa được bật vì provenance chưa rõ.',
              '- Counts zeros giữ nguyên; smoothing chỉ encoder. Other unknown sau khi che raw target. '
              'Output constraints đảm bảo tính hợp lệ, không tự chứng minh accuracy.',
              '- Composition/coverage chỉ chấm trên rows có complete truth; conditional17 '
              'LR diagnostics dùng fixed additive1e-6 để xử lý zeros và có ghi support. '
              'Temporal/feature relationships mô tả vùng có truth; correlation pooled '
              'có thể phản ánh country mix và không phải quan hệ nhân quả.',
              '- LTS raw/solver approximate và Hron Table1/software parity có giới hạn đã ghi '
              'trong nextgen/BASELINES_VI.md; không gọi benchmark này là tái lập tất cả paper tables.',
              f"- Biến thể bổ sung có wall budget {config['supplementary_wall_seconds_per_fold_method']}s "
              'mỗi fold/method, đăng ký trước khi chạy test. Resource-limit là kết quả chi phí, '
              'không phải điểm accuracy và không được đưa vào ranking trên tập fold khác nhau.',
              '- Native missing không có ground truth. CSV final refit dùng toàn bộ observed '
              'sau chọn cấu hình; không dùng nó để báo lại test accuracy.', '',
              '## CSV hoàn chỉnh', '']
    for audit in final_audit:
        csv_path = final_csv_paths[audit['family']].resolve().as_posix()
        lines.append(f"- [{audit['family']}]({csv_path}): {audit['rows']} dòng, "
                     f"điền {audit['native_filled']} missing cells, observed changes=0, invalid rows=0.")
    lines += ['', 'Chi tiết strata, per-feature, location macro, paired bootstrap differences, '
              'temporal/feature diagnostics và failures: summary.json. Toàn bộ primary test '
              'prediction arrays được đọc lại và rescore; xem rescoring.json.']
    (destination / 'report_vi.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return {'primary_jobs': len(rescored), 'supplementary_failures': len(failures), 'final_csv_audit': final_audit}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build_report(args.root, args.output, args.report), indent=2))
