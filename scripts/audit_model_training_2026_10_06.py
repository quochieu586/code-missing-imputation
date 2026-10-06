"""Read-only retrospective training assessment. No fitting, gate changes or test use."""
from __future__ import annotations
import rework_runtime
from rework_runtime import ROOT
import json
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch

from scripts import run_e5_rework as R
from scripts.diagnose_e5_rework import frozen_hashes
from src.evaluation.research_pilot import sha256, write_json

B = ROOT / 'artifacts/e3e4_rework_2026-10-05_b'
OUT = ROOT / 'reports/e5_model_training_2026-10-06'
TRANSFORMS = ('clr16', 'ilr16', 'hkglr16_actual')
MODES = ('no_init', 'mask_only', 'hron_2a')
NOISE_SEED = 20261006


def file_hashes(folder):
    return {str(p.relative_to(ROOT)): sha256(p) for p in folder.rglob('*') if p.is_file()}


def curve_row(blob, branch):
    tr = blob['trace']
    x = np.arange(1, len(tr) + 1)
    train = np.array([r['train_epsilon_mse'] for r in tr])
    val = np.array([r['validation_fixture_loss'] for r in tr])
    best = int(np.argmin(val)) + 1
    assert best == blob['best_epoch'] and float(val.min()) == blob['best_loss']
    mode, transform, seed = blob['key'].split('__')
    return dict(branch=branch, key=blob['key'], mode=mode, transform=transform,
                seed=int(seed.replace('seed', '')), best_epoch=best,
                train_first=float(train[0]), train_last=float(train[-1]),
                val_first=float(val[0]), val_best=float(val.min()), val_last=float(val[-1]),
                train_slope_last10=float(np.polyfit(x[-10:], train[-10:], 1)[0]),
                val_slope_last10=float(np.polyfit(x[-10:], val[-10:], 1)[0]),
                val_late_improvement=float(1 - val[-5:].mean() / val[-10:-5].mean()),
                val_last_over_best=float(val[-1] / val.min()))


def deterministic_loss(model, target, cond, times, windows, ab, ablate=False):
    """Cell-weighted across all windows; row-level noise independent of grouping/width."""
    g = torch.Generator().manual_seed(NOISE_SEED)
    z0_all = torch.from_numpy(target)
    c_all = torch.from_numpy(cond)
    t_all = torch.from_numpy(times)
    alt = c_all.clone()
    alt[:, :17] = 0
    alt[:, 34:51] = 0
    batches = R.groups_of_windows(windows)
    covered = np.concatenate([idx.ravel() for idx in batches])
    assert sorted(covered.tolist()) == list(range(len(target)))
    rows = []
    model.eval()
    with torch.no_grad():
        for step in range(20):
            eps_all = torch.randn(z0_all.shape, dtype=torch.float64, generator=g)
            totals = Counter()
            for idx in batches:
                z0, eps = z0_all[idx], eps_all[idx]
                noisy = R.forward_noise(z0, eps, ab[step])
                pred = model(noisy, c_all[idx], t_all[idx], torch.full((len(idx),), step))
                assert torch.isfinite(pred).all()
                totals['n'] += eps.numel()
                totals['model'] += float((pred - eps).square().sum())
                totals['zero'] += float(eps.square().sum())
                # Simple unconditioned reference, NOT an oracle or a fitted replacement model.
                gaussian = (1 - ab[step]).sqrt() * noisy
                totals['gaussian_linear'] += float((gaussian - eps).square().sum())
                totals['prediction_energy'] += float(pred.square().sum())
                if ablate:
                    p_alt = model(noisy, alt[idx], t_all[idx], torch.full((len(idx),), step))
                    totals['ablation'] += float((p_alt - eps).square().sum())
                    totals['prediction_change'] += float((p_alt - pred).square().sum())
            n = totals['n']
            rows.append(dict(step=step, alpha_bar=float(ab[step]), n_coordinates=n,
                epsilon_mse=totals['model']/n, zero_reference_mse=totals['zero']/n,
                gaussian_linear_reference_mse=totals['gaussian_linear']/n,
                ablation_mse=totals['ablation']/n if ablate else None,
                prediction_change_mse=totals['prediction_change']/n if ablate else None,
                prediction_energy=totals['prediction_energy']/n))
    return rows


def main():
    if (OUT / 'scope.json').exists():
        raise FileExistsError('Assessment already started; choose a new output version to rerun')
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    before = file_hashes(B)
    frozen_before = frozen_hashes()
    baseline = json.loads((ROOT / 'artifacts/e3e4_rework_2026-10-05/frozen_hashes_before.json').read_text())
    assert baseline == frozen_before
    scope = dict(retrospective=True, no_training=True, no_canada=True, no_checkpoint_selection=True,
        curves='all 192 registered checkpoints', diagnostic_transforms=list(TRANSFORMS),
        diagnostic_modes=list(MODES), seeds=[42,43,44], branches=['main','cap8'],
        diagnostic_checkpoints=54, mask='training bank0; frozen scoring bank0 on validation',
        noise_seed=NOISE_SEED, steps=list(range(20)),
        aggregation='sum squared errors / all coordinates, then mean across 20 steps',
        ablation='zero raw visible-value and initializer-value slots; preserve masks/provenance/time',
        scope_note='Written before fresh checkpoint evaluation; retrospective subset, not a new preregistration')
    write_json(OUT / 'scope.json', scope)
    curves = []
    for branch, cache in [('main', B/'cache_validation'), ('cap8', B/'cache_cap8')]:
        for jp in sorted(cache.glob('*__seed*.json')):
            blob = json.loads(jp.read_text())
            assert len(blob['trace']) == 40
            curves.append(curve_row(blob, branch))
    curves = pd.DataFrame(curves)
    assert len(curves) == 192
    curves.to_csv(OUT / 'learning_curves_summary.csv', index=False)
    curve_summary = {}
    for branch, group in curves.groupby('branch'):
        curve_summary[branch] = dict(n=len(group), best_epoch40=int((group.best_epoch==40).sum()),
            best_epoch_ge35=int((group.best_epoch>=35).sum()),
            negative_train_slope=int((group.train_slope_last10<0).sum()),
            negative_val_slope=int((group.val_slope_last10<0).sum()),
            median_late_val_improvement=float(group.val_late_improvement.median()),
            best_epoch_median=float(group.best_epoch.median()),
            val_last_gt_best_5pct=int((group.val_last_over_best>1.05).sum()))
    df = pd.read_csv(ROOT/'data/covariants.csv')
    names = json.loads((ROOT/'artifacts/e0_extended/manifest.json').read_text())['variant_columns']
    fold = json.loads((ROOT/'artifacts/e0_extended/cv_splits.json').read_text())[0]
    outer = df.iloc[fold['train_rows']]
    pool = outer[outer.location!='Denmark']
    val = outer[outer.location=='Denmark']
    assert 'Canada' not in set(pool.location) | set(val.location)
    raw_pool = pool[names].to_numpy(float)
    full = pool.loc[np.isfinite(raw_pool).all(1) & (np.nansum(raw_pool,1)>0)]
    raw_train = full[names].to_numpy(float)
    truth = val[names].to_numpy(float)
    val_ok = np.isfinite(truth).all(1) & (truth.sum(1)>0)
    val_full = val.loc[val_ok]
    raw_val = truth[val_ok]
    raw_scale = R.raw_scale_for(raw_pool)
    origin = pd.to_datetime(pool.date).min().toordinal()
    times_train = R.time_encode(full.date, origin)
    times_val = R.time_encode(val_full.date, origin)
    tm = R.load_transform_module()
    _, ab = R.schedule()
    print('Precomputing train-only cross-fit conditions (no fitting)', flush=True)
    conditions_train = R.precompute_train_conditions(full, pool, names, raw_scale)
    hidden = np.load(B/'validation_random_cell_bank0.npz')['bank0']
    visible = ~hidden
    query = np.where(visible, truth, np.nan)
    init, fb, ek, _ = R.initialize_crossfit(query, val.index.to_numpy(), val.location.to_numpy(), pool, names)
    conditions_val = {mode: R.condition_features(query, visible, 'init' if mode in init else mode,
        raw_scale, init.get(mode), fb.get(mode), ek.get(mode))[val_ok] for mode in MODES}
    fixture_visible = R.masks(len(val_full),0)
    scope['fixture_rows_with_different_mask'] = int(np.any(fixture_visible != visible[val_ok],axis=1).sum())
    scope['train_rows'] = len(full)
    scope['validation_rows'] = len(val_full)
    scope['train_positive_rows_by_location'] = {str(k):int(v) for k,v in full.location.value_counts().items()}
    budgets = {}
    for branch, length in [('main',8), ('cap8',4)]:
        windows = R.window_indices(pool, full.index.to_numpy(), length)
        n_updates = len(R.groups_of_windows(windows))
        budgets[branch] = dict(window_length=length, n_windows=len(windows),
            window_length_distribution={str(k):int(v) for k,v in Counter(map(len,windows)).items()},
            updates_per_epoch=n_updates, updates_40_epochs=n_updates*40,
            raw_row_presentations_40_epochs=len(full)*40, n_unique_training_masks=4,
            rows=len(full), locations=full.location.nunique())
    parameters = []
    for branch in ('main','cap8'):
        for tname in R.ALL_TRANSFORMS:
            refs = R.hkglr_refs(tname,pool,names)
            transform = tm.make_transform(tname,17,.5,refs if tname.startswith('hk') else None)
            k = transform.forward(raw_train).shape[-1]
            channels = R.MODEL_CHANNELS[tname] if branch=='main' else 8
            model = R.CSDICoreRework(k,channels=channels,heads=1)
            components = {name:sum(p.numel() for p in sub.parameters()) for name,sub in model.named_children()}
            parameters.append(dict(branch=branch,transform=tname,latent_dim=k,channels=channels,
                parameter_count=sum(p.numel() for p in model.parameters()), **components))
    pd.DataFrame(parameters).to_csv(OUT/'parameter_counts.csv',index=False)
    diagnostic = []
    step_rows = []
    for branch,cache,length in [('main',B/'cache_validation',8),('cap8',B/'cache_cap8',4)]:
        train_windows = R.window_indices(pool,full.index.to_numpy(),length)
        val_windows = R.window_indices(val,val_full.index.to_numpy(),length)
        for tname in TRANSFORMS:
            refs = R.hkglr_refs(tname,pool,names)
            transform = tm.make_transform(tname,17,.5,refs if tname.startswith('hk') else None)
            target = transform.forward(raw_train)
            mean = target.mean(0)
            scale = max(float(np.sqrt(np.mean((target-mean)**2))),1e-8)
            ztrain = (target-mean)/scale
            zval = (transform.forward(raw_val)-mean)/scale
            for mode in MODES:
                for seed in (42,43,44):
                    key = f'{mode}__{tname}__seed{seed}'
                    blob = json.loads((cache/f'{key}.json').read_text())
                    model = R.CSDICoreRework(target.shape[-1],channels=blob['channels'],heads=1)
                    model.load_state_dict(torch.load(cache/f'{key}.pt',weights_only=True,map_location='cpu'))
                    train_result = deterministic_loss(model,ztrain,conditions_train[(mode,0)],times_train,train_windows,ab)
                    val_result = deterministic_loss(model,zval,conditions_val[mode],times_val,val_windows,ab,ablate=True)
                    if mode=='mask_only':
                        assert all(r['prediction_change_mse']==0 and r['ablation_mse']==r['epsilon_mse'] for r in val_result)
                    avg = lambda rows, attr: float(np.mean([r[attr] for r in rows]))
                    trloss = avg(train_result,'epsilon_mse')
                    valloss = avg(val_result,'epsilon_mse')
                    diag = dict(branch=branch,key=key,mode=mode,transform=tname,seed=seed,best_epoch=blob['best_epoch'],
                        small_fixture_loss=blob['best_loss'],train_epsilon_mse=trloss,validation_epsilon_mse=valloss,
                        val_train_gap=valloss-trloss,val_train_ratio=valloss/trloss,
                        validation_zero_reference=avg(val_result,'zero_reference_mse'),
                        validation_gaussian_linear_reference=avg(val_result,'gaussian_linear_reference_mse'),
                        validation_ablation_mse=avg(val_result,'ablation_mse'),
                        value_ablation_loss_change=avg(val_result,'ablation_mse')-valloss,
                        value_ablation_prediction_rms=float(np.sqrt(avg(val_result,'prediction_change_mse'))),
                        prediction_rms=float(np.sqrt(avg(val_result,'prediction_energy'))),
                        train_standardized_energy=float(np.mean(ztrain**2)),val_standardized_energy=float(np.mean(zval**2)))
                    assert abs(diag['train_standardized_energy']-1)<1e-12
                    diagnostic.append(diag)
                    for label, rows in [('train',train_result),('validation',val_result)]:
                        step_rows += [dict(branch=branch,key=key,mode=mode,transform=tname,seed=seed,partition=label,**r) for r in rows]
                    print(f'{branch} {key}: train={trloss:.4f}, full-val={valloss:.4f}, gap={valloss-trloss:+.4f}',flush=True)
    pd.DataFrame(diagnostic).to_csv(OUT/'checkpoint_full_loss.csv',index=False)
    pd.DataFrame(step_rows).to_csv(OUT/'checkpoint_step_loss.csv',index=False)
    assert before == file_hashes(B)
    assert frozen_before == frozen_hashes()
    assert not (B/'cache_test').exists() and not (B/'canada_evaluated.lock').exists()
    write_json(OUT/'assessment.json',dict(scope=scope,budgets=budgets,curves=curve_summary,
        frozen_inputs_unchanged=True,rework_artifacts_unchanged=True,canada_evaluated=False,
        checkpoint_evaluations=len(diagnostic),wall_seconds=time.perf_counter()-started,
        input_hashes=before,script_sha256=sha256(__file__)))
    print(json.dumps(dict(curves=curve_summary,budgets=budgets,checkpoint_evaluations=len(diagnostic)),indent=2),flush=True)


if __name__ == '__main__':
    main()
