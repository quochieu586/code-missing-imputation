"""Reproduce supporting summaries for the read-only E5 model assessment."""
import rework_runtime
from rework_runtime import ROOT
import json
import numpy as np
import pandas as pd
import torch
from scripts import run_e5_rework as R
from src.evaluation.research_pilot import sha256, write_json

B = ROOT/'artifacts/e3e4_rework_2026-10-05_b'
OUT = ROOT/'reports/e5_model_training_2026-10-06'


def main():
    assert (OUT/'assessment.json').exists()
    d = pd.read_csv(OUT/'checkpoint_full_loss.csv')
    s = pd.read_csv(OUT/'checkpoint_step_loss.csv')
    mean = d.groupby(['branch','transform','mode']).mean(numeric_only=True)
    std = d.groupby(['branch','transform','mode']).std(numeric_only=True,ddof=1)
    mean.to_csv(OUT/'full_loss_means.csv')
    std.to_csv(OUT/'full_loss_seed_sd.csv')
    fold = json.loads((ROOT/'artifacts/e0_extended/cv_splits.json').read_text())[0]
    names = json.loads((ROOT/'artifacts/e0_extended/manifest.json').read_text())['variant_columns']
    df = pd.read_csv(ROOT/'data/covariants.csv')
    outer = df.iloc[fold['train_rows']]
    pool = outer[outer.location!='Denmark']
    raw = pool[names].to_numpy(float)
    full = pool.loc[np.isfinite(raw).all(1)&(np.nansum(raw,1)>0)]
    checks = dict(zero_fraction_train=float((full[names].to_numpy()==0).mean()),
        zero_fraction_by_location={str(loc):float((g[names].to_numpy()==0).mean()) for loc,g in full.groupby('location')},
        expected_random_whole_column_length8=(5/17)**8,
        expected_random_whole_column_length4=(5/17)**4)
    for wl in (8,4):
        windows = R.window_indices(pool,full.index.to_numpy(),wl)
        matches = []
        for bank in range(4):
            hidden = ~R.masks(len(full),bank)
            matches += [int(hidden[idx].all(0).sum()) for idx in windows if len(idx)==wl]
        checks[f'wl{wl}'] = dict(full_windows_x_banks=len(matches),
            window_banks_with_any_whole_column=int((np.array(matches)>0).sum()),whole_columns_total=sum(matches),
            window_banks_with_two_or_more_whole_columns=int((np.array(matches)>=2).sum()))
        batches = R.groups_of_windows(windows)
        checks[f'wl{wl}']['batch_row_counts'] = [int(idx.size) for idx in batches]
        checks[f'wl{wl}']['short_tail_rows'] = int(sum(len(idx) for idx in windows if len(idx)<wl))
        checks[f'wl{wl}']['short_batches_fraction_epoch_log'] = sum(idx.shape[1]<wl for idx in batches)/len(batches)
    checks['alpha_bar'] = [float(v) for v in R.schedule()[1]]
    write_json(OUT/'mask_and_schedule_checks.json',checks)
    cap_duplicates = []
    for mode in R.ALL_MODES:
        for seed in (42,43,44):
            key1,key2 = f'{mode}__clr16__seed{seed}',f'{mode}__clr17__seed{seed}'
            cache = B/'cache_cap8'
            state1 = torch.load(cache/f'{key1}.pt',weights_only=True,map_location='cpu')
            state2 = torch.load(cache/f'{key2}.pt',weights_only=True,map_location='cpu')
            assert state1.keys()==state2.keys() and all(torch.equal(state1[k],state2[k]) for k in state1)
            a,b = np.load(cache/f'{key1}.npz'),np.load(cache/f'{key2}.npz')
            assert a.files==b.files
            assert all(np.array_equal(a[k],b[k]) for k in a.files if k.startswith('latent__'))
            # CLR16 subtracts the mean again in its inverse; softmax is translation invariant.
            # Predictions may differ by floating-point rounding while weights/latents are exact.
            assert all(np.allclose(a[k],b[k],rtol=1e-12,atol=1e-12) for k in a.files)
            cap_duplicates.append(dict(keys=[key1,key2],state_and_latents_exact=True,
                max_prediction_absolute_difference=max(float(np.max(np.abs(a[k]-b[k])))
                    for k in a.files if k.startswith('pred__'))))
    bybranch = {}
    for branch in ('main','cap8'):
        subset = d[(d.branch==branch)&(d['mode']!='mask_only')]
        ratio = subset.value_ablation_prediction_rms/subset.prediction_rms
        bybranch[branch] = dict(value_ablation_relative_prediction_rms_min=float(ratio.min()),
            value_ablation_relative_prediction_rms_max=float(ratio.max()),
            max_abs_ablation_loss_change=float(subset.value_ablation_loss_change.abs().max()))
    bands = []
    for label, lo, hi in [('early',0,4),('middle',5,14),('late',15,19)]:
        group = s[(s.partition=='validation') & (s['mode']=='no_init') & s.step.between(lo,hi)]
        for (branch,t),g in group.groupby(['branch','transform']):
            bands.append(dict(branch=branch,transform=t,step_band=label,first_step=lo,last_step=hi,
                epsilon_mse=float(g.epsilon_mse.mean()),
                gaussian_linear_reference_mse=float(g.gaussian_linear_reference_mse.mean())))
    pd.DataFrame(bands).to_csv(OUT/'noise_step_bands.csv',index=False)
    write_json(OUT/'supporting_checks.json',dict(capacity_clr_duplicate_pairs=cap_duplicates,
        value_ablation=bybranch,all54_validation_losses_below_train=bool((d.val_train_gap<0).all()),
        val_train_gap_min=float(d.val_train_gap.min()),val_train_gap_max=float(d.val_train_gap.max()),
        fixture_over_full_val_loss_min=float((d.small_fixture_loss/d.validation_epsilon_mse).min()),
        fixture_over_full_val_loss_max=float((d.small_fixture_loss/d.validation_epsilon_mse).max()),
        note='Retrospective diagnostic summaries; no changes to cached weights, predictions or gates',
        source_script_sha256=sha256(__file__)))
    print(json.dumps(dict(capacity_clr_duplicate_pairs=len(cap_duplicates),ablation=bybranch),indent=2))


if __name__=='__main__':
    main()
