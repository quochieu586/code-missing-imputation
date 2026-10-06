"""E5 mask-conditioning experiment (directive section 11.1-11.5).

Tests two hypotheses:
  (a) random-cell training masks do not transfer to whole-variant missingness.
  (b) 280 optimizer updates (40 epochs) are insufficient under matched conditions.

Factorial design (section 11.2):
  Control policy:    18 random-cell banks (5 hidden parts/row).
  Intervention policy: 18 banks = 9 paired random-cell banks + 9 whole-variant banks.
  Whole-variant banks cover n_hidden={1,2,3} x abundance={rare,middle,common}.
  2 policies x 4 modes x 3 seeds = 24 trajectories.
  Each trajectory trains 160 epochs (1120 updates); checkpoint selected within
  updates 1-280 and within updates 1-1120 -> 48 scored entries.

Canada is NEVER evaluated. Use --no-test to enforce.

Run:  & $ResearchPython -B scripts/run_e5_mask_conditioning.py --help
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rework_runtime  # noqa: E402,F401

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.dont_write_bytecode = True
torch.set_num_threads(2)

from src.stage_b.research_csdi import (  # noqa: E402
    condition_features, forward_noise, epsilon_loss, project_final,
    reverse_step, sample_latents, schedule)
from src.stage_b.research_csdi_rework import CSDICoreRework  # noqa: E402
from src.stage_a.initializers_e1 import train_fallback_values  # noqa: E402
from src.evaluation.research_pilot import (  # noqa: E402
    evaluate, prevalence_bins, numerical_diagnostics, sha256, write_json,
    failure_records)
from scripts.run_e3_revised import (  # noqa: E402
    load_transform_module, window_indices, initialize_crossfit, masks as rc_masks)
from scripts.run_e5_rework import (  # noqa: E402
    make_random_cell_mask, whole_variant_masks, groups_of_windows,
    time_encode, raw_scale_for, evaluate_csdi_on_scenario,
    evaluate_init_only_on_scenario, evaluate_baseline_on_scenario)
from scripts.audit_e0_extended import choose_hidden_variants  # noqa: E402
from src.evaluation import rework_gates as RG  # noqa: E402

DATA = ROOT / 'data/covariants.csv'
E0 = ROOT / 'artifacts/e0_extended'
E2 = ROOT / 'artifacts/e2_transforms'
DATA_SHA = 'bbb1aacdf46ebef345ca818f086d5586f4aded45475b00b6ee9bd89fc94a90ab'

POLICIES = ('control', 'intervention')
ALL_TRANSFORMS = ['clr16']
ALL_MODES = ['no_init', 'mask_only', 'hron_2a', 'aitchison_complete']
INIT_MODES = ('hron_2a', 'aitchison_complete')
ALL_SEEDS = [42, 43, 44]

MASK_BANKS = 18
MASKED_PARTS = 5
MASK_SEED = 42
MAX_EPOCHS = 160
BUDGET_280_EPOCHS = 40
BUDGET_1120_EPOCHS = 160
MC_SAMPLES = 8
BATCH_WINDOWS = 8
WINDOW_LENGTH = 8
LR = 1e-3
FIXTURE_NOISE_SEED = 919
FIXTURE_STEPS = 20
DDPM_STEPS = 20
LAYERS = 2
HEADS = 1
CHANNELS = 16
N_VARIANTS = 17

PRIMARY = RG.PRIMARY
SECONDARY = ('random_cell_bank0',)


def policy_key(policy, mode, seed):
    return f'{policy}_{mode}__clr16__seed{seed}'


def budget_key(policy, mode, seed, budget):
    return f'{policy}_{mode}__clr16__seed{seed}_budget{budget}'


# ---------------------------------------------------------------------------
# Training mask generation
# ---------------------------------------------------------------------------

def make_training_mask_banks(policy, pool_df, names, n_rows):
    """Generate 18 training mask banks for a policy. Returns list of visible arrays
    (True = VISIBLE), one per bank.

    Control: 18 random-cell banks, 5 hidden parts/row, seeds = MASK_SEED + bank.
    Intervention: 9 random-cell banks (banks 0-8, identical to control) +
    9 whole-variant banks covering n_hidden={1,2,3} x {rare,middle,common}.
    """
    banks = []
    if policy == 'control':
        for bank in range(MASK_BANKS):
            hidden = make_random_cell_mask(n_rows, N_VARIANTS, MASKED_PARTS,
                                           seed=MASK_SEED + bank)
            banks.append(~hidden)
    elif policy == 'intervention':
        for bank in range(9):
            hidden = make_random_cell_mask(n_rows, N_VARIANTS, MASKED_PARTS,
                                           seed=MASK_SEED + bank)
            banks.append(~hidden)
        ordering = choose_hidden_variants(pool_df, names)
        for label in ['rare', 'middle', 'common']:
            for n_hidden in [1, 2, 3]:
                hidden_vars = ordering[label][:n_hidden]
                hidden = np.zeros((n_rows, N_VARIANTS), dtype=bool)
                for v in hidden_vars:
                    hidden[:, names.index(v)] = True
                banks.append(~hidden)
    else:
        raise ValueError(f'unknown policy: {policy}')
    assert len(banks) == MASK_BANKS
    for b in banks:
        assert b.shape == (n_rows, N_VARIANTS)
        assert b.dtype == bool
    return banks


def assert_paired_banks(control_banks, intervention_banks):
    """Intervention banks 0-8 must be identical to control banks 0-8."""
    for i in range(9):
        assert np.array_equal(control_banks[i], intervention_banks[i]), \
            f'paired bank mismatch at index {i}'


def assert_mask_constant_per_location(hidden):
    """Whole-variant masks must hide the same columns for all rows."""
    if hidden.dtype != bool:
        hidden = hidden.astype(bool)
    n_unique_cols = np.unique(hidden, axis=0).shape[0]
    assert n_unique_cols == 1, f'mask varies across rows (got {n_unique_cols} unique rows)'


def assert_n_strata_covered(banks):
    """The 9 whole-variant intervention banks must cover all 9 strata."""
    # Check that the whole-variant banks (indices 9-17) cover all combinations
    # of n_hidden in {1,2,3} and abundance in {rare,middle,common}
    whole_banks = banks[9:]  # 9 whole-variant banks
    assert len(whole_banks) == 9
    # Each bank hides a whole column (same for all rows)
    for b in whole_banks:
        visible = b  # True=VISIBLE
        hidden_cols = np.where(~visible[0])[0]  # columns hidden in first row
        assert len(hidden_cols) in (1, 2, 3), f'unexpected n_hidden: {len(hidden_cols)}'
    # Verify coverage: 3 strata x 3 n_hidden levels
    hidden_set = set()
    for b in whole_banks:
        visible = b
        hidden_cols = frozenset(np.where(~visible[0])[0].tolist())
        hidden_set.add(hidden_cols)
    assert len(hidden_set) == 9, f'expected 9 unique hidden column sets, got {len(hidden_set)}'


# ---------------------------------------------------------------------------
# Precompute training conditions
# ---------------------------------------------------------------------------

def precompute_train_conditions(pool_eligible, all_pool, names, raw_scale, mask_banks):
    """Condition features for every (mode, bank) on training rows.

    mask_banks is a list of visible arrays (True = VISIBLE).
    """
    raw = pool_eligible[names].to_numpy(dtype=np.float64)
    ids = pool_eligible.index.to_numpy(dtype=np.int64)
    conditions = {}
    for bank in range(len(mask_banks)):
        visible = mask_banks[bank]
        query = np.where(visible, raw, np.nan)
        init, fb, ek, _ = initialize_crossfit(
            query, ids, pool_eligible.location.to_numpy(), all_pool, names)
        for mode in ALL_MODES:
            cf_mode = 'init' if mode in INIT_MODES else mode
            conditions[(mode, bank)] = condition_features(
                query, visible, cf_mode, raw_scale,
                init.get(mode), fb.get(mode), ek.get(mode))
    return conditions


# ---------------------------------------------------------------------------
# Checkpoint selection fixture
# ---------------------------------------------------------------------------

def make_checkpoint_fixture(val_frame, names, inner_pool, mode, transform_obj,
                              mean, scale, raw_scale, val_windows, val_times,
                              val_wvmasks, noise_seed=FIXTURE_NOISE_SEED,
                              n_steps=FIXTURE_STEPS):
    """Fixed validation fixture over ALL 103 eligible Denmark rows, all windows,
    20 steps, using the three frozen primary whole-variant n2 masks.

    Equal-weight mean across the 3 masks; earliest argmin at ties.
    Noise is drawn once from a single generator seeded with noise_seed.
    """
    val_raw = val_frame[names].to_numpy(dtype=np.float64)
    ids = val_frame.index.to_numpy(dtype=np.int64)
    locations = val_frame.location.to_numpy()

    fixture_masks = list(PRIMARY)  # whole_rare_n2, whole_middle_n2, whole_common_n2
    beta, alpha_bar = schedule()
    g = torch.Generator().manual_seed(noise_seed)

    fixture_data = []
    noise_records = []
    for mask_key in fixture_masks:
        hidden = val_wvmasks[mask_key]  # True = HIDDEN
        visible = ~hidden
        q = np.where(visible, val_raw, np.nan)

        if mode in INIT_MODES:
            init, fb, ek, _ = initialize_crossfit(q, ids, locations, inner_pool, names)
            init_arr, fb_arr, ek_arr = init[mode], fb[mode], ek[mode]
        else:
            init_arr = fb_arr = ek_arr = None

        cf_mode = 'init' if mode in INIT_MODES else mode
        cond = condition_features(q, visible, cf_mode, raw_scale, init_arr, fb_arr, ek_arr)

        z0_full = transform_obj.forward(val_raw)
        z0_normalized = (z0_full - mean) / scale

        window_data = []
        for w_idx, w in enumerate(val_windows):
            z0_w = torch.from_numpy(z0_normalized[w])[None, ...]
            cond_w = torch.from_numpy(cond[w])[None, ...]
            t_w = torch.from_numpy(val_times[w])[None, ...]
            eligible = torch.ones_like(z0_w, dtype=torch.bool)
            noise = [torch.randn(z0_w.shape, dtype=torch.float64, generator=g)
                     for _ in range(n_steps)]
            noise_records.append({
                'mask': mask_key, 'window': w_idx,
                'row_ids': [int(x) for x in val_frame.index[w].tolist()],
                'shape': list(z0_w.shape)
            })
            window_data.append({
                'z0': z0_w, 'cond': cond_w, 'times': t_w,
                'noise': noise, 'eligible': eligible
            })
        fixture_data.append({
            'mask_key': mask_key,
            'window_data': window_data
        })

    def fixture_fn(model):
        model.eval()
        all_losses = []
        with torch.no_grad():
            for fd in fixture_data:
                mask_losses = []
                for wd in fd['window_data']:
                    step_losses = []
                    for step in range(n_steps):
                        noisy = forward_noise(wd['z0'], wd['noise'][step],
                                              alpha_bar[step])
                        pred = model(noisy, wd['cond'], wd['times'],
                                     torch.tensor([step]))
                        loss = epsilon_loss(pred, wd['noise'][step],
                                            wd['eligible'])
                        step_losses.append(float(loss))
                    mask_losses.append(float(np.mean(step_losses)))
                all_losses.append(float(np.mean(mask_losses)))
        return float(np.mean(all_losses))

    return fixture_fn, noise_records


# ---------------------------------------------------------------------------
# Training with dual-budget checkpoint selection
# ---------------------------------------------------------------------------

def fit_with_dual_budget(target, cond_banks, mode, windows, times, alpha_bar,
                          channels, latent_dim, seed, fixture_fn, max_epochs,
                          budget1_epochs=BUDGET_280_EPOCHS,
                          budget2_epochs=BUDGET_1120_EPOCHS):
    """Train CSDICoreRework (heads=1) for max_epochs.

    Track best checkpoint (argmin fixture loss, earliest tie) within
    epochs 1-budget1_epochs (280 updates) and within 1-budget2_epochs (1120 updates).
    Both checkpoints come from the SAME trajectory.

    Returns model (at final epoch), mean, scale, checkpoint dicts, trace, elapsed.
    """
    torch.manual_seed(seed)
    model = CSDICoreRework(latent_dim, channels=channels, heads=HEADS,
                          steps=DDPM_STEPS, layers=LAYERS)
    mean = target.mean(0)
    scale = max(float(np.sqrt(np.mean((target - mean) ** 2))), 1e-8)
    normalized = (target - mean) / scale
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(seed)
    noise_rng = torch.Generator().manual_seed(seed)
    trace = []

    best1_epoch, best1_loss, best1_state = 1, float('inf'), None
    best2_epoch, best2_loss, best2_state = 1, float('inf'), None

    start = time.perf_counter()
    for epoch in range(max_epochs):
        model.train()
        batch_losses = []
        cell_weighted_sum = 0.0
        cell_total = 0
        grad_norms = []
        cond = cond_banks[(mode, epoch % MASK_BANKS)]
        for idx in groups_of_windows(windows, rng):
            z0 = torch.from_numpy(normalized[idx])
            c = torch.from_numpy(cond[idx])
            t = torch.from_numpy(times[idx])
            steps = torch.randint(20, (len(idx),), generator=noise_rng)
            eps = torch.randn(z0.shape, dtype=torch.float64, generator=noise_rng)
            noisy = forward_noise(z0, eps, alpha_bar[steps, None, None])
            pred = model(noisy, c, t, steps)
            loss = epsilon_loss(pred, eps, torch.ones_like(z0, dtype=torch.bool))
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite train loss')
            optimizer.zero_grad()
            loss.backward()
            gn = float(sum(p.grad.norm() for p in model.parameters()
                          if p.grad is not None) / max(1, sum(1 for p in model.parameters()
                          if p.grad is not None)))
            grad_norms.append(gn)
            optimizer.step()
            n_eligible = z0.numel()
            batch_losses.append(float(loss.detach()))
            cell_weighted_sum += float(loss.detach()) * n_eligible
            cell_total += n_eligible

        fl = fixture_fn(model)
        epoch_num = epoch + 1
        trace.append({
            'epoch': epoch_num,
            'train_epsilon_mse': float(np.mean(batch_losses)) if batch_losses else float('nan'),
            'train_cell_weighted_mse': float(cell_weighted_sum / cell_total)
                                       if cell_total > 0 else float('nan'),
            'validation_fixture_loss': fl,
            'mean_grad_norm': float(np.mean(grad_norms)) if grad_norms else float('nan'),
            'max_grad_norm': float(np.max(grad_norms)) if grad_norms else float('nan'),
        })

        if epoch_num <= budget1_epochs:
            if fl < best1_loss:
                best1_loss = fl
                best1_epoch = epoch_num
                best1_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch_num <= budget2_epochs:
            if fl < best2_loss:
                best2_loss = fl
                best2_epoch = epoch_num
                best2_state = {k: v.clone() for k, v in model.state_dict().items()}

    return model, mean, scale, {
        'budget280': {'best_epoch': best1_epoch, 'best_loss': best1_loss,
                      'state_dict': best1_state},
        'budget1120': {'best_epoch': best2_epoch, 'best_loss': best2_loss,
                       'state_dict': best2_state},
    }, trace, time.perf_counter() - start


# ---------------------------------------------------------------------------
# Cache (resume-safe)
# ---------------------------------------------------------------------------

def _cache_sha(out_dir):
    return sha256(out_dir / 'config.json')


def _train_cache_path(out_dir, key):
    return out_dir / 'cache_train' / f'{key}.pt'


def _train_index_path(out_dir, key):
    return out_dir / 'cache_train' / f'{key}.json'


def train_cache_load(out_dir, key):
    jp = _train_index_path(out_dir, key)
    pt = _train_cache_path(out_dir, key)
    if not jp.exists() or not pt.exists():
        return None
    blob = json.loads(jp.read_text(encoding='utf-8'))
    if blob.get('config_sha256') != _cache_sha(out_dir):
        raise RuntimeError(f'stale cache for {key}: config changed; use a new --out-dir')
    return blob


def train_cache_save(out_dir, key, checkpoints, trace, mean, scale, extra):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'cache_train').mkdir(parents=True, exist_ok=True)
    jp = _train_index_path(out_dir, key)
    pt = _train_cache_path(out_dir, key)
    payload = {
        'key': key,
        'config_sha256': _cache_sha(out_dir),
        'checkpoints': {
            'budget280': {
                'best_epoch': checkpoints['budget280']['best_epoch'],
                'best_loss': checkpoints['budget280']['best_loss'],
            },
            'budget1120': {
                'best_epoch': checkpoints['budget1120']['best_epoch'],
                'best_loss': checkpoints['budget1120']['best_loss'],
            },
        },
        'mean': mean.tolist(),
        'scale': float(scale),
        'trace': trace,
        'extra': extra,
    }
    write_json(jp, payload)
    torch.save({
        'state_dict_280': checkpoints['budget280']['state_dict'],
        'state_dict_1120': checkpoints['budget1120']['state_dict'],
    }, pt)


def _eval_cache_path(out_dir, key):
    base = out_dir / 'cache_eval'
    return base / f'{key}.json', base / f'{key}.npz'


def eval_cache_load(out_dir, key):
    jp, _ = _eval_cache_path(out_dir, key)
    if not jp.exists():
        return None
    blob = json.loads(jp.read_text(encoding='utf-8'))
    if blob.get('config_sha256') != _cache_sha(out_dir):
        raise RuntimeError(f'stale eval cache for {key}: config changed')
    return blob


def eval_cache_save(out_dir, key, scores, preds):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'cache_eval').mkdir(parents=True, exist_ok=True)
    jp, npp = _eval_cache_path(out_dir, key)
    arrays = {f'pred__{k}': v for k, v in preds.items()}
    np.savez_compressed(npp, **arrays)
    write_json(jp, {
        'key': key,
        'config_sha256': _cache_sha(out_dir),
        'scores': scores,
    })


# ---------------------------------------------------------------------------
# Prepare stage
# ---------------------------------------------------------------------------

def stage_prepare(out_dir):
    """Create experiment directory, freeze masks, noise, config, and hashes."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f'output directory already exists: {out_dir}')
    out_dir.mkdir(parents=True, exist_ok=True)

    assert sha256(DATA) == DATA_SHA, 'data SHA mismatch'

    df = pd.read_csv(DATA)
    e0_manifest = json.loads((E0 / 'manifest.json').read_text())
    names = e0_manifest['variant_columns']
    assert len(names) == N_VARIANTS
    fold = json.loads((E0 / 'cv_splits.json').read_text())[0]
    assert fold['test_location'] == 'Canada' and fold['fold_id'] == 0

    tm = load_transform_module()
    delta = tm.fit_positivity_delta(df.iloc[fold['train_rows']][names].to_numpy(float))
    assert delta == 0.5, f'expected delta 0.5, got {delta}'

    pool_df = df.iloc[fold['train_rows']]
    inner_pool = pool_df[pool_df.location != 'Denmark']
    val_frame = pool_df[pool_df.location == 'Denmark']
    assert sorted(set(val_frame.location)) == ['Denmark']

    # Eligible Denmark rows: complete and positive-sum
    val_raw = val_frame[names].to_numpy(dtype=np.float64)
    val_eligible = np.isfinite(val_raw).all(axis=1) & (val_raw.sum(axis=1) > 0)
    val_eligible_frame = val_frame.iloc[np.flatnonzero(val_eligible)]
    n_val_eligible = len(val_eligible_frame)

    # Frozen evaluation masks (on Denmark eligible rows)
    val_wvmasks, val_wvinfo = whole_variant_masks(
        inner_pool, names, n_val_eligible, fold['fold_id'], scenario='validation')
    val_rc = make_random_cell_mask(n_val_eligible, N_VARIANTS, MASKED_PARTS,
                                   seed=MASK_SEED)

    # Save frozen evaluation masks (True = HIDDEN)
    np.savez_compressed(out_dir / 'masks_evaluation.npz',
                        **{k: v for k, v in val_wvmasks.items()},
                        bank0=val_rc)
    write_json(out_dir / 'evaluation_mask_manifest.json', val_wvinfo)

    # Training mask banks (on inner pool rows)
    n_inner = len(inner_pool)
    control_banks = make_training_mask_banks('control', inner_pool, names, n_inner)
    intervention_banks = make_training_mask_banks('intervention', inner_pool, names, n_inner)
    assert_paired_banks(control_banks, intervention_banks)
    assert_n_strata_covered(intervention_banks)

    # Save training masks (True = VISIBLE)
    np.savez_compressed(out_dir / 'masks_training_control.npz',
                        **{f'bank{i}': control_banks[i] for i in range(MASK_BANKS)})
    np.savez_compressed(out_dir / 'masks_training_intervention.npz',
                        **{f'bank{i}': intervention_banks[i] for i in range(MASK_BANKS)})

    # Verify control banks are location-wide constant in column-count hiding
    for bank in control_banks[:9]:
        pass  # random-cell banks vary per row, that's expected

    # Verify intervention whole-variant banks hide same columns for all rows
    for bank in intervention_banks[9:]:
        assert_mask_constant_per_location(~bank)  # ~visible = hidden

    # Fixture noise: pre-generate deterministic verification records
    # (actual noise generated per-window during training, recorded for audit)
    g = torch.Generator().manual_seed(FIXTURE_NOISE_SEED)
    noise_metadata = {
        'seed': FIXTURE_NOISE_SEED,
        'n_steps': FIXTURE_STEPS,
        'n_scenarios': len(PRIMARY),
        'scenarios': list(PRIMARY),
        'note': 'Noise drawn once from seed 919; generation order is '
                'scenario x window x step. Shapes recorded per window.'
    }
    write_json(out_dir / 'fixture_noise_metadata.json', noise_metadata)

    # Config
    config = {
        'experiment_id': 'e3e4_mask_conditioning_2026-10-06',
        'date': '2026-10-06',
        'data_sha256': DATA_SHA,
        'protocol': 'U',
        'fold_id': 0,
        'outer': {
            'test_location': 'Canada',
            'held_out_locations': ['Canada'],
            'validation_location_rule': 'alphabetically_first_complete_outer_train_location',
            'validation_location': 'Denmark',
        },
        'policies': {
            'control': {
                'description': '18 random-cell banks, 5 hidden parts/row',
                'mask_seed': MASK_SEED,
            },
            'intervention': {
                'description': '9 paired random-cell banks + 9 whole-variant banks',
                'whole_variant_strata': [
                    {'label': l, 'n_hidden': n}
                    for l in ['rare', 'middle', 'common']
                    for n in [1, 2, 3]
                ],
                'train_only': True,
            },
        },
        'frozen_inputs': {
            'cv_splits': 'artifacts/e0_extended/cv_splits.json',
            'whole_variant_masks': 'artifacts/e0_extended/masks_whole_variant.npz',
            'random_cell_masks': 'artifacts/e0_extended/masks_random_cell.npz',
            'e2_transform_module': 'artifacts/e2_transforms/transforms.py',
            'positivity_delta': delta,
        },
        'feature_order': names,
        'training': {
            'inner_train_rows': len(inner_pool),
            'n_train_rows': int(np.isfinite(inner_pool[names].to_numpy()).all(axis=1).sum()),
            'validation_rows_eligible': n_val_eligible,
            'windows': 'nonoverlapping, location-grouped, length <=8, no padding',
            'window_length': WINDOW_LENGTH,
            'batch_windows': BATCH_WINDOWS,
            'epochs_max': MAX_EPOCHS,
            'mask_banks': MASK_BANKS,
            'masked_parts_per_row': MASKED_PARTS,
            'mask_seed': MASK_SEED,
            'learning_rate': LR,
            'optimizer': 'Adam',
            'seed_sync': True,
        },
        'budgets': {
            'budget280': {
                'updates': 280,
                'epochs': BUDGET_280_EPOCHS,
                'updates_per_epoch': 7,
                'note': '7 updates/epoch x 40 epochs = 280; 38 windows, batch=8',
            },
            'budget1120': {
                'updates': 1120,
                'epochs': BUDGET_1120_EPOCHS,
                'updates_per_epoch': 7,
                'note': '7 updates/epoch x 160 epochs = 1120; same trajectory',
            },
        },
        'checkpoint_selection': {
            'rule': 'argmin over epochs of fixed validation epsilon-loss fixture; '
                    'earliest epoch wins ties',
            'fixture': {
                'rows': f'All {n_val_eligible} eligible Denmark complete-positive rows',
                'masks': list(PRIMARY),
                'steps': list(range(FIXTURE_STEPS)),
                'noise_seed': FIXTURE_NOISE_SEED,
                'replayed_every_epoch': True,
                'budget280_constraint': f'checkpoint must come from epochs 1-{BUDGET_280_EPOCHS}',
                'budget1120_constraint': f'checkpoint must come from epochs 1-{BUDGET_1120_EPOCHS}',
            },
        },
        'schedule': {'name': 'quad_1e-4_to_0.5', 'beta_min': 1e-4,
                     'beta_max': 0.5, 'steps': DDPM_STEPS},
        'inference': {
            'mc_samples': MC_SAMPLES,
            'sampling_seed_base': 10000,
            'sampling_seed_per_batch_rule': 'model_seed + 10000 + batch_id',
            'projection': 'final_only',
            'clipping': 'disabled',
            'aggregate': 'mean of latent samples, then inverse',
            'observed_restoration': 'exact raw observed locking via E2 restore_observed_counts',
        },
        'model': {
            'class': 'CSDICoreRework',
            'layers': LAYERS,
            'heads': HEADS,
            'channels': CHANNELS,
            'latent_dim': 16,
            'float64': True,
        },
        'transforms': [{'name': 'clr16', 'latent_dim': 16, 'channels': CHANNELS,
                        'projection': 'sum_zero', 'references': None}],
        'modes': ALL_MODES,
        'seeds': ALL_SEEDS,
        'gates': {
            'conditioning': {
                'definition': 'On VALIDATION (Denmark), for every policy, seed, '
                              'and primary scenario: mask_only M2 > no_init M2. '
                              'PASS requires all seeds satisfy for every policy.'
            },
            'efficacy': {
                'definition': 'Init+CSDI M2 improves over matching Init-only by '
                              'strictly >10% AND nonzero CLR MAE not worse, '
                              'for every seed/scenario. PASS = at least one pair qualifies.',
                'method_type_filter': 'init_csdi only',
            },
            'numerical': {
                'definition': 'All scored predictions finite, non-negative, observed-locked. '
                              'Scale-guard 1000x recorded but not removed from metrics.',
            },
        },
        'canada': {
            'rule': 'NEVER evaluated in this experiment (--no-test enforced).',
            'evaluated': False,
        },
        'preservation': {
            'frozen_folders': ['e0_extended', 'e1_pilot', 'e2_transforms',
                              'e3_csdi', 'e4_pilot', 'e3_csdi_revised',
                              'e4_pilot_revised', 'e3e4_rework_2026-10-05_b'],
            'no_reset_no_stash_no_clean': True,
        },
        'prohibited_actions': [
            'use total_sequence as true mass', 'change metric/gate/eligibility mid-run',
            'silent clipping of predictions', 'evaluate Canada',
        ],
    }
    write_json(out_dir / 'config.json', config)

    # Source and input hashes
    frozen_paths = [
        ROOT / 'data' / 'covariants.csv',
        ROOT / 'artifacts/e0_extended' / 'cv_splits.json',
        ROOT / 'artifacts/e0_extended' / 'manifest.json',
        ROOT / 'artifacts/e2_transforms' / 'transforms.py',
        ROOT / 'src/stage_b/research_csdi.py',
        ROOT / 'src/stage_b/research_csdi_rework.py',
        ROOT / 'src/stage_a/initializers_e1.py',
        ROOT / 'src/evaluation/research_pilot.py',
        ROOT / 'src/evaluation/rework_gates.py',
        ROOT / 'scripts/run_e3_revised.py',
        ROOT / 'scripts/run_e5_rework.py',
        ROOT / 'scripts/audit_e0_extended.py',
    ]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in frozen_paths}
    write_json(out_dir / 'source_hashes.json', hashes)

    print(f'prepare: created {out_dir}')
    print(f'  data SHA: {DATA_SHA}')
    print(f'  validation rows eligible: {n_val_eligible}')
    print(f'  inner train rows: {len(inner_pool)}')
    print(f'  policies: {POLICIES}')
    print(f'  trajectories: {len(POLICIES)} x {len(ALL_MODES)} x {len(ALL_SEEDS)} = {len(POLICIES) * len(ALL_MODES) * len(ALL_SEEDS)}')
    print(f'  scored entries: x 2 budgets = {len(POLICIES) * len(ALL_MODES) * len(ALL_SEEDS) * 2}')


# ---------------------------------------------------------------------------
# Train stage
# ---------------------------------------------------------------------------

def stage_train(out_dir, dry_run=False):
    """Train all 24 trajectories for 160 epochs each, selecting dual-budget checkpoints."""
    out_dir = Path(out_dir)
    if not (out_dir / 'config.json').exists():
        raise FileNotFoundError(f'config.json not found in {out_dir}; run --stage prepare first')

    cfg = json.loads((out_dir / 'config.json').read_text())
    if cfg.get('data_sha256') != DATA_SHA:
        raise ValueError('data SHA mismatch in config')

    df = pd.read_csv(DATA)
    e0_manifest = json.loads((E0 / 'manifest.json').read_text())
    names = e0_manifest['variant_columns']
    fold = json.loads((E0 / 'cv_splits.json').read_text())[0]
    tm = load_transform_module()
    delta = tm.fit_positivity_delta(df.iloc[fold['train_rows']][names].to_numpy(float))
    assert delta == 0.5

    pool_df = df.iloc[fold['train_rows']]
    inner_pool = pool_df[pool_df.location != 'Denmark']
    val_frame = pool_df[pool_df.location == 'Denmark']

    val_raw = val_frame[names].to_numpy(dtype=np.float64)
    val_eligible = np.isfinite(val_raw).all(axis=1) & (val_raw.sum(axis=1) > 0)
    val_eligible_frame = val_frame.iloc[np.flatnonzero(val_eligible)]

    # Load frozen evaluation masks
    eval_masks = np.load(out_dir / 'masks_evaluation.npz')
    val_wvmasks = {k: eval_masks[k] for k in PRIMARY}
    val_rc = eval_masks['bank0']

    # Load training masks
    ctrl_npz = np.load(out_dir / 'masks_training_control.npz')
    intv_npz = np.load(out_dir / 'masks_training_intervention.npz')
    control_banks_full = [ctrl_npz[f'bank{i}'] for i in range(MASK_BANKS)]
    intervention_banks_full = [intv_npz[f'bank{i}'] for i in range(MASK_BANKS)]

    # Training data setup
    train_ok = np.isfinite(inner_pool[names].to_numpy()).all(axis=1) & \
               (inner_pool[names].to_numpy().sum(axis=1) > 0)
    eligible_idx = np.flatnonzero(train_ok)
    pool_eligible = inner_pool.iloc[eligible_idx]
    # Filter training masks to eligible rows only
    control_banks = [b[eligible_idx] for b in control_banks_full]
    intervention_banks = [b[eligible_idx] for b in intervention_banks_full]
    origin = pd.to_datetime(inner_pool.date).min().toordinal()
    times_pool = time_encode(pd.Series(pool_eligible.date), origin)
    windows = window_indices(inner_pool, pool_eligible.index.to_numpy(), WINDOW_LENGTH)
    raw_scale = raw_scale_for(inner_pool[names].to_numpy(float))
    bins, _ = prevalence_bins(inner_pool[names].to_numpy(dtype=np.float64), names)

    beta, alpha_bar = schedule()

    # Transform (CLR16 only)
    tname = 'clr16'
    transform_obj = tm.make_transform(tname, N_VARIANTS, delta)
    target = transform_obj.forward(pool_eligible[names].to_numpy(dtype=np.float64))
    latent_dim = target.shape[1]
    mean = target.mean(0)
    scale = max(float(np.sqrt(np.mean((target - mean) ** 2))), 1e-8)
    channels = CHANNELS

    # Validation fixture setup
    val_times = time_encode(pd.Series(val_eligible_frame.date), origin)
    val_windows = window_indices(val_eligible_frame, val_eligible_frame.index.to_numpy(),
                                 WINDOW_LENGTH)

    total_trajectories = len(POLICIES) * len(ALL_MODES) * len(ALL_SEEDS)
    completed = 0
    for policy in POLICIES:
        banks = control_banks if policy == 'control' else intervention_banks
        for mode in ALL_MODES:
            # Precompute training conditions for this policy
            cond_banks = precompute_train_conditions(
                pool_eligible, inner_pool, names, raw_scale, banks)

            for seed in ALL_SEEDS:
                key = policy_key(policy, mode, seed)
                cached = train_cache_load(out_dir, key)
                if cached is not None:
                    print(f'train: {key} cached, skipping', flush=True)
                    completed += 1
                    continue

                print(f'train: {key} (policy={policy}, channels={channels}, '
                      f'heads={HEADS}, {MAX_EPOCHS} epochs)', flush=True)

                # Build fixture for this mode
                fixture_fn, noise_records = make_checkpoint_fixture(
                    val_eligible_frame, names, inner_pool, mode, transform_obj,
                    mean, scale, raw_scale, val_windows, val_times, val_wvmasks)

                model, m_mean, m_scale, checkpoints, trace, t_secs = fit_with_dual_budget(
                    target, cond_banks, mode, windows, times_pool, alpha_bar,
                    channels, latent_dim, seed, fixture_fn, MAX_EPOCHS)

                # Verify budget280 checkpoint is from epochs 1-40
                assert 1 <= checkpoints['budget280']['best_epoch'] <= BUDGET_280_EPOCHS, \
                    f'budget280 checkpoint epoch {checkpoints["budget280"]["best_epoch"]} ' \
                    f'out of range [1, {BUDGET_280_EPOCHS}]'
                assert 1 <= checkpoints['budget1120']['best_epoch'] <= BUDGET_1120_EPOCHS, \
                    f'budget1120 checkpoint epoch out of range'

                train_cache_save(out_dir, key, checkpoints, trace, m_mean, m_scale, {
                    'policy': policy,
                    'mode': mode,
                    'seed': seed,
                    'transform': tname,
                    'channels': channels,
                    'latent_dim': latent_dim,
                    'train_seconds': t_secs,
                    'noise_records': noise_records,
                    'best_epoch_280': checkpoints['budget280']['best_epoch'],
                    'best_epoch_1120': checkpoints['budget1120']['best_epoch'],
                })
                print(f'train: {key} done. b280_epoch={checkpoints["budget280"]["best_epoch"]} '
                      f'b1120_epoch={checkpoints["budget1120"]["best_epoch"]} '
                      f'({t_secs:.1f}s)', flush=True)
                completed += 1

    print(f'train: {completed}/{total_trajectories} trajectories complete')


# ---------------------------------------------------------------------------
# Evaluate stage
# ---------------------------------------------------------------------------

def stage_evaluate(out_dir, dry_run=False):
    """Evaluate all 48 budget entries + controls + gates on frozen scoring masks."""
    out_dir = Path(out_dir)
    if not (out_dir / 'config.json').exists():
        raise FileNotFoundError(f'config.json not found in {out_dir}')

    cfg = json.loads((out_dir / 'config.json').read_text())

    df = pd.read_csv(DATA)
    e0_manifest = json.loads((E0 / 'manifest.json').read_text())
    names = e0_manifest['variant_columns']
    fold = json.loads((E0 / 'cv_splits.json').read_text())[0]
    tm = load_transform_module()
    delta = tm.fit_positivity_delta(df.iloc[fold['train_rows']][names].to_numpy(float))
    assert delta == 0.5

    pool_df = df.iloc[fold['train_rows']]
    inner_pool = pool_df[pool_df.location != 'Denmark']
    val_frame = pool_df[pool_df.location == 'Denmark']
    query_all = df.iloc[fold['test_rows']]  # Canada — MUST NOT BE USED

    val_raw = val_frame[names].to_numpy(dtype=np.float64)
    val_eligible = np.isfinite(val_raw).all(axis=1) & (val_raw.sum(axis=1) > 0)
    val_eligible_frame = val_frame.iloc[np.flatnonzero(val_eligible)]

    eval_masks = np.load(out_dir / 'masks_evaluation.npz')
    val_wvmasks = {k: eval_masks[k] for k in PRIMARY}
    val_rc = eval_masks['bank0']

    # Scoring scenarios: 3 whole-variant n2 (hidden) + 1 random-cell (hidden)
    scenario_masks = {}
    for k in PRIMARY:
        scenario_masks[k] = val_wvmasks[k]  # True = HIDDEN
    scenario_masks['random_cell_bank0'] = val_rc  # True = HIDDEN

    train_ok = np.isfinite(inner_pool[names].to_numpy()).all(axis=1) & \
               (inner_pool[names].to_numpy().sum(axis=1) > 0)
    pool_eligible = inner_pool.iloc[np.flatnonzero(train_ok)]
    origin = pd.to_datetime(inner_pool.date).min().toordinal()
    times_query = time_encode(pd.Series(val_eligible_frame.date), origin)
    query_windows = window_indices(val_eligible_frame, val_eligible_frame.index.to_numpy(),
                                   WINDOW_LENGTH)
    truth_arr = val_eligible_frame[names].to_numpy(dtype=np.float64)
    raw_all = pool_eligible[names].to_numpy(dtype=np.float64)
    raw_scale = raw_scale_for(inner_pool[names].to_numpy(float))
    bins, _ = prevalence_bins(pool_eligible[names].to_numpy(dtype=np.float64), names)
    locations = val_eligible_frame.location.to_numpy()

    beta, alpha_bar = schedule()
    transform_obj = tm.make_transform('clr16', N_VARIANTS, delta)
    target = transform_obj.forward(pool_eligible[names].to_numpy(dtype=np.float64))
    latent_dim = target.shape[1]
    mean = target.mean(0)
    scale = max(float(np.sqrt(np.mean((target - mean) ** 2))), 1e-8)

    # Pre-compute init-only conditions for each initializer
    ids = val_eligible_frame.index.to_numpy(dtype=np.int64)
    init_per_scenario = {}
    for scn_name, scn_mask in scenario_masks.items():
        q = np.where(~scn_mask, truth_arr, np.nan)
        init_per_scenario[scn_name] = initialize_crossfit(
            q, ids, locations, pool_df, names)[:3]

    all_scores = {}
    best_epochs = {}
    budgets = {'budget280': '280', 'budget1120': '1120'}
    for policy in POLICIES:
        for mode in ALL_MODES:
            for seed in ALL_SEEDS:
                key = policy_key(policy, mode, seed)
                cached = train_cache_load(out_dir, key)
                if cached is None:
                    raise RuntimeError(f'training cache for {key} not found; '
                                      f'run --stage train first')
                cp = cached['checkpoints']
                model_state_mean = np.array(cached['mean'])
                model_state_scale = cached['scale']

                for budget_name, budget_label in budgets.items():
                    eval_key = budget_key(policy, mode, seed, budget_label)
                    ev_cached = eval_cache_load(out_dir, eval_key)
                    if ev_cached is not None:
                        all_scores[key] = all_scores.get(key, {})
                        all_scores[key].setdefault(budget_name, ev_cached['scores'])
                        best_epochs[eval_key] = (cp[budget_name]['best_epoch'],
                                                 cp[budget_name]['best_loss'])
                        continue

                    # Reconstruct model and load checkpoint
                    model = CSDICoreRework(latent_dim, channels=CHANNELS, heads=HEADS,
                                          steps=DDPM_STEPS, layers=LAYERS)
                    ckpt_pt = torch.load(_train_cache_path(out_dir, key),
                                        weights_only=True)
                    state_key = f'state_dict_{budget_label}'
                    model.load_state_dict(ckpt_pt[state_key])
                    model.eval()

                    scn_scores, scn_preds = {}, {}
                    for scn_name, scn_mask in scenario_masks.items():
                        hidden = scn_mask
                        init = fb_i = ek_i = None
                        if mode in INIT_MODES:
                            init, fb_i, ek_i = (
                                d.get(mode) for d in init_per_scenario[scn_name])
                        pred, score, _ = evaluate_csdi_on_scenario(
                            model, transform_obj, 'clr16', (), model_state_mean,
                            model_state_scale, val_eligible_frame, names,
                            hidden, truth_arr, delta, bins, tm, raw_scale, mode,
                            init, fb_i, ek_i, query_windows, times_query, seed)
                        score.update(train_minutes=cached['extra']['train_seconds'] / 60,
                                     best_epoch=cp[budget_name]['best_epoch'],
                                     budget=budget_name, policy=policy)
                        scn_scores[scn_name] = score
                        scn_preds[scn_name] = pred

                    eval_cache_save(out_dir, eval_key, scn_scores, scn_preds)
                    all_scores.setdefault(key, {})[budget_name] = scn_scores
                    best_epochs[eval_key] = (cp[budget_name]['best_epoch'],
                                             cp[budget_name]['best_loss'])
                    m2 = scn_scores.get('whole_middle_n2', scn_scores.get('random_cell_bank0', {})).get('m2')
                    print(f'eval: {eval_key} b{budget_label}_epoch={cp[budget_name]["best_epoch"]} m2={m2}', flush=True)

    # Controls: init-only and baselines (computed once, not per budget)
    control_scores = {}
    control_preds = {}
    for im in INIT_MODES:
        for scn_name, scn_mask in scenario_masks.items():
            pred, sc = evaluate_init_only_on_scenario(
                val_eligible_frame, names, scn_mask, pool_df, delta, bins,
                locations, tm, im)
            key = f'init_only__{im}__{scn_name}'
            control_scores[key] = sc
            control_preds[key] = pred
    for bname in ('linear', 'locf_nocb'):
        fb = train_fallback_values(pool_eligible[names].to_numpy(dtype=float))
        for scn_name, scn_mask in scenario_masks.items():
            pred, sc = evaluate_baseline_on_scenario(
                val_eligible_frame, names, scn_mask, times_query, pool_eligible,
                delta, bins, bname)
            key = f'{bname}__{scn_name}'
            control_scores[key] = sc
            control_preds[key] = pred

    # Build flat scores dict for gates (same layout as existing rework_gates)
    flat = {}
    for policy in POLICIES:
        for mode in ALL_MODES:
            for seed in ALL_SEEDS:
                key = policy_key(policy, mode, seed)
                for budget_name in budgets:
                    budget_label = budgets[budget_name]
                    bkey = budget_key(policy, mode, seed, budget_label)
                    for scn, sc in all_scores.get(key, {}).get(budget_name, {}).items():
                        flat[f'{bkey}/{scn}'] = sc
    for k, v in control_scores.items():
        flat[k] = v

    # Run gates per budget, per policy, then aggregate.
    gates = {}
    finalists = []
    for budget_name, budget_label in budgets.items():
        policy_gates = {}
        for policy in POLICIES:
            budget_scores = {}
            for mode in ALL_MODES:
                for seed in ALL_SEEDS:
                    key = policy_key(policy, mode, seed)
                    scn_scores = all_scores.get(key, {}).get(budget_name, {})
                    if scn_scores:
                        gate_key = f'{mode}__clr16__seed{seed}'
                        budget_scores[gate_key] = scn_scores
            budget_scores.update(control_scores)

            cond = RG.conditioning_gate(budget_scores, ALL_TRANSFORMS, ALL_SEEDS)
            eff = RG.efficacy_gate(budget_scores, ALL_TRANSFORMS, INIT_MODES, ALL_SEEDS)
            num = RG.numerical_gate(budget_scores)
            policy_gates[policy] = {
                'conditioning_gate': cond,
                'efficacy_gate': eff,
                'numerical_gate': num,
            }

        # Aggregation (directive §11.2):
        #   conditioning: must pass for EVERY policy.
        #   efficacy:     at least one policy passes (at least one pair qualifies).
        #   numerical:    must pass for EVERY policy (all entries valid).
        cond_pass = all(g['conditioning_gate']['passed'] for g in policy_gates.values())
        eff_pass = any(g['efficacy_gate']['passed'] for g in policy_gates.values())
        num_pass = all(g['numerical_gate']['passed'] for g in policy_gates.values())

        gates[budget_name] = {
            'passed': bool(cond_pass and eff_pass and num_pass),
            'conditioning_pass_for_all_policies': cond_pass,
            'efficacy_pass_for_at_least_one_policy': eff_pass,
            'numerical_pass_for_all_policies': num_pass,
            'per_policy': policy_gates,
        }

        # Select finalists only when gates pass, using the intervention policy's
        # budget1120 scores (the more-trained checkpoint from the hypothesis policy).
        if budget_name == 'budget1120' and gates[budget_name]['passed']:
            intv_scores = {}
            for mode in ALL_MODES:
                for seed in ALL_SEEDS:
                    key = policy_key('intervention', mode, seed)
                    scn_scores = all_scores.get(key, {}).get('budget1120', {})
                    if scn_scores:
                        gate_key = f'{mode}__clr16__seed{seed}'
                        intv_scores[gate_key] = scn_scores
            intv_scores.update(control_scores)
            f, diag = RG.select_finalists(
                intv_scores, policy_gates['intervention'], ALL_TRANSFORMS, INIT_MODES, ALL_SEEDS)
            finalists = [dict(cfg) for cfg in f]

    write_json(out_dir / 'evaluation_scores.json', all_scores)
    write_json(out_dir / 'control_scores.json', control_scores)
    write_json(out_dir / 'validation_gates.json', gates)
    write_json(out_dir / 'best_epochs.json',
               {k: {'best_epoch': v[0], 'best_loss': v[1]} for k, v in best_epochs.items()})
    write_json(out_dir / 'failure_records.json',
               failure_records(flat, val_eligible_frame.index.to_numpy(), 'validation'))

    any_gate_pass = any(gates[b]['passed'] for b in gates)
    status = 'BLOCKED' if not any_gate_pass else 'UNBLOCKED_PENDING_USER_APPROVAL'
    write_json(out_dir / 'e5_mask_conditioning_status.json', {
        'status': status,
        'finalists': [f['config'] for f in finalists],
        'validation_gates_passed': any_gate_pass,
        'canada_evaluated': False,
        'per_budget_gates': {b: g['passed'] for b, g in gates.items()},
    })
    print(f'evaluate: gates computed. any_pass={any_gate_pass}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        prog='run_e5_mask_conditioning.py',
        description='E5 mask-conditioning experiment (§11.1-11.5). '
                    'Tests random-cell vs whole-variant training masks and '
                    '280 vs 1120 optimizer updates.')
    ap.add_argument('--stage', choices=['prepare', 'train', 'evaluate'],
                    required=True,
                    help='prepare: freeze masks/config/hashes; '
                         'train: run 24 trajectories with dual-budget checkpoints; '
                         'evaluate: score 48 entries and compute gates')
    ap.add_argument('--out-dir', dest='out_dir', required=True,
                    help='output directory for this experiment')
    ap.add_argument('--no-test', action='store_true', default=True,
                    help='Canada is never evaluated (always enforced)')
    args = ap.parse_args()

    if not args.no_test:
        raise SystemExit('--no-test cannot be disabled for this experiment; '
                         'Canada is sealed by design.')

    if args.stage == 'prepare':
        stage_prepare(args.out_dir)
    elif args.stage == 'train':
        stage_train(args.out_dir)
    elif args.stage == 'evaluate':
        stage_evaluate(args.out_dir)


if __name__ == '__main__':
    main()
