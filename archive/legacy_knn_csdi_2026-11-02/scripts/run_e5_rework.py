"""E5 rework experiment runner (Phase C -> D).

Implements the registered preregistration in artifacts/e3e4_rework_2026-10-05_b/.
Reuses frozen diffusion numerics from src/stage_b/research_csdi.py and E2
transforms. The model core is CSDICoreRework with single-head attention
policy (channels 16/17/8 varies width only). Config.json enforced before training.

Mask convention: True = HIDDEN (passed to evaluate). condition_features
receives visible = ~hidden.

Run:  .venv-e3-revised/Scripts/python -B -m scripts.run_e5_rework [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rework_runtime  # noqa: E402,F401  (must precede numpy/torch: adds research site-packages)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.dont_write_bytecode = True
torch.set_num_threads(2)

from src.stage_b.research_csdi import (  # noqa: E402
    condition_features, forward_noise, epsilon_loss, sample_latents, schedule)
from src.stage_b.research_csdi_rework import CSDICoreRework  # noqa: E402
from src.stage_a.initializers_e1 import train_fallback_values  # noqa: E402
from src.evaluation.research_pilot import (  # noqa: E402
    evaluate, prevalence_bins, numerical_diagnostics, sha256, write_json,
    write_hashes, failure_records)
from scripts.run_e3_revised import (  # noqa: E402
    load_transform_module, window_indices, initialize_crossfit, masks)
from scripts.run_e4_revised import (  # noqa: E402
    temporal_baseline)
from scripts.audit_e0_extended import choose_hidden_variants  # noqa: E402
from src.evaluation import rework_gates as RG  # noqa: E402

B = ROOT / 'artifacts/e3e4_rework_2026-10-05_b'
DATA = ROOT / 'data/covariants.csv'
E0 = ROOT / 'artifacts/e0_extended'
E1 = ROOT / 'artifacts/e1_pilot'
E2 = ROOT / 'artifacts/e2_transforms'
DATA_SHA = 'bbb1aacdf46ebef345ca818f086d5586f4aded45475b00b6ee9bd89fc94a90ab'

ALL_TRANSFORMS = ['clr16', 'clr17', 'ilr16', 'hkglr16_actual',
                  'hkglr16_random_seed42', 'hkglr16_random_seed43',
                  'hkglr16_random_seed44', 'hkglr16_high']
ALL_MODES = ['no_init', 'mask_only', 'hron_2a', 'aitchison_complete']
INIT_MODES = ['hron_2a', 'aitchison_complete']
ALL_SEEDS = [42, 43, 44]
MODEL_CHANNELS = {
    'clr16': 16, 'clr17': 17, 'ilr16': 16,
    'hkglr16_actual': 16, 'hkglr16_random_seed42': 16,
    'hkglr16_random_seed43': 16, 'hkglr16_random_seed44': 16,
    'hkglr16_high': 16,
}
MAX_EPOCHS = 40
MC_SAMPLES = 8
BATCH_WINDOWS = 8
WINDOW_LENGTH = 8
MASK_BANKS = 4
MASKED_PARTS = 5
MASK_SEED = 42
LR = 1e-3
FIXTURE_NOISE_SEED = 919
GATE_LABELS = ('rare', 'middle', 'common')
GATE_N = (1, 2, 3)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def groups_of_windows(windows, rng=None):
    groups = {}
    for ids in windows:
        groups.setdefault(len(ids), []).append(ids)
    batches = []
    for length in sorted(groups):
        group = groups[length]
        order = np.arange(len(group)) if rng is None else rng.permutation(len(group))
        for i in range(0, len(order), BATCH_WINDOWS):
            batches.append(np.stack([group[k] for k in order[i:i + BATCH_WINDOWS]]))
    if rng is not None:
        rng.shuffle(batches)
    return batches


def hkglr_refs(name, pool_df, names):
    rates = pool_df[names].isna().mean()
    if 'random_seed' in name:
        seed = int(name.rsplit('_', 1)[-1].replace('seed', ''))
        alphabetical = sorted(names)
        rng = np.random.default_rng(seed)
        chosen = sorted(rng.choice(alphabetical, size=5, replace=False).tolist())
        return tuple(names.index(n) for n in chosen)
    if 'high' in name:
        ordered = sorted(names, key=lambda n: (-float(rates[n]), n))[:5]
        return tuple(names.index(n) for n in ordered)
    ordered = sorted(names, key=lambda n: (float(rates[n]), n))[:5]
    return tuple(names.index(n) for n in ordered)


def make_random_cell_mask(n_rows, n_cols=17, n_hide=5, seed=42):
    """True = HIDDEN."""
    rng = np.random.default_rng(seed)
    hidden = np.zeros((n_rows, n_cols), dtype=bool)
    for i in range(n_rows):
        hidden[i, rng.choice(n_cols, n_hide, replace=False)] = True
    return hidden


def whole_variant_masks(pool_df, names, n_rows, fold_id=0, scenario='validation'):
    """True = HIDDEN. Matches E0 write_masks / choose_hidden_variants."""
    if scenario == 'test':
        npz = np.load(E0 / 'masks_whole_variant.npz')
        result = {}
        for label in GATE_LABELS:
            for n in GATE_N:
                key = f'fold{fold_id}_whole_{label}_n{n}'
                result[f'whole_{label}_n{n}'] = npz[key]
        return result
    ordering = choose_hidden_variants(pool_df, names)
    result, info = {}, {}
    for label in GATE_LABELS:
        for n_hidden in GATE_N:
            hidden_vars = ordering[label][:n_hidden]
            hidden = np.zeros((n_rows, len(names)), dtype=bool)
            for v in hidden_vars:
                hidden[:, names.index(v)] = True
            key = f'whole_{label}_n{n_hidden}'
            result[key] = hidden
            info[key] = {'label': label, 'n_hidden': n_hidden,
                         'hidden_variants': hidden_vars,
                         'reference_order': ordering[label]}
    return result, info


def time_encode(series, origin):
    return (pd.to_datetime(series).map(pd.Timestamp.toordinal).to_numpy(dtype=np.float64) - origin) / 14.0


def raw_scale_for(raw_all):
    return float(np.log1p(np.nanmax(raw_all)))


# ---------------------------------------------------------------------------
# Training conditions
# ---------------------------------------------------------------------------

def precompute_train_conditions(pool_eligible, all_pool, names, raw_scale):
    """Condition features for every (mode, bank) on training rows.

    Uses masks() which returns True=VISIBLE (condition_features convention).
    """
    raw = pool_eligible[names].to_numpy(dtype=np.float64)
    ids = pool_eligible.index.to_numpy(dtype=np.int64)
    conditions = {}
    for bank in range(MASK_BANKS):
        visible = masks(len(pool_eligible), bank)  # True = VISIBLE
        query = np.where(visible, raw, np.nan)
        init, fb, ek, _ = initialize_crossfit(
            query, ids, pool_eligible.location.to_numpy(), all_pool, names)
        for mode in ALL_MODES:
            cf_mode = 'init' if mode in init else mode
            conditions[(mode, bank)] = condition_features(
                query, visible, cf_mode, raw_scale,
                init.get(mode), fb.get(mode), ek.get(mode))
    return conditions


# ---------------------------------------------------------------------------
# Checkpoint-selection fixture
# ---------------------------------------------------------------------------

def make_fixture(val_query, val_visible, val_times, val_window_indices,
                 init_dict, fb_dict, ek_dict, raw_scale, transform_obj, mean, scale,
                 mode):
    """Deterministic per-epoch validation epsilon-loss fixture.

    First Denmark window (<=8 rows), bank-0 random-cell mask, 20 steps.
    Noise from FIXTURE_NOISE_SEED drawn once, replayed every epoch.
    val_visible = True=VISIBLE (from masks()). init_dict/fb_dict/ek_dict are
    per-initializer arrays from initialize_crossfit (None for no_init/mask_only).
    """
    w = np.asarray(val_window_indices)
    q = np.where(val_visible, val_query, np.nan)
    cf_mode = 'init' if mode in init_dict else mode
    init_arr = init_dict.get(mode) if init_dict is not None and mode in init_dict else None
    fb_arr = fb_dict.get(mode) if fb_dict is not None else None
    ek_arr = ek_dict.get(mode) if ek_dict is not None else None
    cond = condition_features(q, val_visible, cf_mode, raw_scale, init_arr, fb_arr, ek_arr)
    # Target = the full-LR validation label (as in training); only the CONDITION is masked.
    assert np.isfinite(val_query[w]).all() and (val_query[w].sum(1) > 0).all()
    z0 = torch.from_numpy(((transform_obj.forward(val_query[w]) - mean) / scale))[None, ...]
    cond_w = torch.from_numpy(cond[w])[None, ...]
    t_w = torch.from_numpy(val_times[w])[None, ...]
    eligible = torch.ones_like(z0, dtype=torch.bool)
    g = torch.Generator().manual_seed(FIXTURE_NOISE_SEED)
    noise_list = [torch.randn(z0.shape, dtype=torch.float64, generator=g) for _ in range(20)]
    beta, ab = schedule()

    def fixture_fn(model):
        model.eval()
        losses = []
        with torch.no_grad():
            for step in range(20):
                noisy = forward_noise(z0, noise_list[step], ab[step])
                pred = model(noisy, cond_w, t_w, torch.tensor([step]))
                losses.append(float(epsilon_loss(pred, noise_list[step], eligible)))
        return float(np.mean(losses))

    return fixture_fn


# ---------------------------------------------------------------------------
# Training with checkpoint selection
# ---------------------------------------------------------------------------

def fit_with_checkpoint(target, cond_banks, mode, windows, times, alpha_bar,
                        channels, latent_dim, seed, fixture_fn, max_epochs):
    """Train CSDICoreRework (heads=1); argmin validation fixture loss over epochs."""
    torch.manual_seed(seed)
    model = CSDICoreRework(latent_dim, channels=channels, heads=1, steps=20, layers=2)
    mean = target.mean(0)
    scale = max(float(np.sqrt(np.mean((target - mean) ** 2))), 1e-8)
    normalized = (target - mean) / scale
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(seed)
    noise_rng = torch.Generator().manual_seed(seed)
    trace = []
    best_epoch, best_loss, best_state = 1, float('inf'), None
    start = time.perf_counter()
    for epoch in range(max_epochs):
        model.train()
        batch_losses = []
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
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        fl = fixture_fn(model)
        trace.append({'epoch': epoch + 1,
                      'train_epsilon_mse': float(np.mean(batch_losses)) if batch_losses else float('nan'),
                      'validation_fixture_loss': fl})
        if fl < best_loss:
            best_loss = fl
            best_epoch = epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean, scale, best_epoch, best_loss, trace, time.perf_counter() - start


# ---------------------------------------------------------------------------
# Scenario evaluation
# ---------------------------------------------------------------------------

def evaluate_csdi_on_scenario(model, transform_obj, name, refs, model_mean, model_scale,
                              query_frame, names, hidden, truth_arr, delta, bins,
                              tm, raw_scale, mode, init, fb, ek,
                              query_windows, times_query, seed):
    """Sample CSDI model on one scenario (hidden = True=HIDDEN)."""
    visible = ~hidden
    q = np.where(visible, truth_arr, np.nan)
    cf_mode = 'init' if mode in INIT_MODES else mode
    cond = condition_features(q, visible, cf_mode, raw_scale, init, fb, ek)
    beta, alpha_bar = schedule()
    n = len(query_frame)
    latents = np.empty((MC_SAMPLES, n, model.latent_dim), dtype=np.float64)
    for batch_id, idx in enumerate(groups_of_windows(query_windows)):
        result, _ = sample_latents(
            model, torch.from_numpy(cond[idx]), torch.from_numpy(times_query[idx]),
            beta, alpha_bar, torch.from_numpy(model_mean), model_scale, name, refs or (),
            samples=MC_SAMPLES, seed=seed + 10000 + batch_id)
        for b, ids in enumerate(idx):
            latents[:, ids, :] = result[:, b, :, :].numpy()
    comp = transform_obj.inverse(latents.mean(0))
    pred, rest = tm.restore_observed_counts(comp, q)
    score = evaluate(truth_arr, pred, hidden, q, delta, bins)
    score.update(
        method_type='init_csdi' if mode in INIT_MODES else f'{mode}_csdi',
        initializer=mode if mode in INIT_MODES else None,
        transform=name, model_seed=seed,
        parameter_count=sum(p.numel() for p in model.parameters()),
        latent_dim=model.latent_dim, channels=model.channels,
        mc_samples=MC_SAMPLES,
        numerical_diagnostics=numerical_diagnostics(pred, q),
        restoration={'n_no_positive_visible_anchor': rest['n_no_positive_visible_anchor'],
                     'scales': np.atleast_1d(rest['scales']).tolist()})
    if mode in INIT_MODES:
        h = hidden
        score['init_provenance'] = {
            'fallback_fraction_hidden': float(np.asarray(fb)[h].mean()) if h.any() else None,
            'effective_k_mean_hidden': float(np.asarray(ek)[h].mean()) if h.any() else None}
    return pred, score, latents


def evaluate_init_only_on_scenario(query_frame, names, hidden, pool_df, delta, bins,
                                   locations, tm, init_mode):
    """Compute init-only (no CSDI) result for one scenario mask."""
    truth_arr = query_frame[names].to_numpy(dtype=np.float64)
    visible = ~hidden
    q = np.where(visible, truth_arr, np.nan)
    ids = query_frame.index.to_numpy(dtype=np.int64)
    init, fb, ek, prov = initialize_crossfit(q, ids, locations, pool_df, names)
    # Same convention as legacy E3/E4: score the initializer's raw output directly
    # (observed cells already restored exactly, zeros stay zero for the metric's delta rule).
    pred = init[init_mode]
    assert np.array_equal(pred[visible], q[visible]), 'initializer must lock observed cells'
    rest = {'n_no_positive_visible_anchor': int((np.where(visible, q, 0.).sum(1) == 0).sum()),
            'scales': []}
    score = evaluate(truth_arr, pred, hidden, q, delta, bins)
    score.update(
        method_type='init_only', initializer=init_mode, transform=None,
        numerical_diagnostics=numerical_diagnostics(pred, q),
        restoration={'n_no_positive_visible_anchor': rest['n_no_positive_visible_anchor'],
                     'scales': np.atleast_1d(rest['scales']).tolist()})
    return pred, score


def evaluate_baseline_on_scenario(query_frame, names, hidden, times_query, pool_eligible,
                                  delta, bins, bname):
    """Linear/LOCF temporal baseline on one scenario."""
    truth_arr = query_frame[names].to_numpy(dtype=np.float64)
    visible = ~hidden
    q = np.where(visible, truth_arr, np.nan)
    fb = train_fallback_values(pool_eligible[names].to_numpy(dtype=float))
    pred, diag = temporal_baseline(q, times_query, fb, bname)
    assert np.array_equal(pred[visible], q[visible])  # observed locked; hidden never read from truth
    score = evaluate(truth_arr, pred, hidden, q, delta, bins)
    score.update(method_type='classical_baseline', source=bname,
                   baseline_diagnostics=diag,
                   numerical_diagnostics=numerical_diagnostics(pred, q),
                   baseline_family=bname)
    return pred, score


# ---------------------------------------------------------------------------
# Per-model cache (resume-safe; a killed run loses at most one model)
# ---------------------------------------------------------------------------

def _config_sha():
    return sha256(B / 'config.json')


def cache_paths(cache_dir, key):
    return cache_dir / f'{key}.json', cache_dir / f'{key}.npz', cache_dir / f'{key}.pt'


def cache_load(cache_dir, key):
    jp, _, _ = cache_paths(cache_dir, key)
    if not jp.exists():
        return None
    blob = json.loads(jp.read_text(encoding='utf-8'))
    if blob.get('config_sha256') != _config_sha() or blob.get('mc_samples') != MC_SAMPLES \
            or blob.get('max_epochs') != MAX_EPOCHS:
        raise RuntimeError(f'stale cache for {key}: config/MC/epochs changed; use a new folder')
    return blob


def cache_save(cache_dir, key, scores, preds, latents, state_dict, best_epoch, best_loss, extra):
    cache_dir.mkdir(parents=True, exist_ok=True)
    jp, npp, pt = cache_paths(cache_dir, key)
    arrays = {f'pred__{k}': v for k, v in preds.items()}
    arrays.update({f'latent__{k}': v for k, v in latents.items()})
    np.savez_compressed(npp, **arrays)
    torch.save(state_dict, pt)
    # JSON written last: its existence marks a complete cache entry.
    write_json(jp, {'key': key, 'config_sha256': _config_sha(), 'mc_samples': MC_SAMPLES,
                    'max_epochs': MAX_EPOCHS, 'best_epoch': best_epoch,
                    'best_loss': best_loss, 'scores': scores, **extra})


# ---------------------------------------------------------------------------
# Partition runner
# ---------------------------------------------------------------------------

def all_configs(transforms, modes):
    return [(m, t) for t in transforms for m in modes]


def run_csdi_partition(label, pool_df, query_frame, names, tm, delta, scenario_masks,
                       cache_dir, configs, seeds, val_fixture=None, epochs_fixed=None,
                       window_length=WINDOW_LENGTH, channels_override=None,
                       with_controls=True):
    """Train (or load cached) models for ``configs`` x ``seeds``; score every
    scenario in ``scenario_masks`` (True = HIDDEN).

    val_fixture given  -> checkpoint chosen by the validation fixture.
    epochs_fixed given -> train exactly that many epochs per model key (no selection).
    """
    ids = query_frame.index.to_numpy(dtype=np.int64)
    truth_arr = query_frame[names].to_numpy(dtype=np.float64)
    raw_all = pool_df[names].to_numpy(dtype=np.float64)
    train_ok = np.isfinite(raw_all).all(axis=1) & (np.nansum(raw_all, axis=1) > 0)
    pool_eligible = pool_df.iloc[np.flatnonzero(train_ok)]
    origin = pd.to_datetime(pool_df.date).min().toordinal()
    times_pool = time_encode(pd.Series(pool_eligible.date), origin)
    times_query = time_encode(pd.Series(query_frame.date), origin)
    windows = window_indices(pool_df, pool_eligible.index.to_numpy(), window_length)
    query_windows = window_indices(query_frame, ids, window_length)
    raw_scale = raw_scale_for(raw_all)
    bins, _ = prevalence_bins(raw_all, names)
    beta, alpha_bar = schedule()
    locations = query_frame.location.to_numpy()

    keys = [(m, t, s) for (m, t) in configs for s in seeds]
    pending = [k for k in keys if cache_load(cache_dir, f'{k[0]}__{k[1]}__seed{k[2]}') is None]
    cond_banks = (precompute_train_conditions(pool_eligible, pool_df, names, raw_scale)
                  if pending else None)

    init_per_scenario = {}
    if pending or with_controls:
        for scn_name, scn_mask in scenario_masks.items():
            q = np.where(~scn_mask, truth_arr, np.nan)
            init_per_scenario[scn_name] = initialize_crossfit(q, ids, locations, pool_df, names)[:3]

    all_scores, best_epochs = {}, {}
    cur_t, transform_obj, refs, target, latent_dim, mean, scale = None, None, None, None, None, None, None
    for mode, tname, seed in keys:
        key = f'{mode}__{tname}__seed{seed}'
        blob = cache_load(cache_dir, key)
        if blob is not None:
            all_scores[key] = blob['scores']
            best_epochs[key] = (blob['best_epoch'], blob['best_loss'])
            continue
        if cur_t != tname:
            cur_t = tname
            refs = hkglr_refs(tname, pool_df, names)
            transform_obj = tm.make_transform(
                tname, 17, delta, refs if tname.startswith('hk') else None)
            target = transform_obj.forward(pool_eligible[names].to_numpy(dtype=np.float64))
            latent_dim = target.shape[1]
            mean = target.mean(0)
            scale = max(float(np.sqrt(np.mean((target - mean) ** 2))), 1e-8)
        channels = channels_override or MODEL_CHANNELS[tname]
        print(f'{label}: fit {key} (ld={latent_dim}, ch={channels}, heads=1, '
              f'wl={window_length}, max_epochs={MAX_EPOCHS})', flush=True)
        if epochs_fixed is None:
            fixture_fn = make_fixture(
                val_fixture['query'], val_fixture['visible'], val_fixture['times'],
                val_fixture['window'][0], val_fixture['init'], val_fixture['fb'],
                val_fixture['ek'], val_fixture['raw_scale'], transform_obj, mean, scale, mode)
            model, m_mean, m_scale, best_epoch, best_loss, trace, t_secs = fit_with_checkpoint(
                target, cond_banks, mode, windows, times_pool, alpha_bar, channels,
                latent_dim, seed, fixture_fn, MAX_EPOCHS)
        else:
            n_ep = epochs_fixed[key]
            model, m_mean, m_scale, trace, t_secs = fit_transferred(
                target, cond_banks, mode, windows, times_pool, alpha_bar, channels,
                latent_dim, seed, n_ep)
            best_epoch, best_loss = n_ep, None

        scn_scores, scn_preds, scn_lat = {}, {}, {}
        for scn_name, scn_mask in scenario_masks.items():
            init = fb_i = ek_i = None
            if mode in INIT_MODES:
                init, fb_i, ek_i = (d.get(mode) for d in init_per_scenario[scn_name])
            pred, score, lat = evaluate_csdi_on_scenario(
                model, transform_obj, tname, refs, m_mean, m_scale, query_frame, names,
                scn_mask, truth_arr, delta, bins, tm, raw_scale, mode, init, fb_i, ek_i,
                query_windows, times_query, seed)
            score.update(train_seconds=t_secs, best_epoch=best_epoch,
                         validation_fixture_loss=best_loss, training_trace=trace)
            scn_scores[scn_name], scn_preds[scn_name], scn_lat[scn_name] = score, pred, lat
        cache_save(cache_dir, key, scn_scores, scn_preds, scn_lat, model.state_dict(),
                   best_epoch, best_loss, {'trace': trace, 'channels': channels,
                                           'window_length': window_length,
                                           'train_seconds': t_secs})
        all_scores[key] = scn_scores
        best_epochs[key] = (best_epoch, best_loss)
        m2 = scn_scores.get('whole_middle_n2', scn_scores.get('random_cell_bank0', {})).get('m2')
        print(f'{label}: {key} best_epoch={best_epoch} m2={m2}', flush=True)

    if not with_controls:
        return all_scores, best_epochs
    bl_preds = {}
    for scn_name, scn_mask in scenario_masks.items():
        for im in INIT_MODES:
            pred, sc = evaluate_init_only_on_scenario(
                query_frame, names, scn_mask, pool_df, delta, bins, locations, tm, im)
            all_scores[f'init_only__{im}__{scn_name}'] = sc
            bl_preds[f'init_only__{im}__{scn_name}'] = pred
        for bname in ('linear', 'locf_nocb'):
            pred, sc = evaluate_baseline_on_scenario(
                query_frame, names, scn_mask, times_query, pool_eligible, delta, bins, bname)
            all_scores[f'{bname}__{scn_name}'] = sc
            bl_preds[f'{bname}__{scn_name}'] = pred
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_dir / 'controls_predictions.npz', **bl_preds)
    return all_scores, best_epochs


def fit_transferred(target, cond_banks, mode, windows, times, alpha_bar,
                    channels, latent_dim, seed, epochs):
    """Train CSDICoreRework for a fixed epoch count (no checkpoint selection);
    identical RNG streams to fit_with_checkpoint so transferred epochs are exact."""
    torch.manual_seed(seed)
    model = CSDICoreRework(latent_dim, channels=channels, heads=1, steps=20, layers=2)
    mean = target.mean(0)
    scale = max(float(np.sqrt(np.mean((target - mean) ** 2))), 1e-8)
    normalized = (target - mean) / scale
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(seed)
    noise_rng = torch.Generator().manual_seed(seed)
    trace = []
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        batch_losses = []
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
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        trace.append({'epoch': epoch + 1,
                      'train_epsilon_mse': float(np.mean(batch_losses)) if batch_losses else float('nan')})
    return model, mean, scale, trace, time.perf_counter() - start


# ---------------------------------------------------------------------------
# Restoration sensitivity (registered, descriptive, non-gating) - section 10
# ---------------------------------------------------------------------------

def restoration_sensitivity(cache_dir, keys, scenario_masks, truth_arr, delta, bins,
                            train_row_sums):
    """Replace scale=1 on no-positive-anchor rows by the train-only median row total.
    Control (legacy) numbers are untouched; this only reports the alternative."""
    s_train = float(np.median(train_row_sums))
    out = {'train_only_scale': s_train, 'models': {}}
    for key in keys:
        npz = np.load(cache_paths(cache_dir, key)[1])
        for scn, hidden in scenario_masks.items():
            if f'pred__{scn}' not in npz:
                continue
            pred = npz[f'pred__{scn}']
            q = np.where(~hidden, truth_arr, np.nan)
            visible = np.isfinite(q)
            no_anchor = np.where(visible, q, 0.).sum(1) == 0
            alt = pred.copy()
            alt[no_anchor] = np.where(visible[no_anchor], q[no_anchor], pred[no_anchor] * s_train)
            ctl = evaluate(truth_arr, pred, hidden, q, delta, bins)
            alt_sc = evaluate(truth_arr, alt, hidden, q, delta, bins)
            out['models'][f'{key}/{scn}'] = {
                'n_no_anchor_rows': int(no_anchor.sum()),
                'control_m2': ctl['m2'], 'sensitivity_m2': alt_sc['m2'],
                'control_scale_guard_rows': numerical_diagnostics(pred, q)['n_scale_guard_rows'],
                'sensitivity_scale_guard_rows': numerical_diagnostics(alt, q)['n_scale_guard_rows'],
                'clipping_fraction': 0.0, 'role': 'descriptive_non_gating'}
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(v):
    return 'n/a' if v is None else f'{v:.4f}'


def write_report(gates, finalists, diagnostic, test_scores, canada_ran, cap8_summary, status):
    cond = gates['conditioning_gate']
    eff = gates['efficacy_gate']
    num = gates['numerical_gate']
    L = ['# E5 Rework Experiment Report', '',
         f'Status: **{status}** (read gate fields, not COMPLETE).', '',
         '## Protocol',
         '- 8 transforms x 4 modes x 3 seeds = 96 validation models (Denmark), heads=1, layers=2, '
         'lr 1e-3, max 40 epochs, checkpoint = argmin fixed validation epsilon-loss fixture, MC=8.',
         '- Primary scenario whole-variant (rare/middle/common, n=2) is gated; random-cell is '
         'secondary and NOT gated. Scenarios are never averaged in the gate tables.', '',
         '## Validation gates (preregistration section 11)',
         f'- Overall: **{gates["passed"]}**',
         f'- Conditioning (Mask-only M2 > No-init M2; all seeds, primary scenarios, transforms): '
         f'**{cond["passed"]}** ({sum(not v["passed"] for v in cond["pairs"].values())} of '
         f'{len(cond["pairs"])} transform x scenario cells failed)',
         f'- Efficacy (Init+CSDI >10% M2 over same-initializer Init-only, nonzero CLR MAE not worse, '
         f'all seeds and primary scenarios; at least one pair): **{eff["passed"]}** '
         f'({sum(v["passed"] for v in eff["pairs"].values())} of {len(eff["pairs"])} pairs qualify)',
         f'- Numerical/restoration: **{num["passed"]}** ({num["n_checked"]} results checked, '
         f'{len(num["violations"])} violations)', '',
         '## Selection',
         f'- Finalists: {[f["config"] for f in finalists]}',
         '- Validation-ranked top-3 (diagnostic only; NOT finalists unless all gates passed): '
         + ', '.join(f'{d["config"]} (mean M2 {d["rank_stat"]:.3f}, qualified={d["qualified"]})'
                     for d in diagnostic), '',
         '## Canada (Phase D)',
         ('Canada was evaluated exactly once for the frozen finalists.' if canada_ran else
          '**Not evaluated.** Validation gates did not pass, so Canada stays sealed '
          '(preregistration section 12).'), '']
    if canada_ran and test_scores:
        L += ['| Method | whole_rare_n2 | whole_middle_n2 | whole_common_n2 | random_cell_bank0 |',
              '|---|---|---|---|---|']
        for k in sorted(test_scores):
            v = test_scores[k]
            if isinstance(v, dict) and 'm2' not in v:
                L.append(f'| {k} | ' + ' | '.join(_fmt(v.get(m, {}).get('m2')) for m in (
                    'whole_rare_n2', 'whole_middle_n2', 'whole_common_n2',
                    'random_cell_bank0')) + ' |')
        L.append('')
    if cap8_summary:
        L += ['## Capacity-8 branch (descriptive, non-gated, random-cell only)',
              f'- Conditioning (random-cell): {cap8_summary["conditioning_passed"]}',
              f'- Efficacy pairs qualifying (random-cell): {cap8_summary["efficacy_pairs_passing"]}', '']
    L += ['## Limitations',
          '- One outer fold, one validation location (Denmark, 103 rows), 3 seeds: no p-values, '
          'no generalisation claim.',
          '- Training masks are random-cell banks while the primary scenario is whole-variant '
          '(train/inference mask mismatch inherited from the registered design).',
          '- Protocol U: row totals unidentified; raw-count metrics are diagnostic only.', '']
    (B / 'report.md').write_text('\n'.join(L), encoding='utf-8')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_val_fixture(inner_pool, val_frame, names, window_length, wv, rc):
    """Fixture rows = Denmark COMPLETE-POSITIVE rows only (config.json), because the
    fixture target is the full-LR label exactly as in training."""
    raw = val_frame[names].to_numpy(dtype=np.float64)
    eligible = np.isfinite(raw).all(axis=1) & (raw.sum(axis=1) > 0)
    frame = val_frame.iloc[np.flatnonzero(eligible)]
    ids = frame.index.to_numpy(np.int64)
    windows = window_indices(frame, ids, window_length)
    query = frame[names].to_numpy(dtype=np.float64)
    origin = pd.to_datetime(inner_pool.date).min().toordinal()
    times = time_encode(pd.Series(frame.date), origin)
    visible = masks(len(frame), 0)  # True = VISIBLE
    q = np.where(visible, query, np.nan)
    init, fb, ek, _ = initialize_crossfit(q, ids, frame.location.to_numpy(), inner_pool, names)
    return {'window': windows, 'query': query, 'visible': visible, 'times': times,
            'init': init, 'fb': fb, 'ek': ek,
            'raw_scale': raw_scale_for(inner_pool[names].to_numpy(float)),
            'masks': wv, 'random_cell_bank0': rc}


def main():
    global MAX_EPOCHS, MC_SAMPLES
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['train', 'finalize', 'all'], default='all')
    ap.add_argument('--branch', choices=['main', 'cap8'], default='main')
    ap.add_argument('--transforms', default=None, help='comma list (shard) for --stage train')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    global B
    if args.dry_run:
        MAX_EPOCHS, MC_SAMPLES = 2, 2
        real = B
        B = real / '_dryrun'
        B.mkdir(parents=True, exist_ok=True)
        shutil.copy(real / 'config.json', B / 'config.json')
    start = time.perf_counter()
    B.mkdir(parents=True, exist_ok=True)
    assert sha256(DATA) == DATA_SHA, 'data SHA mismatch'
    if (B / 'e5_status.json').exists() and args.stage != 'train' and args.branch == 'main':
        raise FileExistsError('completed E5 run exists; preserve it')

    cfg = json.loads((B / 'config.json').read_text(encoding='utf-8'))
    assert cfg['data_sha256'] == DATA_SHA and cfg['seeds'] == ALL_SEEDS
    assert cfg['modes'] == ALL_MODES and [t['name'] for t in cfg['transforms']] == ALL_TRANSFORMS
    assert {t['name']: t['channels'] for t in cfg['transforms']} == MODEL_CHANNELS
    assert args.dry_run or (cfg['training']['epochs_max'] == MAX_EPOCHS
                            and cfg['inference']['mc_samples'] == MC_SAMPLES)

    df = pd.read_csv(DATA)
    e0_manifest = json.loads((E0 / 'manifest.json').read_text())
    names = e0_manifest['variant_columns']
    assert names == cfg['feature_order']
    fold = json.loads((E0 / 'cv_splits.json').read_text())[0]
    assert fold['test_location'] == 'Canada' and fold['fold_id'] == 0
    tm = load_transform_module()
    delta = tm.fit_positivity_delta(df.iloc[fold['train_rows']][names].to_numpy(float))
    assert delta == 0.5, f'expected delta 0.5, got {delta}'

    pool_df = df.iloc[fold['train_rows']]
    query_all = df.iloc[fold['test_rows']]
    inner_pool = pool_df[pool_df.location != 'Denmark']
    val_frame = pool_df[pool_df.location == 'Denmark']
    assert sorted(set(val_frame.location)) == ['Denmark']

    val_wvmasks, val_wvinfo = whole_variant_masks(inner_pool, names, len(val_frame),
                                                   fold['fold_id'], scenario='validation')
    val_rc = make_random_cell_mask(len(val_frame), 17, MASKED_PARTS, seed=MASK_SEED)
    mask_files = [B / 'validation_whole_variant_masks.npz', B / 'validation_random_cell_bank0.npz']
    if mask_files[0].exists() and mask_files[1].exists():
        # masks were frozen before any scoring; later runs must reproduce them exactly
        frozen = np.load(mask_files[0])
        for k, v in val_wvmasks.items():
            assert np.array_equal(frozen[k], v), f'frozen validation mask changed: {k}'
        assert np.array_equal(np.load(mask_files[1])['bank0'], val_rc)
    else:
        np.savez_compressed(mask_files[0], **val_wvmasks)
        np.savez_compressed(mask_files[1], bank0=val_rc)
        write_json(B / 'validation_mask_manifest.json', val_wvinfo)

    e0_npz = np.load(E0 / 'masks_whole_variant.npz')
    regen, _ = whole_variant_masks(pool_df, names, len(query_all), fold['fold_id'], 'validation')
    for label in GATE_LABELS:
        for n in GATE_N:
            assert np.array_equal(regen[f'whole_{label}_n{n}'],
                                  e0_npz[f'fold0_whole_{label}_n{n}']), 'frozen E0 test mask changed'

    transforms = ALL_TRANSFORMS if not args.dry_run else ['clr16']
    modes = ALL_MODES if not args.dry_run else ['hron_2a']
    seeds = ALL_SEEDS if not args.dry_run else [42]
    inits = INIT_MODES if not args.dry_run else modes
    if args.branch == 'cap8':
        return run_cap8(args, inner_pool, val_frame, names, tm, delta, val_wvmasks, val_rc,
                        transforms, modes, seeds, inits)

    shard_transforms = transforms
    if args.transforms:
        shard_transforms = [t for t in args.transforms.split(',') if t in ALL_TRANSFORMS]
    gate_scn = {k: val_wvmasks[k] for k in RG.PRIMARY}
    gate_scn['random_cell_bank0'] = val_rc
    cache_val = B / 'cache_validation'
    fixture = build_val_fixture(inner_pool, val_frame, names, WINDOW_LENGTH, val_wvmasks, val_rc)

    if args.stage in ('train', 'all'):
        print('=== Phase C: validation training ===', flush=True)
        run_csdi_partition('validation', inner_pool, val_frame, names, tm, delta, gate_scn,
                           cache_val, all_configs(shard_transforms, modes), seeds,
                           val_fixture=fixture, with_controls=False)
        if args.stage == 'train':
            print('shard trained; run --stage finalize after all shards finish')
            return

    # finalize: all models must be cached (no silent partial run)
    full_cfg = all_configs(transforms, modes)
    missing = [f'{m}__{t}__seed{s}' for (m, t) in full_cfg for s in seeds
               if cache_load(cache_val, f'{m}__{t}__seed{s}') is None]
    if missing:
        raise RuntimeError(f'cannot finalize: {len(missing)} models missing, e.g. {missing[:3]}')
    scores, val_best = run_csdi_partition(
        'validation', inner_pool, val_frame, names, tm, delta, gate_scn, cache_val,
        full_cfg, seeds, val_fixture=fixture)

    gates = RG.validation_gates(scores, transforms, inits, seeds)
    finalists, diagnostic = RG.select_finalists(scores, gates, transforms, inits, seeds)
    write_json(B / 'validation_gates.json', gates)
    write_json(B / 'validation_scores.json', scores)
    write_json(B / 'validation_best_epochs.json',
               {k: {'best_epoch': v[0], 'best_loss': v[1]} for k, v in val_best.items()})
    write_json(B / 'validation_finalists.json',
               {'finalists': finalists, 'diagnostic_top3_not_finalists': diagnostic,
                'rule': 'finalists only if every validation gate passed (config.json selection)'})
    flat = {}
    for k, v in scores.items():
        if 'm2' in v:
            flat[k] = v
        else:
            flat.update({f'{k}/{scn}': sc for scn, sc in v.items()})
    write_json(B / 'failure_records_validation.json',
               failure_records(flat, val_frame.index.to_numpy(), 'validation'))
    write_json(B / 'scale_guard_validation.json', RG.scale_guard_summary(scores))

    # Registered restoration sensitivity (descriptive) on the validation-ranked top-3
    ok = inner_pool[names].notna().all(axis=1) & (inner_pool[names].sum(axis=1) > 0)
    bins_v, _ = prevalence_bins(inner_pool[names].to_numpy(dtype=np.float64), names)
    sens_keys = [f'{d["mode"]}__{d["transform"]}__seed{seeds[0]}' for d in diagnostic]
    write_json(B / 'restoration_sensitivity_validation.json', restoration_sensitivity(
        cache_val, sens_keys, gate_scn, val_frame[names].to_numpy(dtype=np.float64), delta,
        bins_v, inner_pool.loc[ok, names].sum(axis=1).to_numpy()))

    # Phase D: Canada stays sealed unless every validation gate passed; evaluated at most once.
    canada_ran, test_scores = False, {}
    lock = B / 'canada_evaluated.lock'
    if finalists and not args.dry_run:
        if lock.exists():
            raise FileExistsError('Canada was already evaluated once; never re-evaluate')
        lock.write_text(json.dumps({'finalists': [f['config'] for f in finalists]}))
        print('=== Phase D: Canada pass (frozen finalists, once) ===', flush=True)
        t_masks = {f'whole_{l}_n2': e0_npz[f'fold0_whole_{l}_n2'] for l in GATE_LABELS}
        t_masks['random_cell_bank0'] = np.load(
            E0 / 'masks_random_cell.npz')['fold0_random_r0.30_seed42']
        f_tr = sorted({f['transform'] for f in finalists})
        f_cfg = sorted({(f['mode'], f['transform']) for f in finalists} |
                       {(m, t) for t in f_tr for m in ('no_init', 'mask_only')},
                       key=lambda c: (c[1], c[0]))
        epochs = {f'{m}__{t}__seed{s}': val_best[f'{m}__{t}__seed{s}'][0]
                  for (m, t) in f_cfg for s in ALL_SEEDS}
        test_scores, _ = run_csdi_partition(
            'test', pool_df, query_all, names, tm, delta, t_masks, B / 'cache_test', f_cfg,
            ALL_SEEDS, epochs_fixed=epochs)
        write_json(B / 'test_scores.json', test_scores)
        canada_ran = True

    status = 'UNBLOCKED_PENDING_USER_APPROVAL' if finalists else 'BLOCKED'
    cap = B / 'capacity8_summary.json'
    cap8 = json.loads(cap.read_text(encoding='utf-8')) if cap.exists() else None
    write_report(gates, finalists, diagnostic, test_scores, canada_ran, cap8, status)
    write_json(B / 'e5_status.json', {
        'e5_status': status, 'finalists': [f['config'] for f in finalists],
        'validation_gates_passed': gates['passed'], 'canada_evaluated': canada_ran,
        'e5_not_run_without_user_approval': True,
        'runtime_seconds_finalize': time.perf_counter() - start})
    print(json.dumps({'e5_status': status, 'finalists': [f['config'] for f in finalists],
                      'validation_gates_passed': gates['passed'],
                      'canada_evaluated': canada_ran}, indent=2))


def run_cap8(args, inner_pool, val_frame, names, tm, delta, val_wvmasks, val_rc,
             transforms, modes, seeds, inits):
    """Registered capacity branch: channels=8, window_length=4, random-cell only,
    validation only, descriptive and non-gated (preregistration section 8)."""
    wl, ch = 4, 8
    shard_transforms = transforms
    if args.transforms:
        shard_transforms = [t for t in args.transforms.split(',') if t in ALL_TRANSFORMS]
    fixture = build_val_fixture(inner_pool, val_frame, names, wl, val_wvmasks, val_rc)
    scn = {'random_cell_bank0': val_rc}
    cache = B / 'cache_cap8'
    if args.stage in ('train', 'all'):
        run_csdi_partition('cap8', inner_pool, val_frame, names, tm, delta, scn, cache,
                           all_configs(shard_transforms, modes), seeds, val_fixture=fixture,
                           window_length=wl, channels_override=ch, with_controls=False)
        if args.stage == 'train':
            return
    full = all_configs(transforms, modes)
    missing = [f'{m}__{t}__seed{s}' for (m, t) in full for s in seeds
               if cache_load(cache, f'{m}__{t}__seed{s}') is None]
    if missing:
        raise RuntimeError(f'cap8 finalize: {len(missing)} models missing, e.g. {missing[:3]}')
    scores, _ = run_csdi_partition(
        'cap8', inner_pool, val_frame, names, tm, delta, scn, cache, full, seeds,
        val_fixture=fixture, window_length=wl, channels_override=ch)
    cond = RG.conditioning_gate(scores, transforms, seeds, ('random_cell_bank0',))
    eff = RG.efficacy_gate(scores, transforms, inits, seeds, ('random_cell_bank0',))
    summary = {'channels': ch, 'window_length': wl, 'scenarios': ['random_cell_bank0'],
               'role': 'descriptive_non_gated',
               'conditioning_passed': cond['passed'],
               'efficacy_pairs_passing': sum(v['passed'] for v in eff['pairs'].values()),
               'conditioning': cond, 'efficacy': eff}
    write_json(B / 'capacity8_summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('conditioning_passed', 'efficacy_pairs_passing')}))


if __name__ == '__main__':
    main()
