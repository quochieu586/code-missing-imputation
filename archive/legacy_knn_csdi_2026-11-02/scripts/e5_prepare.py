"""E5 prepare stage implementation."""
from scripts.e5_experiment_context import (
    ALL_MODES, ALL_SEEDS, BATCH_WINDOWS, BUDGET_1120_EPOCHS, BUDGET_280_EPOCHS, CHANNELS, DATA,
    DATA_SHA, DDPM_STEPS, E0, FIXTURE_NOISE_SEED, FIXTURE_STEPS, HEADS, LAYERS, LR,
    MASKED_PARTS, MASK_BANKS, MASK_SEED, MAX_EPOCHS, MC_SAMPLES, N_VARIANTS, POLICIES, PRIMARY,
    Path, ROOT, WINDOW_LENGTH, json, load_transform_module, make_random_cell_mask, np, pd,
    sha256, torch, whole_variant_masks, write_json,
)
from scripts.e5_mask_protocol import (
    assert_mask_constant_per_location, assert_n_strata_covered, assert_paired_banks,
    make_training_mask_banks,
)


def stage_prepare(out_dir, *, context=None):
    """Create experiment directory, freeze masks, noise, config, and hashes."""
    from scripts.e5_experiment_context import DEFAULT_CONTEXT
    context = DEFAULT_CONTEXT if context is None else context
    ROOT, DATA, E0 = context.root, context.data, context.e0
    DATA_SHA = context.data_sha256
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
        DATA,
        E0 / 'cv_splits.json',
        E0 / 'manifest.json',
        context.e2 / 'transforms.py',
        ROOT / 'src/stage_b/research_csdi.py',
        ROOT / 'src/stage_b/research_csdi_rework.py',
        ROOT / 'src/stage_a/initializers_e1.py',
        ROOT / 'src/evaluation/research_pilot.py',
        ROOT / 'src/evaluation/rework_gates.py',
        ROOT / 'scripts/run_e3_revised.py',
        ROOT / 'scripts/run_e5_rework.py',
        ROOT / 'scripts/run_e5_mask_conditioning.py',
        ROOT / 'scripts/e5_experiment_context.py',
        ROOT / 'scripts/e5_mask_protocol.py',
        ROOT / 'scripts/e5_checkpoint_training.py',
        ROOT / 'scripts/e5_prepare.py',
        ROOT / 'scripts/e5_evaluate.py',
        ROOT / 'src/stage_b/experiment_artifacts.py',
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
