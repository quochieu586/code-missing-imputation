"""E5 evaluate stage implementation."""
from scripts.e5_experiment_context import (
    ALL_MODES, ALL_SEEDS, ALL_TRANSFORMS, CHANNELS, CSDICoreRework, DATA, DDPM_STEPS, E0,
    HEADS, INIT_MODES, LAYERS, N_VARIANTS, POLICIES, PRIMARY, Path, RG, WINDOW_LENGTH,
    evaluate_baseline_on_scenario, evaluate_csdi_on_scenario, evaluate_init_only_on_scenario,
    failure_records, initialize_crossfit, json, load_transform_module, np, pd, prevalence_bins,
    raw_scale_for, schedule, time_encode, torch, train_fallback_values, window_indices,
    write_json,
)
from src.stage_b.experiment_artifacts import (
    _train_cache_path, eval_cache_load, eval_cache_save, train_cache_load,
)
from scripts.e5_mask_protocol import (
    budget_key, policy_key,
)


def stage_evaluate(out_dir, dry_run=False, *, context=None):
    """Evaluate all 48 budget entries + controls + gates on frozen scoring masks."""
    from scripts.e5_experiment_context import DEFAULT_CONTEXT
    context = DEFAULT_CONTEXT if context is None else context
    DATA, E0 = context.data, context.e0
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
    query_all = df.iloc[fold['test_rows']]  # Canada â€” MUST NOT BE USED

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

        # Aggregation (directive Â§11.2):
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
