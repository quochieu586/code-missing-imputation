"""E5 checkpoint training stage implementation."""
from scripts.e5_experiment_context import (
    ALL_MODES, ALL_SEEDS, BUDGET_1120_EPOCHS, BUDGET_280_EPOCHS, CHANNELS, CSDICoreRework,
    DATA, DATA_SHA, DDPM_STEPS, E0, FIXTURE_NOISE_SEED, FIXTURE_STEPS, HEADS, INIT_MODES,
    LAYERS, LR, MASK_BANKS, MAX_EPOCHS, N_VARIANTS, POLICIES, PRIMARY, Path, WINDOW_LENGTH,
    condition_features, epsilon_loss, forward_noise, groups_of_windows, initialize_crossfit,
    json, load_transform_module, np, pd, prevalence_bins, raw_scale_for, schedule, time,
    time_encode, torch, window_indices,
)
from scripts.e5_mask_protocol import (
    policy_key, precompute_train_conditions,
)
from src.stage_b.experiment_artifacts import (
    train_cache_load, train_cache_save,
)


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

def stage_train(out_dir, dry_run=False, *, context=None):
    """Train all 24 trajectories for 160 epochs each, selecting dual-budget checkpoints."""
    from scripts.e5_experiment_context import DEFAULT_CONTEXT
    context = DEFAULT_CONTEXT if context is None else context
    DATA, E0, DATA_SHA = context.data, context.e0, context.data_sha256
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
