"""E5 mask protocol stage implementation."""
from scripts.e5_experiment_context import (
    ALL_MODES, INIT_MODES, MASKED_PARTS, MASK_BANKS, MASK_SEED, N_VARIANTS,
    choose_hidden_variants, condition_features, initialize_crossfit, make_random_cell_mask, np,
)


def policy_key(policy, mode, seed):
    return f'{policy}_{mode}__clr16__seed{seed}'

def budget_key(policy, mode, seed, budget):
    return f'{policy}_{mode}__clr16__seed{seed}_budget{budget}'

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
