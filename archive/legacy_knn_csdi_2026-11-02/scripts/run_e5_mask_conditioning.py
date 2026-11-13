"""CLI orchestration for the frozen E5 mask-conditioning protocol.

Scientific functions are re-exported for existing callers; implementations live
in protocol, preparation, checkpoint training, evaluation and artifact modules.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rework_runtime
import torch
from scripts.e5_experiment_context import (
    ALL_MODES, ALL_SEEDS, ALL_TRANSFORMS, BATCH_WINDOWS, BUDGET_1120_EPOCHS, BUDGET_280_EPOCHS,
    CHANNELS, DATA, DATA_SHA, DDPM_STEPS, E0, E2, FIXTURE_NOISE_SEED, FIXTURE_STEPS, HEADS,
    INIT_MODES, LAYERS, LR, MASKED_PARTS, MASK_BANKS, MASK_SEED, MAX_EPOCHS, MC_SAMPLES,
    N_VARIANTS, POLICIES, PRIMARY, RG, ROOT, SECONDARY, WINDOW_LENGTH,
)
from scripts.e5_mask_protocol import (
    policy_key, budget_key, make_training_mask_banks, assert_paired_banks,
    assert_mask_constant_per_location, assert_n_strata_covered, precompute_train_conditions,
)
from scripts.e5_checkpoint_training import (
    make_checkpoint_fixture, fit_with_dual_budget, stage_train,
)
from scripts.e5_prepare import (
    stage_prepare,
)
from scripts.e5_evaluate import (
    stage_evaluate,
)
from src.stage_b.experiment_artifacts import (
    _cache_sha, _train_cache_path, eval_cache_load, eval_cache_save, train_cache_load,
    train_cache_save,
)

def main():
    ap = argparse.ArgumentParser(
        prog='run_e5_mask_conditioning.py',
        description='E5 mask-conditioning experiment (Â§11.1-11.5). '
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
    torch.set_num_threads(2)
    main()
