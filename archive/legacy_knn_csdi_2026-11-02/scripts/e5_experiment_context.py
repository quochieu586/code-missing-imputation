"""Shared protocol constants and scientific dependencies for E5 stages."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rework_runtime  # noqa: E402,F401

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.dont_write_bytecode = True


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


@dataclass(frozen=True)
class ExperimentContext:
    """Explicit input locations; scientific constants above remain frozen."""
    root: Path = ROOT
    data: Path = DATA
    e0: Path = E0
    e2: Path = E2
    data_sha256: str = DATA_SHA


DEFAULT_CONTEXT = ExperimentContext()


