"""Evaluation v1 commands.

    python -m longitudinal_impute.evaluation check    --config configs/evaluation/eval_v1.toml
    python -m longitudinal_impute.evaluation masks    --config ...   # once: writes masks_manifest.json
    python -m longitudinal_impute.evaluation run      --config ... --runs <dir>
    python -m longitudinal_impute.evaluation evaluate --config ... --runs <dir> --exploratory-runs <dir> [<dir> ...] --out <dir>

Every command reads the frozen configuration and first refuses a confirmation seed already used by steps
1-7 (configs/evaluation/used_seeds.json); masks, run and evaluate also check the frozen sha256 of the
configuration against masks_manifest.json. --runs holds the confirmation runs <pass>_<mask>
(pass: main, micoda_other, residual); --exploratory-runs the seed 0-4 runs of the same passes, read by the
native tier 2 check. Paths in the configuration are relative to the repository root.
"""
import argparse
from pathlib import Path

from . import pipeline
from .report import ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m longitudinal_impute.evaluation', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', nargs='?', choices=('check', 'masks', 'run', 'evaluate'), default='evaluate')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/evaluation/eval_v1.toml')
    parser.add_argument('--runs', type=Path, help='confirmation run directories <pass>_<mask> (run, evaluate)')
    parser.add_argument('--exploratory-runs', type=Path, nargs='+',
                        help='directories with the seed 0-4 runs <pass>_<mask> (evaluate)')
    parser.add_argument('--out', type=Path, help='output directory (evaluate)')
    args = parser.parse_args(argv)
    needed = {'run': ('runs',), 'evaluate': ('runs', 'exploratory_runs', 'out')}.get(args.command, ())
    missing = [f"--{name.replace('_', '-')}" for name in needed if getattr(args, name) is None]
    if missing:
        parser.error(f'{args.command} needs {", ".join(missing)}')
    config, files = pipeline.setup(args.config)
    if args.command == 'check':
        print(f'ok: {len(files)} files of steps 1-7 in the used-seed snapshot, no confirmation seed among them')
    elif args.command == 'masks':
        pipeline.draw_masks(config, args.config)
    elif args.command == 'run':
        pipeline.run_all(config, args.config, args.runs)
    else:
        pipeline.evaluate(config, files, args.out, args.config, args.runs, args.exploratory_runs)


if __name__ == '__main__':
    main()
