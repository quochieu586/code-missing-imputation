"""Full-table COVID-19 imputation without splits or fit-ID files."""
import argparse
import json
from pathlib import Path
from .pipeline import run
from .baselines.hron import HronKNN
from .baselines.micoda import MICoDa
from .baselines.tsagris import TsagrisKNN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--schema-config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mask-rate', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    result = run(args.data, args.schema_config, args.output,
                 models={'hron': HronKNN, 'micoda': MICoDa, 'tsagris': TsagrisKNN},
                 mask_rate=args.mask_rate, seed=args.seed)
    print(json.dumps(result, indent=2, allow_nan=False))
    if any(r['status'] != 'completed' or r.get('evaluation_status') == 'failed' for r in result.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
