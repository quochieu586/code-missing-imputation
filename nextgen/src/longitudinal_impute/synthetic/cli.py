"""Explicit Schouten-only CLI. CSV masks never infer targets or private columns."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tomllib

from .capabilities import NotReadyError, capabilities
from .reference import reproduce_table1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("capabilities")
    gen = sub.add_parser("generate")
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--profile", choices=("easy", "medium", "hard", "all"), default="all")
    gen.add_argument("--scale", choices=("small", "standard", "large"), default="standard")
    gen.add_argument("--seed", type=int, default=2026)
    gen.add_argument("--mask-spec", type=Path, action="append", default=[])
    table = sub.add_parser("table1")
    table.add_argument("--output", type=Path, required=True)
    table.add_argument("--seed", type=int, default=2018)
    table.add_argument("--replicates", type=int, default=3)
    mask = sub.add_parser("mask")
    mask.add_argument("--data", type=Path, required=True)
    mask.add_argument("--schema-config", type=Path, required=True)
    mask.add_argument("--spec", type=Path, required=True,
                      help="JSON: scenario, mechanism, case_probability, seed, and declared patterns/frequencies or wave_time")
    mask.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "capabilities":
            print(json.dumps(capabilities(), indent=2)); return 0
        if args.command == "generate":
            from .generator import GeneratorConfig, export_dataset
            specs = [json.loads(p.read_text(encoding="utf-8")) for p in args.mask_spec]
            profiles = ("easy", "medium", "hard") if args.profile == "all" else (args.profile,)
            from .service import MaskSpec
            from .publication import staged_directory
            specs = [MaskSpec.parse(s, components=12) for s in specs]
            final_output = args.output
            with staged_directory(final_output) as staging:
                args.output = staging
                for profile in profiles:
                    export_dataset(args.output / profile, GeneratorConfig(profile=profile, scale=args.scale,
                                   seed=args.seed), mask_specs=specs, rscript=args.rscript)
                index = {"format_version": "synthetic-suite-v2", "shared_seed": args.seed,
                         "scale": args.scale, "profiles": []}
                for profile in profiles:
                    path = args.output / profile / "manifest.json"
                    report = json.loads(path.read_text(encoding="utf-8"))
                    index["profiles"].append({"profile": profile, "bundle": profile,
                        "manifest": f"{profile}/manifest.json",
                        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "masks": [m["name"] for m in report["masks"]]})
                (args.output / "dataset_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
            args.output = final_output
            print(json.dumps({"status": "COMPLETE", "profiles": profiles, "output": str(args.output)}))
            return 0
        if args.command == "table1":
            result = reproduce_table1(args.output, seed=args.seed, replicates=args.replicates, rscript=args.rscript)
            print(json.dumps({k: result[k] for k in ("status", "design_runs", "replicates")}, indent=2))
            return 0
        return _mask_csv(args)
    except NotReadyError as error:
        print(json.dumps({"status": "NOT-READY", "error": str(error)}), file=sys.stderr)
        return 2
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(json.dumps({"status": "ERROR", "error": str(error)}), file=sys.stderr)
        return 1


def _mask_csv(args):
    from ..baselines.csv_io import schema_from_config, load_csv
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    from .service import MaskSpec, apply_mask, write_mask
    from .publication import staged_directory
    spec = MaskSpec.parse(spec)  # Reject unsupported specs before data/output I/O.
    schema_bytes = args.schema_config.read_bytes()
    schema = schema_from_config(tomllib.loads(schema_bytes.decode('utf-8')))
    spec = MaskSpec.parse(spec, components=len(schema.target_columns))
    raw, _, rows = load_csv(args.data, schema)
    result = apply_mask(raw, spec, rscript=args.rscript)
    with staged_directory(args.output) as staging:
        write_mask(staging, raw, result, spec,
                   input_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(), schema_bytes=schema_bytes, source_rows=rows)
    print(json.dumps({'status': 'COMPLETED', 'mask_id': result.diagnostics['mask_id'],
                      'achieved_target_cell_rate': result.diagnostics['achieved_target_cell_rate']}))
    return 0
