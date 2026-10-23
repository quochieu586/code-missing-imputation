"""Read-only config inspection, usable on Windows and Kaggle without torch."""
import argparse
import json
from pathlib import Path
import tomllib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect-config', required=True, type=Path)
    args = parser.parse_args()
    with args.inspect_config.open('rb') as stream:
        config = tomllib.load(stream)
    print(json.dumps({'status': 'scaffold_only', 'config': config},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
