"""Two local subprocesses with disjoint fold outputs; no agent delegation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .engine import load_run
from .artifacts import publish_json
from .completion import RunArtifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        parser.error('Use 1 or 2 workers on the measured local runtime')
    manifest, raw, _, _ = load_run(args.root, args.output)
    artifacts = RunArtifacts(args.output, manifest, raw)
    pending = list(range(manifest['config']['folds']))
    active, completed = {}, []
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1')
    logdir = args.output / 'execution'; logdir.mkdir(exist_ok=True)
    while pending or active:
        while pending and len(active) < args.workers:
            fold = pending.pop(0)
            completion = args.output / f'fold{fold}' / 'completion.json'
            if completion.exists():
                artifacts.fold(fold)
                completed.append(fold)
                continue
            attempt = 1
            while (logdir / f'fold{fold}_attempt{attempt}.txt').exists(): attempt += 1
            log = (logdir / f'fold{fold}_attempt{attempt}.txt').open('x', encoding='utf-8')
            command = [sys.executable, '-B', '-m', 'longitudinal_impute.benchmark.engine', 'all',
                       '--root', str(args.root.resolve()), '--output', str(args.output.resolve()), '--fold', str(fold)]
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            active[fold] = (process, log, time.monotonic())
            print(json.dumps({'event': 'started', 'fold': fold, 'pid': process.pid}), flush=True)
        for fold in list(active):
            process, log, start = active[fold]
            code = process.poll()
            if code is None:
                continue
            log.close()
            del active[fold]
            if code:
                # Other owned fold processes keep their live handles; caller
                # must inspect them rather than restart duplicate jobs.
                raise RuntimeError(f'Fold {fold} exited {code}; see execution log. Other live PIDs: '
                                   + str({f: p[0].pid for f, p in active.items()}))
            inventory = artifacts.fold_inventory(fold)
            publish_json(args.output / f'fold{fold}' / 'completion.json', {
                'status': 'done', **artifacts.identity(fold), 'artifacts': inventory,
                'seconds': time.monotonic() - start})
            completed.append(fold)
            print(json.dumps({'event': 'completed', 'fold': fold}), flush=True)
        state = {'completed': sorted(completed), 'pending': pending,
                 'active_pids': {str(f): x[0].pid for f, x in active.items()}}
        (logdir / 'state.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
        if active: time.sleep(2)
    print(json.dumps({'status': 'all_folds_complete', 'folds': sorted(completed)}), flush=True)


if __name__ == '__main__':
    main()
