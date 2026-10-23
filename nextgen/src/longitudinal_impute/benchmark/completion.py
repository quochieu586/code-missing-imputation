"""Hierarchical completion contract: a marker never substitutes for its children."""
import json
from pathlib import Path
import numpy as np
from .artifacts import (artifact_path, evaluation_identity, file_hash, get_mask,
                        prediction_artifact_path, final_csv_artifact_path,
                        read_completed_evaluation)
from .candidates import supplementary_candidate


class RunArtifacts:
    """Validate required artifacts without running a model or changing files.

    Incomplete jobs may be retried by the engine in a new attempt. Existing
    completion markers with missing/corrupt children fail closed and are kept.
    New parent markers seal the exact required artifact paths and their hashes.
    """

    def __init__(self, output, manifest, raw):
        self.output = Path(output).resolve()
        self.manifest, self.config = manifest, manifest['config']
        self.row_count, self.feature_count = raw.counts.shape
        self.splits = self._read(self.output / 'folds.json')['folds']

    @staticmethod
    def _read(path):
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise ValueError(f'Missing or invalid completion artifact: {path}') from exc
        if not isinstance(record, dict):
            raise ValueError(f'Expected an artifact object: {path}')
        return record

    def identity(self, fold=None, method=None):
        values = {key: self.manifest[key] for key in ('source_hash', 'config_hash')}
        if fold is not None:
            values['fold'] = int(fold)
        if method is not None:
            values['method'] = method
        return values

    @staticmethod
    def _matches(record, expected, path):
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError(f'Completed artifact identity mismatch: {path}')

    @staticmethod
    def _sealed(record, inventory, path):
        if record.get('artifacts') != inventory:
            raise ValueError(f'Completed artifact inventory missing or corrupt: {path}')

    def add(self, inventory, path):
        path = Path(path).resolve()
        try:
            key = path.relative_to(self.output).as_posix()
        except ValueError as exc:
            raise ValueError('Artifact escapes run directory') from exc
        if not path.is_file():
            raise ValueError(f'Missing required artifact: {path}')
        inventory[key] = file_hash(path)

    def _evaluation(self, job, ids, scenario, seed, inventory):
        hidden = get_mask(self.output, scenario, seed)
        identity = evaluation_identity(self.output, job, ids, scenario, seed, hidden)
        record = read_completed_evaluation(job, ids, scenario, seed, hidden, identity, self.feature_count)
        if record is None:
            raise ValueError(f'Missing required evaluation marker: {job}')
        self.add(inventory, job / 'result.json')
        self.add(inventory, prediction_artifact_path(job, record))
        return record

    def candidate(self, fold, family, index):
        directory = self.output / f'fold{fold}' / 'validation' / family / f'candidate{index}'
        marker = directory / 'candidate.json'
        if not marker.exists():
            return None
        record = self._read(marker)
        expected = {'candidate': self.config['candidates'][family][index], 'index': index}
        self._matches(record, expected, marker)
        inventory = {}
        if record.get('status') == 'ok':
            scores = []
            for seed in self.config['mask_seeds']:
                result = self._evaluation(directory / f'seed{seed}', self.splits[fold]['validation_rows'],
                                          'whole_feature', seed, inventory)
                scores.append(result['metrics']['overall']['ratio_mae'])
            score = record.get('ratio_mae')
            if score is None or not np.isfinite(score) or score != float(np.mean(scores)):
                raise ValueError(f'Candidate score differs from evaluation artifacts: {marker}')
        elif record.get('status') not in ('not_applicable', 'numerical_failure') or not record.get('error'):
            raise ValueError(f'Invalid candidate terminal status: {marker}')
        self.add(inventory, marker)
        return record, inventory

    def validation_inventory(self, fold, tables):
        if set(tables) != set(self.config['candidates']):
            raise ValueError('Validation families differ from frozen configuration')
        inventory = {}
        for family, candidates in self.config['candidates'].items():
            if len(tables[family]) != len(candidates):
                raise ValueError('Validation candidate count differs from frozen configuration')
            for index, expected in enumerate(tables[family]):
                cached = self.candidate(fold, family, index)
                if cached is None or cached[0] != expected:
                    raise ValueError('Selection table differs from candidate artifacts')
                inventory.update(cached[1])
        return inventory

    def selection(self, fold):
        marker = self.output / f'fold{fold}' / 'selection.json'
        record = self._read(marker)
        self._matches(record, self.identity(fold), marker)
        tables = record.get('validation', {})
        inventory = self.validation_inventory(fold, tables)
        selected = {}
        for family, rows in tables.items():
            valid = [row for row in rows if row['status'] == 'ok']
            if not valid:
                raise ValueError(f'No successful validation candidate: {family}')
            selected[family] = min(valid, key=lambda row: (row['ratio_mae'], row['index']))
        self._matches(record, {'selected': selected}, marker)
        self._sealed(record, inventory, marker)
        return record

    def supplementary_inventory(self, fold, method, record):
        directory = self.output / f'fold{fold}' / 'supplementary' / method
        inventory = {}
        status = record.get('status')
        if status == 'ok':
            for scenario in self.config['supplementary_scenarios']:
                for seed in self.config['mask_seeds']:
                    self._evaluation(directory / f'{scenario}_{seed}', self.splits[fold]['test_rows'],
                                     scenario, seed, inventory)
        elif status in ('not_applicable', 'numerical_failure', 'resource_limit', 'failed_runtime'):
            if not record.get('error'):
                raise ValueError(f'Missing supplementary failure reason: {directory}')
            if status in ('resource_limit', 'failed_runtime'):
                if not record.get('execution_log'):
                    raise ValueError(f'Missing supplementary execution log: {directory}')
                log = artifact_path(self.output, record, 'execution_log', '')
                self.add(inventory, log)
                if file_hash(log) != record.get('execution_log_hash'):
                    raise ValueError(f'Corrupt supplementary execution log: {log}')
        else:
            raise ValueError(f'Invalid supplementary terminal status: {directory}')
        return inventory

    def supplementary(self, fold, method, selection=None):
        marker = self.output / f'fold{fold}' / 'supplementary' / method / 'finished.json'
        if not marker.exists():
            return None
        record = self._read(marker)
        self._matches(record, self.identity(fold, method), marker)
        if selection is None:
            selection = self.selection(fold)['selected']
        self._matches(record, {'candidate': supplementary_candidate(self.config, selection, method)}, marker)
        inventory = self.supplementary_inventory(fold, method, record)
        self._sealed(record, inventory, marker)
        return record

    def fold_inventory(self, fold):
        selection = self.selection(fold)
        inventory = dict(selection['artifacts'])
        base = self.output / f'fold{fold}'
        self.add(inventory, base / 'selection.json')
        for family in self.config['candidates']:
            for scenario in self.config['scenarios']:
                for seed in self.config['mask_seeds']:
                    self._evaluation(base / 'test' / family / f'{scenario}_{seed}',
                                     self.splits[fold]['test_rows'], scenario, seed, inventory)
        for method in self.config['supplementary_methods']:
            record = self.supplementary(fold, method, selection['selected'])
            if record is None:
                raise ValueError(f'Missing supplementary completion: fold{fold}/{method}')
            inventory.update(record['artifacts'])
            self.add(inventory, base / 'supplementary' / method / 'finished.json')
        return inventory

    def fold(self, fold):
        marker = self.output / f'fold{fold}' / 'completion.json'
        record = self._read(marker)
        self._matches(record, {'status': 'done', **self.identity(fold)}, marker)
        self._sealed(record, self.fold_inventory(fold), marker)
        return record

    def final(self, family, candidate):
        directory = self.output / 'final' / family
        marker = directory / 'manifest.json'
        if not marker.exists():
            return None
        record = self._read(marker)
        self._matches(record, {**self.identity(), 'family': family, 'candidate': candidate,
            'dataset_hash': self.manifest['dataset_hash'], 'rows': self.row_count}, marker)
        csv = final_csv_artifact_path(directory, record)
        inventory = {}
        self.add(inventory, csv)
        if inventory[csv.relative_to(self.output).as_posix()] != record.get('csv_hash'):
            raise ValueError(f'Corrupt final CSV: {csv}')
        self._sealed(record, inventory, marker)
        return record
