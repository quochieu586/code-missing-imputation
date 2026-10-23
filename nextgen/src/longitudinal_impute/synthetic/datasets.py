"""Public conditioning and evaluator-only truth boundaries for v1/v2 bundles."""
import csv
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import tomllib
import numpy as np
from ..baselines.csv_io import load_csv, schema_from_config
from ..baselines.panels import hide_targets, subset


def _manifest(root):
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'COMPLETE' or manifest.get('format_version') not in ('synthetic-panel-v1', 'synthetic-panel-v2'):
        raise ValueError('Incomplete or unsupported dataset bundle')
    dims = manifest['dimensions']
    if any(type(dims[k]) is not int or dims[k] <= 0 for k in ('entities','timepoints','components')):
        raise ValueError('Invalid bundle dimensions')
    return manifest


def _verify(root, manifest, paths):
    for relative in paths:
        if hashlib.sha256((root/relative).read_bytes()).hexdigest() != manifest['files_sha256'].get(relative):
            raise ValueError(f'Dataset checksum mismatch: {relative}')


def _schema(root, manifest):
    schema = schema_from_config(tomllib.loads((root/'schema.toml').read_text(encoding='utf-8')))
    names = [schema.entity_column, schema.time_column, schema.total_column, *schema.target_columns, *schema.metadata_columns]
    if (len(names) != len(set(names)) or len(schema.target_columns) != manifest['dimensions']['components'] or
        schema.metadata_columns != ('entity_trait_01',)):
        raise ValueError('Invalid synthetic schema')
    return schema


def load_conditioning(bundle, mask_name, *, split=None):
    """Read/check only public files; private/ may be absent."""
    root = Path(bundle).resolve()
    manifest = _manifest(root)
    if mask_name not in {m['name'] for m in manifest['masks']}:
        raise ValueError('Unknown conditioning mask')
    prefix = f'conditioning/{mask_name}'
    _verify(root, manifest, ['schema.toml', 'entities.csv', *(f'{prefix}/{f}' for f in
                            ('conditioning.csv','manifest.json','hidden_mask.csv'))])
    schema = _schema(root, manifest)
    raw, fields, _ = load_csv(root/prefix/'conditioning.csv', schema)
    mask_manifest = json.loads((root/prefix/'manifest.json').read_text(encoding='utf-8'))
    hidden = np.loadtxt(root/prefix/'hidden_mask.csv', delimiter=',', ndmin=2)
    dims = manifest['dimensions']
    if (raw.counts.shape != (dims['entities']*dims['timepoints'],dims['components']) or
        fields != mask_manifest['conditioning_columns'] or fields !=
        [schema.entity_column,schema.time_column,schema.total_column,*schema.target_columns,*schema.metadata_columns] or
        list(raw.row_ids) != mask_manifest['row_ids'] or hidden.shape != raw.counts.shape or
        not np.isin(hidden, [0,1]).all() or not np.array_equal(hidden.astype(bool), ~raw.original_observed) or
        mask_manifest['schema_sha256'] != hashlib.sha256((root/'schema.toml').read_bytes()).hexdigest()):
        raise ValueError('Conditioning schema/row/mask mismatch')
    identity = json.dumps(list(raw.row_ids), ensure_ascii=False).encode('utf-8')
    mask_id = hashlib.sha256(identity+hidden.astype(bool).tobytes()).hexdigest()
    if mask_id != mask_manifest['diagnostics']['mask_id']:
        raise ValueError('Conditioning mask identity mismatch')
    if (raw.counts[raw.original_observed] != np.floor(raw.counts[raw.original_observed])).any() or (raw.totals != np.floor(raw.totals)).any():
        raise ValueError('Synthetic counts/totals must be integers')
    with (root/'entities.csv').open(encoding='utf-8',newline='') as f:
        entities = list(csv.DictReader(f))
    entity_splits = {r['entity_id']:r['split'] for r in entities}
    if (len(entity_splits) != len(entities) or len(entities) != dims['entities'] or
        set(entity_splits) != set(raw.entity_ids) or set(entity_splits.values()) != {'train','validation','test'} or
        any((raw.entity_ids == e).sum() != dims['timepoints'] for e in entity_splits)):
        raise ValueError('Invalid entity split contract')
    panel, _ = hide_targets(raw, np.zeros_like(raw.original_observed), mask_id=mask_id,
                            approved_metadata=('entity_trait_01',))
    if split is not None:
        if split not in ('train','validation','test'):
            raise ValueError('Unknown split')
        panel = subset(panel,np.flatnonzero([entity_splits[e] == split for e in raw.entity_ids]))
    return panel


def load_evaluation(bundle, mask_name, *, split=None):
    """Verify all private truth before creating labels; align each source by row ID.

    v1 has no NPZ IDs: its checked entity-major/date grid is the identity contract.
    v2 stores explicit row_ids with axes (entity,time), independently of CSV order.
    """
    from ..contracts import EvaluationLabels
    root = Path(bundle).resolve()
    manifest = _manifest(root)
    panel = load_conditioning(root, mask_name)
    _verify(root,manifest,['private/complete_counts.csv','private/ground_truth.npz'])
    schema = _schema(root,manifest)
    raw, _, _ = load_csv(root/'private/complete_counts.csv',schema)
    with np.load(root/'private/ground_truth.npz',allow_pickle=False) as source:
        truth = {k:source[k].copy() for k in source.files}
    e,t,d = (manifest['dimensions'][k] for k in ('entities','timepoints','components'))
    shapes = {'p':(e,t,d),'counts':(e,t,d),'strata':(e,t,d),'N':(e,t),
              'residual_p':(e,t),'residual_counts':(e,t),'split':(e,),
              'entity_trait_01':(e,),'onsets':(e,d),'peaks':(e,d),'entity_shifts':(e,)}
    expected_keys = set(shapes) | ({'row_ids'} if manifest['format_version']=='synthetic-panel-v2' else set())
    if set(truth) != expected_keys or any(truth[k].shape != shape for k,shape in shapes.items()):
        raise ValueError('Invalid private truth keys/shapes')
    if any(not np.isfinite(truth[k]).all() for k in shapes if k != 'split'):
        raise ValueError('Nonfinite private truth')
    with (root/'entities.csv').open(encoding='utf-8',newline='') as f:
        entities=list(csv.DictReader(f))
    canonical = tuple(json.dumps([r['entity_id'],(date(2000,1,1)+timedelta(days=i)).isoformat()],separators=(',',':'))
                      for r in entities for i in range(t))
    if 'row_ids' in truth:
        if truth['row_ids'].shape != (e,t) or tuple(truth['row_ids'].ravel()) != canonical:
            raise ValueError('Invalid private row identity grid')
        truth_ids = tuple(truth['row_ids'].ravel())
    else:
        truth_ids = canonical
        if raw.row_ids != canonical:
            raise ValueError('Legacy v1 private CSV order differs from NPZ identity contract')
    if set(raw.row_ids) != set(truth_ids) or set(panel.row_ids) != set(truth_ids):
        raise ValueError('Private/public row identity mismatch')
    index = {key:i for i,key in enumerate(raw.row_ids)}
    csv_ids = np.array([index[key] for key in truth_ids])
    p,x,n = truth['p'],truth['counts'],truth['N']
    rp,rx = truth['residual_p'],truth['residual_counts']
    if ((p < 0).any() or (rp <= 0).any() or (p.sum(axis=2) >= 1).any() or
        not np.allclose(p.sum(axis=2)+rp,1,rtol=0,atol=1e-14) or
        any((v < 0).any() or (v != np.floor(v)).any() for v in (x,rx,n)) or (n <= 0).any() or
        not np.array_equal(x.sum(axis=2)+rx,n) or
        not np.array_equal(truth['strata'],np.where(p==0,0,np.where(x==0,1,2))) or (x[p==0] != 0).any() or
        not np.array_equal(raw.counts[csv_ids],x.reshape(-1,d)) or
        not np.array_equal(raw.totals[csv_ids],n.ravel()) or
        not np.array_equal(truth['split'],[r['split'] for r in entities]) or
        not np.array_equal(np.asarray(raw.metadata['entity_trait_01'][csv_ids],float),np.repeat(truth['entity_trait_01'],t))):
        raise ValueError('Inconsistent private truth')
    index = {key:i for i,key in enumerate(truth_ids)}
    ids = np.array([index[key] for key in panel.row_ids])
    full_counts = x.reshape(-1,d)[ids]
    if (not np.array_equal(panel.totals,n.ravel()[ids]) or
        not np.array_equal(panel.counts[panel.conditioning_mask],full_counts[panel.conditioning_mask])):
        raise ValueError('Public observations differ from private truth')
    if split is not None:
        selected = load_conditioning(root,mask_name,split=split)
        index = {key:i for i,key in enumerate(panel.row_ids)}
        take = np.array([index[key] for key in selected.row_ids])
        ids,full_counts,panel = ids[take],full_counts[take],selected
    hidden = ~panel.conditioning_mask
    labels = EvaluationLabels(panel.row_ids,np.where(hidden,full_counts,np.nan),hidden)
    latent = {k:truth[k].reshape(-1,d)[ids] for k in ('p','strata')}
    latent.update({k:truth[k].ravel()[ids] for k in ('residual_p','residual_counts','N')})
    return labels,latent
