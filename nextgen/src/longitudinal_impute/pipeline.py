"""Run existing models directly on the full COVID-19 table."""
import json
from pathlib import Path
import time
import numpy as np
from .data import load_csv, make_mask, model_input, restore_counts, write_completed_csv
from .evaluation import distribution_scores


def impute(raw, model, hidden):
    values = model_input(raw, hidden, positive=model.input_domain == 'positive')
    model.fit(values)
    return restore_counts(raw, model.predict(values) * raw.totals[:, None], hidden)


def run(data, schema_config, output, *, models, mask_rate=0., seed=0):
    """Fit every row. Artificial masking is a separate optional MSE run."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    raw = load_csv(data, schema_config)
    hidden = make_mask(raw, 'random_cell', mask_rate, seed)[0]
    output.mkdir(parents=True)
    np.save(output / 'evaluation_mask.npy', hidden)
    summary = {}
    for name, factory in models.items():
        start = time.perf_counter()
        try:
            completed = impute(raw, factory(), np.zeros_like(hidden))
            write_completed_csv(output / f'{name}.csv', raw, completed)
            scores = distribution_scores(raw, completed)
            scores['status'] = 'completed'
        except (ValueError, RuntimeError) as exc:
            summary[name] = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
            (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
            continue
        scores['evaluation_cells'] = int(hidden.sum())
        scores['mse'] = None
        if hidden.any():
            try:
                evaluated = impute(raw, factory(), hidden)
                scores['mse'] = float(np.mean((evaluated[hidden] - raw.counts[hidden]) ** 2))
                scores['evaluation_status'] = 'completed'
            except (ValueError, RuntimeError) as exc:
                scores['evaluation_status'] = 'failed'
                scores['evaluation_error'] = f'{type(exc).__name__}: {exc}'
        scores['seconds'] = time.perf_counter() - start
        summary[name] = scores
        (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    return summary
