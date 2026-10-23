"""Train-fitted numeric/categorical metadata encoder for linear baselines.

Missing and unknown categories remain distinct before constant-column removal.
An intercept is added by regression, not by this encoder. Numerical rank is
checked by the core; metadata encoding does not silently use a pseudoinverse.
"""
import numpy as np
from .errors import BaselineError


class MetadataEncoder:
    def __init__(self, *, numeric=(), categorical=()):
        if set(numeric) & set(categorical) or len(set(numeric)) != len(numeric) or len(set(categorical)) != len(categorical):
            raise BaselineError('Metadata lists must be unique and disjoint')
        self.numeric, self.categorical = tuple(numeric), tuple(categorical)

    def _column(self, panel, name):
        if name not in panel.metadata or name not in panel.metadata_observed:
            raise BaselineError(f'Metadata {name} is absent or lacks a missing mask')
        value = np.asarray(panel.metadata[name])
        mask = np.asarray(panel.metadata_observed[name])
        if value.shape != (len(panel.row_ids),) or mask.shape != value.shape or mask.dtype != bool:
            raise BaselineError('Invalid metadata shape or mask type')
        return value, mask

    def fit(self, panel):
        self.numeric_state_, self.vocab_ = {}, {}
        for name in self.numeric:
            value, mask = self._column(panel, name)
            try:
                observed = value[mask].astype(float)
            except (ValueError, TypeError) as exc:
                raise BaselineError(f'Non-numeric visible metadata {name}') from exc
            if not np.isfinite(observed).all():
                raise BaselineError('Visible numeric metadata contains infinity/NaN')
            fill = float(np.median(observed)) if len(observed) else 0.
            scale = float(np.std(observed)) if len(observed) else 1.
            self.numeric_state_[name] = (fill, scale if scale > 0 else 1.)
        for name in self.categorical:
            value, mask = self._column(panel, name)
            self.vocab_[name] = tuple(sorted(set(str(v) for v in value[mask])))
        encoded, names = self._encode(panel)
        # Always-constant columns cannot be estimated alongside an intercept.
        self.keep_ = np.ptp(encoded, axis=0) > 0 if encoded.shape[1] else np.zeros(0, bool)
        self.feature_names_ = tuple(n for n, keep in zip(names, self.keep_) if keep)
        self.dropped_constant_ = tuple(n for n, keep in zip(names, self.keep_) if not keep)
        return self

    def _encode(self, panel):
        columns, names = [], []
        for name in self.numeric:
            value, mask = self._column(panel, name)
            fill, scale = self.numeric_state_[name]
            x = np.full(len(value), fill)
            x[mask] = value[mask].astype(float)
            if not np.isfinite(x).all():
                raise BaselineError('Nonfinite numeric metadata')
            columns.extend(((x - fill) / scale, (~mask).astype(float)))
            names.extend((name, name + ':missing'))
        for name in self.categorical:
            value, mask = self._column(panel, name)
            text = np.array([str(v) for v in value])
            vocab = self.vocab_[name]
            # First known category is reference; unknown != missing in design.
            for category in vocab[1:]:
                columns.append((mask & (text == category)).astype(float))
                names.append(name + ':' + category)
            columns.extend(((~mask).astype(float), (mask & ~np.isin(text, vocab)).astype(float)))
            names.extend((name + ':missing', name + ':unknown'))
        return (np.column_stack(columns) if columns else np.empty((len(panel.row_ids), 0))), names

    def transform(self, panel):
        if not hasattr(self, 'keep_'):
            raise BaselineError('Fit metadata encoder before transform')
        encoded, _ = self._encode(panel)
        return encoded[:, self.keep_]
