"""Counts/N/Other and preprocessing adaptations around paper cores.

FitScope is enforced before donors or metadata state are fitted. Every core
receives extended proportions: Other is unknown for partial conditioning rows.
Hron's positive replacement and no-anchor/no-donor fallback are explicit.
"""
import copy
import numpy as np
from ..contracts import ImputationResult
from ..constraints import CountBudgetConstraint
from ..metadata import MetadataEncoder
from .common import BaselineError, closure
from .diagnostics import PredictionDiagnostics
from .registry import configure_core
from ..errors import BaselineInputError, NotFittedError, PredictionUnavailableError
from .panels import extended_proportions, fit_subset, safe_counts


class CountBaselineAdapter:
    def __init__(self, core, *, metadata_encoder=None, zero_delta=None,
                 fallback='error', degenerate_policies=False, input_domain=None,
                 prediction_policy=None):
        if fallback not in ('error', 'train_mean'):
            raise BaselineError('fallback must be error or train_mean')
        if zero_delta is not None and (not np.isfinite(zero_delta) or zero_delta <= 0):
            raise BaselineError('zero_delta must be finite positive')
        self.core = configure_core(core, degenerate_policies=degenerate_policies)
        self.encoder = metadata_encoder or MetadataEncoder()
        self.zero_delta, self.fallback = zero_delta, fallback
        self.degenerate_policies = degenerate_policies
        self.input_domain = input_domain or getattr(core, 'input_domain', None)
        self.prediction_policy = prediction_policy or getattr(core, 'prediction_policy', None)
        if self.input_domain not in ('positive', 'nonnegative'):
            raise BaselineError('Adapter requires an explicit core input domain')
        if self.prediction_policy not in ('rowwise_adaptation', 'population_adaptation'):
            raise BaselineError('Adapter requires an explicit prediction policy')
        if self.prediction_policy == 'population_adaptation' and fallback == 'train_mean':
            raise BaselineError('train_mean fallback requires rowwise adaptation to preserve successful query rows')

    def _positive(self, values):
        x = values.copy()
        x[x == 0] = self.zero_delta_
        return x

    def fit(self, panel, scope):
        # Keep all learned adapter/core state detached until the whole fit works.
        candidate = copy.copy(self)
        candidate.__dict__.pop('last_prediction_', None)
        candidate._fit_in_place(panel, scope)
        self.__dict__ = candidate.__dict__
        return self

    def _fit_in_place(self, panel, scope):
        train = fit_subset(panel, scope)
        values = extended_proportions(train)
        self.width_ = values.shape[1] - 1
        self.fit_scope_, self.fit_row_ids_ = scope, train.row_ids
        self.encoder = copy.deepcopy(self.encoder).fit(train)
        covariates = self.encoder.transform(train)
        # Independent per-part observed means; declared fallback when no donors.
        count = np.isfinite(values).sum(axis=0)
        if (count == 0).any() and self.fallback == 'train_mean':
            raise BaselineError('Train-mean fallback cannot identify entirely unobserved parts (including Other)')
        means = np.divide(np.nansum(values, axis=0), count, out=np.zeros_like(count, dtype=float), where=count > 0)
        self.fallback_values_ = closure(means)
        self.core_ = copy.deepcopy(self.core)
        if self.input_domain == 'positive':
            positive = values[np.isfinite(values) & (values > 0)]
            if not len(positive):
                raise BaselineError('No positive parts to fit Hron zero replacement')
            self.zero_delta_ = self.zero_delta or float(positive.min() / 2)
            values = self._positive(values)
        else:
            self.zero_delta_ = None
        self.core_.fit(values, covariates)
        return self

    def predict_result(self, panel):
        if not hasattr(self, 'core_'):
            raise NotFittedError('Fit count adapter before prediction')
        raw = safe_counts(panel)
        if raw.shape[1] != self.width_:
            raise BaselineInputError('Target width differs from fit schema')
        values = extended_proportions(panel)
        encoded = self._positive(values) if self.input_domain == 'positive' else values.copy()
        covariates = self.encoder.transform(panel)
        output = np.empty_like(encoded)
        row_events, fallback_records = [], []
        summaries = PredictionDiagnostics()
        if self.prediction_policy == 'rowwise_adaptation':
            for i in range(len(raw)):
                try:
                    result = self.core_.predict_result(encoded[i:i + 1], covariates[i:i + 1])
                except PredictionUnavailableError as exc:
                    if self.fallback == 'error':
                        raise
                    output[i] = self.fallback_values_
                    fallback_records.append({'row': i, 'reason': str(exc), 'error_code': exc.code,
                                             'policy': 'train_mean'})
                    continue
                output[i] = result.values[0]
                summaries.add(result.diagnostics, i)
                for event in result.diagnostics.get('row_events', []):
                    normalized = dict(event)
                    normalized['row'] = i
                    row_events.append(normalized)
        else:
            result = self.core_.predict_result(encoded, covariates)
            output[:] = result.values
            summaries.add(result.diagnostics)
            for event in result.diagnostics.get('row_events', []):
                normalized = dict(event)
                row = normalized.get('row')
                if row is not None and 0 <= int(row) < len(raw):
                    normalized['row'] = int(row)
                row_events.append(normalized)
        proposal = output * np.asarray(panel.totals)[:, None]
        repaired = CountBudgetConstraint().apply(proposal, panel)
        diagnostics = dict(repaired.diagnostics)
        fit = getattr(self.core_, 'fit_summary_',
                      getattr(getattr(self.core_, 'fit_result_', None), 'diagnostics', {}))
        diagnostics.update(summaries.export(fit))
        diagnostics.update({'run_kind': 'dataset_adaptation', 'core_type': type(self.core_).__name__,
                            'core_row_events': row_events, 'fallback_records': fallback_records,
                            'input_domain': self.input_domain, 'prediction_policy': self.prediction_policy,
                            'fit_row_ids': list(self.fit_row_ids_),
                            'fit_protocol': self.fit_scope_.protocol, 'split_id': self.fit_scope_.split_id,
                            'fit_mask_id': self.fit_scope_.mask_id, 'zero_delta': self.zero_delta_,
                            'metadata_features': list(self.encoder.feature_names_),
                            'metadata_constant_columns_dropped': list(self.encoder.dropped_constant_),
                            'fit_diagnostics': getattr(getattr(self.core_, 'fit_result_', None), 'diagnostics', None)})
        return ImputationResult(panel.row_ids, repaired.counts, diagnostics)

    def predict(self, panel):
        result = self.predict_result(panel)
        self.last_prediction_ = result.diagnostics
        return result.counts
