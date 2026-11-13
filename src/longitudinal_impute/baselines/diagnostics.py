"""Canonical diagnostics boundary shared by cores, adapters and reports."""
import copy


def canonical_diagnostics(diagnostics):
    data = copy.deepcopy(diagnostics)
    events = data.get('row_events', data.get('records', []))
    tuning = {key: data[key] for key in ('cv', 'pattern_cv_sizes') if key in data}
    excluded = {'row_events', 'records', 'fit_diagnostics', 'cv', 'pattern_cv_sizes',
                'fit_summary', 'prediction_summary', 'tuning'}
    return {'row_events': events,
            'fit_summary': data.get('fit_summary', data.get('fit_diagnostics', {})),
            'prediction_summary': data.get('prediction_summary',
                                           {k: v for k, v in data.items() if k not in excluded}),
            'tuning': data.get('tuning', tuning)}


class PredictionDiagnostics:
    """Keep row summaries, with one tuning table per pattern across queries."""
    def __init__(self):
        self.rows, self.events, self.tuning = [], [], {}

    def add(self, diagnostics, row=None):
        fields = {'row_events', 'fit_summary', 'prediction_summary', 'tuning'}
        data = diagnostics if fields <= diagnostics.keys() else canonical_diagnostics(diagnostics)
        self.rows.append({'row': row, **data['prediction_summary']})
        for event in data['row_events']:
            self.events.append({**event, **({'row': row} if row is not None else {})})
        for category, tables in data['tuning'].items():
            destination = self.tuning.setdefault(category, {})
            for pattern, table in tables.items():
                destination.setdefault(pattern, table)

    def export(self, fit):
        return {'fit_summary': canonical_diagnostics(fit)['prediction_summary'],
                'prediction_summary': self.rows, 'row_events': self.events,
                'tuning': self.tuning}
