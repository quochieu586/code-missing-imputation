from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from patsy import dmatrix


SCHEMA_VERSION = "2.1"


@dataclass
class ZINBEncoderConfig:
    time_spline_df: int = 4
    time_spline_type: str = "natural"
    use_lag_lead: bool = True
    use_availability_indicators: bool = True
    center_spline: bool = True

    @classmethod
    def from_zinb_config(cls, config) -> "ZINBEncoderConfig":
        return cls(
            time_spline_df=getattr(config, "time_spline_df", 4),
            time_spline_type=getattr(config, "time_spline_type", "natural"),
            use_lag_lead=getattr(config, "use_lag_lead", True),
            use_availability_indicators=getattr(config, "use_availability_indicators", True),
            center_spline=getattr(config, "center_spline", True),
        )


class DesignEncoder:

    def __init__(self, config: ZINBEncoderConfig | None = None, variant_name: str = ""):
        self.config = config or ZINBEncoderConfig()
        self.variant_name = variant_name
        self._fitted = False
        self._design_info = None
        self._spline_col_means: np.ndarray | None = None
        self._spline_projection: np.ndarray | None = None
        self._n_spline_components: int = 0
        self._active_zi_indices: np.ndarray | None = None
        self._active_nb_indices: np.ndarray | None = None
        self._full_zi_names: list[str] = []
        self._full_nb_names: list[str] = []
        self.schema: dict = {}

    def fit(
        self,
        total_seq: np.ndarray,
        day_index: np.ndarray,
        lag_counts: np.ndarray,
        lag_avail: np.ndarray,
        lead_counts: np.ndarray,
        lead_avail: np.ndarray,
    ) -> "DesignEncoder":
        total_seq = np.asarray(total_seq, dtype=np.float64)
        day_index = np.asarray(day_index, dtype=np.float64)
        lag_counts = np.asarray(lag_counts, dtype=np.float64)
        lag_avail = np.asarray(lag_avail, dtype=np.float64)
        lead_counts = np.asarray(lead_counts, dtype=np.float64)
        lead_avail = np.asarray(lead_avail, dtype=np.float64)

        self._validate_inputs(total_seq, day_index, lag_counts, lag_avail, lead_counts, lead_avail)

        spline_formula = f"0 + cr(day_index, df={self.config.time_spline_df})"
        spline_design = dmatrix(spline_formula, {"day_index": day_index}, return_type="dataframe")
        self._design_info = spline_design.design_info

        raw_spline = np.asarray(spline_design, dtype=np.float64)

        if self.config.center_spline:
            self._spline_col_means = raw_spline.mean(axis=0)
            centered = raw_spline - self._spline_col_means
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            tol = 1e-10
            keep = S > tol
            self._spline_projection = Vt[keep].T
            self._n_spline_components = int(keep.sum())
        else:
            self._spline_col_means = np.zeros(raw_spline.shape[1])
            self._spline_projection = np.eye(raw_spline.shape[1])
            self._n_spline_components = raw_spline.shape[1]

        X_zi_full, X_nb_full, _ = self._build_matrices(
            total_seq, day_index, lag_counts, lag_avail, lead_counts, lead_avail
        )
        zi_names_full, nb_names_full = self._build_full_names()

        zi_var = X_zi_full.var(axis=0)
        nb_var = X_nb_full.var(axis=0)

        self._active_zi_indices = np.where(zi_var > 1e-12)[0]
        self._active_nb_indices = np.where(nb_var > 1e-12)[0]

        if 0 not in self._active_zi_indices:
            self._active_zi_indices = np.sort(np.concatenate([[0], self._active_zi_indices]))
        if 0 not in self._active_nb_indices:
            self._active_nb_indices = np.sort(np.concatenate([[0], self._active_nb_indices]))

        self._full_zi_names = zi_names_full
        self._full_nb_names = nb_names_full

        active_zi_names = [zi_names_full[i] for i in self._active_zi_indices]
        active_nb_names = [nb_names_full[i] for i in self._active_nb_indices]

        spline_names = [f"cspline_{i}" for i in range(self._n_spline_components)]

        self.schema = {
            "schema_version": SCHEMA_VERSION,
            "variant_name": self.variant_name,
            "spline_df": self.config.time_spline_df,
            "spline_type": self.config.time_spline_type,
            "center_spline": self.config.center_spline,
            "day_index_min": float(day_index.min()),
            "day_index_max": float(day_index.max()),
            "spline_column_names": spline_names,
            "n_spline_components": self._n_spline_components,
            "spline_col_means": self._spline_col_means.tolist(),
            "spline_projection": self._spline_projection.tolist(),
            "use_lag_lead": self.config.use_lag_lead,
            "use_availability_indicators": self.config.use_availability_indicators,
            "full_zi_names": zi_names_full,
            "full_nb_names": nb_names_full,
            "active_zi_indices": self._active_zi_indices.tolist(),
            "active_nb_indices": self._active_nb_indices.tolist(),
            "zi_predictor_names": active_zi_names,
            "nb_predictor_names": active_nb_names,
            "offset_policy": "log_total_seq",
        }
        self._fitted = True
        return self

    def transform(
        self,
        total_seq: np.ndarray,
        day_index: np.ndarray,
        lag_counts: np.ndarray,
        lag_avail: np.ndarray,
        lead_counts: np.ndarray,
        lead_avail: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self._fitted, "Encoder not fitted"

        total_seq = np.asarray(total_seq, dtype=np.float64)
        day_index = np.asarray(day_index, dtype=np.float64)
        lag_counts = np.asarray(lag_counts, dtype=np.float64)
        lag_avail = np.asarray(lag_avail, dtype=np.float64)
        lead_counts = np.asarray(lead_counts, dtype=np.float64)
        lead_avail = np.asarray(lead_avail, dtype=np.float64)

        self._validate_inputs(total_seq, day_index, lag_counts, lag_avail, lead_counts, lead_avail)

        X_zi_full, X_nb_full, offset = self._build_matrices(
            total_seq, day_index, lag_counts, lag_avail, lead_counts, lead_avail
        )

        X_zi = X_zi_full[:, self._active_zi_indices]
        X_nb = X_nb_full[:, self._active_nb_indices]

        return X_zi, X_nb, offset

    def _build_matrices(
        self,
        total_seq: np.ndarray,
        day_index: np.ndarray,
        lag_counts: np.ndarray,
        lag_avail: np.ndarray,
        lead_counts: np.ndarray,
        lead_avail: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_obs = len(total_seq)

        spline_design = dmatrix(self._design_info, {"day_index": day_index}, return_type="dataframe")
        raw_spline = np.asarray(spline_design, dtype=np.float64)
        centered = raw_spline - self._spline_col_means
        spline_array = centered @ self._spline_projection

        intercept = np.ones((n_obs, 1), dtype=np.float64)
        log_total_seq = np.log(np.maximum(total_seq, 1.0)).reshape(-1, 1)

        zi_parts = [intercept, spline_array, log_total_seq]
        nb_parts = [intercept, spline_array]

        if self.config.use_lag_lead:
            lag_pos = (lag_counts > 0).astype(np.float64).reshape(-1, 1)
            lead_pos = (lead_counts > 0).astype(np.float64).reshape(-1, 1)
            log1p_lag = (np.log1p(np.maximum(lag_counts, 0)) * lag_avail).reshape(-1, 1)
            log1p_lead = (np.log1p(np.maximum(lead_counts, 0)) * lead_avail).reshape(-1, 1)

            if self.config.use_availability_indicators:
                zi_parts.extend([lag_pos, lead_pos, lag_avail.reshape(-1, 1), lead_avail.reshape(-1, 1)])
            else:
                zi_parts.extend([lag_pos, lead_pos])

            nb_parts.extend([log1p_lag, log1p_lead])

        X_zi = np.hstack(zi_parts)
        X_nb = np.hstack(nb_parts)
        offset = np.log(np.maximum(total_seq, 1.0))
        return X_zi, X_nb, offset

    def _build_full_names(self) -> tuple[list[str], list[str]]:
        spline_names = [f"cspline_{i}" for i in range(self._n_spline_components)]
        zi_names = ["intercept"] + spline_names + ["log_total_seq"]
        nb_names = ["intercept"] + spline_names

        if self.config.use_lag_lead:
            if self.config.use_availability_indicators:
                zi_names.extend(["lag_positive", "lead_positive", "lag_avail", "lead_avail"])
            else:
                zi_names.extend(["lag_positive", "lead_positive"])
            nb_names.extend(["log1p_lag_x_avail", "log1p_lead_x_avail"])

        return zi_names, nb_names

    def check_rank(self, X: np.ndarray, name: str = "design") -> bool:
        return np.linalg.matrix_rank(X) == X.shape[1]

    def get_schema(self) -> dict:
        assert self._fitted, "Encoder not fitted"
        return dict(self.schema)

    @classmethod
    def from_schema(cls, schema: dict, config: ZINBEncoderConfig | None = None) -> "DesignEncoder":
        encoder = cls(config=config, variant_name=schema.get("variant_name", ""))
        if config is None:
            encoder.config = ZINBEncoderConfig(
                time_spline_df=schema.get("spline_df", 4),
                time_spline_type=schema.get("spline_type", "natural"),
                use_lag_lead=schema.get("use_lag_lead", True),
                use_availability_indicators=schema.get("use_availability_indicators", True),
                center_spline=schema.get("center_spline", True),
            )
        encoder.schema = dict(schema)
        encoder._spline_col_means = np.array(schema.get("spline_col_means", []), dtype=np.float64)
        encoder._n_spline_components = schema.get("n_spline_components", encoder.config.time_spline_df)
        encoder._active_zi_indices = np.array(schema.get("active_zi_indices", []), dtype=int)
        encoder._active_nb_indices = np.array(schema.get("active_nb_indices", []), dtype=int)
        encoder._full_zi_names = schema.get("full_zi_names", [])
        encoder._full_nb_names = schema.get("full_nb_names", [])

        if "spline_projection" in schema:
            encoder._spline_projection = np.array(schema["spline_projection"], dtype=np.float64)
        else:
            encoder._spline_projection = np.eye(encoder._n_spline_components)

        n_raw_cols = encoder.config.time_spline_df
        if len(encoder._spline_col_means) == 0:
            encoder._spline_col_means = np.zeros(n_raw_cols)

        dummy_day = np.linspace(
            float(schema.get("day_index_min", 0.0)),
            float(schema.get("day_index_max", 100.0)),
            max(n_raw_cols + 2, 10),
        )
        spline_formula = f"0 + cr(day_index, df={encoder.config.time_spline_df})"
        dummy_design = dmatrix(spline_formula, {"day_index": dummy_day}, return_type="dataframe")
        encoder._design_info = dummy_design.design_info

        if len(encoder._active_zi_indices) == 0:
            zi_names = schema.get("zi_predictor_names", [])
            nb_names = schema.get("nb_predictor_names", [])
            encoder._active_zi_indices = np.arange(len(zi_names))
            encoder._active_nb_indices = np.arange(len(nb_names))
            encoder._full_zi_names = zi_names
            encoder._full_nb_names = nb_names

        encoder._fitted = True
        return encoder

    def save_schema(self, path: str | Path) -> None:
        assert self._fitted, "Encoder not fitted"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.schema, f, indent=2)

    @classmethod
    def load_schema(cls, path: str | Path) -> "DesignEncoder":
        with Path(path).open() as f:
            schema = json.load(f)
        return cls.from_schema(schema)

    def _validate_inputs(
        self,
        total_seq: np.ndarray,
        day_index: np.ndarray,
        lag_counts: np.ndarray,
        lag_avail: np.ndarray,
        lead_counts: np.ndarray,
        lead_avail: np.ndarray,
    ) -> None:
        n = len(total_seq)
        for name, arr in [
            ("day_index", day_index),
            ("lag_counts", lag_counts),
            ("lag_avail", lag_avail),
            ("lead_counts", lead_counts),
            ("lead_avail", lead_avail),
        ]:
            assert len(arr) == n, f"{name} length mismatch: {len(arr)} != {n}"
        assert np.all(total_seq >= 0), "total_seq must be non-negative"
        assert np.all(lag_avail >= 0) and np.all(lag_avail <= 1), "lag_avail must be 0/1"
        assert np.all(lead_avail >= 0) and np.all(lead_avail <= 1), "lead_avail must be 0/1"