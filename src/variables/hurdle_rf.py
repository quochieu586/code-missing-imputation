import pandas as pd
import numpy as np

from src.variables import Parameter, VariableModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.stats import bernoulli, nbinom, truncnorm


class HurdleRFVariableModel(VariableModel):
    name: str
    missing_indices: pd.Index
    zi_predictors: list[str]
    rf_predictors: list[str]
    zi_model: RandomForestClassifier
    rf_model: RandomForestRegressor
    residual_std: float    # store residuals for later use in sampling

    def __init__(self, name: str, missing_indices: pd.Index, zi_predictors: list[str], rf_predictors: list[str]):
        self.name = name
        self.missing_indices = missing_indices
        self.zi_predictors = zi_predictors
        self.rf_predictors = rf_predictors
    
    def _fit_distribution(
            self, X_train_zi: pd.DataFrame, X_train_rf: pd.DataFrame, y_train: pd.Series,
        ) -> list[Parameter]:
        """
        Fit the hurdle model to the training data. Return the fitted parameters with its values.
        """
        # --- Stage 1: Zero-Inflated ---
        # Label: 1 if zero, 0 if non-zero
        y_binary = (y_train == 0).astype(int)

        # Use Random Forest Classifier to predict the probability of zero vs non-zero
        clf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
        clf.fit(X_train_zi, y_binary)

        # --- Stage 2: Count Model ---
        # Filter out zero values for the count model
        non_zero_indices = y_train[y_train > 0].index
        X_train_rf_non_zero = X_train_rf.loc[non_zero_indices]
        y_train_non_zero = y_train.loc[non_zero_indices]

        # Use Random Forest Regressor to predict the count values for non-zero data
        reg = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
        reg.fit(X_train_rf_non_zero, y_train_non_zero)

        # Store the fitted models for later use in sampling
        self.zi_model = clf
        self.rf_model = reg

        parameters: list[Parameter] = []
        for i, predictor in enumerate(self.zi_predictors):
            parameters.append(Parameter(name=predictor, value=clf.feature_importances_[i], component="zi"))

        for i, predictor in enumerate(self.rf_predictors):
            parameters.append(Parameter(name=predictor, value=reg.feature_importances_[i], component="rf"))

        return parameters
    
    def fit(self, imputed_data: pd.DataFrame) -> list[Parameter]:
        """
        Fit the hurdle model to the training data. Return the fitted parameters with its values.
        """
        columns = list(set([self.name] + self.zi_predictors + self.rf_predictors))
        data = imputed_data[columns].copy()
        train_data = data.drop(index=self.missing_indices)

        X_train_zi = train_data[self.zi_predictors]
        X_train_rf = train_data[self.rf_predictors]

        y_train = train_data[self.name]

        parameters = self._fit_distribution(X_train_zi, X_train_rf, y_train)
        self.residual_std = np.std(y_train - self.rf_model.predict(X_train_rf))

        return parameters

    def sample(self, imputed_data: pd.DataFrame) -> list:
        """
        Vectorized sampling for all missing rows.
        Returns an array aligned with self.missing_indices.
        """

        # --- Extract rows ---
        rows = imputed_data.loc[self.missing_indices]

        X_zi = rows[self.zi_predictors]
        X_rf = rows[self.rf_predictors]

        # --- Model predictions (vectorized) ---
        prob_zero = self.zi_model.predict_proba(X_zi)[:, 1]   # shape (n,)
        mu = self.rf_model.predict(X_rf)                      # shape (n,)

        sigma = self.residual_std

        # --- Constraint ---
        total_seq = rows["total_sequence"].values

        covariate_columns = [c for c in imputed_data.columns if c.startswith("covid_")]
        covariates = imputed_data[covariate_columns].values
        current_var = imputed_data[self.name].values

        sum_others = covariates.sum(axis=1) - current_var
        sum_others = sum_others[self.missing_indices]

        max_allowed = total_seq - sum_others
        max_allowed = np.maximum(max_allowed, 0.0)

        # --- Zero model ---
        is_zero = np.random.binomial(1, prob_zero)

        # --- Initialize output ---
        values = np.zeros_like(mu)

        # --- Mask for valid truncated sampling ---
        valid_mask = (max_allowed > 0) & (is_zero == 0)

        if np.any(valid_mask):
            mu_v = mu[valid_mask]
            max_v = max_allowed[valid_mask]

            if sigma < 1e-6:
                values[valid_mask] = np.clip(mu_v, 0.0, max_v)
            else:
                a = (0.0 - mu_v) / sigma
                b = (max_v - mu_v) / sigma

                values[valid_mask] = truncnorm.rvs(a, b, loc=mu_v, scale=sigma)

        return values.tolist()

    def __str__(self) -> str:
        def _format_list(lst, max_items=5):
            if len(lst) <= max_items:
                return ", ".join(lst)
            return ", ".join(lst[:max_items]) + f", ... (+{len(lst) - max_items})"
        
        return (
            f"HurdleRFVariableModel(\n"
            f"  name={self.name},\n"
            f"  n_missing={len(self.missing_indices)},\n"
            f"  zi_predictors=[{_format_list(self.zi_predictors)}],\n"
            f"  rf_predictors=[{_format_list(self.rf_predictors)}]\n"
            f")"
        )