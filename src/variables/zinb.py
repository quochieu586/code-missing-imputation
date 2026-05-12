import pandas as pd
import numpy as np
import statsmodels.api as sm

from .base import VariableModel, Parameter
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialResultsWrapper, ZeroInflatedNegativeBinomialP
from scipy.stats import nbinom, bernoulli

class ZINBVariableModel(VariableModel):
    name: str
    missing_indices: pd.Index
    zi_predictors: list[str]
    nb_predictors: list[str]
    fitted_results: ZeroInflatedNegativeBinomialResultsWrapper

    def __init__(self, name: str, missing_indices: pd.Index, zi_predictors: list[str], nb_predictors: list[str]):
        self.name = name
        self.missing_indices = missing_indices
        self.zi_predictors = zi_predictors
        self.nb_predictors = nb_predictors

    def fit(self, imputed_data: pd.DataFrame) -> list[Parameter]:
        """
        Input: data is the imputed dataset of the previous iteration.
        Return: parameters of the current distribution and fitted values for the missing indices.
        """
        rows = imputed_data.drop(index=self.missing_indices).copy()

        X_train_zi = sm.add_constant(rows[self.zi_predictors])
        X_train_nb = sm.add_constant(rows[self.nb_predictors])
        y_train = rows[self.name]

        model = ZeroInflatedNegativeBinomialP(endog=y_train, exog=X_train_nb, exog_infl=X_train_zi, inflation='logit')
        self.fitted_results = model.fit()

        # Extract and return the fitted parameters
        parameters = []
        for name, value in self.fitted_results.params.items():
            if name.startswith("inflate_"):
                name = name[len("inflate_"):]
                component = "zi"
            else:
                component = "nb"
            
            parameters.append(Parameter(name=name, value=value, component=component))

        return parameters

    def sample(self, imputed_data: pd.DataFrame) -> pd.Series:
        """
        Isolate the sampling logic to this method, which can be called after fitting the model. 
        This allows us to sample data later.
        """
        # --- Extract rows ---
        rows = imputed_data.loc[self.missing_indices]

        X_zi = sm.add_constant(rows[self.zi_predictors])
        X_nb = sm.add_constant(rows[self.nb_predictors])


        # --- Model predictions (vectorized) ---
        prob_zero = self.fitted_results.predict(X_nb, exog_infl=X_zi, which='prob-zero')   # shape (n,)
        mu = self.fitted_results.predict(X_nb, exog_infl=X_zi, which='mean-main')          # shape (n,)

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

        # --- Generate NB samples ---
        # --- NB parameters ---
        alpha = self.fitted_results.params['alpha']
        n_param = 1 / alpha
        p_param = n_param / (n_param + mu)

        # --- Initialize ---
        nb_counts = np.zeros_like(mu, dtype=int)

        # --- Mask: only sample where non-zero AND feasible ---
        valid_mask = (is_zero == 0) & (max_allowed > 0)

        if np.any(valid_mask):
            n_v = n_param
            p_v = p_param[valid_mask]
            M_v = max_allowed[valid_mask].astype(int)

            # Compute CDF at truncation point
            cdf_M = nbinom.cdf(M_v, n_v, p_v)

            # Avoid numerical issues
            cdf_M = np.clip(cdf_M, 1e-12, 1.0)

            # Sample uniform
            u = np.random.uniform(0, cdf_M)

            # Inverse CDF (vectorized)
            nb_counts_valid = nbinom.ppf(u, n_v, p_v)

            nb_counts[valid_mask] = nb_counts_valid.astype(int)
        
        imputed_values = np.where(is_zero == 1, 0, nb_counts)

        return imputed_values.tolist()