import numpy as np
import pandas as pd
import statsmodels.api as sm

from dataclasses import dataclass
from abc import ABC, abstractmethod
from scipy.stats import nbinom, bernoulli
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP, ZeroInflatedNegativeBinomialResultsWrapper

@dataclass
class Parameter:
    name: str
    value: any
    component: str = ""  # e.g. "zi", "nb"

class VariableModel(ABC):
    name: str
    missing_indices: pd.Index

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, imputed_data: pd.DataFrame) -> list[Parameter]:
        pass

    @abstractmethod
    def sample(self, imputed_data: pd.DataFrame) -> pd.Series:
        pass