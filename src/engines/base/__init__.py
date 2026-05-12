from abc import ABC, abstractmethod

class BaseImputer(ABC):
    @abstractmethod
    def fit(self, X, y=None):
        """
        Impute missing values in the dataset X. Return the imputed dataset.
        """
        pass

    @abstractmethod
    def print_history(self):
        """
        Print the history of imputation steps taken by the imputer.
        """
        pass