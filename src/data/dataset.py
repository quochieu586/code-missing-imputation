import pandas as pd

class Dataset:
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def get_observed_indices(self, col: str):
        return self.df[col].notna()

    def get_missing_indices(self, col: str):
        return self.df[col].isna()
    
    def process_categorical(self, col: str):
        """
        Convert categorical column to numerical using one-hot encoding.
        """
        return pd.get_dummies(self.df[col], prefix=col)
    
    