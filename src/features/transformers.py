import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Custom Transformer (누수 제거 핵심)
class Log2Transformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log2(X + 1.0)


class LowExpressionFilter(BaseEstimator, TransformerMixin):
    """
    fold 안에서만 gene 필터링 수행 (누수 방지)
    """
    def __init__(self, min_nonzero_frac=0.34):
        self.min_nonzero_frac = min_nonzero_frac
        self.keep_genes_ = None

    def fit(self, X, y=None):
        # X: samples x genes
        nonzero_frac = (X > 0).sum(axis=0) / X.shape[0]
        self.keep_genes_ = nonzero_frac >= self.min_nonzero_frac
        return self

    def transform(self, X):
        return X.loc[:, self.keep_genes_]
