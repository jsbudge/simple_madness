from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, FunctionTransformer
from sklearn.pipeline import Pipeline
import numpy as np


class LegendreScalarPolynomialFeatures(TransformerMixin, BaseEstimator):

    """
    This polynomial feature uses a Legendre basis to create uncorrelated features. In order to use it,
    it needs all features to be within [-1, 1].

    """
    def __init__(self, degree=2, include_bias=False):
        self.degree = degree
        self.include_bias = include_bias

    def fit(self, X, y=None):
        # There is nothing to learn
        # Legendre polynomials do not depend on the training data.
        return self

    def __sklearn_is_fitted__(self):
        # See above - it's always "fitted" by definition
        return True

    def transform(self, X, y=None):
        # create a Vandermonde matrix for each feature, and create a 3D array
        # of shape
        vander = np.polynomial.legendre.legvander(X, self.degree)
        if not self.include_bias:
            # discard the column of ones for each feature
            vander = vander[..., 1:]

        # reshape to concatenate the Vandermonde matrices horizontally
        n_rows = X.shape[0]
        result = vander.reshape(n_rows, -1)
        return result


class SeasonalSplit:
    def __init__(self, random_state=None):
        self.random_state = random_state

    def split(self, X, y=None):
        seasons = list(set(X.index.get_level_values(1)))
        for s in seasons:
            yield X.loc[X.index.get_level_values(1) == s].index, X.loc[X.index.get_level_values(1) != s].index

    def get_n_splits(self, X, y=None, groups=None):
        return len(list(set(X.index.get_level_values(1))))


def get_legendre_pipeline(degree=2, include_bias=False):
    return Pipeline([
        ('quantile', QuantileTransformer()),
        ('post-mapper', FunctionTransformer(lambda x: 2 * x - 1)),
        ('polynomial_features', LegendreScalarPolynomialFeatures(degree=degree, include_bias=include_bias)),
    ])
