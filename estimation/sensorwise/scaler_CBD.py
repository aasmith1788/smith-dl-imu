
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 데이터셋 불러올때 필요한 것
class MinMaxScalerSensor(MinMaxScaler):
    def fit(self, X, y=None):
        x = np.reshape(X, newshape=(X.shape[0] * X.shape[1], -1))
        super().fit(x, y=y)

    def transform(self, X):
        x = np.reshape(X, newshape=(X.shape[0] * X.shape[1], -1))
        return np.reshape(super().transform(x), newshape=X.shape)

    def inverse_transform(self, X):
        x = np.reshape(X, newshape=(X.shape[0] * X.shape[1], -1))
        return np.reshape(super().inverse_transform(x), newshape=X.shape)