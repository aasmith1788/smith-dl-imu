import keras.backend as K
import tensorflow as tf
# 원래의 범위로 값 평가하기
# 원래 정확도 복구해서 tensorboard에 기록하기
# 통합은 의미 없는 수치!
def rescaled_RMSE(y_true, y_pred):
    y_true = tf.transpose(tf.reshape(tf.squeeze(y_true), [3,-1]))
    y_pred = tf.transpose(tf.reshape(tf.squeeze(y_pred), [3,-1]))
    y_true = (y_true - K.constant(load_scaler4Y_angle.min_)) / K.constant(load_scaler4Y_angle.scale_)
    y_pred = (y_pred - K.constant(load_scaler4Y_angle.min_)) / K.constant(load_scaler4Y_angle.scale_)
    # default is RMSE, squaredbool, default=True If True returns MSE value, if False returns RMSE value.
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    rescaled_RMSE = K.sqrt(K.mean(K.square(y_pred - y_true)))
    print(f"\nUsing {numFold} fold scaler")
    return rescaled_RMSE

def X_Axis_RMSE(y_true, y_pred):
    NumAxis = 0
    y_true = tf.transpose(tf.reshape(tf.squeeze(y_true), [3,-1]))[:,NumAxis]
    y_pred = tf.transpose(tf.reshape(tf.squeeze(y_pred), [3,-1]))[:,NumAxis]
    print(y_true.shape)
    y_true = (y_true - K.constant(load_scaler4Y_angle.min_[NumAxis])) / K.constant(load_scaler4Y_angle.scale_[NumAxis])
    y_pred = (y_pred - K.constant(load_scaler4Y_angle.min_[NumAxis])) / K.constant(load_scaler4Y_angle.scale_[NumAxis])
    # default is RMSE, squaredbool, default=True If True returns MSE value, if False returns RMSE value.
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    X_Axis_RMSE = K.sqrt(K.mean(K.square(y_pred - y_true)))
    print(f"\nUsing {numFold} fold scaler")
    return X_Axis_RMSE
    
def Y_Axis_RMSE(y_true, y_pred):
    NumAxis = 1
    y_true = tf.transpose(tf.reshape(tf.squeeze(y_true), [3,-1]))[:,NumAxis]
    y_pred = tf.transpose(tf.reshape(tf.squeeze(y_pred), [3,-1]))[:,NumAxis]
    print(y_true.shape)
    y_true = (y_true - K.constant(load_scaler4Y_angle.min_[NumAxis])) / K.constant(load_scaler4Y_angle.scale_[NumAxis])
    y_pred = (y_pred - K.constant(load_scaler4Y_angle.min_[NumAxis])) / K.constant(load_scaler4Y_angle.scale_[NumAxis])
    # default is RMSE, squaredbool, default=True If True returns MSE value, if False returns RMSE value.
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    Y_Axis_RMSE = K.sqrt(K.mean(K.square(y_pred - y_true)))
    print(f"\nUsing {numFold} fold scaler")
    return Y_Axis_RMSE

def Z_Axis_RMSE(y_true, y_pred):
    NumAxis = 2
    y_true = tf.transpose(tf.reshape(tf.squeeze(y_true), [3,-1]))[:,NumAxis]
    y_pred = tf.transpose(tf.reshape(tf.squeeze(y_pred), [3,-1]))[:,NumAxis]
    print(y_true.shape)
    y_true = (y_true - K.constant(load_scaler4Y_angle.min_[NumAxis])) / K.constant(load_scaler4Y_angle.scale_[NumAxis])
    y_pred = (y_pred - K.constant(load_scaler4Y_angle.min_[NumAxis])) / K.constant(load_scaler4Y_angle.scale_[NumAxis])
    # default is RMSE, squaredbool, default=True If True returns MSE value, if False returns RMSE value.
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    Z_Axis_RMSE = K.sqrt(K.mean(K.square(y_pred - y_true)))
    print(f"\nUsing {numFold} fold scaler")
    return Z_Axis_RMSE

def rescaled_RMSE_pct(y_true, y_pred):
    y_true = tf.transpose(tf.reshape(tf.squeeze(y_true), [3,-1]))
    y_pred = tf.transpose(tf.reshape(tf.squeeze(y_pred), [3,-1]))
    y_true = (y_true - K.constant(load_scaler4Y_moBHWT.min_)) / K.constant(load_scaler4Y_moBHWT.scale_)
    y_pred = (y_pred - K.constant(load_scaler4Y_moBHWT.min_)) / K.constant(load_scaler4Y_moBHWT.scale_)
    # default is RMSE, squaredbool, default=True If True returns MSE value, if False returns RMSE value.
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    rescaled_RMSE_pct = 100 * K.sqrt(K.mean(K.square(y_pred - y_true))) / (K.max(y_true) - K.min(y_true))
    print(f"\nUsing {numFold} fold scaler")
    return rescaled_RMSE_pct

def X_Axis_RMSE_pct(y_true, y_pred):
    NumAxis = 0
    y_true = tf.transpose(tf.reshape(tf.squeeze(y_true), [3,-1]))[:,NumAxis]
    y_pred = tf.transpose(tf.reshape(tf.squeeze(y_pred), [3,-1]))[:,NumAxis]
    print(y_true.shape)
    y_true = (y_true - K.constant(load_scaler4Y_moBHWT.min_[NumAxis])) / K.constant(load_scaler4Y_moBHWT.scale_[NumAxis])
    y_pred = (y_pred - K.constant(load_scaler4Y_moBHWT.min_[NumAxis])) / K.constant(load_scaler4Y_moBHWT.scale_[NumAxis])
    # default is RMSE, squaredbool, default=True If True returns MSE value, if False returns RMSE value.
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    X_Axis_RMSE_pct = 100 * K.sqrt(K.mean(K.square(y_pred - y_true))) / (K.max(y_true) - K.min(y_true))
    print(f"\nUsing {numFold} fold scaler")
    return X_Axis_RMSE_pct
    
def Y_Axis_RMSE_pct(y_true, y_pred):
    NumAxis = 1
    y_true = tf.transpose(tf.reshape(tf.squeeze(y_true), [3,-1]))[:,NumAxis]
    y_pred = tf.transpose(tf.reshape(tf.squeeze(y_pred), [3,-1]))[:,NumAxis]
    print(y_true.shape)
    y_true = (y_true - K.constant(load_scaler4Y_moBHWT.min_[NumAxis])) / K.constant(load_scaler4Y_moBHWT.scale_[NumAxis])
    y_pred = (y_pred - K.constant(load_scaler4Y_moBHWT.min_[NumAxis])) / K.constant(load_scaler4Y_moBHWT.scale_[NumAxis])
    # default is RMSE, squaredbool, default=True If True returns MSE value, if False returns RMSE value.
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    Y_Axis_RMSE_pct = 100 * K.sqrt(K.mean(K.square(y_pred - y_true))) / (K.max(y_true) - K.min(y_true))
    print(f"\nUsing {numFold} fold scaler")
    return Y_Axis_RMSE_pct

def Z_Axis_RMSE_pct(y_true, y_pred):
    NumAxis = 2
    y_true = tf.transpose(tf.reshape(tf.squeeze(y_true), [3,-1]))[:,NumAxis]
    y_pred = tf.transpose(tf.reshape(tf.squeeze(y_pred), [3,-1]))[:,NumAxis]
    print(y_true.shape)
    y_true = (y_true - K.constant(load_scaler4Y_moBHWT.min_[NumAxis])) / K.constant(load_scaler4Y_moBHWT.scale_[NumAxis])
    y_pred = (y_pred - K.constant(load_scaler4Y_moBHWT.min_[NumAxis])) / K.constant(load_scaler4Y_moBHWT.scale_[NumAxis])
    # default is RMSE, squaredbool, default=True If True returns MSE value, if False returns RMSE value.
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    Z_Axis_RMSE_pct = 100 * K.sqrt(K.mean(K.square(y_pred - y_true))) / (K.max(y_true) - K.min(y_true))
    print(f"\nUsing {numFold} fold scaler")
    return Z_Axis_RMSE_pct
