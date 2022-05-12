import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Nadam
from keras.callbacks import EarlyStopping
import numpy as np
from os.path import join
from pickle import load
import shutil
from sklearn.metrics import mean_squared_error
import keras.backend as K
import random
import os
import datetime

time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)

#### 설정 영역
modelVersion = 'Dense_1st'
nameDataset = 'IWALQQ_2nd'
goal = 'angle'
# 데이터 셋 준비
relativeDir = '../preperation/SAVE_dataSet'
dataSetDir = join(relativeDir,nameDataset)
# 학습된 모델은 백업안하는 공간에서 저장하기
SaveDir = '/restricted/projectnb/movelab/bcha/IMUforKnee/trainedModel/'
# epochs
epochs = 1000
# Model 생성, compile
def create_model():

    model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(4242,)),
    tf.keras.layers.Dense(6000, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4000, activation='relu'),  
    tf.keras.layers.Dense(303, activation='linear'),  
    ])

    return model

learningRate = 0.001 # 두번째 실험
patience = 10
myoptim=Nadam(learning_rate=learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
early_stopping = EarlyStopping(monitor='val_loss',patience = patience, mode='min') # 일단은 적당히 epoch 주고 돌리기

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
    y_true = (y_true - K.constant(load_scaler4Y_angle.min_)) / K.constant(load_scaler4Y_angle.scale_)
    y_pred = (y_pred - K.constant(load_scaler4Y_angle.min_)) / K.constant(load_scaler4Y_angle.scale_)
    # default is RMSE, squaredbool, default=True If True returns MSE value, if False returns RMSE value.
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    rescaled_RMSE_pct = 100 * K.sqrt(K.mean(K.square(y_pred - y_true))) / (K.max(y_true) - K.min(y_true))
    print(f"\nUsing {numFold} fold scaler")
    return rescaled_RMSE_pct

def X_Axis_RMSE_pct(y_true, y_pred):
    NumAxis = 0
    y_true = tf.transpose(tf.reshape(tf.squeeze(y_true), [3,-1]))[:,NumAxis]
    y_pred = tf.transpose(tf.reshape(tf.squeeze(y_pred), [3,-1]))[:,NumAxis]
    y_true = (y_true - K.constant(load_scaler4Y_angle.min_[NumAxis])) / K.constant(load_scaler4Y_angle.scale_[NumAxis])
    y_pred = (y_pred - K.constant(load_scaler4Y_angle.min_[NumAxis])) / K.constant(load_scaler4Y_angle.scale_[NumAxis])
    # default is RMSE, squaredbool, default=True If True returns MSE value, if False returns RMSE value.
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    X_Axis_RMSE_pct = 100 * K.sqrt(K.mean(K.square(y_pred - y_true))) / (K.max(y_true) - K.min(y_true))
    print(f"\nUsing {numFold} fold scaler")
    return X_Axis_RMSE_pct
    
def Y_Axis_RMSE_pct(y_true, y_pred):
    NumAxis = 1
    y_true = tf.transpose(tf.reshape(tf.squeeze(y_true), [3,-1]))[:,NumAxis]
    y_pred = tf.transpose(tf.reshape(tf.squeeze(y_pred), [3,-1]))[:,NumAxis]
    y_true = (y_true - K.constant(load_scaler4Y_angle.min_[NumAxis])) / K.constant(load_scaler4Y_angle.scale_[NumAxis])
    y_pred = (y_pred - K.constant(load_scaler4Y_angle.min_[NumAxis])) / K.constant(load_scaler4Y_angle.scale_[NumAxis])
    # default is RMSE, squaredbool, default=True If True returns MSE value, if False returns RMSE value.
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    Y_Axis_RMSE_pct = 100 * K.sqrt(K.mean(K.square(y_pred - y_true))) / (K.max(y_true) - K.min(y_true))
    print(f"\nUsing {numFold} fold scaler")
    return Y_Axis_RMSE_pct

def Z_Axis_RMSE_pct(y_true, y_pred):
    NumAxis = 2
    y_true = tf.transpose(tf.reshape(tf.squeeze(y_true), [3,-1]))[:,NumAxis]
    y_pred = tf.transpose(tf.reshape(tf.squeeze(y_pred), [3,-1]))[:,NumAxis]
    y_true = (y_true - K.constant(load_scaler4Y_angle.min_[NumAxis])) / K.constant(load_scaler4Y_angle.scale_[NumAxis])
    y_pred = (y_pred - K.constant(load_scaler4Y_angle.min_[NumAxis])) / K.constant(load_scaler4Y_angle.scale_[NumAxis])
    # default is RMSE, squaredbool, default=True If True returns MSE value, if False returns RMSE value.
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    Z_Axis_RMSE_pct = 100 * K.sqrt(K.mean(K.square(y_pred - y_true))) / (K.max(y_true) - K.min(y_true))
    print(f"\nUsing {numFold} fold scaler")
    return Z_Axis_RMSE_pct

for numFold in range(0,5): # 5-fold crossvalidation
    # 각 fold 별로 별도로 표기하기
    log_dir = join("logs","fit",modelVersion,nameDataset,time +'_'+ str(numFold) + '_fold_' + goal)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    print(f"Num of Fold: {numFold}")
    # 데이터 불러오기
    # 모든 데이터는 scaled된 데이터임!
    load_train = np.load(join(dataSetDir,f"{numFold}_fold_final_train.npz"))
    load_test = np.load(join(dataSetDir,f"{numFold}_fold_final_test.npz"))
    print(f'loaded Train shape: {load_train["final_X_train"].shape}, {load_train["final_Y_angle_train"].shape}, {load_train["final_Y_moBWHT_train"].shape}')
    print(f'loaded Test shape: {load_test["final_X_test"].shape}, {load_test["final_Y_angle_test"].shape}, {load_test["final_Y_moBWHT_test"].shape}')
    # sclaer 불러오기
    # Here scaler is MinMaxScaler!
    load_scaler4X = load(open(join(dataSetDir,f"{numFold}_fold_scaler4X.pkl"), 'rb'))
    load_scaler4Y_angle = load(open(join(dataSetDir,f"{numFold}_fold_scaler4Y_angle.pkl"), 'rb'))

    # https://wandb.ai/sauravm/Optimizers/reports/How-to-Compare-Keras-Optimizers-in-Tensorflow--VmlldzoxNjU1OTA4
    # Nadam을 선택한 이유
    model = create_model()
    model.compile(optimizer=myoptim,
              loss='mean_absolute_error',
              metrics=[X_Axis_RMSE_pct, Y_Axis_RMSE_pct, Z_Axis_RMSE_pct])
    # 차원 축소
    X_train = np.squeeze(load_train["final_X_train"], axis=2)
    Y_angle_train = np.squeeze(load_train["final_Y_angle_train"], axis=2)

    X_test = np.squeeze(load_test["final_X_test"], axis=2)
    Y_angle_test = np.squeeze(load_test["final_Y_angle_test"], axis=2)

    # 요건 나중에... [early_stopping,]
    history = model.fit(X_train, Y_angle_train, validation_data=(X_test,Y_angle_test), epochs=epochs, callbacks=[tensorboard_callback])
    model.save(join(SaveDir,modelVersion,nameDataset,f'{goal}_{numFold}_fold'))
