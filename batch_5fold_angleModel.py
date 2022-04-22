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

from orientpine import X_Axis_RMSE_pct, Y_Axis_RMSE_pct, Z_Axis_RMSE_pct

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)

print(tf.__version__)
print(keras.__version__)

# Model 생성, compile
def create_model():

    model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(4242,)),
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(6000, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4000, activation='relu'),  
    tf.keras.layers.Dense(303, activation='linear'),  
    ])

    return model
model = create_model().summary()
# 모델 형식을 남기기 위한 저장!


learningRate = 0.001 # 두번째 실험
patience = 10
myoptim=Nadam(learning_rate=learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
early_stopping = EarlyStopping(monitor='val_loss',patience = patience, mode='min') # 일단은 적당히 epoch 주고 돌리기

# 데이터 셋 준비
# 데이터 셋 준비
dataSetDir = 'DATASET'
scalerDir  = 'SCALER'

# 준비된 K-fold data iterations
time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


# shutil.rmtree('./logs', ignore_errors=True) # 삭제는 신중ㅎ히
# tensorboard 동작시키는 법 : tensorboard --logdir logs/fit

goal = 'angle'
epochs = 1000
for numFold in range(0,5): # 5-fold crossvalidation
    # 각 fold 별로 별도로 표기하기
    log_dir = "logs/fit/" + time +'_'+ str(numFold) + '_fold_' + goal  
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    print(f"Num of Fold: {numFold}")
    # 데이터 불러오기
    # 모든 데이터는 scaled된 데이터임!
    load_train = np.load(join(dataSetDir,f"{numFold}_fold_final_train.npz"))
    load_test = np.load(join(dataSetDir,f"{numFold}_fold_final_test.npz"))
    print(f'loaded Train shape: {load_train["final_X_train"].shape}, {load_train["final_Y_angle_train"].shape}, {load_train["final_Y_moBHWT_train"].shape}')
    print(f'loaded Test shape: {load_test["final_X_test"].shape}, {load_test["final_Y_angle_test"].shape}, {load_test["final_Y_moBHWT_test"].shape}')
    # sclaer 불러오기
    # Here scaler is MinMaxScaler!
    load_scaler4X = load(open(join(scalerDir,f"{numFold}_fold_scaler4X.pkl"), 'rb'))
    load_scaler4Y_angle = load(open(join(scalerDir,f"{numFold}_fold_scaler4Y_angle.pkl"), 'rb'))
    load_scaler4Y_moBHWT = load(open(join(scalerDir,f"{numFold}_fold_scaler4Y_moBHWT.pkl"), 'rb'))

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

