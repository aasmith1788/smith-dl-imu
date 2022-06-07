import torch
import numpy as np
from os.path import join

class Dataset4regressor(torch.utils.data.Dataset): 
  def __init__(self, dataSetDir, dataType, sess, numFold):
      self.dataType = dataType # angle or moBWHT
      self.sess = sess # train or test
      self.load_Data_X = torch.from_numpy(np.load(join(dataSetDir,f"{str(numFold)}_fold_final_{sess}.npz"))[f'final_X_{self.sess}'])
      self.load_Data_Y = torch.from_numpy(np.load(join(dataSetDir,f"{str(numFold)}_fold_final_{sess}.npz"))[f'final_Y_{self.dataType}_{self.sess}'])
      # regressor는 Dense layer이므로, AE 데이터 셋을 dense 용으로 만들어줘야함
      # 3 * 101 => 1 * 303
      self.load_Data_Y = np.reshape(self.load_Data_Y, newshape=(-1,self.load_Data_Y.shape[1]*self.load_Data_Y.shape[2],1), order='F')
    
      self.load_Data_X = torch.squeeze(self.load_Data_X.type(torch.FloatTensor))
      self.load_Data_Y = torch.squeeze(self.load_Data_Y.type(torch.FloatTensor))
      
  def __len__(self):
      return len(self.load_Data_X)

  def __getitem__(self, idx):
    X = self.load_Data_X[idx]
    Y = self.load_Data_Y[idx]
    return X, Y

class Dataset4autoencoder(torch.utils.data.Dataset): 
  def __init__(self, dataSetDir, dataType, sess, numFold):
      self.dataType = dataType # angle or moBWHT
      self.sess = sess # train or test
      self.load_Data_X = torch.from_numpy(np.load(join(dataSetDir,f"{str(numFold)}_fold_final_{sess}.npz"))[f'final_X_{self.sess}'])
      self.load_Data_X = torch.squeeze(self.load_Data_X.type(torch.FloatTensor))

  def __len__(self):
      return len(self.load_Data_X)

  def __getitem__(self, idx):
    X = self.load_Data_X[idx]
    return X

class DatasetWithDG4regressor(torch.utils.data.Dataset): 
  def __init__(self, dataSetDir, dataType, sess, numFold):
      self.dataType = dataType # angle or moBWHT
      self.sess = sess # train or test
      self.load_Data_X_imu = torch.from_numpy(np.load(join(dataSetDir,f"{str(numFold)}_fold_final_{sess}.npz"))[f'final_X_{sess}'])
      self.load_Data_X_dg = torch.from_numpy(np.load(join(dataSetDir,f"{str(numFold)}_fold_final_{sess}.npz"))[f'final_DG_{sess}'])
      self.load_Data_Y = torch.from_numpy(np.load(join(dataSetDir,f"{str(numFold)}_fold_final_{sess}.npz"))[f'final_Y_{dataType}_{sess}'])
      # regressor는 Dense layer이므로, AE 데이터 셋을 dense 용으로 만들어줘야함
      # 3 * 101 => 1 * 303
      self.load_Data_Y = np.reshape(self.load_Data_Y, newshape=(-1,self.load_Data_Y.shape[1]*self.load_Data_Y.shape[2],1), order='F')
    
      self.load_Data_X_imu = torch.squeeze(self.load_Data_X_imu.type(torch.FloatTensor))
      self.load_Data_X_dg = torch.squeeze(self.load_Data_X_dg.type(torch.FloatTensor))
      self.load_Data_Y = torch.squeeze(self.load_Data_Y.type(torch.FloatTensor))
      
  def __len__(self):
      return len(self.load_Data_X_imu)

  def __getitem__(self, idx):
    X_imu = self.load_Data_X_imu[idx]
    X_dg = self.load_Data_X_dg[idx]
    Y = self.load_Data_Y[idx]
    return X_imu,X_dg, Y