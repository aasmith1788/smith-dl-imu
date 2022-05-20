import torch
import numpy as np
from os.path import join

class Dataset4predictor(torch.utils.data.Dataset): 
  def __init__(self, dataSetDir, dataType, sess, numFold):
      self.dataType = dataType # angle or moBWHT
      self.sess = sess # train or test
      self.load_Data_X = torch.from_numpy(np.load(join(dataSetDir,f"{str(numFold)}_fold_final_{sess}.npz"))[f'final_X_{self.sess}'])
      self.load_Data_Y = torch.from_numpy(np.load(join(dataSetDir,f"{str(numFold)}_fold_final_{sess}.npz"))[f'final_Y_{self.dataType}_{self.sess}'])
      
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