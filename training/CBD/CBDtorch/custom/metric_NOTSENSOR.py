import torch

def nRMSE_Axis_TLPerbatch(pred, target,axis,load_scaler4Y, device):
    dict_axis = {
    'x': 0,
    "y": 1,
    "z": 2,
    }
    axis = dict_axis[axis]
    nRMSE_perbatch = 0
    # 필요한 sclaer 불러오기
    
    batchNum = len(target)
    for bat in range(batchNum): # batch 내를 순회
        pred_axis   = torch.transpose(torch.reshape(torch.squeeze(pred[bat]), [3,-1]), 0, 1)[:,axis]
        target_axis = torch.transpose(torch.reshape(torch.squeeze(target[bat]), [3,-1]), 0, 1)[:,axis]
        pred_axis = (pred_axis - torch.tensor(load_scaler4Y.min_[axis], device=device)) / torch.tensor(load_scaler4Y.scale_[axis],device=device)
        target_axis = (target_axis - torch.tensor(load_scaler4Y.min_[axis],device=device)) / torch.tensor(load_scaler4Y.scale_[axis],device=device)
        nRMSE = 100 * torch.sqrt(torch.mean(torch.square(pred_axis - target_axis))) / (torch.max(target_axis) - torch.min(target_axis))
        nRMSE_perbatch += nRMSE
    return nRMSE_perbatch