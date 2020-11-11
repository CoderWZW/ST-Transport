from loader.dataloader_STGCN import LoaderSTGCN_4
from Glob.glob import p_parse
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
# from utils.lossFunction import Loss

from model.stgcn import STGCN

def testNetSTGCN(args, model_path='./save/snap_STGCN/'):
    test_dataset = LoaderSTGCN_4(args, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=24, num_workers=args.num_workers,
                             pin_memory=True, drop_last=False, shuffle=False)
    
    snapList = os.listdir(os.path.join(model_path))
    snapList.sort(key=lambda x: int(x.split('.')[0].split('_')[1]))

    best_model = snapList[185]
    # best_model = snapList[293]

    model = STGCN(num_nodes=307, num_features=3, num_timesteps_input=12, num_timesteps_output=1)

    # adjacency = np.load(args.data_path_adjacency, allow_pickle=True)
    # norm_adjacency = adjacency + np.eye(adjacency.shape[0])
    # norm_adjacency = torch.Tensor(norm_adjacency)

    adjacency = np.load(args.data_path_adjacency, allow_pickle=True)
    adjacency = adjacency + np.eye(adjacency.shape[0])
    adjacency = torch.Tensor(adjacency)
    D = adjacency.sum(1)
    D = torch.diag(torch.pow(D, -0.5))
    A = D.mm(adjacency).mm(D)

    if args.cuda:
        model = model.cuda()
        A = A.cuda()
    
    model.load_state_dict(torch.load(os.path.join(
        model_path, best_model), map_location='cpu')['state_dict'])
    model.eval()
    loss_func = torch.nn.MSELoss()

    val_res_list = []
    val_label_list = []

    with torch.no_grad():
        for _, pack in enumerate(test_loader):
            pack = list(map(lambda item: item.numpy(), pack))
            seqs = torch.Tensor(pack[:-1])
            labels = torch.Tensor(pack[-1])
            if args.cuda:
                seqs = seqs.cuda()
                labels = labels.cuda()
            seqs = seqs.permute(1,2,0,3)
            out = model.forward(A_hat=A, X=seqs)
            for i in out.detach().cpu().numpy():
                val_res_list.append(i)
            for i in labels.detach().cpu().numpy():
                val_label_list.append(i)
        val_res_list = torch.Tensor(val_res_list)
        val_label_list = torch.Tensor(val_label_list)
        RMSE_loss = (loss_func(val_res_list, val_label_list).item())**0.5
    MAPE_loss = MAPE(y_true=val_label_list.detach().cpu().numpy(), y_pred=val_res_list.detach().cpu().numpy())
    MAE_loss = MAE(y_true=val_label_list.detach().cpu().numpy(), y_pred=val_res_list.detach().cpu().numpy())
    print('RMSE: {}   MAPE: {}   MAE: {}'.format(RMSE_loss, MAPE_loss, MAE_loss))

def MAPE(y_true, y_pred):
    # idx = (y_true>20).nonzero()
    # return np.mean(np.abs(y_true[idx] - y_pred[idx]) / y_true[idx])
    return np.mean(np.abs(y_true - y_pred) / y_true)

# def MAPE(y_true, y_pred):
#     a = np.array(list(range(744)))
#     sunday = []
#     monday = []
#     tuesday = []
#     wednesday = []
#     thursday = []
#     friday = []
#     saturday = []
#     sunday = []

#     weekdays = []
#     weekend = []

#     for i in a:
#         r = i%(24*7)
#         if 24*0<=r<24*1:
#             sunday.append(i)
#             weekend.append(i)
#         elif 24*1<=r<24*2:
#             monday.append(i)
#             weekdays.append(i)
#         elif 24*2<=r<24*3:
#             tuesday.append(i)
#             weekdays.append(i)
#         elif 24*3<=r<24*4:
#             wednesday.append(i)
#             weekdays.append(i)
#         elif 24*4<=r<24*5:
#             thursday.append(i)
#             weekdays.append(i)
#         elif 24*5<=r<24*6:
#             friday.append(i)
#             weekdays.append(i)
#         elif 24*6<=r<24*7:
#             saturday.append(i)
#             weekend.append(i)    

#     y_pred = y_pred[weekend]
#     y_true = y_true[weekend]
#     idx = (y_true>20).nonzero()
#     return np.mean(np.abs(y_true[idx] - y_pred[idx]) / y_true[idx])

def MAE(y_true, y_pred):
    return np.mean(abs(y_pred-y_true))

if __name__ == "__main__":
    args = p_parse()
    testNetSTGCN(args)