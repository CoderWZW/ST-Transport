from loader.dataloader import Loader
from Glob.glob import p_parse
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from model.model import STGCN
from model.STGAT import STGAT
import sys

sys.path.append("..") 

MODEL_ROOT = 'save/snap_mymodel/'
IMG_ROOT = 'save/img/'
VAL_ROOT = 'save/img/mymodel'

# 训练过程
def train(args):
    # 训练数据加载
    print(torch.__version__)
    train_dataset = Loader(args, mode = 'train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True, shuffle=True)

    seed = 1337
    torch.manual_seed(seed)

    model = Our_model(num_nodes=307, num_features=3, num_timesteps_input=12, num_timesteps_output=1)

    adjacency = np.load(args.data_path_adjacency, allow_pickle=True)
    adjacency = adjacency + np.eye(adjacency.shape[0])
    adjacency = torch.Tensor(adjacency)
    D = adjacency.sum(1)
    D = torch.diag(torch.pow(D, -0.5))
    A = D.mm(adjacency).mm(D)

    if args.cuda:
        torch.cuda.manual_seed(seed)
        model = model.cuda()
        A = A.cuda()

    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)

    train_loss_list = []

    for epoch in range(args.max_epoches):

        # if 100 < epoch < 150:
        #     for para_group in opt.param_groups:
        #         para_group['lr'] = args.lr*0.5
        # if epoch > 150:
        #     for para_group in opt.param_groups:
        #         para_group['lr'] = args.lr*0.1

        train_loss = 0.0
        step = 0
        for _, pack in enumerate(train_loader):
            step += 1
            pack = list(map(lambda item: item.numpy(), pack))
            seqs = torch.Tensor(pack[:-1])
            labels = torch.Tensor(pack[-1])
            if args.cuda:
                seqs = seqs.cuda()
                labels = labels.cuda()
            seqs = seqs.permute(1,2,0,3)
            out = model.forward(A_hat=A, X=seqs)
            # print(out.shape)
            # print(labels.shape)
            loss_func = torch.nn.MSELoss()
            loss = loss_func(out, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= step
        train_loss = train_loss**0.5
        train_loss_list.append(train_loss)
        print('epoch:{} train_loss:{}'.format(epoch, train_loss))
        snap_shot = {'state_dict': model.state_dict()}
        if os.path.exists(MODEL_ROOT) == False:
            os.mkdir(MODEL_ROOT)
        torch.save(snap_shot, MODEL_ROOT + 'snap_{}.pth.tar'.format(epoch))

        plt.figure()
        plt.plot(range(len(train_loss_list)), train_loss_list, label='train_loss')
        plt.legend()

        if os.path.exists(IMG_ROOT) == False:
            os.mkdir(IMG_ROOT)
        plt.savefig(IMG_ROOT+'train_loss.png')
        plt.close()

def val(args, model_path=MODEL_ROOT):
    val_dataset = Loader(args, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=256, num_workers=args.num_workers,
                            pin_memory=True, drop_last=False, shuffle=True)
    
    snapList = os.listdir(os.path.join(model_path))
    snapList.sort(key=lambda x: int(x.split('.')[0].split('_')[1]))
    val_loss_list = []
    best_loss = None
    best_model = None

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
        A = A.cuda()

    for snap in snapList:
        model = STGCN(num_nodes=307, num_features=3, num_timesteps_input=12, num_timesteps_output=1)

        if args.cuda:
            model = model.cuda()
        model.load_state_dict(torch.load(os.path.join(
            model_path, snap), map_location='cpu')['state_dict'])
        model.eval()
        loss_func = torch.nn.MSELoss()

        val_res_list = []
        val_label_list = []

        with torch.no_grad():
            for _, pack in enumerate(val_loader):
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
                # val_res_list.append(out.detach().cpu().numpy())
                # val_label_list.append(labels.detach().cpu().numpy())
            val_res_list = torch.Tensor(val_res_list)
            val_label_list = torch.Tensor(val_label_list)
            loss = (loss_func(val_res_list, val_label_list).item())**0.5
        val_loss_list.append(loss)
        print('model: {} val_loss: {}'.format(snap, loss))

        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_model = int(snap.split('.')[0].split('_')[1])
        
        
        if os.path.exists(VAL_ROOT) == False:
            os.mkdir(VAL_ROOT)
        plt.figure()
        plt.plot(range(len(val_loss_list)), val_loss_list, label='val_loss')
        plt.legend()
        plt.savefig('./'+VAL_ROOT+'/val_loss.png')
        plt.close()
    
    print('best loss: {}  model: {}'.format(best_loss, best_model))

def test(args, model_path=MODEL_ROOT):
    test_dataset = Loader(args, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=24, num_workers=args.num_workers,
                             pin_memory=True, drop_last=False, shuffle=False)
    
    snapList = os.listdir(os.path.join(model_path))
    snapList.sort(key=lambda x: int(x.split('.')[0].split('_')[1]))

    best_model = snapList[179]
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
    # model.load_state_dict(torch.load(os.path.join(
    #     model_path, best_model), map_location='cpu')['state_dict'])
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
    idx = (y_true>1).nonzero()
    # return np.mean(np.abs(y_true[idx] - y_pred[idx]) / y_true[idx])
    return np.mean(np.abs(y_true - y_pred) / y_true)

def MAE(y_true, y_pred):
    return np.mean(abs(y_pred-y_true))

if __name__ == "__main__":
    args = p_parse()
    print(args)
    #train(args)
    #val(args)
    test(args)
