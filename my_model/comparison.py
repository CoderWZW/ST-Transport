from loader.XGBdataloader import XGBLoader
from Glob.glob import p_parse
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
#from utils.lossFunction import Loss

from sklearn.linear_model import Lasso, LassoCV, MultiTaskLassoCV
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor, XGBRFRegressor, XGBModel

from statsmodels.tsa.arima.model import ARIMA

def getData(args):
    data = np.load(args.data_path, allow_pickle=True)['data'][:,:,0]
    print(data.shape)
    data = data.reshape(data.shape[0], 307)
    print(data.shape)
    train = data[:-19*288]
    test = data[-19*288:]
    print(train.shape)
    print(test.shape)
    print('data split')
    return train, test

def getData4XGBoost(args):
    train_dataset = XGBLoader(args, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True, shuffle=False)
    train_seq_list = []
    train_label_list = []
    for _, pack in enumerate(train_loader):
        pack = list(map(lambda item: item.numpy(), pack))
        seqs = pack[:-1]
        labels = pack[-1]
        train_seq_list.append(seqs)
        train_label_list.append(labels)
    train_seq_list = torch.Tensor(train_seq_list)
    train_label_list = torch.Tensor(train_label_list)

    train_seq_list = train_seq_list.view(train_seq_list.shape[0], train_seq_list.shape[1], -1)
    train_seq_list = train_seq_list.permute(0,2,1)
    train_seq_list = train_seq_list.contiguous().view(train_seq_list.shape[0]*train_seq_list.shape[1], -1).numpy()

    train_label_list = train_label_list.view(train_label_list.shape[0], train_label_list.shape[1], -1)
    train_label_list = train_label_list.permute(0,2,1)
    train_label_list = train_label_list.contiguous().view(train_label_list.shape[0]*train_label_list.shape[1], -1).numpy()

    print(train_seq_list.shape)
    print(train_label_list.shape)


    test_dataset = XGBLoader(args, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers,
                             pin_memory=True, drop_last=True, shuffle=False)
    test_seq_list = []
    test_label_list = []
    for _, pack in enumerate(test_loader):
        pack = list(map(lambda item: item.numpy(), pack))
        seqs = pack[:-1]
        labels = pack[-1]
        test_seq_list.append(seqs)
        test_label_list.append(labels)
    test_seq_list = torch.Tensor(test_seq_list)
    test_label_list = torch.Tensor(test_label_list)
    
    test_seq_list = test_seq_list.view(test_seq_list.shape[0], test_seq_list.shape[1], -1)
    test_seq_list = test_seq_list.permute(0,2,1)
    test_seq_list = test_seq_list.contiguous().view(test_seq_list.shape[0]*test_seq_list.shape[1], -1).numpy()

    test_label_list = test_label_list.view(test_label_list.shape[0], test_label_list.shape[1], -1)
    test_label_list = test_label_list.permute(0,2,1)
    test_label_list = test_label_list.contiguous().view(test_label_list.shape[0]*test_label_list.shape[1], -1).numpy()

    print(test_seq_list.shape)
    print(test_label_list.shape)

    return train_seq_list, train_label_list, test_seq_list, test_label_list

def MAPE(y_true, y_pred):
    idx = (y_true>5).nonzero()
    return np.mean(np.abs(y_true[idx] - y_pred[idx]) / y_true[idx])

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

def RMSE(y_pred, y_true):
    return np.mean((y_true - y_pred)**2)**0.5

def ridgeRegression(args):
    train, test = getData(args)
    print(len(train))
    res = []
    trainX = [[i] for i in range(40*288)]
    testX = [[i] for i in range(40*288,59*288)]
    trainX = np.array(trainX)
    testX = np.array(testX)

    print(len(trainX),len(trainX[0]))
    print(len(testX))
    for i in range(307):
        model = Ridge(alpha=0.001,normalize=True)
        model.fit(trainX, train[:,i])
        pred = model.predict(testX)
        res.append(pred)
    res = np.array(res)
    res = res.transpose((1,0))

    print('Ridge: ')
    print('RMSE: {}'.format(RMSE(y_pred=res, y_true=test)))
    print('MAPE: {}'.format(MAPE(y_pred=res, y_true=test)))
    print('MAE: {}'.format(MAE(y_pred=res, y_true=test)))

def lassoRegression(args):
    train, test = getData(args)
    res = []
    trainX = [[i] for i in range(40*288)]
    testX = [[i] for i in range(40*288,59*288)]
    trainX = np.array(trainX)
    testX = np.array(testX)
    
    for i in range(307):
        model = Lasso(alpha=0.001, normalize=True, precompute=False, warm_start=True, copy_X=False, max_iter=5000)
        model.fit(trainX, train[:,i])
        pred = model.predict(testX)
        res.append(pred)
    res = np.array(res)
    res = res.transpose((1,0))
    print(res.shape)

    print('Lasso: ')
    print('RMSE: {}'.format(RMSE(y_pred=res, y_true=test)))
    print('MAPE: {}'.format(MAPE(y_pred=res, y_true=test)))
    print('MAE: {}'.format(MAE(y_pred=res, y_true=test)))

def XGBoost(args):
    print('XGBOOST:')
    trainX, trainY, testX, testY = getData4XGBoost(args)
    print(trainX.shape)
    print(trainY.shape)
    print(testX.shape)
    print(testY.shape)
    print('data prepare')
    model = XGBRegressor(max_depth=200, n_jobs=-1, objective='reg:linear', booster='gblinear',learning_rate=0.1)
    model.fit(trainX, trainY)
    pred = model.predict(testX)
    pred = pred.reshape((-1,1))
    print(pred.shape)
    print(testY.shape)
    pred = pred.reshape((2592,307,1))
    testY = testY.reshape((2592,307,1))

    print('XGBoost: ')
    print('RMSE: {}'.format(RMSE(y_pred=pred, y_true=testY)))
    print('MAPE: {}'.format(MAPE(y_pred=pred, y_true=testY)))
    print('MAE: {}'.format(MAE(y_pred=pred, y_true=testY)))

def arima(args):
    train, test = getData(args)
    train = train.transpose((1,0))
    test = test.transpose((1,0))
    res_list = []
    for i in range(121):
        model = ARIMA(train[i], order=(5,1,5)).fit()
        res = model.forecast(744)
        res_list.append(res)
    res_list = np.array(res_list)

    # res = np.load('./save/arima_res.npy',allow_pickle=True)
    test = test.reshape((test.shape[0],11,11))
    print(res.shape)
    print(test.shape)
    print('ARIMA: ')
    # print('RMSE: {}'.format(RMSE(y_pred=res, y_true=test)))
    print('MAPE: {}'.format(MAPE(y_pred=res, y_true=test)))
    # print('MAE: {}'.format(MAE(y_pred=res, y_true=test)))
        


if __name__ == "__main__":
    
    args = p_parse()
    ridgeRegression(args)
    lassoRegression(args)
    XGBoost(args)
    #arima(args)
    