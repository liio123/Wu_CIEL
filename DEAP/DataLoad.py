import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold

# 加载数据
data_de_dir = 'G:\XXX/2025NIPS\DEAP特征\DE/'
data_pcc_dir = 'G:\XXX/2025NIPS\DEAP特征\PCC/'
data_de_list = os.listdir(data_de_dir)
data_pcc_list = os.listdir(data_pcc_dir)
data_de_list.sort(key=lambda x: x.split('.')[0])
data_pcc_list.sort(key=lambda x: x.split('.')[0])

X_de_test = np.load(data_de_dir+data_de_list[i]).astype(np.float32)
X_pcc_test = np.load(data_pcc_dir+data_pcc_list[i]).astype(np.float32)
X_de_test = torch.tensor(X_de_test, dtype=torch.float32)
X_pcc_test = torch.tensor(X_pcc_test, dtype=torch.float32)

# data_de_list.remove(data_de_list[i])
# data_pcc_list.remove(data_pcc_list[i])
#
# train_data_de_list = []
# train_data_pcc_list = []
# it_de = iter(data_de_list)
# it_pcc = iter(data_pcc_list)
# for idx in it_de:
#     data = np.load(data_de_dir+idx)
#     train_data_de_list.append(data)
# for idx in it_pcc:
#     data = np.load(data_pcc_dir+idx)
#     train_data_pcc_list.append(data)
# X_de_train = np.concatenate(train_data_de_list).astype(np.float32)
# X_pcc_train = np.concatenate(train_data_pcc_list).astype(np.float32)
# X_de_train = torch.tensor(X_de_train, dtype=torch.float32)
# X_pcc_train = torch.tensor(X_pcc_train, dtype=torch.float32)
# print(X_pcc_train.shape)

labels = np.load("G:\XXX/2025NIPS\DEAP特征/label/dominance/all.npy")

# Y_train = np.concatenate((labels[0:i*400,:],labels[(i+1)*400:,:]),axis=0)
# Y_train = torch.tensor(Y_train, dtype=torch.int64).squeeze_(1)
#
Y_test = labels[i*400:(i+1)*400,:]
Y_test = torch.tensor(Y_test, dtype=torch.int64).squeeze_(1)
print(Y_test)

MyDataset = TensorDataset(X_de_test, X_pcc_test, Y_test)
kfold = KFold(n_splits=5, shuffle=True)

# train_dataset =TensorDataset(X_de_train, X_pcc_train, Y_train)
# test_dataset = TensorDataset(X_de_test, X_pcc_test, Y_test)

# print("训练集测试集已划分完成............")
batch_size = 80
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
print("dataloader已完成装载............")
