import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch import nn
from Index_calculation import testclass
from DEAP.Model import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time

batch_size = 80
num_epochs = 200
learning_rate = 0.005
channel_num = 32
band_num = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def divide_data_based_on_labels(datas, labels):

    # print(datas.shape)
    # print(labels.shape)

    index_dict = {}
    datas_dict = {}

    # print(labels)
    for i, num in enumerate(labels):
        # new_num = one_hot_to_number(num)
        num = num.item()
        if num not in index_dict:

            index_dict[int(num)] = [i]
            datas_dict[int(num)] = [datas[i,:]]
            # print("aaa")
        else:
            index_dict[int(num)].append(i)
            datas_dict[int(num)].append(datas[i, :])
            # print("bbb")


    unique_labels = list(index_dict.keys())
    return datas_dict, unique_labels

def tSNE_2D(datas, labels, label_names):
    datas = datas.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    # print(type(datas))
    datas = datas.reshape(datas.shape[0], -1)

    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    datas_tsne = tsne.fit_transform(datas)
    # print(type(datas_tsne))

    print(datas_tsne.shape)
    print(labels.shape)

    datas_tsne, unique_labels = divide_data_based_on_labels(datas_tsne, labels)
    # print("a:", unique_labels)
    # print(datas_tsne)
    # print(np.array(unique_labels).shape)

    # datas_tsne = np.array(datas_tsne[1])
    # print(datas_tsne[:,0])

    # markers = ['*', 'o', 'v', 'd', 'x']
    # markers = ['o', 'o', 'o', 'o', 'o']
    # markers = ['.', '.', '.', '.', '.']

    markers = ['*', 'o', 'v', '+']

    # Plot the data in 2D
    plt.figure(figsize=[15, 10])
    plt.xticks([])
    plt.yticks([])
    index = 0
    for label in unique_labels:
        print(label)
        datas_array_tsne = np.array(datas_tsne[label])
        print(label_names)
        plt.scatter(datas_array_tsne[:, 0], datas_array_tsne[:, 1], marker=markers[index])
        index += 1
    # plt.title(title)
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    # plt.legend(prop={'family' : 'Times New Roman', 'size' : 22}, labels=["happy", "neural", "sad", "fear"])
    plt.savefig('G:/吴老师的论文/2025NIPS/模型可视化/TSNE_valence.pdf', dpi=120, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

def tSNE_3D(datas, labels, label_names):
    datas = datas.cpu().detach().numpy()
    labels = labels.cpu()
    # print(type(datas))
    # print(datas.shape)
    datas = datas.reshape(datas.shape[0], -1)


    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=3, random_state=42)
    datas_tsne = tsne.fit_transform(datas)

    # print(type(datas_tsne))

    # print(datas_tsne.shape)

    datas_tsne, unique_labels = divide_data_based_on_labels(datas_tsne,labels)

    # datas_tsne = np.array(datas_tsne[1])
    # print(datas_tsne[:,0])

    markers = ['.','o','d']
    # markers = ['o', 'o', 'o', 'o', 'o']

    # Plot the data in 2D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    index = 0
    for label in unique_labels:
        print(label)
        datas_array_tsne = np.array(datas_tsne[label])
        ax.scatter(datas_array_tsne[:, 0], datas_array_tsne[:, 1], datas_array_tsne[:, 2], label=label_names[label], marker=markers[index])

        index += 1

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    plt.legend()
    plt.show()

start_time = time.time()
model = PINN_VMC(hidden_dim=32, output_dim=2, xdim = [batch_size, channel_num, band_num], kadj=2, num_out=16, dropout=0.5).to(device)
model.load_state_dict(torch.load("G:/吴老师的论文/2025NIPS/保存的模型/DEAP/valence/1.pth"))

#测试集-DE
X_test = torch.tensor(np.load("G:\吴老师的论文/2025NIPS\DEAP特征\DE/1.npy").real.astype(float), dtype=torch.float)
PCC_test = torch.tensor(np.load("G:\吴老师的论文/2025NIPS\DEAP特征\PCC/1.npy").real.astype(float), dtype=torch.float)

# X_test = X_test[100:200,:,:]


#测试集-label
i=4
labels = np.load("G:\吴老师的论文/2025NIPS\DEAP特征/label/valence/all.npy")

# Y_train = np.concatenate((labels[0:i*400,:],labels[(i+1)*400:,:]),axis=0)
# Y_train = torch.tensor(Y_train, dtype=torch.int64).squeeze_(1)
#
Y_test = labels[i*400:(i+1)*400,:]
Y_test = torch.tensor(Y_test, dtype=torch.int64).squeeze_(1)
# testData = TensorDataset(X_test_DE, X_test_PCC, X_test_DE, X_test_PCC, Y_test)
#
# test_dataloader = DataLoader(testData, batch_size=512, shuffle=True, drop_last=True)


test_len = X_test.shape[0]

test_loss_plt = []
test_acc_plt = []
Test_Loss_list = []
Test_Accuracy_list = []
total_test_acc = 0

X_test = X_test.to(device)
PCC_test = PCC_test.to(device)

Y_test = Y_test.to(device)

output = model(X_test, PCC_test)

test_acc = (output.argmax(dim=1) == Y_test).sum()
# TP_TN_FP_FN = G.Compute_TP_TN_FP_FN(test_label, label, matrix)
test_acc_plt.append(test_acc)
total_test_acc += test_acc
end_time = time.time()

print("Test Accuracy: {:.4f}".format(total_test_acc / test_len), "Running time:",end_time-start_time, "s")
#
# print(output.argmax(dim=1).shape)
# print(Y_test)
# from matplotlib import font_manager
# # my_font = font_manager.FontProperties(fname="C:\Windows\Fonts\MSYHL.TTC")
# t, p = stats.ttest_ind(np.array(output.argmax(dim=1)),np.array(Y_test))
# print(p)
#
# x = range(0,600)
#
# # 绘制图像 label 设置标签 color设置颜色 linestyle 设置线条 linewidth 设置线条粗细 alpha设置透明度
# plt.plot(x, np.array(output.argmax(dim=1)), label='predicted_label', color='red', linestyle='--')
# plt.plot(x, np.array(Y_test), label='true_label', color='green', linestyle='--')
#
# # 设置X刻度
# # _xtick_labels = ['{}岁'.format(i) for i in x]
# # plt.xticks(x, _xtick_labels, rotation=45, fontproperties=my_font)
#
# # 设置X，Y轴标签
# plt.xlabel('Number of Samples')
# plt.ylabel('Class')
# # plt.title('在11岁到30岁每年交往女(男)朋友的数目', fontproperties=my_font)
# plt.yticks([i for i in range(0, 2) if i % 1 == 0])
# # plt.yticks(range(0,9)) 设置网格单位间距
#
# # 绘制网格
# # plt.grid(alpha=0.8)  # alpha调整网格透明度
#
# # 添加图例 先写label参数 再用plt.lenged()
# plt.legend(loc='lower right')  # 显示中文设置prop参数 loc='upper left'将图例移到左上方
# plt.savefig("E:/博士成果/跟吴老师的第一篇文章/配图/label_1.pdf")
# plt.savefig("E:/博士成果/跟吴老师的第一篇文章/配图/label_1.tif")
# # 展示图形
# plt.show()

tSNE_2D(output, Y_test, "012")
# tSNE_3D(output, Y_test, "012")