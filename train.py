import numpy as np
from sklearn.datasets import load_digits
import scipy.io as scio
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import f1_score
from utils import *
from load_data import *
from model import *
import torch

use_gpu = torch.cuda.is_available()
device = torch.device('cuda:0')
num_epochs = 50

def data_load(Dataname):
    if Dataname in ['Mnist']:
        # train
        path = '../data/{}/{}_train.mat'.format(Dataname, Dataname)
        data = scio.loadmat(path)
        labels = data['Y']
        labels = labels.reshape(-1, )
        train_labels = process_y(labels, num_classes=max(labels) - min(labels) + 1)
        train_X = data['X']
        train_X = train_X.T
        # train_X = process_x(np.transpose(train_X))

        # test
        path = '../data/{}/{}_test.mat'.format(Dataname, Dataname)
        data = scio.loadmat(path)
        labels = data['Y']
        labels = labels.reshape(-1, )
        test_labels = process_y(labels, num_classes=max(labels) - min(labels) + 1)
        test_X = data['X']
        test_X = test_X.T
        # test_X = process_x(np.transpose(test_X))
        c = train_labels.shape[0]
        d = train_X.shape[0]

        X_train = torch.FloatTensor(train_X.T)
        y_train = torch.FloatTensor(train_labels)

        X_test = torch.FloatTensor(test_X.T)
        y_test = torch.FloatTensor(test_labels)

        return X_train, X_test, y_train, y_test, c, d
    elif Dataname in ['FashionMnist', 'cifar10']:
        # train
        path = '../data/{}_train.mat'.format(Dataname)
        data = scio.loadmat(path)
        labels = data['train_label']
        labels = labels.reshape(-1, )
        train_labels = process_y(labels, num_classes=max(labels) - min(labels) + 1)
        train_X = data['train_data']
        # train_X = process_x(np.transpose(train_X.T))

        # test
        path = '../data/{}_test.mat'.format(Dataname)
        data = scio.loadmat(path)
        labels = data['test_label']
        labels = labels.reshape(-1, )
        test_labels = process_y(labels, num_classes=max(labels) - min(labels) + 1)
        test_X = data['test_data']
        # test_X = process_x(np.transpose(test_X.T))
        c = train_labels.shape[0]
        d = train_X.shape[1]

        X_train = torch.FloatTensor(train_X)
        y_train = torch.FloatTensor(train_labels)

        X_test = torch.FloatTensor(test_X)
        y_test = torch.FloatTensor(test_labels)
    else:
        path = '../data/{}.mat'.format(Dataname)
        data = scio.loadmat(path)
        labels = data['Y'].astype(int)
        labels = labels.reshape(-1, )
        labels = process_y(labels, num_classes=max(labels) - min(labels) + 1)
        if Dataname in ['mnist_mini', 'att40']:
            X = data['X']
        else:
            X = data['X'].T
        # X = process_x(X)

        X = (X).astype(np.float)

        c = labels.shape[0]
        d = X.shape[0]
        labels = np.argmax(labels, axis=0)

        X_train, X_test, y_train, y_test = train_test_split(X.T, labels, test_size=0.3, random_state=42)
        y_train = process_y(y_train, num_classes=c)
        y_test = process_y(y_test, num_classes=c)

        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)

        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)

    return X_train, X_test, y_train, y_test, c, d

def fit(dimension_list, Index, X_train, X_test, y_train, y_test, d, c):
    model = Manifold_NN(dimension_list, lamda=0.1, gama=0.3, use_gpu=use_gpu, device=device)
    if use_gpu:
        model = model.to(device)
    index = 0
    Train_Loss = []
    Train_Acc = []
    Test_Loss = []
    Test_Acc = []
    F1 = []
    for i in range(num_epochs):
        print('*' * 15, 'Epoch {}'.format(i), '*' * 15)
        model.eval()
        if model.use_gpu:
            X_train = X_train.to(model.device)
            y_train = y_train.to(model.device)
        # train
        emb = model(X_train.T)
        layer_num = len(d_list) - 1
        for j in range(layer_num):
            layer_loss = model.loose_backward_layer(y_train, index=Index[index])
            index = (index + 1) % len(Index)

        train_loss = model.cal_loss(y_train)
        # train
        emb = model(X_train.T)
        pred = torch.argmax(emb, dim=0)
        train_acc = (pred == torch.argmax(y_train, dim=0)).float().mean()
        # test
        model.eval()
        if model.use_gpu:
            X_test = X_test.to(model.device)
            y_test = y_test.to(model.device)
        emb = model(X_test.T)
        test_loss = model.cal_loss(y_test)
        pred = torch.argmax(emb, dim=0)
        test_acc = (pred == torch.argmax(y_test, dim=0)).float().mean()
        if use_gpu:
            gt = torch.argmax(y_test, dim=0).cpu().numpy()
            pd = pred.cpu().numpy()
            f1 = f1_score(gt, pd, average='macro')
        print('train_loss: {}, train_ACC: {}, test_loss: {}, test_ACC: {}, test_F1: {}'.
              format(train_loss, train_acc, test_loss, test_acc, f1))
        Train_Loss.append(train_loss.item())
        Train_Acc.append(train_acc.item())
        Test_Loss.append(test_loss.item())
        Test_Acc.append(test_acc.item())
        F1.append(f1)
    return max(Test_Acc), max(F1)

if __name__ == "__main__":
    ATT40 = "att40"
    WAVEFORM = "waveform"
    UMIST = "umist"
    MNIST_MINI = "mnist_mini"
    Data_dir = [ATT40, WAVEFORM, UMIST, MNIST_MINI]
    Data_name = ["att40", "waveform", "umist", "mnist_mini"]

    dimension_list = [[64], [10, 4], [64, 32], [31, 16]]
    Index = [[2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1]]

    for i, data_dir in enumerate(Data_dir):
        X_train, X_test, y_train, y_test, c, d = data_load(data_dir)
        d_list = dimension_list[2]
        Idx = Index[2]
        d_list.insert(0, d)
        d_list.insert(len(d_list), c)
        acc, f1 = fit(d_list, Idx, X_train, X_test, y_train, y_test, d, c)
        print("Data: {}, Acc: {:.4f}, F1: {:.4f}"
              .format(Data_name[i], acc, f1))