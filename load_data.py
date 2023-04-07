import numpy as np
import scipy.io as scio
import torch
from sklearn.model_selection import train_test_split
import torchvision
from torchvision import transforms


YALE = 'Yale'
UMIST = 'UMIST'
THREE_RINGS = 'three_rings'

def load_data(name):
    path = './data/{}.mat'.format(name)
    data = scio.loadmat(path)
    labels = data['Y'].astype(int)
    labels = labels.reshape(-1, )
    labels = process_y(labels, num_classes=max(labels)-min(labels)+1)
    if name in ['mnist_mini', 'att40']:
        X=data['X']
    else:
        X = data['X'].T
    X = process_x(X)
    return X, labels

def process_y(Labels, num_classes):
    '''
    function: translate the (numbers, 1) into (types, numbers)
    :param Labels: the data label
    :return:
    Y: the data label (types, numbers)
    '''
    c = num_classes
    n = len(Labels)
    if np.min(Labels) > 0:
        Labels = Labels - np.min(Labels)
    Y = np.ones(shape=[c, n]) * 0
    for i in range(n):
        Y[Labels[i], i] = 1
    return Y

def one_hot(Y, num_classes):
    res = torch.zeros(len(Y), num_classes)
    res = res.scatter_(1, Y.view(-1, 1), 1)
    return res

def process_x(Data):
    '''
    function: normalize the data
    :param Data: input Data
    :return:
    X: normalized data
    '''
    # Min = np.min(Data)
    # Max = np.max(Data)
    # X = (Data - Min) / (Max - Min)
    col_mean = np.mean(Data, axis=0)
    col_std = np.std(Data, axis=0)
    col_max = np.max(Data, axis=0)
    col_min = np.min(Data, axis=0)
    X = (Data - col_mean.reshape(1,-1)) / col_std.reshape(1,-1)
    return X

def gen_data(filename, ratio):
    X, labels = load_data(filename)
    X = (X).astype(np.float)
    if filename == 'cifar10_train' or filename == 'cifar10_test':
        X = process_x(X)
    # X = process_x(X)

    c = labels.shape[0]
    d = X.shape[0]
    labels = np.argmax(labels, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X.T, labels, test_size=ratio, random_state=42)
    y_train = process_y(y_train, num_classes=c)
    y_test = process_y(y_test, num_classes=c)

    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)

    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    return X_train, X_test, y_train, y_test, c, d

def load_writtendata(name):
    # train
    path = './data/{}/{}_train.mat'.format(name, name)
    data = scio.loadmat(path)
    labels = data['Y']
    labels = labels.reshape(-1, )
    train_labels = process_y(labels, num_classes=max(labels)-min(labels)+1)
    train_X = data['X']
    # train_X = train_X.T
    train_X = process_x(np.transpose(train_X))

    #test
    path = './data/{}/{}_test.mat'.format(name, name)
    data = scio.loadmat(path)
    labels = data['Y']
    labels = labels.reshape(-1, )
    test_labels = process_y(labels, num_classes=max(labels) - min(labels) + 1)
    test_X = data['X']
    test_X = process_x(np.transpose(test_X))
    c = train_labels.shape[0]
    d = train_X.shape[0]

    X_train = torch.FloatTensor(train_X.T)
    y_train = torch.FloatTensor(train_labels)

    X_test = torch.FloatTensor(test_X.T)
    y_test = torch.FloatTensor(test_labels)

    return X_train, X_test, y_train, y_test, c, d

def load_fashion(dataname):
    '''
    train_dataset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST/',
                                                 train=True,
                                                 transform=transforms.ToTensor(),
                                                 download=True)

    test_dataset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST/',
                                                train=False,
                                                transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=60000,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=10000,
                                              shuffle=False)
    for i, (img, target) in enumerate(train_loader):
        train_data = (img.view(60000, -1)).numpy()
        train_label = target.numpy()
    scio.savemat('./data/{}_train.mat'.format(dataname), {'train_data': train_data, 'train_label': train_label})

    for i, (img, target) in enumerate(test_loader):
        test_data = (img.view(10000, -1)).numpy()
        test_label = target.numpy()
    scio.savemat('./data/{}_test.mat'.format(dataname), {'test_data': test_data, 'test_label': test_label})
    '''
    # train
    path = './data/{}_train.mat'.format(dataname)
    data = scio.loadmat(path)
    labels = data['train_label']
    labels = labels.reshape(-1, )
    train_labels = process_y(labels, num_classes=max(labels) - min(labels) + 1)
    train_X = data['train_data']
    train_X = process_x(np.transpose(train_X.T))

    # test
    path = './data/{}_test.mat'.format(dataname)
    data = scio.loadmat(path)
    labels = data['test_label']
    labels = labels.reshape(-1, )
    test_labels = process_y(labels, num_classes=max(labels) - min(labels) + 1)
    test_X = data['test_data']
    test_X = process_x(np.transpose(test_X.T))
    c = train_labels.shape[0]
    d = train_X.shape[1]

    X_train = torch.FloatTensor(train_X)
    y_train = torch.FloatTensor(train_labels)

    X_test = torch.FloatTensor(test_X)
    y_test = torch.FloatTensor(test_labels)
    return X_train, X_test, y_train, y_test, c, d
