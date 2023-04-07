import numpy as np
from scipy.stats import ortho_group
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from torch.nn.parameter import Parameter
from utils import *

class Manifold_NN(nn.Module):
    def __init__(self, dimension_list, lamda=0.1, k=1.0, gama=1.0, use_gpu=True, device='cuda:0'):
        super(Manifold_NN, self).__init__()
        self.weight_list = dimension_list
        self.lamda = lamda  # activation param
        self.k = k  # activation ratio
        self.gama = gama  # regularization parameter
        self.use_gpu = use_gpu
        self.device = device
        self.build_network()


    '''
    def __init__(self, input_nodes=1, hidden1_nodes=4, hidden2_nodes=4, output_nodes=1, lamda=0.1, k=1.0, gama=1.0):
        super(Manifold_NN, self).__init__()
        self.input_nodes = input_nodes  # d0
        self.hidden1_nodes = hidden1_nodes  # d1
        self.hidden2_nodes = hidden2_nodes  # d2
        self.output_nodes = output_nodes  # d3
        self.lamda = lamda # regularization parameter
        self.k = k # activation ratio
        self.gama = gama #
        self.weight_list = [input_nodes, hidden1_nodes, hidden2_nodes, output_nodes]
        self.build_network()
    '''


    def build_network(self):
        '''utilize the orthogonal matrix init the W, only reduction demension: d^m < d^(m-1)'''
        i = 0
        n = len(self.weight_list)
        L = []
        W = []
        B = []
        while i < n - 1:
            '''reduction: d1 < d0'''
            d0 = self.weight_list[i]
            d1 = self.weight_list[i + 1]
            if self.use_gpu:
                w = Parameter(torch.FloatTensor(d0, d1).cuda(self.device))
                b = Parameter(torch.FloatTensor(d1, 1).cuda(self.device))
            else:
                w = Parameter(torch.FloatTensor(d0, d1))
                b = Parameter(torch.FloatTensor(d1, 1))
            '''init with std'''
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            b.data.uniform_(-stdv, stdv)
            W.append(w)
            B.append(b)
            i = i + 1
        self.W = W
        self.b = B


    def forward(self, x):
        # inputs: d0 * n
        i = 0
        layer_num = len(self.W)
        input = x
        X = []  # save the input
        Z = []  # save the latent res without activation
        A = []  # save the final res with activation
        while i < layer_num:
            X.append(input)
            if i == layer_num - 1:
                '''last layer without activation function'''
                z = torch.mm(self.W[i].T, input) + self.b[i]
                z = F.normalize(z, dim=0)
                a = z
                Z.append(z)
                A.append(a)  # aim to keep the same length
            else:
                z = torch.mm(self.W[i].T, input) + self.b[i]
                a = torch.sigmoid(z)
                a = F.normalize(a, dim=0)
                Z.append(z)
                A.append(a)
            input = a
            i = i + 1
        self.X = X
        self.Z = Z
        self.A = A
        return self.A[-1]

    def cal_loss(self, Y):
        '''cal the BP loss of the model with the regularization term'''
        H_m = self.A[-1]
        loss = torch.norm(H_m - Y)
        loss = 0.5 * (loss ** 2)
        '''
        for w in self.W:
            loss = loss + self.lamda * (torch.norm(w))**2
        '''
        # loss = loss / Y.shape[1]
        return loss

    def cal_svmloss(self, Y):
        '''cal the svmloss'''
        X = self.X[-1].detach()
        W = self.W[-1]
        b = self.b[-1]
        M = self.cal_M(X, Y, W, b)
        alphas, lamda = self.cal_alphas(X, Y, M, W, b)
        M = M.detach()
        alphas = alphas.detach()
        lamda = lamda.detach()
        Loss = self.cal_obj(X, Y, self.gama, W, b, alphas, lamda, M)
        # loss = torch.mm(self.W.t(), X) + self.b - Y
        # loss = torch.pow(loss, 2)
        # loss = torch.sum(loss)
        # Loss = loss + self.gama * (torch.norm(self.W))**2
        return Loss


    def get_params(self):
        params = self.W + self.b
        return params

    def cal_M(self, X, Y, W, b):
        '''
        M = Y .* (W^T * X) + Y .* b - 1
        :param X: feature matrix, d * n
        :param Y: label matrix, c * n
        :param W: weight matrix, d * c
        :param b: bias vector, c * 1
        :return:
        :param M: slack variable matrix, c * n
        '''
        M = Y * (torch.mm(W.t(), X)) + Y * b - 1
        M = torch.clamp(M, min=0)
        return M

    def cal_alphas(self, X, Y, M, W, b):
        '''
        the activation parameter for the model
        :param X: feature matrix, d * n
        :param Y: label matrix, c * n
        :param M: slack matrix, c * n
        :param W: weight matrix, d * c
        :param b: bias vector, c * 1
        :return:
        '''
        c, n = Y.shape
        k = round(self.k * n) - 2

        f = torch.mm(W.t(), X) + b - Y - Y * M
        f = torch.sum(f ** 2, axis=0) # sum according to col
        f = f.t()
        f_sorted = torch.sort(f)[0]
        lower_bound = 0.5 * (k * f_sorted[k] - torch.sum(f_sorted[:k]))
        upper_bound = 0.5 * (k * f_sorted[k + 1] - torch.sum(f_sorted[:k]))
        lamda = (lower_bound + upper_bound) / 2
        lamda = k / 2 * f_sorted[k + 1] - 0.5 * torch.sum(f_sorted[:k])
        t = torch.sum(f_sorted[:k]) + 2 * lamda
        t = t / (2 * k * lamda)
        alphas = t - f / (2 * lamda)
        alphas = torch.clamp(alphas, min=0.0)
        alphas = alphas.reshape(-1, 1)
        return alphas, lamda

    def cal_W(self, X, Y, alphas, M):
        '''
        cal the weight matrix, d * n
        :param X: feature matrix, d * n
        :param Y: label matrix, c * n
        :param alphas: activation parameter vector,
        :param M: slack variable, c * n
        :return:
        '''
        d, n = X.shape
        c, _ = Y.shape
        gama = self.gama
        weight_u = alphas / torch.sum(alphas)
        X_weight = torch.mul(X, weight_u.reshape(1, -1))
        X_weight_mean = torch.sum(X_weight, dim=1)
        XH = X - X_weight_mean.reshape(-1, 1)
        S = 0
        for i in range(XH.size(1)):
            temp = torch.mm(XH[:, i].reshape(-1, 1), XH[:, i].t().reshape(1,-1))
            S = S + alphas[i]*temp
        if self.use_gpu:
            S = S + gama * torch.eye(d).cuda(self.device)
        else:
            S = S + gama * torch.eye(d)

        m1 = torch.ge(S, S.t()).float()
        m2 = torch.gt(S.t(), S).float()
        S = S * m1 + S.t() * m2
        Z = Y + torch.mul(Y, M)
        if self.use_gpu:
            S_12 = tensor_1_2_inv(S.cpu()).to(self.device)
        else:
            S_12 = tensor_1_2_inv(S)

        '''solve the centralized method2, B=S^-1 X H D H^T Z^T= S^-1 (HX) D (HZ)^T'''
        Z_weight = torch.mul(Z, weight_u.reshape(1, -1))
        Z_weight_mean = torch.sum(Z_weight, dim=1)
        HZ = Z - Z_weight_mean.reshape(-1, 1)
        XHDHT = 0
        for i in range(XH.size(1)):
            temp = torch.mm(XH[:, i].reshape(-1, 1), HZ[:, i].t().reshape(1, -1))
            XHDHT = XHDHT + alphas[i] * temp
        B = torch.mm(S_12, XHDHT)
        U, sigma, V = torch.svd(B)  # torch.svd: return (U, sigma, V), numpy.linalg.svd: return (U, sigma, VT)
        Q = torch.mm(U, V.t())
        W = torch.mm(S_12, Q)
        b = torch.mm((Z - torch.mm(W.t(), X)), alphas) / torch.sum(alphas)
        b = b.reshape(-1, 1)
        return W, b


    def cal_obj(self, X, Y, gama, W, b, alphas, lamda, M):
        T = torch.mm(W.t(), X) + b - Y - torch.mul(Y, M)
        t2 = torch.pow(T, 2)
        t = torch.sum(t2, dim=0)  # sum according by row
        t = t ** 2
        # alphas = alphas.reshape((n, 1))
        #t = t.reshape((1, n))
        # a = torch.mul(t, alphas)
        a = t.reshape(-1,1) * alphas
        obj = torch.sum(a) + lamda * (torch.norm(alphas))**2 + gama * torch.pow(torch.norm(W), 2)
        return obj


    def optimize_svm(self, X, Y, W, bias, iteration=20):
        '''
        optimize the svm decision layer
        :param X: the feature matrix, d * n
        :param Y: the groundtruth label, c * n
        :param W: the weight matrix, d * c
        :param bias: the bias vector, c * 1
        :param iteration: the max iteration
        :return:
        '''
        gt = torch.argmax(Y, dim=0)
        d, n = X.shape
        err = 1
        iter = 1
        obj = []
        oldW = 0

        while err>10**-2 and iter<iteration:
            # update M
            M = self.cal_M(X, Y, W, bias)
            # update alpha
            alphas, lamda = self.cal_alphas(X, Y, M, W, bias)
            # alphas = torch.ones(size=[n, 1]) / n
            # if self.use_gpu:
            #     alphas = alphas.to(self.device)
            # lamda = 0
            # update W, b
            W, b = self.cal_W(X, Y, alphas, M)
            diff = torch.norm(input=(W - oldW))
            obj.append(self.cal_obj(X, Y, self.gama, W, b, alphas, lamda, M).item())
            print('diff: %.5f, obj: %.5f' % (diff.item(), obj[iter - 1]))
            if iter > 1:
                err = abs(obj[iter - 1] - obj[iter - 2])
            iter = iter + 1
            oldW = W

        return W, b, obj[-1]

    def layer_loss(self, X, Y, W, b, alpha=-1):
        '''
        cal the loss of the layer
        :param X: input matrix, d * n
        :param Y: output matrix, c * n
        :param W: weigth matrix, d * c
        :param b: bias vector, c * 1
        :param alpha: scaling parameter
        :return:
        :param loss
        '''
        if alpha > 0:
            loss = torch.mm(alpha * W.t(), X) + b - Y
        else:
            loss = torch.mm(W.t(), X) + b - Y
        l = loss.cpu().data.numpy()
        loss = torch.norm(loss)
        loss = loss ** 2
        loss = loss + self.gama * (torch.norm(W))**2
        return loss

    def cal_bias(self, W, X, Z, alpha=-1):
        '''
        cal the bias vector, c * 1
        :param W: weight matrix, d * c
        :param X: input matrix, d * n
        :param Z: ouput matrix without activation, c * n
        :param alpha: the scaling parameter
        :return:
        :param bias: bias vector, c * 1
        '''
        _, n = X.shape
        if alpha >= 0:
            bias = (Z - torch.mm(alpha * W.t(), X)) / n
        else:
            bias = (Z - torch.mm(W.t(), X)) / n
        if self.use_gpu:
            bias = torch.mm(bias, torch.ones(size=(n, 1), dtype=torch.float32).to(self.device))
        else:
            bias = torch.mm(bias, torch.ones(size=(n, 1), dtype=torch.float32))
        return bias

    def centrailize(self, X):
        '''
        cal the centralized matrix X
        :param X: data matrix, d * n
        :return:
        :param X_c: the centrilzed matrix, d * n
        '''
        _, n = X.shape
        X_sum = (torch.sum(X, dim=1)).reshape(-1, 1)
        X_mean = X_sum / n
        X_c = X - X_mean
        return X_c

    def optimize_ridge_regress(self, input, latent, output, weight, bias, activation=1, iteration=20):
        """
        optimize the individual layer without orthogonal layer, via ridge regression
        :param input: input of the layer, d * n
        :param latent: output of the layer without activation, c * n
        :param output: output of the layer, c * n
        :param weight: the weight matrix, d * c
        :param bias: the bias vector, c * 1
        :param activation: activation function type
        :return:
        """
        d, n = input.shape
        c = output.shape[0]
        # initialize the parameters
        err = 1
        iter = 1
        obj = []
        X = input
        Z = latent
        Y = output
        old_w = 0
        W = weight
        while err>10**-2 and iter<iteration:
            obj.append(self.layer_loss(X, Z, W, bias))
            diff = torch.norm(W - old_w)
            print('diff: %.5f, obj: %.5f' % (diff.item(), obj[iter - 1].item()))
            # update the bias
            bias = self.cal_bias(W, X, Z)
            '''relaxtion: min ||CX^TW-CZ^T||
                        let: CX^T=X_c, CZ^T=Y_c, don't cal C to reduce the complexity of model
                        C is centrailized matrix, it means minus the average'''
            X_c = self.centrailize(X)
            Y_c = self.centrailize(Z)
            # update the W
            if self.use_gpu:
                G = torch.mm(X_c, X_c.t()) + self.gama * torch.eye(d, dtype=torch.float32).to(self.device)
            else:
                G = torch.mm(X_c, X_c.t()) + self.gama * torch.eye(d, dtype=torch.float32)
            W = 2 * torch.mm(X_c, Y_c.t())
            W = torch.mm(torch.inverse(G+G.t()), W)
            '''
            x_norm = torch.mm(X_c, X_c.t())
            W = torch.mm(X_c, Y_c.t())
            W = torch.mm(torch.inverse(x_norm), W)
            '''
            if iter > 1:
                err = abs(obj[iter - 1].item() - obj[iter - 2].item())
            iter = iter + 1
        pred = torch.mm(W.T, X) + bias
        pred = torch.argmax(pred, dim=0)
        return W, bias, obj[-1]

    def normalize(self, data):
        '''
        normalize the data according to col
        :param data: d * n
        :return:
        '''
        a = torch.max(data, dim=0)
        range = torch.max(data, dim=0)[0] - torch.min(data, dim=0)[0] + 0.001 # avoid the 1
        a = (data - torch.min(data, dim=0)[0] + 0.0001) / range
        return (data - torch.min(data, dim=0)[0] + 0.0001) / range # avoid the 0

    def convert_activate(self, output, type=1):
        '''
        cal the convert of the actiavtion
        :param output: the output of the layer
        :param typer: the type of the activation
        :return:
        '''
        if type==1:
            res = output / (1-output)
            res = torch.log(res)
        elif type==2:
            res = (1+output) / (1-output)
            res = torch.log(res) / 2.0
        return res

    def convert_A(self, W, Y, b):
        '''
        cal the output of layer, back forward
        :param W: the weight matrix, d * c
        :param Y: the label matrix, c * n
        :param b: the bias vector, c * 1
        :return:
        :param output: the output of last layer == the input of now layer, c * n
        '''
        use_gpu = self.use_gpu
        device = self.device
        a = torch.mm(W, W.T).data.numpy()
        if use_gpu:
            t_1 = torch.mm(W, W.T) + 0.00001 * torch.eye(W.shape[0]).to(device)
        else:
            t_1 = torch.mm(W, W.T) + 0.00001 * torch.eye(W.shape[0])
        t_1_inv = torch.inverse(t_1)
        output = torch.mm(t_1, W)
        output = torch.mm(output, Y-b)
        return output

    def revise_latent(self, input):
        lower = torch.zeros(input.shape) + 0.0001
        upper = torch.ones(input.shape) - 0.0001
        if self.use_gpu:
            lower = lower.to(self.device)
            upper = upper.to(self.device)
        out = torch.where(input<1, input, upper)
        out = torch.where(out>0, out, lower)
        return out



    def loose_backward_layer(self, Y, index):
        '''
        only optimize the decision layer with SVM, other with closed-form ridge regression
        :param Y: the ground truth of the network
        :param index: the index of optimizing layer
        :return:
        '''
        print('*' * 10, 'Optimize layer {}:'.format(index), '*' * 10)
        layer_num = len(self.W)
        i = layer_num - 1
        input = self.X[i].detach()
        latent = Y
        output = Y
        W = self.W[i].detach()
        bias = self.b[i].detach()
        LOSS = []
        while i + 1 > index:
            output = torch.mm(W, (latent - bias))
            # output = self.convert_A(W, latent, bias)
            # output = self.cal_output(input, W, latent, bias)
            # latent = self.normalize(output)
            latent = F.softmax(output, dim=1)
            latent = self.revise_latent(latent)
            latent = self.convert_activate(latent)
            # latent = self.cal_output(input, W, latent, bias)
            # output = F.sigmoid(latent)
            i = i - 1
            if i < 0:
                break
            input = self.X[i].detach()
            W = self.W[i].detach()
            bias = self.b[i].detach()
        if index == layer_num:
            '''optimize the decision layer with SVM'''
            W, bias, loss = self.optimize_svm(input, output, W, bias, iteration=5)
            # W, bias, loss = self.optimize_ridge_regress(input, latent, output, W, bias, activation=1, iteration=20)
        else:
            '''optimize the others with ridge regression'''
            # W, bias, loss = self.optimize_svm(input, latent, W, bias, iteration=20)
            W, bias, loss = self.optimize_ridge_regress(input, latent, output, W, bias, activation=1, iteration=20)
        LOSS.append(loss)
        # update the parameter
        self.W[i] = Variable(W, requires_grad=True)
        self.b[i] = Variable(bias, requires_grad=True)
        return LOSS