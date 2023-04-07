import numpy as np
import torch
import matplotlib.pyplot as plt

def mat_1_2_inv(M):
    '''
    function: calculate the -1/2 of input matrix
    :param M: input matrix, numpy
    :return:
    M_12: the result
    '''
    eigen_val, eigen_vec = np.linalg.eig(M)
    eigen = np.diag(eigen_val ** (-0.5))
    M_12 = np.matmul(eigen_vec, eigen)
    M_12 = np.matmul(M_12, np.linalg.inv(eigen_vec)).astype(float)  # caculate the -1/2 of S_t
    return M_12

def tensor_1_2_inv(T):
    # eigen_val, eigen_vec = torch.eig(T, eigenvectors=True)
    # eigen = torch.diag(eigen_val[:, 0] ** (-0.5))
    # T_12 = torch.mm(eigen_vec, eigen)
    # T_12 = torch.mm(T_12, torch.inverse(eigen_vec))
    M_12 = mat_1_2_inv(T.numpy())
    T_12 = torch.FloatTensor(M_12)
    return T_12

def show_curve(ys, title):
    """
    plot curlve for Loss and Accuacy
    Args:
        ys: loss or acc list
        title: loss or accuracy
    """
    plt.figure()
    x = np.array(range(len(ys)))
    y = np.array(ys)
    plt.plot(x, y, c='b')
    plt.axis()
    plt.title('{} curve'.format(title))
    plt.xlabel('epoch')
    plt.ylabel('{}'.format(title))
    plt.show()