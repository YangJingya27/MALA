import numpy as np
from matplotlib import pyplot as plt

samp_num = 50000

np.random.seed(42)
dimension = 60
lam = 0.5
stateS = np.zeros((dimension, samp_num))
beta = 0.01
plt.close('all')

nx = dimension
x = np.zeros(nx)
x[:nx // 2] = 1
x[nx // 2:3 * nx // 4] = -0.5
n = np.random.normal(0, 0.1, nx)
data = x + n
Y_arr = np.zeros((dimension, 1))
for i in range(dimension):
    Y_arr[i, 0] = data[i]


def DX_1(xy):
    d = len(xy)
    DX1 = 0
    for z in range(d - 1):
        dx_1 = abs(xy[z + 1] - xy[z])
        DX1 = DX1 + dx_1
    return DX1


# D = np.zeros([dimension, dimension])
# for i in range(1, dimension):
#     D[i, i-1] = -1
#     D[i, i] = 1


# def gradient_U(mn, xy):  # 直接返回一个梯度向量
#     DX = D.dot(xy)
#     B = (DX.T.dot(DX)+0.001)**0.5
#     gradient = 1/(2*0.1**2) * -2 * (mn-xy) + lam*(D.T.dot(D)).dot(xy) / B
#     return gradient


# mn代表y的数据（已经知道），xy是未知的x的状态
# 定义proposal distribution，q(xy_1|xy) ####################看到这里了
# def q(xy_1, xy):
#     miu = (1-beta**2)**0.5*xy
#     cov_inv = 1/beta**2
#     X = cov_inv*((xy_1 - miu).T.dot(xy_1 - miu))
#     prob = -0.5 * X
#     return prob[0, 0]


# mn代表y的数据（已经知道），xy是未知的x的状态
# 定义target distribution
def p(mn, xy):
    cov1_inv = 1 / (0.1 ** 2)  # p(y|x)的方差的逆
    X = 0.5*((mn - xy).T.dot(mn - xy))*cov1_inv + lam*DX_1(xy)
    return -1*X[0, 0]


# mn代表y的数据（已经知道），xy是未知的x的状态

arr = np.ones((dimension, 1)) * 0.1
accept_num = 0
A = np.zeros((dimension, 1))
for i in range(samp_num):
    Z = np.random.randn(dimension, 1)
    arr_star = (1 - beta**2)*arr + beta * Z
    # arr是x，arr_star是x*，作为下一步的候选值。都是d*1矩阵
    px_part = p(Y_arr, arr_star) - p(Y_arr, arr)
    # qx_part = q(arr, arr_star)-q(arr_star, arr)
    alpha1 = min(1, np.exp(px_part))
    # print(alpha1)
    u = np.random.rand(1)[0]
    if u < alpha1:
        arr = arr_star
        accept_num += 1

    stateS[:, [i]] = arr
# stateS = stateS[:, 3000:]
for i in range(dimension):
    A[i] = np.mean(stateS[i])

accept_rate = accept_num / samp_num * 100
print(beta, accept_rate)

plt.figure(figsize=(10, 5))
plt.plot(x, 'k', lw=3, label='x')
plt.plot(Y_arr, '.k', label='y=x+n')
plt.plot(A, 'k', label='x*')
plt.legend()
plt.title('Model and data')
plt.show()
