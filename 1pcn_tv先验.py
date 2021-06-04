import numpy as np
from matplotlib import pyplot as plt

samp_num = 100000

np.random.seed(42)
dimension = 23
stateS = np.zeros((dimension, samp_num))
beta = 0.005
lam = 500
plt.close('all')

x = np.linspace(0, 1, dimension)
interval0 = [1 if (i < 1//3) else 0 for i in x]
interval1 = [1 if (i >= 1/3 and i < 2/3) else 0 for i in x]
interval2 = [1 if (i >= 2/3) else 0 for i in x]
u = 0 * interval0 + 1 * interval1 + 0*interval2
n = np.random.normal(0, 0.02, dimension)
data = u + n
Y_arr = np.zeros((dimension, 1))
for i in range(dimension):
    Y_arr[i, 0] = data[i]
# print(u)
# print(Y_arr)


def DX_1(xy):
    DX1 = 0
    for z in range(dimension - 1):
        dx_1 = abs(xy[z + 1] - xy[z])
        DX1 = DX1 + dx_1
    return DX1


# delta_t = 1/dimension
# d = 0.02
# r = 0.01
# 定义协方差矩阵K
# K = np.zeros((dimension, dimension))
# for i in range(dimension):
#     for j in range(dimension):
#         K[i, j] = r * np.exp(-0.5 * (abs(i-j) * delta_t / d)**2)
# I = np.identity(dimension)
# Sigma = K + 0.02**2 * I

# 定义target distribution
# def p(y_arr, u_arr):
#     part_u = (u_arr.T.dot(np.linalg.inv(K))).dot(u_arr)
#     part_y = ((y_arr - u_arr).T.dot(np.linalg.inv(Sigma))).dot(y_arr - u_arr)
#     X = 0.5 * (part_u+part_y)+lam * DX_1(u_arr)
#     return -1*X[0, 0]


def p(y_arr, u_arr):
    cov1_inv = 1 / (0.02 ** 2)  # p(y|x)的方差的逆
    X = 0.5 * ((y_arr - u_arr).T.dot(y_arr - u_arr)) * cov1_inv + lam * DX_1(u_arr)
    return -1*X[0, 0]


# mn代表y的数据（已经知道），xy是未知的x的状态

arr = np.ones((dimension, 1)) * 0.1
accept_num = 0
A = np.zeros((dimension, 1))
for i in range(samp_num):
    Z = np.random.randn(dimension, 1)
    # mean = np.zeros(dimension)
    # cov = K
    # Z = np.random.multivariate_normal(mean, cov, (1,))
    arr_star = (1 - beta ** 2) * arr + beta * Z
    # arr_star = (1 - beta**2)*arr + beta * Z.T
    # arr是x，arr_star是x*，作为下一步的候选值。都是d*1矩阵
    px_part = p(Y_arr, arr_star) - p(Y_arr, arr)
    # qx_part = q(arr, arr_star)-q(arr_star, arr)
    alpha1 = min(1, np.exp(px_part))
    # print(alpha1)
    m = np.random.rand(1)[0]
    if m < alpha1:
        arr = arr_star
        accept_num += 1

    stateS[:, [i]] = arr
stateS = stateS[:, 3000:]
for i in range(dimension):
    A[i] = np.mean(stateS[i])

accept_rate = accept_num / samp_num * 100
print(beta, accept_rate)
# print(A)
plt.figure(figsize=(10, 5))
plt.plot(x, u, 'k', lw=3, label='x')
plt.plot(x, Y_arr, '.k', label='y=x+n')
plt.plot(x, A, 'k', label='x*')  # 可能有问题
plt.legend()
plt.title('Model and data')
plt.show()
