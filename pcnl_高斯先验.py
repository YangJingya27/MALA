import numpy as np
from matplotlib import pyplot as plt

samp_num = 100000

np.random.seed(42)
dimension = 23
stateS = np.zeros((dimension, samp_num))
delta = 0.001
d = 0.01
plt.close('all')

x = np.linspace(0, 1, dimension)
interval0 = [1 if (i < 1/3) else 0 for i in x]
interval1 = [1 if (1/3 <= i < 2/3) else 0 for i in x]
interval2 = [1 if (i >= 2/3) else 0 for i in x]
u = 0 * interval0 + 1 * interval1 + 0*interval2
n = np.random.normal(0, 0.02, dimension)
data = u + n
Y_arr = np.zeros((dimension, 1))
for i in range(dimension):
    Y_arr[i, 0] = data[i]


delta_t = 1/dimension
r = 0.2  # 0.2
# 定义协方差矩阵K
K = np.zeros((dimension, dimension))
for i in range(dimension):
    for j in range(dimension):
        K[i, j] = r * np.exp(-0.5 * (abs(i-j) * delta_t / d)**2)
Sigma = 0.02**2  # y的协方差矩阵(一个单位矩阵）
K_inv = np.linalg.inv(K)


# 定义target distribution 的指数部分（带负号）
def p(y_arr, u_arr):
    part_u = (u_arr.T.dot(K_inv)).dot(u_arr)
    part_y = ((y_arr - u_arr).T.dot(y_arr - u_arr))*1/Sigma
    X = -0.5 * (part_u+part_y)
    return X[0, 0]


# 定义梯度
def gradient_u(y_arr, u_arr):
    gradient = - 1/Sigma * (y_arr-u_arr)
    return gradient


# 定义proposal distribution p(u,u*)
def q(y_arr, u_arr, u_arr1):
    part1 = 0.5 * (y_arr - u_arr).T.dot(y_arr - u_arr) * 1/Sigma
    part2 = 0.5 * (u_arr1 - u_arr).T.dot(gradient_u(y_arr, u_arr))
    part3 = delta/4 * (u_arr + u_arr1).T.dot(gradient_u(y_arr, u_arr))
    part4 = delta/4 * (gradient_u(y_arr, u_arr).T.dot(K)).dot(gradient_u(y_arr, u_arr))
    return part1 + part2 + part3 + part4


arr = np.ones((dimension, 1)) * 0.1
accept_num = 0
np.random.seed(42)
A = np.zeros((dimension, 1))
for i in range(samp_num):
    # Z = np.random.randn(dimension, 1)
    mean = np.zeros(dimension)
    cov = K
    Z = np.random.multivariate_normal(mean, cov, (1,))
    # arr_star = (1 - beta**2)*arr + beta * Z
    arr_star = ((2-delta) * arr - 2*delta * K.dot(gradient_u(Y_arr, arr)) + (8 * delta)**0.5 * Z.T) * 1/(2+delta)
    # arr是x，arr_star是x*，作为下一步的候选值。都是d*1矩阵
    qx_part = q(Y_arr, arr, arr_star) - q(Y_arr, arr_star, arr)
    # qx_part = q(arr, arr_star)-q(arr_star, arr)
    alpha1 = min(1, np.exp(qx_part))
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
print(delta, accept_rate)
plt.figure(figsize=(10, 5))
plt.plot(x, u, 'k', lw=3, label='x')
plt.plot(x, Y_arr, '.k', label='y=x+n')
plt.plot(x, A, 'k', label='x*')  # 可能有问题
plt.legend()
plt.title('d=%f,delta=%f' % (d, delta))
plt.show()
