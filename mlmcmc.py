import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import lagrange
from scipy import interpolate

# 维数和样本数目不同
samp_num = np.array([20, 10])  # 2个level对应的样本数目

L = 2  # mlmcmc的层数
np.random.seed(42)
dimension = np.array([11, 21])  # 2个level 对应的维数。M1=M0*s
samp = []
for i in range(L):
    samp_i = [[0]*dimension[i]]*samp_num[i]
    samp.append(samp_i)
samp = np.array(samp, dtype=object)
for i in range(L):
    samp[i] = np.array(samp[i]).T

beta = 0.001
lam = 300
plt.close('all')


# 真实数据,分成2个维度便于后续的讨论.
data = []
for l in range(L):
    x = np.linspace(0, 1, dimension[l])
    interval0 = [1 if (i < 1//3) else 0 for i in x]
    interval1 = [1 if (1/3 <= i < 2/3) else 0 for i in x]
    interval2 = [1 if (i >= 2/3) else 0 for i in x]
    u = 0 * interval0 + 1 * interval1 + 0*interval2
    n = np.random.normal(0, 0.02, dimension[l])
    real_data = u + n
    real_data = real_data.tolist()
    data.append(real_data)


def DX_1(xy, dim):
    DX1 = 0
    for z in range(dim - 1):  # 这里的维度是变化的
        dx_1 = abs(xy[z + 1] - xy[z])
        DX1 = DX1 + dx_1
    return DX1


d = 0.02
r = 0.1
K = []
K_inv = []
Sigma = 0.02**2
for l in range(L):
    delta_t = 1/dimension[l]
    # 定义协方差矩阵K
    K_l = np.zeros((dimension[l], dimension[l]))
    for i in range(dimension[l]):
        for j in range(dimension[l]):
            K_l[i, j] = r * np.exp(-0.5 * (abs(i-j) * delta_t / d)**2)
    Kl_inv = np.linalg.inv(K_l)
    K.append(K_l)
    K_inv.append(Kl_inv)


# target distribution
def p(y_arr, u_arr, dim, k):
    part_u = (u_arr.T.dot(k)).dot(u_arr)
    part_y = (y_arr - u_arr).T.dot(y_arr - u_arr) * 1/Sigma
    X = 0.5 * (part_u + part_y)+lam * DX_1(u_arr, dim)
    return -1*X[0, 0]


# B代表的是每一个维度的采样结果
B = []
for l in range(L):
    accept_num = 0
    accept_num1 = 0
    A = np.zeros((dimension[l], 1))
    # A_l = np.zeros((dimension[l], 1))
    # 用于当l = 0
    arr = np.ones((dimension[l], 1)) * 0.1
    mean = np.zeros(dimension[l])
    cov = K[l]

    # 用于当l不等于0
    if l != 0:
        # on level l-1
        arr_0 = np.ones((dimension[l - 1], 1)) * 0.1
        mean_0 = np.zeros(dimension[l - 1])
        cov_0 = K[l - 1]
        # level l
        delta_dimension = dimension[l] - dimension[l - 1]
        arr_f = np.ones((delta_dimension, 1)) * 0.1
        arr_c = np.ones((dimension[l - 1], 1)) * 0.1
        mean_f = np.zeros(delta_dimension)
        delta_t = 1 / delta_dimension
        #     定义协方差矩阵K
        K_1 = np.zeros((delta_dimension, delta_dimension))
        for i in range(delta_dimension):
            for j in range(delta_dimension):
                K_1[i, j] = r * np.exp(-0.5 * (abs(i - j) * delta_t / d) ** 2)
        cov_f = K_1
        K_1_inv = np.linalg.inv(K_1)

        x = np.linspace(0, 1, delta_dimension)
        interval0 = [1 if (i < 1 // 3) else 0 for i in x]
        interval1 = [1 if (1 / 3 <= i < 2 / 3) else 0 for i in x]
        interval2 = [1 if (i >= 2 / 3) else 0 for i in x]
        u = 0 * interval0 + 1 * interval1 + 0 * interval2
        n = np.random.normal(0, 0.02, delta_dimension)
        data_arr = u + n
#  接下来是mlmcmc的部分
    for i in range(samp_num[l]):
        if l == 0:
            # Z = np.random.randn(dimension, 1)
            # arr_star = (1 - beta ** 2) * arr + beta * Z
            Z = np.random.multivariate_normal(mean, cov, (1,))
            arr_star = (1 - beta**2)**0.5*arr + beta * Z.T
            # arr是x，arr_star是x*，作为下一步的候选值。都是d*1矩阵
            px_part = p(np.array(data[l]), arr_star, dimension[l], K_inv[l]) - p(np.array(data[l]), arr, dimension[l], K_inv[l])
            # qx_part = q(arr, arr_star)-q(arr_star, arr)
            alpha1 = min(1, np.exp(px_part))
            # print(alpha1)
            m = np.random.rand(1)[0]
            if m < alpha1:
                arr = arr_star
                accept_num += 1
            samp[l][:, [i]] = arr
            accept_rate = accept_num / samp_num[l]
            print("l=0", accept_rate)
        else:
            # on level l-1
            Z = np.random.multivariate_normal(mean_0, cov_0, (1,))
            arr_star_0 = (1 - beta ** 2) ** 0.5 * arr_0 + beta * Z.T
            px_part0 = p(np.array(data[l-1]), arr_star_0, dimension[l-1], K_inv[l-1]) - p(np.array(data[l-1]), arr_0, dimension[l-1], K_inv[l-1])
            alpha0 = min(1, np.exp(px_part0))
            m = np.random.rand(1)[0]
            if m < alpha0:
                arr_0 = arr_star_0
                accept_num += 1
            accept_rate = accept_num / samp_num[l-1]
            print("l=%f" % l, accept_rate)

            # on level l
            # 首先使用pcn生成FINE部分，这个部分的维数是delta_dimension
            Z_f = np.random.multivariate_normal(mean_f, cov_f, (1,))
            arr_star_f = (1 - beta ** 2) ** 0.5 * arr_f + beta * Z_f.T
            # 将粗糙部分与F部分连接在一起
            arr_star_c = arr_0
            arr_star = np.concatenate((arr_star_f, arr_star_c))
            p_part_c = p(np.array(data[l-1]), arr_c, dimension[l-1], K_inv[l-1]) - p(np.array(data[l-1]), arr_star_c, dimension[l-1], K_inv[l-1])
            p_part = p(np.array(data[l]), arr_star, dimension[l], K_inv[l]) - p(np.array(data[l]), arr, dimension[l], K_inv[l])
            alpha1 = min(1, np.exp(px_part + p_part_c))
            m = np.random.rand(1)[0]
            if m < alpha1:
                arr = arr_star
                accept_num1 += 1
            accept_rate1 = accept_num1 / samp_num[l]

            #  关于插值方法,使得arr,arr_0两者维数相同。使用拉格朗日插值法
            arr0_new = np.zeros((dimension[l], 1), dtype=object)
            x = arr[:dimension[l-1]]   # arr维数为dimension(l),截取前dimension(l-1)
            y = arr_0  # 维数为dimension(l-1)
            x = np.asarray(x).squeeze()
            y = np.asarray(y).squeeze()
            f = lagrange(x, y)
            arr0_new = f(arr)
            samp[l][:, [i]] = arr - arr0_new
            print(accept_rate1)
    for j in range(dimension[l]):
        A[j] = np.mean(samp[l][j])
    B.append(A)

# 由于mlmcmc相加项的维数依然不同，故再次使用插值方法,希望得到dimension(l)的维数
x = B[0]   # arr维数为dimension(l),截取前dimension(l-1)
y = B[1][:dimension[0]]  # 维数为dimension(l-1)
x = np.asarray(x).squeeze()
y = np.asarray(y).squeeze()
f = lagrange(x, y)
B0_new = f(B[1])

data_mlmcmc = B0_new + B[1]

########################################################
plt.figure(figsize=(10, 5))
plt.plot(x, u, 'k', lw=3, label='x')
plt.plot(x, np.array(data[L-1]), '.k', label='y=x+n')
plt.plot(x, data_mlmcmc, 'k', label='x*')  # 可能有问题
plt.legend()
plt.title('Model and data')
plt.show()
