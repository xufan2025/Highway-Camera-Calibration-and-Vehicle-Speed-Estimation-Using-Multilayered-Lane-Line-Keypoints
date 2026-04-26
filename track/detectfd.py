# -- coding: utf-8 --
import statistics

import numpy as np
import matplotlib.pyplot as plt
import cmath
import math
import os

from matplotlib import font_manager
from matplotlib import rcParams
size = 14  # 全局字体大小
# 设置英文字体
config = {
    "font.family": 'serif',
    "font.size": size,
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman'],
}
rcParams.update(config)
# 设置中文宋体
fontcn = {'family': 'SimSun', 'size': size}
label_size = size
text_size = size
rcParams.update(config)
H = 6000
D = 3750
C = 15000  # 道路的基本信息
wide = 1920
height = 1080
A = np.zeros((3, 3), dtype=np.float64)  # 最小二乘拟合算法Au=b
u = np.zeros((3, 1), dtype=np.float64)
b = np.zeros((3, 1), dtype=np.float64)
A = np.matrix(A)
# u = np.matrix(u)
b = np.matrix(b)
jiaojv_cede = np.matrix([[963, 926, 1706, 926], [875, 586, 1289, 586], [843, 455, 1129, 455]], dtype=np.float64)  # 道路上虚线端点的坐标以及与其纵坐标相同的实线上的坐标
jiaojv = np.zeros((3, 6), dtype=np.float64)
jiaojv = np.matrix(jiaojv)
for a in range(3):
    jiaojv[a, 0] = jiaojv_cede[a, 0] - wide / 2
    jiaojv[a, 1] = jiaojv_cede[a, 1] - height / 2
    jiaojv[a, 2] = jiaojv_cede[a, 2] - wide / 2
    jiaojv[a, 3] = jiaojv_cede[a, 3] - height / 2
size_jiaojv = jiaojv.shape
print(jiaojv)

S = []
for theta in np.arange(0.01, (1.0 + 0.01), 0.01):  # theta取值进行轮训
    for i in range(size_jiaojv[0]):
        A[i, :] = [(-D ** 2) / (jiaojv[i, 2] - jiaojv[i, 0]) ** 2, theta ** 2, 2 * i * C * (theta ** 3)]
        b[i] = (D ** 2) * (jiaojv[i, 1] ** 2) / (jiaojv[i, 2] - jiaojv[i, 0]) ** 2 - (H ** 2) * (theta ** 2) - (
                i ** 2) * (C ** 2) * (theta ** 4)

    u = np.linalg.pinv(A) * b
    # print(u[0, 0])
    fr = cmath.sqrt(u[0, 0])  # 求得焦距f与像元大小的比值
    # print(fr)
    Yr = u[2, 0]
    Yr = int(Yr)
    # tempindex = np.matrix([], dtype=np.float64)
    if (Yr - Yr.real == 0) and (fr - fr.real == 0):
        if float(Yr) > 1000 and float(fr.real) > 0:
            S.extend([Yr, fr.real, theta.real])

S = np.matrix(S).reshape((int(len(S) / 3), 3)).T

x = np.zeros((3, 1), dtype=np.float64)
y = np.zeros((3, 1), dtype=np.float64)
m = np.zeros((3, 1), dtype=np.float64)
n = np.zeros((3, 1), dtype=np.float64)
x = np.matrix(x)
y = np.matrix(y)
m = np.matrix(m)
n = np.matrix(n)

x[:] = jiaojv[:, 0]
y[:] = jiaojv[:, 1]
m[:] = jiaojv[:, 2]
n[:] = jiaojv[:, 3]

start = 0
S_size = S.shape
Y = np.zeros((3, S_size[1]), dtype=np.float64)
X = np.zeros((3, S_size[1]), dtype=np.float64)
N = np.zeros((3, S_size[1]), dtype=np.float64)
M = np.zeros((3, S_size[1]), dtype=np.float64)
deltatheta = np.zeros((S_size[1]))
K1_list = []
K1_Mean = []
# K1_Std = []

K2_list = []
K2_Mean = []
# K2_Std = []

for i in range(start, S_size[1]):
    for j in range(0, 3):
        Y[j, i - start] = (S[0, i] + j * C * S[2, i]).real
        X[j, i - start] = (x[j] * cmath.sqrt(H ** 2 + Y[j, i - start] ** 2) / cmath.sqrt(S[1, i] ** 2 + y[j] ** 2)).real
        N[j, i - start] = (S[0, i] + j * C * S[2, i]).real
        M[j, i - start] = (m[j] * cmath.sqrt(H ** 2 + N[j, i - start] ** 2) / cmath.sqrt(S[1, i] ** 2 + n[j] ** 2)).real  # 通过图上坐标反演出所取点的实际坐标


    k1_1 = (Y[0, i - start] - Y[1, i - start]) / (X[0, i - start] - X[1, i - start])
    k1_2 = (Y[1, i - start] - Y[2, i - start]) / (X[1, i - start] - X[2, i - start])
    k1_mean = (k1_1 + k1_2) / 2
    k1_std = math.sqrt((k1_1 - k1_mean)**2 + (k1_2 - k1_mean)**2)
    K1_Mean.append(k1_mean)
    # K1_Std.append(k1_std)
    K1_Std = statistics.mean(K1_Mean)

    k2_1 = (N[0, i - start] - N[1, i - start]) / (M[0, i - start] - M[1, i - start])
    k2_2 = (N[1, i - start] - N[2, i - start]) / (M[1, i - start] - M[2, i - start])
    k2_mean = (k2_1 + k2_2) / 2
    k2_std = math.sqrt((k2_1 - k2_mean)**2 + (k2_2 - k2_mean)**2)
    K2_Mean.append(k2_mean)
    # K2_Std.append(k2_std)
    K2_Std = statistics.mean(K2_Mean)

    K1 = np.polyfit(X[:, i - start], Y[:, i - start], 1)
    K2 = np.polyfit(M[:, i - start], N[:, i - start], 1)  # 将所得实际坐标进行曲线拟合，得到图上虚线和实现对应的线
    math.atan(K1[1]) / math.pi * 180


    # 画图代码
    fig = plt.figure(figsize=(7, 5.2))
    # plt.title("The parallelism of two lines", fontsize='20', loc='center', color='black')
    plt.plot(X[:, i - start] / 1000, Y[:, i - start] / 1000, label = '$left$',linewidth = 3)
    #
    plt.plot(M[:, i - start] / 1000, N[:, i - start] / 1000, label = '$right$',linewidth = 3)
    # plt.title("Reconstruction of the lane lines", fontsize='20', loc='center', color='black')

    plt.ylabel('X/m', fontsize=16)
    plt.xlabel('Y/m', fontsize=16)
    plt.legend(loc='upper right', prop=fontcn)
    plt.grid(linestyle='--')
    # plt.show()
    # plt.xlim(-4.3, 12)
    plt.xlim(-3, 7)
    # plt.xlabel(u"X/m", fontsize=16)
    # plt.ylabel(u"Y/m", fontsize=16)
    plt.savefig(os.path.join(r"C:\Users\zgshang\Desktop\new/" + str(i+1) + ".png"))



    if math.atan(K1[0]) / math.pi * 180 > 0:
        deltatheta[i - start] = abs(
            90 - math.atan(K1[0]) / math.pi * 180 - math.acos(S[2, i]) / math.pi * 180)  # 筛选的实际物理条件
    else:
        deltatheta[i - start] = abs(math.atan(K1[0]) / math.pi * 180 - math.acos(S[2, i]) / math.pi * 180 + 90)
        # plt.show()
    # if math.atan(K2[0]) / math.pi * 180 > 0:
    #     deltatheta[i - start] = abs(
    #         90 - math.atan(K2[0]) / math.pi * 180 - math.acos(S[2, i]) / math.pi * 180)  # 筛选的实际物理条件
    # else:
    #     deltatheta[i - start] = abs(math.atan(K2[0]) / math.pi * 180 - math.acos(S[2, i]) / math.pi * 180 + 90)
    #     # plt.show()
min_deltatheta = deltatheta[0]
min_index = 0
for p in range(len(deltatheta)):
    if deltatheta[p] < min_deltatheta:
        min_deltatheta = deltatheta[p]
        min_index = p

best = np.zeros((3, 1), dtype=np.float64)
best = S[:, min_index]
Yr, f, theta = best
# print(Yr, f, theta)
# plt.plot(deltatheta)
# plt.show()
# plt.savefig('C:/Users/zgshang/Desktop/picture/62.png')
# print(deltatheta)
# print(S)

x = np.arange(61)
# y1_1 = []
# y1_2 = []
# for i in range(len(K1_Mean)):
#     # y1_1.append(K1_Mean[i] + K1_Std[i])
#     # y1_2.append(K1_Mean[i] - K1_Std[i])
#     y1_1.append(K1_Mean[i] + K1_Std)
#     y1_2.append(K1_Mean[i] - K1_Std)
#
# y1_1 = np.array(y1_1)
# y1_2 = np.array(y1_2)
#
# y2_1 = []
# y2_2 = []
# for i in range(len(K2_Mean)):
#     # y2_1.append(K2_Mean[i] + K2_Std[i])
#     # y2_2.append(K2_Mean[i] - K2_Std[i])
#     y2_1.append(K2_Mean[i] + K2_Std)
#     y2_2.append(K2_Mean[i] - K2_Std)
# y2_1 = np.array(y2_1)
# y2_2 = np.array(y2_2)
#
# ax = plt.subplot()
# ax.plot(x, K1_Mean, 'r', linewidth=1)    #绘制图中的均值线
# ax.plot(x, K2_Mean, 'b', linewidth=1)    #绘制图中的均值线
# #下面绘制填充区间
#
# ax.fill_between(x, y1_1, y1_2, alpha=0.2,facecolor='r', where=y1_2 >= y1_1,  interpolate=True)
# ax.fill_between(x, y1_1, y1_2, alpha=0.2,facecolor='r', where=y1_2 <= y1_1,  interpolate=True)
# ax.fill_between(x, y2_1, y2_2, alpha=0.2,facecolor='b', where=y2_2 >= y2_1,  interpolate=True)
# ax.fill_between(x, y2_1, y2_2, alpha=0.2,facecolor='b', where=y2_2 <= y2_1,  interpolate=True)
# plt.show()
#

# K_Mine = []
# for i in range(len(K1_Mean)):
#     k_mine = abs(K1_Mean[i] - K2_Mean[i])
#     K_Mine.append(k_mine)
#
# plt.plot(x, K_Mine, linewidth = 2, label = 'the slope difference')    #绘制图中的均值线)
# x_z = x = np.arange(-2, 63)
# z = np.ones(65) * 0.2
#
# plt.plot(x_z, z, linewidth = 1.5, color = 'gray', linestyle = '--', label = r'$\varepsilon$')
# plt.xlim = (-0.01, 61)
# # plt.annotate(r'$\varepsilon$', xy=(0.99, 1), xytext=(0.99-0.12,0.23), color = 'gray', size = '20')
# plt.arrow(10, 0.2, 0, -0.25, linewidth=1, color='gray', head_length=0.1, head_width=1)
# # plt.title("The Curve of slope difference between two lines", fontsize='20', loc='center', color='black')
# plt.xlabel('The num of $S_{i}$', fontsize=16)
# plt.ylabel('Difference of slope', fontsize=16)
# plt.legend(loc='upper right', prop=fontcn)
# # plt.show()
# plt.savefig(os.path.join("C:/Users\zgshang\Desktop/test/difference.png"))