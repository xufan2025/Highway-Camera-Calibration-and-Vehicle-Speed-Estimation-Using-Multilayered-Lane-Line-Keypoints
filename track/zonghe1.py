# -- coding: utf-8 --
import cv2
import numpy as np
import cmath
import math

from numpy import vstack
from sympy import symbols, Eq, solve
import time


def get_zuobiao(wide, height):
    get_zuobiao_jiaojv1 = np.matrix([[962, 924, 1347, 924, 1709, 924],
                                    [874, 584, 1088, 584, 1288, 584],
                                    [842, 454, 989, 454, 1127, 454]], dtype=np.float64)
    get_zhuanhuan_zuobiao1 = np.zeros((3, 6), dtype=np.float64)
    get_zhuanhuan_zuobiao1 = np.matrix(get_zhuanhuan_zuobiao1)
    for a in range(get_zuobiao_jiaojv1.shape[0]):
        for b in range(0, get_zuobiao_jiaojv1.shape[1], 2):
            get_zhuanhuan_zuobiao1[a, b] = get_zuobiao_jiaojv1[a, b] - wide / 2
        for c in range(1, get_zuobiao_jiaojv1.shape[1], 2):
            get_zhuanhuan_zuobiao1[a, c] = get_zuobiao_jiaojv1[a, c] - height / 2

    get_zuobiao_jiaojv2 = np.matrix([[155, 602, 385, 602, 620, 602],
                                    [358, 458, 512, 458, 670, 458],
                                    [462, 386, 579, 386, 696, 386]], dtype=np.float64)
    get_zhuanhuan_zuobiao2 = np.zeros((3, 6), dtype=np.float64)
    get_zhuanhuan_zuobiao2 = np.matrix(get_zhuanhuan_zuobiao2)
    for a in range(get_zuobiao_jiaojv2.shape[0]):
        for b in range(0, get_zuobiao_jiaojv2.shape[1], 2):
            get_zhuanhuan_zuobiao2[a, b] = get_zuobiao_jiaojv2[a, b] - wide / 2
        for c in range(1, get_zuobiao_jiaojv2.shape[1], 2):
            get_zhuanhuan_zuobiao2[a, c] = get_zuobiao_jiaojv2[a, c] - height / 2

    return get_zhuanhuan_zuobiao1, get_zhuanhuan_zuobiao2


def solve_nc(H, D, C, jiaojv):
    A = np.zeros((3, 3), dtype=np.float64)  # 最小二乘拟合算法Au=b
    u = np.zeros((3, 1), dtype=np.float64)
    b = np.zeros((3, 1), dtype=np.float64)
    A = np.matrix(A)
    u = np.matrix(u)
    b = np.matrix(b)

    size_jiaojv = jiaojv.shape
    S = []
    for theta in np.arange(0.01, (1.0 + 0.01), 0.01):  # theta取值进行轮训
        for i in range(size_jiaojv[0]):
            A[i, :] = [(- D ** 2) / (jiaojv[i, 2] - jiaojv[i, 0]) ** 2, theta ** 2, 2 * i * C * (theta ** 3)]
            b[i] = (D ** 2) * (jiaojv[i, 1] ** 2) / (jiaojv[i, 2] - jiaojv[i, 0]) ** 2 - (H ** 2) * (theta ** 2) - (
                    i ** 2) * (C ** 2) * (theta ** 4)
        u = np.linalg.pinv(A) * b
        fr = cmath.sqrt(u[0, 0])  # 求得焦距f与像元大小的比值
        Yr = u[2, 0]
        Yr = int(Yr)
        if (Yr - Yr.real == 0) and (fr - fr.real == 0):
            if float(Yr) > 1000 and float(fr.real) > 0:
                S.extend([Yr, fr.real, theta.real])

    S = np.matrix(S).reshape((int(len(S) / 3), 3)).T
    return S


def find_best(H, D, C, jiaojv):

    S = solve_nc(H, D, C, jiaojv)

    x = jiaojv[:, 0]
    y = jiaojv[:, 1]
    m = jiaojv[:, 2]
    n = jiaojv[:, 3]

    start = 0
    S_size = S.shape
    Y = np.zeros((3, S_size[1]), dtype=np.float64)
    X = np.zeros((3, S_size[1]), dtype=np.float64)
    N = np.zeros((3, S_size[1]), dtype=np.float64)
    M = np.zeros((3, S_size[1]), dtype=np.float64)
    deltatheta = np.zeros((S_size[1]))

    for i in range(start, S_size[1]):
        for j in range(0, 3):
            Y[j, i - start] = (S[0, i] + j * C * S[2, i]).real
            X[j, i - start] = (
                    x[j] * cmath.sqrt(H ** 2 + Y[j, i - start] ** 2) / cmath.sqrt(S[1, i] ** 2 + y[j] ** 2)).real
            N[j, i - start] = (S[0, i] + j * C * S[2, i]).real
            M[j, i - start] = (m[j] * cmath.sqrt(H ** 2 + N[j, i - start] ** 2) / cmath.sqrt(
                S[1, i] ** 2 + n[j] ** 2)).real  # 通过图上坐标反演出所取点的实际坐标

        K1 = np.polyfit(X[:, i - start], Y[:, i - start], 1)
        K2 = np.polyfit(M[:, i - start], N[:, i - start], 1)  # 将所得实际坐标进行曲线拟合，得到图上虚线和实现对应的线
        math.atan(K1[1]) / math.pi * 180
        # plt.figure()
        # plt.plot(X[:, i - start], Y[:, i - start], 'b')
        # plt.plot(M[:, i - start], N[:, i - start], 'r')
        if abs(K1[0] - K2[0]) < 5:
            if math.atan(K1[0]) / math.pi * 180 > 0:
                deltatheta[i - start] = abs(
                    90 - math.atan(K1[0]) / math.pi * 180 - math.acos(S[2, i]) / math.pi * 180)  # 筛选的实际物理条件
            else:
                deltatheta[i - start] = abs(math.atan(K1[0]) / math.pi * 180 - math.acos(S[2, i]) / math.pi * 180 + 90)
                # plt.show()
    min_deltatheta = deltatheta[0]
    min_index = 0
    for p in range(len(deltatheta)):
        if deltatheta[p] < min_deltatheta:
            min_deltatheta = deltatheta[p]
            min_index = p
    best = S[:, min_index]
    Yr = best[0, 0]
    fr = best[1, 0]
    theta = best[2, 0]
    return Yr, fr, theta
    # plt.plot(deltatheta)
    # plt.show()


def linear_functions1(wide, height):
    x_y_1 = get_zuobiao(wide, height)[0]
    x1 = x_y_1[0, 0] + wide / 2
    x2 = x_y_1[1, 0] + wide / 2
    y1 = x_y_1[0, 1] + height / 2
    y2 = x_y_1[1, 1] + height / 2
    k1 = symbols('k1')
    b1 = symbols('b1')
    eq1 = Eq(k1 * y1 + b1 - x1, 0)
    eq2 = Eq(k1 * y2 + b1 - x2, 0)
    E1 = solve([eq1, eq2], [k1, b1])
    k1 = E1[k1]
    b1 = E1[b1]

    return k1, b1


def linear_functions2(wide, height):
    x_y_1 = get_zuobiao(wide, height)[0]
    x5 = x_y_1[0, 4] + wide / 2
    x6 = x_y_1[1, 4] + wide / 2
    y5 = x_y_1[0, 5] + height / 2
    y6 = x_y_1[1, 5] + height / 2
    k2 = symbols('k2')
    b2 = symbols('b2')
    eq5 = Eq(k2 * y5 + b2 - x5, 0)
    eq6 = Eq(k2 * y6 + b2 - x6, 0)
    E2 = solve([eq5, eq6], [k2, b2])
    k2 = E2[k2]
    b2 = E2[b2]

    return k2, b2


def linear_functions3(wide, height):
    x_y_2 = get_zuobiao(wide, height)[1]
    x7 = x_y_2[0, 0] + wide / 2
    x8 = x_y_2[1, 0] + wide / 2
    y7 = x_y_2[0, 1] + height / 2
    y8 = x_y_2[1, 1] + height / 2
    k3 = symbols('k3')
    b3 = symbols('b3')
    eq7 = Eq(k3 * y7 + b3 - x7, 0)
    eq8 = Eq(k3 * y8 + b3 - x8, 0)
    E3 = solve([eq7, eq8], [k3, b3])
    k3 = E3[k3]
    b3 = E3[b3]

    return k3, b3


def linear_functions4(wide, height):
    x_y_2 = get_zuobiao(wide, height)[1]
    x11 = x_y_2[0, 4] + wide / 2
    x12 = x_y_2[1, 4] + wide / 2
    y11 = x_y_2[0, 5] + height / 2
    y12 = x_y_2[1, 5] + height / 2
    k4 = symbols('k4')
    b4 = symbols('b4')
    eq11 = Eq(k4 * y11 + b4 - x11, 0)
    eq12 = Eq(k4 * y12 + b4 - x12, 0)
    E4 = solve([eq11, eq12], [k4, b4])
    k4 = E4[k4]
    b4 = E4[b4]

    return k4, b4


def cesu(x1, y1, y2, H, D, n_lane, t_fps, f, theta, wide, height, k1, k2, k3, k4, b1, b2, b3, b4):
    if y1 < y2:
        a1 = k1 * y1 + b1
        a2 = k2 * y1 + b2
        c1 = k1 * y2 + b1
        c2 = k2 * y2 + b2
        l1 = abs(c1 - c2)
        l2 = abs(a1 - a2)
        y_prev = y2 - height / 2
        y_after = y1 - height / 2
    else:
        a1 = k3 * y1 + b3
        a2 = k4 * y1 + b4
        c1 = k3 * y2 + b3
        c2 = k4 * y2 + b4
        l1 = abs(a1 - a2)
        l2 = abs(c1 - c2)
        y_prev = y1 - height / 2
        y_after = y2 - height / 2
    # Y = symbols('Y')
    # C = symbols('C')
    #
    # eq1 = Eq(f ** 2 + y_prev ** 2 - ((Y ** 2 + H ** 2) * l1 ** 2 * theta ** 2) / (n_lane * D) ** 2, 0)
    # eq2 = Eq(f ** 2 + y_after ** 2 - (((Y + C * theta) ** 2 + H ** 2) * l2 ** 2 * theta ** 2) / (n_lane * D) ** 2, 0)
    # L = solve([eq1, eq2], [Y, C])
    # i = 0
    # sign1 = 0
    # sign2 = 0
    # while sign2 == 0:
    #     sign1 = 0  # 使每次进入while循环时，标志位sign1均为0
    #     for j in range(len(L[i])):  # 对列表中第i个元组进行遍历
    #         if L[i][j] < 0:  # 若元组中有小于0的数
    #             del L[i]  # 则删去该整个元组的元素
    #             sign1 = 1 - sign1  # 则sign1的标志位置1
    #             break  # 删除一个元组就跳出本for循环，利用sign1标志位进行后续的索引i的值的判断
    #
    #     # 根据上面for循环后标志位的值，进行如下判断
    #     if abs(1 - sign1):  # 若sign标志位为0，则表示第i个元组中没有负数，则不删除该元素
    #         i += 1  # 进行下一个元组的访问
    #
    #     if i == len(L):  # 判断索引i是否超过L的元素总数
    #         sign2 = 1  # 若i已超过L的元素总数，那么sign2标志位置1，跳出while循环
    #
    # L1 = L[0][1]
    # L2 = L[0][0]
    # v = (L1 * 3.6) / (1000 * t_fps)
    # A1 = np.array([1, 0, 0])
    # A2 = np.array([1, 2 * theta, theta ** 2])
    A1 = np.array([1, 0])
    A2 = np.array([1, theta])
    A = vstack((A1, A2))
    d1 = np.array([cmath.sqrt((((D * n_lane) ** 2 * (f ** 2 + y_prev ** 2)) / (l1 ** 2 * theta ** 2)) - H ** 2)])
    d2 = np.array([cmath.sqrt((((D * n_lane) ** 2 * (f ** 2 + y_after ** 2)) / (l2 ** 2 * theta ** 2)) - H ** 2)])
    d = vstack((d1, d2))
    A_1 = np.linalg.pinv(A)
    r = np.dot(A_1, d)
    Y = abs(r[0, 0])
    C = abs(r[1, 0])
    end = time.perf_counter()
    v = (C * 3.6) / (1000 * t_fps)
    if y1 < y2:
        X1 = (x1 - wide / 2) * (cmath.sqrt(H ** 2 + Y ** 2) / cmath.sqrt(f ** 2 + (y_after / 2) ** 2))
    else:
        X1 = (x1 - wide / 2) * (cmath.sqrt(H ** 2 + Y ** 2) / cmath.sqrt(f ** 2 + (y_prev / 2) ** 2))
    X1 = X1.real / 1000
    return v, X1


if __name__ == '__main__':
    # x1 = 281
    # y1 = 999
    # y2 = 959
    x1 = 0
    y1 = 584
    y2 = 924
    H = 6000
    D = 3750
    C = 15000
    n_lane = 2
    t_fps = 1 / 30
    # path = r'D:\chen_pythonfile\Yolov5_StrongSORT_OSNet-master\MOT16_eval\1.jpg'  # 用于计算相机内参的图像  ccb修改
    path = r'D:\chen_pythonfile\Yolov5_StrongSORT_OSNet-master\MOT16_eval\biye\2.jpg'  # 用于计算相机内参的图像   zxg修改
    img = cv2.imread(path)
    height = img.shape[0]
    wide = img.shape[1]
    jiaojv = get_zuobiao(wide, height)[0]
    Yr, f, theta = find_best(H, D, C, jiaojv)  # 计算内参
    k1, b1 = linear_functions1(wide, height)
    k2, b2 = linear_functions2(wide, height)
    k3, b3 = linear_functions3(wide, height)
    k4, b4 = linear_functions4(wide, height)
    print(Yr, f, theta)
    v = cesu(x1, y1, y2, H, D, n_lane, t_fps, f, theta, wide, height, k1, k2, k3, k4, b1, b2, b3, b4)[0]  # 进行测速
    print(v)