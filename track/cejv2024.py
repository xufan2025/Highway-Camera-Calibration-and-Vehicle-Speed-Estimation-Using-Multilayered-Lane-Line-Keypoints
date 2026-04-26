"""
用于论文实验部分，打印数据：
                    1、相机参数
                    2、所求点的实际Y坐标
                    3、计算X轴坐标的误差百分比；公式为：|X2 - X1| / (3.75 * n)
                    4、计算Y轴坐标的误差百分比；公式为：|Y2 - Y1| / (15 * n)
"""


import cmath
import math
import time

import cv2 as cv
import cv2
import numpy as np

from sympy import symbols, Eq, solve

"""
这一段代码用于划分图像中的车道部分
"""
pts = []  # 用于存放点
def draw_roi(event, x, y, flags, param):
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击，选择点
        pts.append((x, y))
    if event == cv2.EVENT_RBUTTONDOWN:  # 右键点击，取消最近一次选择的点
        pts.pop()
    if event == cv2.EVENT_MBUTTONDOWN:  # 中键绘制轮廓
        mask = np.zeros(img.shape, np.uint8)
        points = np.array(pts, np.int32)
        points = points.reshape((-1, 1, 2))

        # 画多边形
        mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
        mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255))  # 用于求 ROI
        mask3 = cv2.fillPoly(mask.copy(), [points], (0, 255, 0))  # 用于 显示在桌面的图像
        show_image = cv2.addWeighted(src1=img, alpha=0.8, src2=mask3, beta=0.2, gamma=0)

        cv2.imshow("mask", mask2)
        cv2.imshow("show_img", show_image)

        ROI = cv2.bitwise_and(mask2, img)
        cv2.imshow("ROI", ROI)
        cv2.waitKey(0)

    if len(pts) > 0:
        # 将pts中的最后一点画出来
        cv2.circle(img2, pts[-1], 3, (0, 0, 255), -1)

    if len(pts) > 1:
        # 画线
        for i in range(len(pts) - 1):
            cv2.circle(img2, pts[i], 5, (0, 0, 255), -1)  # x ,y 为鼠标点击地方的坐标
            cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)

    cv2.imshow('image', img2)
    return pts


# 创建图像与窗口并将窗口与回调函数绑定
def image_process(path, thresh):
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_roi)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord("S"):
            data = pts
            break

    cv.imshow('image', img)
    cv.waitKey(0)
    img0 = cv.imread(path, 0)  # 重新以灰度图读取原图，后续找出车道线的处理都在这张图的基础上

    '''2.roi_mask(提取感兴趣的区域)'''
    mask = np.zeros_like(img0)  # 变换为numpy格式的图片
    # print(data)
    mask = cv.fillPoly(mask, np.array([data]), color=255)  # 对感兴趣区域制作掩膜
    # cv.namedWindow('mask', 0)
    # cv.imshow('mask', mask)
    # cv.waitKey(0)
    masked_edge_img = cv.bitwise_and(img0, mask)  # 与运算
    # cv2.imwrite('C:/Users/zgshang/Desktop/picture/100.jpg', masked_edge_img)

    ret, img1 = cv.threshold(masked_edge_img, thresh, 255, cv.THRESH_BINARY)  # 二值化

    '剔除小连通域1'
    contours, hierarchy = cv.findContours(img1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)  # 找出连通域
    for i in range(len(contours)):
        area = cv.contourArea(contours[i])  # 将每一个连通域的面积赋值给area
        if area < 5:  # '设定连通域最小阈值，小于该值被清理'
            cv.drawContours(img1, [contours[i]], 0, 0, -1)
    # cv.waitKey(0)
    img2 = img1
    # cv2.imwrite('C:/Users/zgshang/Desktop/picture/101.jpg', img2)
    '找出车道线每个点的坐标，并绘制在原图上'
    points = []  # 创建一个空列表，用于存放点的坐标信息
    for i in range(height - 1, -1, -1):
        for j in range(wide):
            if img2[i, j] != 0:
                img[i, j] = (0, 0, 255)  # 将值不为零的点用红色代替在原图上，注意这里的img是600*600的原图
                points.append([i + 1, j + 1])
    cv.imshow('image', img)
    cv.waitKey(0)
    return img2


def get_endpoint(wide, height):
    re_xy_all = []
    prev_result1 = []
    prev_result3 = []
    prev_result4 = []
    detect_f = []
    flag = 0
    for i in range(height - 1, -1, -1):  # 从左下开始遍历
        result = []  # 存放二值化之后为255的点
        result1 = []  # 存放第一组点
        result2 = []  # 存放过渡组点
        result3 = []  # 存放第二组点
        result4 = []  # 存放第三组点
        white_num = 0  # 记录白色线段个数
        v = 0  # 初始化白色点的纵坐标
        for j in range(wide):
            value = img2[i, j]
            if value != 0:
                result.append([i + 1, j + 1])  # 因为python0表示1，所以要加1
                p = j + 1  # 当前白色纵坐标
                if p - v >= 25:
                    white_num = white_num + 1
                v = p
        if white_num == 3:
            big_flag = 0
            for q in range(len(result)):
                if q != 0:
                    if result[q][1] - result[q - 1][1] >= 25 and big_flag == 0:  # 遍历每行，如果一行中的横坐标相差超过25，那么保存q
                        a = q
                        big_flag = 1
                        result1 = result[0: a]  # 将q前面的值保存进result1
                        del result[0: a]  # 删除前面q个值
                        result2 = result  # 删除前面q个值之后，剩下的就是中间虚线和后面白实线
                        break
            for k in range(len(result2)):
                if k != 0:
                    if result2[k][1] - result2[k - 1][1] >= 25 and big_flag == 1:
                        b = k
                        result3 = result2[0: b]  # 将k前面的值，也就是中间白实线的放到result3
                        del result2[0: b]  # 删除前面k个值
                        result4 = result2  # 删除前面k个值之后，剩下的就是后面白实线
                        break
        elif white_num == 2:
            result3 = []  # 中间无虚线
        if len(result3) == 0 and len(prev_result3) != 0:
            re_y = i + 1  # 当前行的纵坐标
            x1 = []
            re_xy = []
            for i1 in range(len(prev_result1)):
                x1.append(prev_result1[i1][1])  # 存放第一组的所有横坐标
            re_x1 = int(np.mean(x1))  # 求平均
            x2 = []
            for i2 in range(len(prev_result3)):
                x2.append(prev_result3[i2][1])
            re_x2 = int(np.mean(x2))  # 同上
            x3 = []
            for i3 in range(len(prev_result4)):
                x3.append(prev_result4[i3][1])
            re_x3 = int(np.mean(x3))  # 同上
            w = 0.025 * wide
            if len(prev_result1) < w and len(prev_result3) < w and len(prev_result4) < w:
                re_xy = [re_x1, re_y, re_x3, re_y]  # 输出坐标
                re_xy_all.append(re_xy)

        prev_result1 = result1
        prev_result3 = result3
        prev_result4 = result4

    return re_xy_all


def draw_line(zuobiao, img1, img2):
    for i in range(len(zuobiao)):
        for x in range(zuobiao[i][0] - 1, zuobiao[i][2], 1):
            y = zuobiao[i][1]
            img1[y, x] = (0, 255, 0)
    return img


def CalcuCemeraIR_coordinates(zuobiao, wide, height):
    detect_f = []
    detect_f.append(zuobiao[0])
    detect_f.append(zuobiao[1])
    detect_f.append(zuobiao[2])
    detect_f.append(zuobiao[3])
    # detect_f.append(zuobiao[4])
    # detect_f.append(zuobiao[5])
    get_zuobiao_jiaojv = np.matrix(detect_f, dtype=np.float64)
    detect_f_coordinates = np.zeros((4, 4), dtype=np.float64) # 修改 ccb修改
    # detect_f_coordinates = np.zeros((5, 5), dtype=np.float64) # 修改
    detect_f_coordinates = np.matrix(detect_f_coordinates)
    for a in range(get_zuobiao_jiaojv.shape[0]):
        for b in range(0, get_zuobiao_jiaojv.shape[1], 2):
            detect_f_coordinates[a, b] = get_zuobiao_jiaojv[a, b] - wide / 2
        for c in range(1, get_zuobiao_jiaojv.shape[1], 2):
            detect_f_coordinates[a, c] = get_zuobiao_jiaojv[a, c] - height / 2  # 标定坐标处理
    return detect_f_coordinates


def solve_nc(H, D, C, jiaojv, n_lane, l):
    A = np.zeros((4, 3), dtype=np.float64)  # 最小二乘拟合算法Au=b
    u = np.zeros((3, 1), dtype=np.float64)
    b = np.zeros((4, 1), dtype=np.float64)
    A = np.matrix(A)
    u = np.matrix(u)
    b = np.matrix(b)
    size_jiaojv = jiaojv.shape
    S = []
    for theta in np.arange(0, (1.0 + 0.01), 0.01):  # theta取值进行轮训 # 修改
        for i in range(size_jiaojv[0]):
            A[i, :] = [(-(D * n_lane) ** 2) / l[i] ** 2, theta ** 2, 2 * i * C * (theta ** 3)]
            b[i] = ((D * n_lane) ** 2) * (jiaojv[i, 1] ** 2) / l[i] ** 2 - (H ** 2) * (
                    theta ** 2) - (
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


def find_best(H, D, C, jiaojv, S):
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
                    x[j] * cmath.sqrt(H ** 2 + (Y[j, i - start]+j*C*S[2, i]) ** 2) / cmath.sqrt(S[1, i] ** 2 + y[j] ** 2)).real
            N[j, i - start] = (S[0, i] + j * C * S[2, i]).real
            M[j, i - start] = (m[j] * cmath.sqrt(H ** 2 + (N[j, i - start]+j*C*S[2, i]) ** 2) / cmath.sqrt(
                S[1, i] ** 2 + n[j] ** 2)).real  # 通过图上坐标反演出所取点的实际坐标

        K1 = np.polyfit(X[:, i - start], Y[:, i - start], 1)
        K2 = np.polyfit(M[:, i - start], N[:, i - start], 1)  # 将所得实际坐标进行曲线拟合，得到图上虚线和实现对应的线
        math.atan(K1[1]) / math.pi * 180
        # plt.figure()
        # plt.plot(X[:, i - start], Y[:, i - start], 'b')
        # plt.plot(M[:, i - start], N[:, i - start], 'r')
        if math.atan(K1[0]) / math.pi * 180 > 0:
            deltatheta[i - start] = abs(
                90 - math.atan(K1[0]) / math.pi * 180 - math.acos(S[2, i]) / math.pi * 180)  # 筛选的实际物理条件
        else:
            deltatheta[i - start] = abs(math.atan(K1[0]) / math.pi * 180 - math.acos(S[2, i]) / math.pi * 180 + 90)
    Yr = 0
    fr = 0
    theta = 0
    if len(deltatheta) != 0:
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
    # plt.plot(deltatheta)
    # plt.show()
    return Yr, fr, theta


def linear_functions1(zuobiao):
    x1 = []
    y1 = []
    for i in range(0, zuobiao.shape[0], 1):
        x1.append(zuobiao[i, 0])
        y1.append(zuobiao[i, 1])
    K1 = np.polyfit(y1, x1, 1)
    k1 = K1[0]
    b1 = K1[1]
    return k1, b1


def linear_functions2(zuobiao):
    x1 = []
    y1 = []
    for i in range(0, zuobiao.shape[0], 1):
        x1.append(zuobiao[i, 2])
        y1.append(zuobiao[i, 3])
    K2 = np.polyfit(y1, x1, 1)
    k2 = K2[0]
    b2 = K2[1]
    return k2, b2

def linear_functions3(zuobiao):
    x1 = zuobiao[0, 0]
    x2 = zuobiao[1, 0]
    y1 = zuobiao[0, 1]
    y2 = zuobiao[1, 1]
    k1 = symbols('k1')
    b1 = symbols('b1')
    eq1 = Eq(k1 * y1 + b1 - x1, 0)
    eq2 = Eq(k1 * y2 + b1 - x2, 0)
    E1 = solve([eq1, eq2], [k1, b1])
    k1 = E1[k1]
    b1 = E1[b1]
    return k1, b1


def linear_functions4(zuobiao):
    x3 = zuobiao[0, 2]
    x4 = zuobiao[1, 2]
    y3 = zuobiao[0, 3]
    y4 = zuobiao[1, 3]
    k2 = symbols('k2')
    b2 = symbols('b2')
    eq3 = Eq(k2 * y3 + b2 - x3, 0)
    eq4 = Eq(k2 * y4 + b2 - x4, 0)
    E2 = solve([eq3, eq4], [k2, b2])
    k2 = E2[k2]
    b2 = E2[b2]
    return k2, b2


def draw_line_lane(k1, b1, k2, b2):
    for y in range(height - 1, -1, -1):
        x = k1 * y + b1
        x = int(x)
        img[y, x] = (255, 0, 0)
    for y in range(height - 1, -1, -1):
        x = k2 * y + b2
        x = int(x)
        img[y, x] = (255, 0, 0)
    return img

def draw_line_lane1(k1, b1, k2, b2):
    for y in range(height - 1, -1, -1):
        x = k1 * y + b1
        x = int(x)
        img[y, x] = (0, 255, 0)
    for y in range(height - 1, -1, -1):
        x = k2 * y + b2
        x = int(x)
        img[y, x] = (0, 255, 0)
    return img


def best(jiaojv, H, D, C, zuobiao, k1, b1, k2, b2, n_lane):
    distance_all = []
    l1 = jiaojv[0, 2] - jiaojv[0, 0]
    l2 = jiaojv[1, 2] - jiaojv[1, 0]
    l3 = jiaojv[2, 2] - jiaojv[2, 0]
    l4 = jiaojv[3, 2] - jiaojv[3, 0]
    # l5 = jiaojv[4, 2] - jiaojv[4, 0]
    # l6 = jiaojv[5, 2] - jiaojv[5, 0]
    l1 = int(l1)
    l2 = int(l2)
    l3 = int(l3)
    l4 = int(l4)
    # l5 = int(l5)
    # l6 = int(l6)
    for l_1 in np.arange(l1 - 2, l1 + 3, 1):
        for l_2 in np.arange(l2 - 2, l2 + 3, 1):
            for l_3 in np.arange(l3 - 2, l3 + 3, 1):
                for l_4 in np.arange(l4 - 2, l4 + 3, 1):
                    #     for l_5 in np.arange(l5 - 2, l5 + 3, 1):
                    #         for l_6 in np.arange(l6 - 2, l6 + 3, 1):
                    l = []
                    f_theta_distance = []
                    l.append(l_1)
                    l.append(l_2)
                    l.append(l_3)
                    l.append(l_4)
                    # l.append(l_5)
                    # l.append(l_6)
                    S = solve_nc(H, D, C, jiaojv, n_lane, l)
                    Yr, f, theta = find_best(H, D, C, jiaojv, S)
                    result_Y = detect_ceshi(k1, b1, k2, b2, f, theta, zuobiao)
                    distance = abs(result_Y[1] - result_Y[0] - C / 1000 * theta)
                    f_theta_distance.append(f)
                    f_theta_distance.append(theta)
                    f_theta_distance.append(distance)
                    f_theta_distance.append(Yr)
                    distance_all.append(f_theta_distance)
    min_distance = distance_all[0][2]
    for p in range(len(distance_all)):
        if distance_all[p][2] < min_distance:
            min_distance = distance_all[p][2]
    f = 0
    theta = 0
    fai = 0
    for i in range(len(distance_all)):
        if min_distance == distance_all[i][2]:
            # print(i)
            f = distance_all[i][0]
            theta = distance_all[i][1]
            fai = H / distance_all[i][3]
            print(distance_all[i][3])

    return f, fai, theta


def detect_ceshi(k1, b1, k2, b2, f, theta, ceshi_zuobiao):
    result_Y1 = []
    for q in range(ceshi_zuobiao.shape[0] - 1):
        a1 = k1 * ceshi_zuobiao[q, 1] + b1
        a2 = k2 * ceshi_zuobiao[q, 1] + b2
        l = abs(a1 - a2)
        y = ceshi_zuobiao[q, 1] - height / 2
        Y = cmath.sqrt((((D * n_lane) ** 2 * (f ** 2 + y ** 2)) / (l ** 2 * theta ** 2)) - H ** 2)
        Y = Y.real / 1000
        result_Y1.append(Y)
        result_Y1 = [round(x, 2) for x in result_Y1]
    return result_Y1

def cal_fai(f, theta, k1, b1, k2, b2, height, H):
    v = height/2
    a1 = k1 * v + b1
    a2 = k2 * v + b2
    l = abs(a1 - a2)
    Y = cmath.sqrt(
        (((D * n_lane) ** 2 * (f ** 2 + v ** 2)) / (l ** 2 * theta ** 2)) - H ** 2)
    fai = math.degrees(math.atan(H/Y.real))
    return fai

def distance_measurement(k1, b1, k2, b2, f, theta, wide, height):
    # cesu_zuobiao = np.matrix(
    #     [[503, 890, 1017, 890], [802, 606, 1137, 606], [947, 470, 1195, 470], [1032, 391, 1228, 391], [1087, 340, 1250, 340], [1127, 303, 1266, 303],
    #      [1156, 276, 1156, 276]], dtype=np.float64)
    cesu_zuobiao = np.matrix(
        [[1040, 844, 1481, 844], [955, 640, 1257, 640], [911, 533, 1139, 533], [885, 469, 1068, 469],
         [867, 426, 1021, 426], [854, 394, 986, 394], [844, 371, 960, 371]], dtype=np.float64)
    # cesu_zuobiao = np.matrix(
    #     [[962, 926, 1711, 926], [874, 585, 1289, 585], [842, 455, 1129, 455], [825, 387, 1044, 387], [815, 345, 991, 345],
    #      [809, 317, 957, 317], [805, 297, 933, 297]], dtype=np.float64)
    # cesu_zuobiao = np.matrix(
    #     [[1024, 1049, 1647, 1049], [871, 723, 1267, 723], [802, 573, 1092, 573], [763, 486, 991, 486],
    #      [737, 429, 927, 429], [721, 390, 882, 390], [709, 361, 850, 361]], dtype=np.float64)
    result_calculate = []
    result_truth = []
    result_error = []
    error_dx = []
    result_error_bfb = []
    error_bfb = []
    Y_result = 0
    X_result = []
    X_bfb = []
    for q in range(cesu_zuobiao.shape[0]):
        a1 = k1 * cesu_zuobiao[q, 1] + b1
        a2 = k2 * cesu_zuobiao[q, 1] + b2
        l = abs(a1 - a2)
        Y = cmath.sqrt(
            (((D * n_lane) ** 2 * (f ** 2 + (cesu_zuobiao[q, 1] - height / 2) ** 2)) / (l ** 2 * theta ** 2)) - H ** 2)
        X1 = (cesu_zuobiao[q, 0] - wide / 2) * (
                    cmath.sqrt(H ** 2 + Y ** 2) / cmath.sqrt(f ** 2 + (cesu_zuobiao[q, 1] - height / 2) ** 2))
        X2 = (cesu_zuobiao[q, 2] - wide / 2) * (
                    cmath.sqrt(H ** 2 + Y ** 2) / cmath.sqrt(f ** 2 + (cesu_zuobiao[q, 3] - height / 2) ** 2))
        X1 = X1.real / 1000
        X2 = X2.real / 1000
        Y = Y.real / 1000
        result_calculate.append(Y)
        X_diff = X2 - X1
        X_result.append(X_diff)
        # for c in range(len(result_calculate) - 1):
        #     error = result_calculate[c + 1] - result_calculate[c]
        #     if c % 2 == 0:
        #         error_dx = error - 9 * theta
        #     else:
        #         error_dx = error - 6 * theta
        # result_error.append(error_bfb)
        # for o in range(len(result_calculate) - 1):
        #     error = result_calculate[o + 1] - result_calculate[o]
        #     if o % 2 == 0:
        #         error_bfb = abs(100 - error * 100 / (9 * theta))
        #     else:
        #         error_bfb = abs(100 - error * 100 / (6 * theta))
        # result_error_bfb.append(error_bfb)
    for i in range(len(X_result) - 1):
        bfb = X_result[i] / (3.75 * n_lane)
        X_bfb.append(bfb)
    error = result_calculate[2] - result_calculate[0]
    error_dx = abs(error - 30 * theta)
    error_bfb = abs(100 - error * 100 / (30 * theta))
    result_error_bfb.append(error_bfb)
    result_error.append(error_dx)

    error = result_calculate[4] - result_calculate[0]
    error_dx = abs(error - 60 * theta)
    error_bfb = abs(100 - error * 100 / (60 * theta))
    # result_error_bfb.append(error_bfb)
    # result_error.append(error_dx)
    print("第二段误差为：", error_bfb)

    error = result_calculate[6] - result_calculate[0]
    error_dx = abs(error - 90 * theta)
    error_bfb = abs(100 - error * 100 / (90 * theta))
    # result_error.append(error_dx)
    # result_error_bfb.append(error_bfb)
    print("第三段误差为：", error_bfb)

    error = result_calculate[4] - result_calculate[2]
    error_dx = abs(error - 30 * theta)
    error_bfb = abs(100 - error * 100 / (30 * theta))
    result_error_bfb.append(error_bfb)
    result_error.append(error_dx)

    error = result_calculate[6] - result_calculate[4]
    error_dx = abs(error - 30 * theta)
    error_bfb = abs(100 - error * 100 / (30 * theta))
    result_error.append(error_dx)
    result_error_bfb.append(error_bfb)

    # del result_error[0]
    # del result_error_bfb[0]
    # result_calculate = [round(x, 2) for x in result_calculate]
    # result_error = [round(x, 2) for x in result_error]
    # result_error_bfb = [round(x, 2) for x in result_error_bfb]

    return result_calculate, result_error, result_error_bfb


if __name__ == '__main__':
    # 以下这五个参数根据实际参数改写
    H = 6000            # 相机架设高度    修改
    D = 3750            # 一条车道宽     修改
    C = 15000           # 前后两端点间距   修改
    n_lane = 3          # 车道数           修改
    # thresh = 130        # 二值化阈值    140 140 100 130              ccb修改
    thresh = 130        # 二值化阈值    140 140 100 130              修改

    # path = r'G:\HighwayData\datasets\images\train\23765.jpg'     # 修改
    # path = r'D:\chen_pythonfile\Yolov5_StrongSORT_OSNet-master\MOT16_eval\biye\.jpg'     # 修改  原来ccb
    path = r'D:\chen_pythonfile\Yolov5_StrongSORT_OSNet-master\MOT16_eval\new\4.jpg'     # 修改  zxg
    img = cv2.imread(path)
    height = img.shape[0]
    wide = img.shape[1]

    img2 = image_process(path, thresh)

    endpoint_coordinates = get_endpoint(wide, height)
    # print(endpoint_coordinates)

    img3 = draw_line(endpoint_coordinates, img, img2)
    cv2.imshow('image', img3)
    cv2.waitKey(0)
    # cv2.imwrite('C:/Users/zgshang/Desktop/picture/200.jpg', img3)

    detect_f_coordinates = CalcuCemeraIR_coordinates(endpoint_coordinates, wide, height)

    endpoint_coordinates = np.matrix(endpoint_coordinates, dtype=np.float64)
    print("endpoint_coordinates:", endpoint_coordinates)
    # k1, b1 = linear_functions1(endpoint_coordinates)
    # k2, b2 = linear_functions2(endpoint_coordinates)
    # print(k1,b1,k2,b2)
    k1, b1 = linear_functions3(endpoint_coordinates)
    k2, b2 = linear_functions4(endpoint_coordinates)
    # print(k1,b1,k2,b2)  #ccb修改
    print("k1,b1,k2,b2:",k1,b1,k2,b2)   #zxg修改
    # img = cv2.imread(path)
    # img4 = draw_line_lane(k1, b1, k2, b2)
    # img5 = draw_line_lane1(k3, b3, k4, b4)
    # cv2.imwrite('C:/Users/zgshang/Desktop/picture/300.jpg', img4)
    # cv2.imshow('image', img5)
    # cv2.waitKey(0)

    start_time = time.time()
    f, fai, theta = best(detect_f_coordinates, H, D, C, endpoint_coordinates, k1, b1, k2, b2, n_lane)
    print(f, theta)
    fai = cal_fai(f, theta, k1, b1, k2, b2, height, H)
    print("焦距为：", str(f), "俯仰角为：", str(fai), "旋转角为：", str(math.degrees(math.acos(theta))))  # 相机内参
    result_calculate, result_error, result_error_bfb = distance_measurement(k1, b1, k2, b2, f, theta, wide, height)
    print("result_calculate:",result_calculate)
    print("result_error:", result_error)
    print("result_error_bfb:", result_error_bfb)
    end_time = time.time()    # 程序结束时间
    run_time = end_time - start_time    # 程序的运行时间，单位为秒
    print("代码运行时间为：", run_time)