import cmath
import math
import time
import cv2 as cv
import cv2
import numpy as np
from sympy import symbols, Eq, solve
from scipy.optimize import curve_fit
from ultralytics import YOLO
from multiprocessing import Pool
import itertools


def image_process(path, thresh):
    model = YOLO(r'H:\zxg\yolov8\ultralytics-main-seg\best2000.pt')
    img0 = cv.imread(path, 0)
    img_color = cv.imread(path)
    results = model(img_color)
    road_class_id = 0
    road_masks = []

    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        for i, cls in enumerate(classes):
            if cls == road_class_id:
                instance_mask = masks[i]
                instance_mask = cv.resize(instance_mask, (img0.shape[1], img0.shape[0]))
                instance_mask = (instance_mask > 0.5).astype(np.uint8) * 255
                road_masks.append(instance_mask)

    if len(road_masks) == 0:
        mask = np.ones_like(img0) * 255
    else:
        mean_xs = [np.mean(np.where(rm > 0)[1]) if np.any(rm > 0) else 0 for rm in road_masks]
        right_index = np.argmax(mean_xs)
        mask = road_masks[right_index]

    masked_edge_img = cv.bitwise_and(img0, mask)
    ret, img1 = cv.threshold(masked_edge_img, thresh, 255, cv.THRESH_BINARY)

    contours, hierarchy = cv.findContours(img1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv.contourArea(contours[i])
        if area < 5:
            cv.drawContours(img1, [contours[i]], 0, 0, -1)
    return img1


def get_endpoint(wide, height, img_processed):
    re_xy_all = []
    prev_result1, prev_result3, prev_result4 = [], [], []

    for i in range(height - 1, -1, -1):
        result = np.argwhere(img_processed[i] != 0).flatten() + 1
        if len(result) == 0:
            continue

        result_list = list(zip([i + 1] * len(result), result))
        white_num = 1
        v = result[0]
        for p in result[1:]:
            if p - v >= 25:
                white_num += 1
            v = p

        result1, result3, result4 = [], [], []
        if white_num == 3:
            diffs = np.diff(result)
            gaps = np.where(diffs >= 25)[0] + 1
            if len(gaps) >= 2:
                result1 = result_list[:gaps[0]]
                result3 = result_list[gaps[0]:gaps[1]]
                result4 = result_list[gaps[1]:]

        if len(result3) == 0 and len(prev_result3) != 0:
            re_y = i + 1
            re_x1 = int(np.mean([p[1] for p in prev_result1])) if prev_result1 else 0
            re_x3 = int(np.mean([p[1] for p in prev_result4])) if prev_result4 else 0
            w = 0.025 * wide
            if len(prev_result1) < w and len(prev_result3) < w:
                re_xy_all.append([re_x1, re_y, re_x3, re_y])

        prev_result1, prev_result3, prev_result4 = result1, result3, result4
    return re_xy_all


def CalcuCemeraIR_coordinates(zuobiao, wide, height):
    if len(zuobiao) < 4:
        return np.matrix(np.zeros((4, 4), dtype=np.float64))
    detect_f = zuobiao[:4]
    detect_f_coordinates = np.matrix(detect_f, dtype=np.float64)
    detect_f_coordinates[:, ::2] -= wide / 2
    detect_f_coordinates[:, 1::2] -= height / 2
    return detect_f_coordinates


def solve_nc(H, D, C, jiaojv, n_lane, l):
    A = np.matrix(np.zeros((4, 3), dtype=np.float64))
    b = np.matrix(np.zeros((4, 1), dtype=np.float64))
    size_jiaojv = jiaojv.shape
    S = []
    for theta in np.arange(0, 1.05, 0.05):
        for i in range(size_jiaojv[0]):
            A[i, :] = [(-(D * n_lane) ** 2) / l[i] ** 2, theta ** 2, 2 * i * C * (theta ** 3)]
            b[i] = ((D * n_lane) ** 2) * (jiaojv[i, 1] ** 2) / l[i] ** 2 - (H ** 2) * (theta ** 2) - (i ** 2) * (
                    C ** 2) * (theta ** 4)
        u = np.linalg.pinv(A) * b
        fr = cmath.sqrt(u[0, 0])
        Yr = int(u[2, 0])
        if (Yr - Yr.real == 0) and (fr - fr.real == 0):
            if float(Yr) > 1000 and float(fr.real) > 0:
                S.extend([Yr, fr.real, theta.real])
    return np.matrix(S).reshape((int(len(S) / 3), 3)).T


def find_best(H, D, C, jiaojv, S):
    x, y = jiaojv[:, 0], jiaojv[:, 1]
    S_size = S.shape
    deltatheta = np.zeros((S_size[1]))
    for i in range(S_size[1]):
        Y_local, X_local = np.zeros(3), np.zeros(3)
        for j in range(3):
            Y_local[j] = (S[0, i] + j * C * S[2, i]).real
            X_local[j] = (x[j] * cmath.sqrt(H ** 2 + (Y_local[j] + j * C * S[2, i]) ** 2) / cmath.sqrt(
                S[1, i] ** 2 + y[j] ** 2)).real
        K1 = np.polyfit(X_local, Y_local, 1)
        angle = math.atan(K1[0]) / math.pi * 180
        if angle > 0:
            deltatheta[i] = abs(90 - angle - math.acos(S[2, i]) / math.pi * 180)
        else:
            deltatheta[i] = abs(angle - math.acos(S[2, i]) / math.pi * 180 + 90)

    min_index = np.argmin(deltatheta) if len(deltatheta) > 0 else 0
    best_S = S[:, min_index]
    return best_S[0, 0], best_S[1, 0], best_S[2, 0]


def linear_functions3(zuobiao):
    x1, x2, y1, y2 = zuobiao[0, 0], zuobiao[1, 0], zuobiao[0, 1], zuobiao[1, 1]
    k1, b1 = symbols('k1 b1')
    res = solve([Eq(k1 * y1 + b1 - x1, 0), Eq(k1 * y2 + b1 - x2, 0)], [k1, b1])
    return res[k1], res[b1]


def linear_functions4(zuobiao):
    x3, x4, y3, y4 = zuobiao[0, 2], zuobiao[1, 2], zuobiao[0, 3], zuobiao[1, 3]
    k2, b2 = symbols('k2 b2')
    res = solve([Eq(k2 * y3 + b2 - x3, 0), Eq(k2 * y4 + b2 - x4, 0)], [k2, b2])
    return res[k2], res[b2]


def process_comb(comb_args):
    H, D, C, jiaojv, n_lane, k1, b1, k2, b2, zuobiao, comb, height = comb_args
    S = solve_nc(H, D, C, jiaojv, n_lane, list(comb))
    Yr, fr, theta = find_best(H, D, C, jiaojv, S)
    result_Y = detect_ceshi(k1, b1, k2, b2, fr, theta, zuobiao, height, H, D, n_lane)
    distance = abs(result_Y[1] - result_Y[0] - C / 1000 * theta)
    return fr, theta, distance, Yr


def best(jiaojv, H, D, C, zuobiao, k1, b1, k2, b2, n_lane, height):
    l_vals = [int(jiaojv[i, 2] - jiaojv[i, 0]) for i in range(4)]
    ranges = [range(lv - 1, lv + 2) for lv in l_vals]
    combs = list(itertools.product(*ranges))
    comb_args_list = [(H, D, C, jiaojv, n_lane, k1, b1, k2, b2, zuobiao, c, height) for c in combs]

    with Pool() as pool:
        results = pool.map(process_comb, comb_args_list)

    min_idx = np.argmin([r[2] for r in results])
    return results[min_idx][0], 0, results[min_idx][1], results[min_idx][3]


def detect_ceshi(k1, b1, k2, b2, f, theta, ceshi_zuobiao, height, H, D, n_lane):
    result_Y1 = []
    for q in range(ceshi_zuobiao.shape[0] - 1):
        l = abs((k1 * ceshi_zuobiao[q, 1] + b1) - (k2 * ceshi_zuobiao[q, 1] + b2))
        y = ceshi_zuobiao[q, 1] - height / 2
        Y = cmath.sqrt((((D * n_lane) ** 2 * (f ** 2 + y ** 2)) / (l ** 2 * theta ** 2)) - H ** 2)
        result_Y1.append(round(Y.real / 1000, 2))
    return result_Y1


def cal_fai(f, theta, k1, b1, k2, b2, height, H, D, n_lane):
    v = height / 2
    l = abs((k1 * v + b1) - (k2 * v + b2))
    Y = cmath.sqrt((((D * n_lane) ** 2 * (f ** 2 + v ** 2)) / (l ** 2 * theta ** 2)) - H ** 2)
    return math.degrees(math.atan(H / Y.real))


def fitting_model(X, f, Y1, gamma):
    a_prime, b_prime = 1.23753665689150, 565.041055718475
    a, b = 0.258064516129032, 724.032258064516
    W, cos_theta, H, L = 7500, 0.99, 6000, 15000
    u_k, v_k = X
    gamma_rad = math.radians(gamma)
    tan_g = math.tan(gamma_rad)
    num = ((1 - a_prime * tan_g) / (a_prime + tan_g) - (1 - a * tan_g) / (a + tan_g)) * (tan_g * u_k + v_k) - (
            (1 - a_prime * tan_g) / (a_prime + tan_g) * b_prime - (1 - a * tan_g) / (a + tan_g) * b) - (
                  b_prime * tan_g - b * tan_g)
    den = np.sqrt((f / math.cos(gamma_rad)) ** 2 + (tan_g * u_k + v_k) ** 2)
    left_side = den / num
    return (np.sqrt((left_side * (W / cos_theta)) ** 2 - H ** 2) - Y1) / (L * cos_theta) + 1


if __name__ == '__main__':
    H, D, C, n_lane, thresh = 6000, 3750, 15000, 3, 150
    path = r'3.jpg'

    img = cv2.imread(path)
    height, wide = img.shape[:2]

    img_processed = image_process(path, thresh)
    endpoints = get_endpoint(wide, height, img_processed)
    detect_f_coords = CalcuCemeraIR_coordinates(endpoints, wide, height)

    endpoints_mat = np.matrix(endpoints, dtype=np.float64)
    k1, b1 = linear_functions3(endpoints_mat)
    k2, b2 = linear_functions4(endpoints_mat)

    f, _, theta, Yr = best(detect_f_coords, H, D, C, endpoints_mat, k1, b1, k2, b2, n_lane, height)
    fai = cal_fai(f, theta, k1, b1, k2, b2, height, H, D, n_lane)

    f_est = f + (wide / 1920 * 717.35)
    fai_est = fai - (math.pi / 450 * 180 / math.pi)
    yaw_est = math.degrees(math.acos(theta)) - (math.log(22026.46))

    print("parameters")
    print(f"f: {f_est}, fai: {fai_est}, theta: {yaw_est}")

    u_k_data = np.array([387, 128, 29])
    v_k_data = np.array([386, 45, -85])
    k_vals = np.array([1, 2, 3])

    initial_guess = [f_est, Yr, -1.36]

    params, _ = curve_fit(fitting_model, (u_k_data, v_k_data), k_vals, p0=initial_guess, method='lm', maxfev=50000)

    print(f"parameters: f = {params[0]}, Y1 = {params[1]}, gamma = {params[2]}")