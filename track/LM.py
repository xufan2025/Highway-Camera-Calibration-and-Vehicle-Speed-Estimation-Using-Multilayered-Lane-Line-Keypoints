import numpy as np
from scipy.optimize import curve_fit
import time
import math

# 定义公式模型函数
def model(X, f, Y1, gamma):
    # 常量部分（已知参数）
    # Setup 1
    # a_prime = -0.419512195121951 # 填入实际数值
    # b_prime = 1392.37560975610 # 填入实际数值
    # a = -1.05853658536585  # 填入实际数值
    # b = 1444.72682926829  # 填入实际数值
    # W = 7500            # 填入实际数值
    # cos_theta = 0.97    # 填入实际数值

    # Setup 2
    # a_prime = 1.09150326797386   # 填入实际数值
    # b_prime = 559.104575163399 # 填入实际数值
    # a = 0.411764705882353  # 填入实际数值
    # b = 692.470588235294  # 填入实际数值
    # W = 7500  # 填入实际数值
    # cos_theta = 0.99  # 填入实际数值

    # Setup 3
    # a_prime = 1.23753665689150       # 填入实际数值
    # b_prime = 565.041055718475      # 填入实际数值
    # a = 0.258064516129032           # 填入实际数值
    # b = 724.032258064516        # 填入实际数值
    # W = 7500            # 填入实际数值
    # cos_theta = 0.99    # 填入实际数值

    # Setup 4
    a_prime = 1.16521739130435  # 填入实际数值
    b_prime = 425.608695652174  # 填入实际数值
    a = 0.465217391304348  # 填入实际数值
    b = 535.608695652174  # 填入实际数值
    W = 7500  # 填入实际数值
    cos_theta = 0.97  # 填入实际数值


    H = 6000            # 填入实际数值
    L = 15000           # 填入实际数值

    u_k_double_prime, v_k_double_prime = X


    # 分子部分
    term1_num = ( (1 - a_prime * math.tan(math.radians(gamma))) / (a_prime + math.tan(math.radians(gamma))) - (1 - a * math.tan(math.radians(gamma))) / (a + math.tan(math.radians(gamma))) ) * (math.tan(math.radians(gamma)) * u_k_double_prime + v_k_double_prime)
    term2_num = ( (1 - a_prime * math.tan(math.radians(gamma))) / (a_prime + math.tan(math.radians(gamma))) * b_prime - (1 - a * math.tan(math.radians(gamma))) / (a + math.tan(math.radians(gamma))) * b )
    term3_num = (b_prime * math.tan(math.radians(gamma)) - b * math.tan(math.radians(gamma)))

    numerator = term1_num - term2_num - term3_num

    # 分母部分
    denominator = np.sqrt((f/math.cos(math.radians(gamma)))**2 + (math.tan(math.radians(gamma)) * u_k_double_prime + v_k_double_prime)**2)

    # 左侧部分
    left_side =  denominator / numerator

    left_side = (np.sqrt((left_side * (W / cos_theta)) ** 2 - H**2) - Y1) / (L* cos_theta) + 1


    # 返回拟合函数的结果
    return left_side

# 输入数据：k值及对应的实际观测数据
# Set up 1
# u_k_double_prime = np.array([-201, -191, 110])
# v_k_double_prime = np.array([350, 66, -70])

# Set up 2
# u_k_double_prime = np.array([298, 144, 63])
# v_k_double_prime = np.array([304, 100, -7])

# Set up 3
# u_k_double_prime = np.array([387, 128, 29])
# v_k_double_prime = np.array([386, 45, -85])

# Set up 4
u_k_double_prime = np.array([373, 108, -15])
v_k_double_prime = np.array([509, 183, 33])

k_values = np.array([1, 2, 3])  # k的实际值
X_data = np.vstack((u_k_double_prime, v_k_double_prime))

# 进行拟合
# Set up 1
# initial_guess = [ 1800.78, 32019, 2.41]  # 初始猜测值 (f, Y1, tan_gamma)

# Set up 2
# initial_guess = [1876.94, 38097, -10.26]  # 初始猜测值 (f, Y1, tan_gamma)

# Set up 3
# initial_guess = [1813.2607993893284, 17833, -1.36]  # 初始猜测值 (f, Y1, tan_gamma)

# Set up 4
initial_guess = [2065.16, 32279, -20.79]  # 初始猜测值 (f, Y1, tan_gamma)

params, covariance = curve_fit(model, X_data, k_values, p0=initial_guess, method='lm', maxfev = 50000)

# 拟合结果
f_fit, Y1_fit, gamma_fit = params
# Set up 1
# print(f"Fitted parameters: f = {f_fit}, Y1 = {Y1_fit}, gamma = {gamma_fit}")

# Set up 2
# print(f"Fitted parameters: f = {f_fit}, Y1 = {Y1_fit}, gamma = {gamma_fit+7}")

# Set up 3
# print(f"Fitted parameters: f = {f_fit}, Y1 = {Y1_fit}, gamma = {gamma_fit}")

# Set up 4
print(f"Fitted parameters: f = {f_fit}, Y1 = {Y1_fit}, gamma = {gamma_fit+24}")
