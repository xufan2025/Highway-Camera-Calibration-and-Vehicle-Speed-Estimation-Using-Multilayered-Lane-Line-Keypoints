import cv2
from matplotlib import font_manager
from sympy.physics.control.control_plots import plt
from matplotlib import rcParams
# my_font = font_manager.FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
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


deltatheta = [49.26588487, 49.38213002, 49.44141253, 49.45296018, 49.42179937, 49.35317386,
              49.24942609, 49.11278461, 48.94546308, 48.74969444, 48.52624151, 48.27668882,
              48.0018965,  47.70344035, 47.38144514, 47.03756154, 46.67209836, 46.28568262,
              45.87949182, 45.45329668, 45.00779267, 44.54383086, 44.0613265,  43.56069786,
              43.04227176, 42.50629013, 41.95248409, 41.38120193, 40.79228116, 40.18570542,
              39.56156,    38.91930101, 38.25886017, 37.57973609, 36.88166876, 36.16383468,
              35.42591627, 34.66689037, 33.88603851, 33.08234874, 32.25452089, 31.40131329,
              30.5209036,  29.61177847, 28.67143006, 27.69741399, 26.68679558, 25.63575047,
              24.54009117, 23.39427633, 22.19175123, 20.92410996, 19.58048473, 18.14634992,
              16.60188419, 14.91836262, 13.05153987, 10.92700421,  8.39925109,  5.09241848,
              2.95717035, 5.1028364]

costheta = [0.39, 0.40, 0.41, 0.42, 4.30000000e-01,
            4.40000000e-01, 4.50000000e-01, 4.60000000e-01, 4.70000000e-01,
            4.80000000e-01, 4.90000000e-01, 5.00000000e-01, 5.10000000e-01,
            5.20000000e-01, 5.30000000e-01, 5.40000000e-01, 5.50000000e-01,
            5.60000000e-01, 5.70000000e-01, 5.80000000e-01, 5.90000000e-01,
            6.00000000e-01, 6.10000000e-01, 6.20000000e-01, 6.30000000e-01,
            6.40000000e-01, 6.50000000e-01, 6.60000000e-01, 6.70000000e-01,
            6.80000000e-01, 6.90000000e-01, 7.00000000e-01, 7.10000000e-01,
            7.20000000e-01, 7.30000000e-01, 7.40000000e-01, 7.50000000e-01,
            7.60000000e-01, 7.70000000e-01, 7.80000000e-01, 7.90000000e-01,
            8.00000000e-01, 8.10000000e-01, 8.20000000e-01, 8.30000000e-01,
            8.40000000e-01, 8.50000000e-01, 8.60000000e-01, 8.70000000e-01,
            8.80000000e-01, 8.90000000e-01, 9.00000000e-01, 9.10000000e-01,
            9.20000000e-01, 9.30000000e-01, 9.40000000e-01, 9.50000000e-01,
            9.60000000e-01, 9.70000000e-01, 9.80000000e-01, 9.90000000e-01,
            1.00000000e+00]
fig = plt.figure(figsize=(7,5.2))

# plt.rcParams['font.family']='serif'
plt.plot(costheta, deltatheta,linewidth = 3)
plt.grid(linestyle='--')
# plt.xlabel(u"cos$\Theta$$_{i}$", fontprorties=my_font, fontsize=20)
# plt.title("Rotation Angle Error", fontsize='20', loc='center', color='black')
plt.xlabel(u"$cos\\theta_{i}$", fontsize=18)
plt.ylabel(u"Rotation Angle Error", fontsize=18)
plt.scatter(9.90000000e-01, 2.95717035, color='r', marker='o', s = 70, alpha = 1)
plt.scatter(0.42, 49.45296018, color='r', marker='o', s = 70, alpha = 1)

# plt.annotate(f'({0.99:.2f}, {2.95717035:.2f})', xy=(0.99, 2.95717035), xytext=(0.99-0.12, 2.95717035))
# plt.annotate(f'({0.42:.2f}, {49.45296018:.2f})', xy=(0.42, 49.45296018), xytext=(0.42-0.05, 49.45296018-4))
plt.text(0.415, 49.45296018-3, 'a',fontsize=20, ha='left', rotation=0, wrap=True)
plt.text(0.99-0.03, 2.95717035, 'b',fontsize=20, ha='left', rotation=0, wrap=True)

# plt.xlabel('Predicted label', fontsize=20)
# plt.ylabel('True label', fontsize=20)
# plt.show()
# plt.savefig('C:/Users/zgshang/Desktop/Highway-Paper/picture/62.png')
plt.savefig(r"C:\Users\zgshang\Desktop\new/62.png")
