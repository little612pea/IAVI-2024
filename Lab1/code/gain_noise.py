import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mean_noises = []
std_noises = []
gains = []

for i in range(0,98):
    i = i/10
    # 读取图像
    image = cv2.imread(f'../Images/gain/gain_{i}.png', cv2.IMREAD_GRAYSCALE)

    # 应用高斯滤波（平滑图像）
    gaussian_filtered = cv2.GaussianBlur(image, (5, 5), 0)

    # 计算残差图像（即噪声图像）
    noise_image = image.astype(np.float32) - gaussian_filtered.astype(np.float32)

    # 将噪声图像转换为一维向量
    noise_values = noise_image.flatten()

    # 拟合正态分布
    mean_noise, std_noise = norm.fit(noise_values)

    mean_noises.append(mean_noise)
    std_noises.append(std_noise)
    gains.append(i)

    # 输出噪声均值和标准差
    # print(f"噪声均值: {mean_noise}")
    # print(f"噪声标准差: {std_noise}")

plt.figure(figsize=(6, 4))
plt.plot(gains, std_noises, marker='o')
plt.title('Gain vs. Noise Standard Deviation')
plt.xlabel('Gain')
plt.ylabel('Noise Standard Deviation')
plt.grid(True)
plt.show()