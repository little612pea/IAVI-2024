import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
gain_images = []
gains = []
for i in range(0, 98):
    i = i/10
    image = cv2.imread(f'../Images/gain/gain_{i}.png', cv2.IMREAD_GRAYSCALE)  # 加载为灰度图
    image = image.astype(np.float32)  # 转换为浮点型以便后续处理
    gain_images.append(image)
    gains.append(i)

# 计算每个增益下的图像像素均值
mean_pixel_values = [np.mean(gain_image) for gain_image in gain_images]

# 绘制增益与均值像素值的关系
plt.figure(figsize=(6, 4))
plt.plot(gains, mean_pixel_values, marker='o')
plt.title('Gain vs. Mean Pixel Value')
plt.xlabel('Gain')
plt.ylabel('Mean Pixel Value')
plt.grid(True)
plt.show()