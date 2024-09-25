import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 假设我们有多张同尺寸的图像存储在一个列表中
image_list = [cv2.imread(f'../Images/noise/noise_{i}.png') for i in range(1, 50)]  # 读取图像

# 确保图像尺寸相同
H, W, C = image_list[0].shape  # 获取图像的尺寸 (高度, 宽度, 通道数)
N = len(image_list)  # 图像的数量

# 初始化累计图像，大小与图像相同，类型为float32
accumulated_image = np.zeros((H, W, C), dtype=np.float32)

# 逐像素累加 (保持 float32 类型)
for img in image_list:
    accumulated_image += img.astype(np.float32)

# 计算平均图像
average_image = accumulated_image / N

# 生成差异图像（残差图像）
residual_images = []
for img in image_list:
    # 计算残差图像，使用逐像素相减，保留浮点数
    residual_image = (img.astype(np.float32) - average_image)
    residual_images.append(residual_image)

# 将残差图像转换为灰度图（只分析强度，不分析颜色）
residual_grayscale_images = [cv2.cvtColor(residual_image.astype(np.float32), cv2.COLOR_BGR2GRAY)
                             for residual_image in residual_images]

# 将所有残差图像中的像素值拉平为一个向量（用于统计分析）
residual_values = np.concatenate([residual_image.flatten() for residual_image in residual_grayscale_images])

print(residual_values)

# 统计分析残差像素值：拟合正态分布
mean_value, std_value = norm.fit(residual_values)  # 拟合正态分布，获取均值和标准差

print(f"拟合的正态分布均值: {mean_value}")
print(f"拟合的正态分布标准差: {std_value}")

# 绘制残差像素值的直方图
plt.figure(figsize=(10, 6))
bins = np.arange(-5, 5, 0.01)
plt.hist(residual_values, bins=bins, density=True, alpha=0.6, color='g', label='Residual Data')

# 生成拟合的正态分布曲线
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean_value, std_value)

# 绘制拟合的正态分布曲线
plt.plot(x, p, 'k', linewidth=2, label='Fitted Normal Distribution')
plt.title('Residual Pixel Value Distribution with Normal Fit')
plt.xlabel('Pixel Intensity')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.xlim(-5, 5)
plt.show()