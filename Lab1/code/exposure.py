from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# 打开BMP文件
start = 2000

end = 98000

def exposure(pixel):
        # 获取每个通道的值
        red, green, blue = pixel
        
        # 计算曝光后的值
        red = int(red)
        green = int(green)
        blue = int(blue)

        result = pow(red/255, 2.2) + pow(green/255, 2.2) + pow(blue/255, 2.2)

        light = 0.547373 * pow(result, 1/2.2)

        return light

exposures = []
lights = []

for i in range(start, end, 2000):
    file_name = f'../Images/exposure/exposure_2000_98000/exposure_{i}_us.png'
    image = Image.open(file_name)

    # 将图像转换为RGB模式（可根据具体需求调整）
    image = image.convert('RGB')

    # 获取图像尺寸
    width, height = image.size

    # 遍历每个像素1296, 972
    pixel = image.getpixel((1296, 972))  
    light = exposure(pixel)  

    exposures.append(i)
    lights.append(light)

    print(f"exposure time = {i} 的值: {light}")


plt.scatter(exposures, lights)

# 添加标题和标签
plt.title('Sample Line Plot')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

# 显示图像
plt.show()