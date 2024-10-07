import cv2 as cv
import numpy as np

img = cv.imread('../Images/clarity/blurred_0.png')

if img is None:
    print("Error: Image not found or unable to read.")
else:
    for i in range(1, 11):
        ksize = 4 * i + 1
        blurred_img = cv.GaussianBlur(img, (ksize, ksize), 0)
        cv.imwrite(f'../Images/clarity/blurred_{i}.png', blurred_img)
        print(f'Generated blurred image with kernel size {ksize}.')

print("All blurred images have been generated.")
