import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
from natsort import natsorted

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

gird_rows = 8
gird_cols = 11

objp = np.zeros((gird_rows * gird_cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:gird_cols, 0:gird_rows].T.reshape(-1, 2)

errors = []

# 读取不同清晰度的图像
images = natsorted(glob.glob('../Images/clarity/blurred_*.png'))

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (gird_cols, gird_rows), None)

    if ret == True:
        objpoints = [objp]
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints = [corners2]

        # Calibrate camera with the single image
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # Calculate total error
        imgpoints2, _ = cv.projectPoints(objp, rvecs[0], tvecs[0], mtx, dist)
        error = cv.norm(corners2, imgpoints2, cv.NORM_L2) / len(imgpoints2)
        errors.append(error)
        print(f"Total error for {fname}: {error}")
    else:
        print(f"Chessboard corners not found in image: {fname}")

# 提取图像编号
image_indices = [int(fname.split('_')[-1].split('.')[0]) for fname in images]

# Plot clarity vs total error
plt.figure()
plt.plot(image_indices, errors, marker='o')
plt.xlabel('Image Index (Clarity)')
plt.ylabel('Total Error')
plt.title('Clarity vs Total Error')
plt.grid(True)
plt.savefig('../Images/clarity/clarity_vs_error.png')
plt.show()
