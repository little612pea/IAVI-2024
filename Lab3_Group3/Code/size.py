import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Define chessboard dimensions for different sizes
sizes = {
    'large': (8, 11),
    'mid': (8, 11),  # 确保与工作代码一致
    'small': (8, 11)  # 确保与工作代码一致
}

errors = []

for size, (gird_rows, gird_cols) in sizes.items():
    objp = np.zeros((gird_rows * gird_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:gird_cols, 0:gird_rows].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    images = sorted(glob.glob(f'../Images/size/{size}/*.png'))

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (gird_cols, gird_rows), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
        else:
            print(f"Chessboard corners not found in image: {fname}")
            continue

    if len(objpoints) > 0 and len(imgpoints) > 0:
        # Calibrate camera with the images
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # Calculate total error
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
            mean_error += error

        total_error = mean_error / len(objpoints)
        errors.append((size, total_error))
        print(f"Total error for {size} size: {total_error}")
    else:
        print(f"Skipping calibration for {size} size due to insufficient valid images.")

# Plot size vs total error
sizes, total_errors = zip(*errors)
plt.figure()
plt.bar(sizes, total_errors)
plt.xlabel('Chessboard Size')
plt.ylabel('Total Error')
plt.title('Chessboard Size vs Total Error')
plt.grid(True)
plt.savefig('../ProjectedImages/size_vs_error.png')
plt.show()
