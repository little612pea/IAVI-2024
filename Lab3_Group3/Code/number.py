import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

gird_rows = 8
gird_cols = 11

objp = np.zeros((gird_rows*gird_cols,3), np.float32)
objp[:,:2] = np.mgrid[0:gird_cols,0:gird_rows].T.reshape(-1,2)

# Read images from the number directory
images = sorted(glob.glob('../Images/shift/*.png'))
num_images = len(images)
errors = []

for n in range(1, num_images + 1):
    objpoints = []
    imgpoints = []

    for fname in images[:n]:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (gird_cols, gird_rows), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
        else:
            print(f"Chessboard corners not found in image: {fname}")
            break

    if len(objpoints) == n and len(imgpoints) == n:
        # Calibrate camera with the n images
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # Calculate total error
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
            mean_error += error

        total_error = mean_error / len(objpoints)
        errors.append(total_error)
        print(f"Total error for {n} images: {total_error}")
    else:
        print(f"Skipping calibration for {n} images due to insufficient valid images.")

# Plot number of images vs total error
plt.figure()
plt.plot(range(1, num_images + 1), errors, marker='o')
plt.xlabel('Number of Images')
plt.ylabel('Total Error')
plt.title('Number of Images vs Total Error')
plt.grid(True)
plt.savefig('../ProjectedImages/num_images_vs_error.png')
plt.show()
