import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

gird_rows = 8
gird_cols = 11

objp = np.zeros((gird_rows*gird_cols,3), np.float32)
objp[:,:2] = np.mgrid[0:gird_cols,0:gird_rows].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('../Images/demo/*.png')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (gird_cols, gird_rows), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (gird_cols,gird_rows), corners2, ret)
        cv.imshow('img', cv.resize(img,None,None,fx=0.2,fy=0.2))
        cv.waitKey(1000)
    else:
        print("not found")

cv.destroyAllWindows()

print("Settings Complete...\n----------------------------")

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Calibration Complete...\n----------------------------")

# fix distortion
img = cv.imread('../Images/demo/1.png') # the image to be undistorted
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
if dst is None or dst.size == 0:
    print("Error: undistorted image is empty")
else:
    print("Undistorted image size:", dst.shape)

print("Height, Width:", h, w)
print("Camera Matrix:\n", mtx)
print("Distortion Coefficients:\n", dist)
print("newmtx:\n", newcameramtx)

# 打印 ROI 值
print("ROI:", roi)

# 检查 ROI 是否有效
if roi == (0, 0, 0, 0):
    print("Error: ROI is invalid, skipping cropping")
else:
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    if dst is None or dst.size == 0:
        print("Error: cropped image is empty")
    else:
        print("Cropped image size:", dst.shape)

# 保存图像
cv.imwrite('../ProjectedImages/calibresult.png', dst)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )
