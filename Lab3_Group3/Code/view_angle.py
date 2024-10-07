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

# Read and undistort images from the shift directory
shift_images = sorted(glob.glob('../Images/shift/*.png'))
angles = []
errors = []

# Skip the first and last image
for i in range(1, len(shift_images) - 1):
    angle = int(shift_images[i].split('/')[-1].split('.')[0])
    angles.append(angle)
    
    img_files = [shift_images[i-1], shift_images[i], shift_images[i+1]]
    objpoints = []
    imgpoints = []

    for fname in img_files:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (gird_cols, gird_rows), None)

        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
        else:
            print(f"Chessboard corners not found in image: {fname}")
            break

    if len(objpoints) == 3 and len(imgpoints) == 3:
        # Calibrate camera with the three images
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # Undistort the middle image
        img = cv.imread(shift_images[i])
        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)
        if dst is None or dst.size == 0:
            print("Error: undistorted image is empty")
            continue
        else:
            print(f"Undistorted image size for angle {angle}:", dst.shape)

        # Print ROI value
        print("ROI:", roi)

        # Check if ROI is valid
        if roi == (0, 0, 0, 0):
            print("Error: ROI is invalid, skipping cropping")
        else:
            # Crop the image
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            if dst is None or dst.size == 0:
                print("Error: cropped image is empty")
            else:
                print("Cropped image size:", dst.shape)

        # Save the undistorted image
        # cv.imwrite(f'../ProjectedImages/calibresult_{angle}.png', dst)

        # Calculate total error
        imgpoints2, _ = cv.projectPoints(objp, rvecs[1], tvecs[1], mtx, dist)
        error = cv.norm(imgpoints[1], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        errors.append(error)
        print(f"Total error for angle {angle}: {error}")
    else:
        print(f"Skipping angle {angle} due to insufficient valid images.")

# Sort angles and errors by angle
sorted_indices = np.argsort(angles)
angles = np.array(angles)[sorted_indices]
errors = np.array(errors)[sorted_indices]

# Plot angle vs total error
plt.figure()
plt.plot(angles, errors, marker='o')
plt.xlabel('Angle (degrees)')
plt.ylabel('Total Error')
plt.title('Angle vs Total Error')
plt.grid(True)
plt.savefig('angle_vs_error.png')
plt.show()
