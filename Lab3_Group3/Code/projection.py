import numpy as np
import cv2 as cv
import open3d as o3d
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((8*11,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:11].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('./train/*.png')

# print(images)

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (11,8), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        # cv.drawChessboardCorners(img, (11,8), corners2, ret)
        # cv.imshow('img', cv.resize(img,None,None,fx=0.2,fy=0.2))
        # if cv.waitKey(0) & 0xFF == ord('q'):
        #     break
    else:
        print(f"Chessboard not found in {fname}")

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

def read_ply_file(ply_file):
    """
    Read a PLY file and extract 3D points and color data (if available).
    This function assumes the PLY file contains vertex positions, normals, and RGB colors.
    """
    with open(ply_file, 'r') as f:
        lines = f.readlines()

    header_ended = False
    vertex_data = []
    colors = []
    
    for line in lines:
        if line.startswith("end_header"):
            header_ended = True
            continue
        if header_ended:
            parts = line.strip().split()
            if(len(parts) >= 9): 

                # Parse 3D coordinates (x, y, z)
                x, y, z = map(float, parts[:3])
                vertex_data.append([x, y, z])
                
                # Parse RGB colors (red, green, blue), skipping the normals
                r, g, b = map(int, parts[-3:])  # Color data starts at index 6, skipping nx, ny, nz
                # print([r,g,b])
                colors.append([r, g, b])

    return np.array(vertex_data), np.array(colors)


# 读取3D点数据
ply_file = 'cat.ply'
cat_points, cat_colors = read_ply_file(ply_file)

scale_factor = 30

cat_points = cat_points * scale_factor


for i, fname in enumerate(images):
    img = cv.imread(fname)

    fixed_tvec = tvecs[i].copy()  # Start from the chessboard's translation
    fixed_tvec[0] += 2  # Move the cat 2 squares along the x-axis
    fixed_tvec[1] += 2  # Move the cat 2 squares along the y-axis

    fixed_rvec = rvecs[i].copy()
    
    imgpoints2, _ = cv.projectPoints(cat_points, fixed_rvec, fixed_tvec, mtx, dist)
    
    # 转换为整数坐标
    imgpoints2 = np.int32(imgpoints2).reshape(-1, 2)

    for j, point in enumerate(imgpoints2):
        # Draw small circles for each point
        color = tuple(map(int, cat_colors[j]))
        # print(color)
        img = cv.circle(img, tuple(point), 3, color, -1)

    # 显示投影后的图像
    cv.imshow('Projected Cube', cv.resize(img,None,None,fx=0.5,fy=0.5))
    if cv.waitKey(0) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()