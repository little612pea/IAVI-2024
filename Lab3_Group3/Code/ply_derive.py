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
        cv.drawChessboardCorners(img, (11,8), corners2, ret)
        cv.imshow('img', cv.resize(img,None,None,fx=0.2,fy=0.2))
        if cv.waitKey(0) & 0xFF == ord('q'):
            break

    else:
        print(f"Chessboard not found in {fname}")

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)



def write_ply(filename, points, colors=None):
    ply_header = '''ply
format ascii 1.0
element vertex {0}
property float x
property float y
property float z
'''.format(len(points))
    
    if colors is not None:
        ply_header += '''property uchar red
property uchar green
property uchar blue
'''
    
    ply_header += '''end_header
'''
    
    with open(filename, 'w') as f:
        f.write(ply_header)
        # print(len(colors))
        for i in range(len(points)):
            point_str = f"{points[i, 0]} {points[i, 1]} {points[i, 2]}"
            if colors is not None:
                point_str += f" {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}"
            f.write(point_str + '\n')


camera_centers = []
for rvec, tvec in zip(rvecs, tvecs):
    R, _ = cv.Rodrigues(rvec)  # 将旋转向量转换为旋转矩阵
    camera_center = -np.dot(R.T, tvec)  # 计算摄像机中心
    camera_centers.append(camera_center.flatten())

# 将棋盘格点和摄像机中心点整合
# 假设 objp 是棋盘格的 3D 点

objpoints = np.array(objpoints)
camera_centers = np.array(camera_centers)

# 检查并确保它们是 2D 数组
print(f"objpoints 的维度: {objpoints.ndim}")
print(f"camera_centers 的维度: {camera_centers.ndim}")

if objpoints.ndim == 3:
    objpoints = objpoints.reshape(-1, 3)

all_points = np.vstack((objpoints , camera_centers))  # 合并棋盘格点和摄像机中心


# 创建颜色数据：为棋盘点设为白色，为摄像机中心设为红色
colors = np.vstack([np.full((len(objpoints), 3), [255, 255, 255]),  # 棋盘点为白色
                    np.full((len(camera_centers), 3), [255, 0, 0])])  # 摄像机中心为红色
    

# 导出到 .ply 文件
write_ply('chessboard_camera.ply', all_points, colors)
