import numpy as np
import cv2
import glob

# Wait time to show calibration in 'ms'
WAIT_TIME = 1000

# termination criteria for iterative algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# generalizable checkerboard dimensions
# https://stackoverflow.com/questions/31249037/calibrating-webcam-using-python-and-opencv-error?rq=1
chessboard_rows = 6
chessboard_columns = 9
square_size = 15  # wymiar pola na szachownic w mm

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
# IMPORTANT : Object points must be changed to get real physical distance.
objp = np.zeros((chessboard_rows * chessboard_columns, 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_columns, 0:chessboard_rows].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('img/cal/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # znajdź rogi na szachownicy (wewnętrzne czarne pola)
    ret, corners = cv2.findChessboardCorners(gray, (chessboard_columns,chessboard_rows),None)
    print(f'fname: {fname}, ret: {ret}')

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        # popraw dokładność znalezionych punktów
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # narysuj na obraz znalezione punkty
        img = cv2.drawChessboardCorners(img, (chessboard_columns, chessboard_rows), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(WAIT_TIME)

cv2.destroyAllWindows()
# dokonaj kalibracji, zwróć camera matrix, distortion coefficients, rotation and translation vectors
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

# projection_matrix = camera_matrix * rotation | camera_matrix * translation
rotation_mat = np.zeros(shape=(3, 3))
R = cv2.Rodrigues(rvecs[0], rotation_mat)[0]
P = np.column_stack((np.matmul(mtx,R), tvecs[0]))

# ---------- Saving the calibration -----------------
cv_file = cv2.FileStorage("calibrated.yaml", cv2.FILE_STORAGE_WRITE)
cv_file.write("camera_matrix", mtx)
cv_file.write("dist_coeff", dist)

# note you *release* you don't close() a FileStorage object
cv_file.release()