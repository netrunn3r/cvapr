import numpy as np
import cv2
import glob

def calibrate(cam_id):
    # Wait time to show calibration in 'ms'
    WAIT_TIME = 100

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

    images = glob.glob(f'img/cal/*{cam_id}.jpg')

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
            #img = cv2.drawChessboardCorners(img, (chessboard_columns, chessboard_rows), corners2,ret)
            #cv2.imshow('img',img)
            #cv2.waitKey(WAIT_TIME)

    cv2.destroyAllWindows()
    # dokonaj kalibracji, zwróć camera matrix, distortion coefficients, rotation and translation vectors
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    return (mtx, rvecs, tvecs)
    # ---------- Saving the calibration -----------------
    #cv_file = cv2.FileStorage('calibrated.yaml', cv2.FILE_STORAGE_WRITE)
    #cv_file.write(f'camera_matrix_{cam_id}', mtx)
    #cv_file.write(f'dist_coeff_{cam_id}', dist)
    #cv_file.write(f'rotation_vector_{cam_id}', rvecs)
    #cv_file.write(f'translation_vector_{cam_id}', tvecs)

    # note you *release* you don't close() a FileStorage object
    #cv_file.release()

if __name__ == '__main__':
    mtx_r, rvecs_r, tvecs_r = calibrate('r')
    mtx_l, rvecs_l, tvecs_l = calibrate('l')

    rotation_mat_r = np.zeros(shape=(3, 3))
    rotation_mat_l = np.zeros(shape=(3, 3))
    R_r = cv2.Rodrigues(rvecs_r[0], rotation_mat_r)[0]
    R_l = cv2.Rodrigues(rvecs_l[0], rotation_mat_l)[0]
    P_r = np.column_stack((np.matmul(mtx_r,R_r), tvecs_r[0]))
    P_l = np.column_stack((np.matmul(mtx_l,R_l), tvecs_l[0]))

    # l = np.array([[304], [277]], dtype=np.float)
    # r = np.array([[255], [277]], dtype=np.float)
    points = []
    l_1 = np.array([[233.0, 319.0]])
    r_3 = np.array([[227.0, 231.0]])
    points.append({'l': l_1, 'r': r_3})

    l_2 = np.array([[386.0, 358.0]])
    r_2 = np.array([[341.0, 251.0]])
    points.append({'l': l_2, 'r': r_2})

    l_3 = np.array([[324.0, 373.0]])
    r_1 = np.array([[268.0, 267.0]])
    points.append({'l': l_3, 'r': r_1})

    l_4 = np.array([[471.0, 108.0]])
    r_7 = np.array([[340.0, 96.0]])
    points.append({'l': l_4, 'r': r_7})

    l_5 = np.array([[381.0, 52.0]])
    r_4 = np.array([[314.0, 51.0]])
    points.append({'l': l_5, 'r': r_4})

    l_6 = np.array([[332.0, 91.0]])
    r_5 = np.array([[247.0, 76.0]])
    points.append({'l': l_6, 'r': r_5})

    l_7 = np.array([[285.0, 300.0]])
    r_6 = np.array([[306.0, 213.0]])
    points.append({'l': l_7, 'r': r_6})

    
    for point in points:
        point_3d = cv2.triangulatePoints(P_r, P_l, point['r'].T, point['l'].T)
        print(point_3d/point_3d[3])
    