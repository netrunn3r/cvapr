import numpy as np
import cv2
import glob

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def calibrate(cam_id):
    # stała dająca przerwe w wyświetlaniu skalibrowanych obrazków [ms]
    WAIT_TIME = 100

    # termination criteria for iterative algorithm <- nie mam pojęcia co
    # “Criteria” is our computation criteria to iterate calibration function. You can check OpenCV documentation for the parameters.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # generalizable checkerboard dimensions
    # https://stackoverflow.com/questions/31249037/calibrating-webcam-using-python-and-opencv-error?rq=1
    chessboard_rows = 6  # ilość wierszy minus jeden (chodzi o STYKAJACE sie rogi czarnych kwadratow)
    chessboard_columns = 9  # ilość kolumn minus jeden
    square_size = 15  # wymiar pola na szachownic w mm

    # do kalibracji musimy podać jakieś punkty 3D, które odpowiadają widoku z kamery
    # choć ich nie znamy to zakładamy że Z jest bez zmian, czyli 0, a X i Y zakładamy że szachownica była
    # nieruchoma, a kamera  zmieniała pozycję. Opis tu (sekcja Code, drugi akapit):
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
    # filanie mamy (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    objp = np.zeros((chessboard_rows * chessboard_columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_columns, 0:chessboard_rows].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # osobno obrazki do kalibracji lewej i prawej kamery
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

    # jakiś algorytm z neta to liczenia macierzy rzutowania
    # https://stackoverflow.com/questions/16101747/how-can-i-get-the-camera-projection-matrix-out-of-calibratecamera-return-value
    # nie wiem czy nie robię tu jakiegoś mega uproszczenia bo gość piszę aby iterować po rvecs[] i tvecs dla kolejnych obrazków
    rotation_mat_r = np.zeros(shape=(3, 3))
    rotation_mat_l = np.zeros(shape=(3, 3))
    R_r = cv2.Rodrigues(rvecs_r[0], rotation_mat_r)[0]
    R_l = cv2.Rodrigues(rvecs_l[0], rotation_mat_l)[0]
    P_r = np.column_stack((np.matmul(mtx_r,R_r), tvecs_r[0]))
    P_l = np.column_stack((np.matmul(mtx_l,R_l), tvecs_l[0]))

    # zbiór odpowiadających sobie punktów z kamery lewej i prawej - brane z obrazków z detect_edges.py
    points = []
    l_1 = np.array([[233.0, 319.0]])  # punkt 1 z lewego obrazka o x=233 i y=319
    r_3 = np.array([[227.0, 231.0]])  # odpowiada punktowi 3z prawego obrazka o x=227 i y=231
    points.append({'l': l_1, 'r': r_3})  # points[0] = {'l': [233.0, 319.0], 'r': [227.0, 231.0]}; to będzie pierwszy punkt 3D

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

    points_3d = {}
    for id in 'xyz':
            points_3d[id] = []
    for point in points:
        point_3d = cv2.triangulatePoints(P_r, P_l, point['r'].T, point['l'].T)  # magicznie zamień 2D na 3D
        point_3d /= point_3d[3]  # wszędzie pisali aby nie zapomnieć o podzieleniu przez 4 kolumnę; chyba chodzi o to aby mieć 3 wartości (x,y,z), a nie 4
        for i, id in enumerate('xyz'):
            points_3d[id].append(point_3d[i][0]) # nie wiem czy to jest x, y czy z, ale nie powinno mieć znaczenia
    
    lines = {}
    dist = {}
    middle = {}
    for id in [17,72,23,31,16,65,54,42,57]:  # 17 - linia między pkt 1 i 7, itd
            lines[id] = {}
    for axis in 'xyz':  # punkty końców odcinka
        lines[17][axis] = [points_3d[axis][0], points_3d[axis][6]]
        lines[72][axis] = [points_3d[axis][6], points_3d[axis][1]]
        lines[23][axis] = [points_3d[axis][1], points_3d[axis][2]]
        lines[31][axis] = [points_3d[axis][2], points_3d[axis][0]]
        lines[16][axis] = [points_3d[axis][0], points_3d[axis][5]]
        lines[65][axis] = [points_3d[axis][5], points_3d[axis][4]]
        lines[54][axis] = [points_3d[axis][4], points_3d[axis][3]]
        lines[42][axis] = [points_3d[axis][3], points_3d[axis][1]]
        lines[57][axis] = [points_3d[axis][4], points_3d[axis][6]]

    for id in [17,72,23,31,16,65,54,42,57]:
        x_diff = lines[id]['x'][0] - lines[id]['x'][1]
        y_diff = lines[id]['y'][0] - lines[id]['y'][1]
        z_diff = lines[id]['z'][0] - lines[id]['z'][1]
        x_mid = (lines[id]['x'][0] + lines[id]['x'][1]) / 2
        y_mid = (lines[id]['y'][0] + lines[id]['y'][1]) / 2
        z_mid = (lines[id]['z'][0] + lines[id]['z'][1]) / 2
        dist[id] = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)  # wzór na liczenie odległości między punktami
        middle[id] = {'x': x_mid, 'y': y_mid, 'z': z_mid}  # środek linii do pisania odległości

    fig = plt.figure()
    ax = plt.axes(projection="3d")  # utwórz wykres 3D; można nim ruszać!

    ax.scatter3D(points_3d['x'], points_3d['y'], points_3d['z'], c=points_3d['x'], cmap='hsv')  # wyrysuj wszystkie punkty
    for id in [17,72,23,31,16,65,54,42,57]:  # wyrysuj linie i odległości
        ax.plot3D(lines[id]['x'], lines[id]['y'], lines[id]['z'], 'gray')
        ax.text3D(middle[id]['x'], middle[id]['y'], middle[id]['z'], f'{dist[id]:.2f}')

    #ax.set_yscale('linear')  # oś y ma IMO jakieś dziwne skalowanie, nie udało mi się tego ogarnąć, ale to nie ma znaczenia
    #plt.yticks(range(8, 12))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()