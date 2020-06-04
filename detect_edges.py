import numpy as np
import cv2
from matplotlib import pyplot as plt

# to: https://medium.com/analytics-vidhya/corner-detection-using-opencv-13998a679f76
# rozszerzone na dwa obrazki

def get_points_from_2d(cam_id):
    img = cv2.imread(f'img/box_clean_{cam_id}.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 7, 0.01, 50)
    corners = np.int0(corners)

    i = 1
    for corner in corners:
        x,y = corner.ravel()
        cv2.circle(img,(x,y), 3, 255, -1)
        cv2.putText(img, str(i), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        print(f'{cam_id}_{i} = np.array([[{x}.0, {y}.0]])')
        i += 1

    return img

if __name__ == '__main__':
    img_l = get_points_from_2d('l')
    img_r = get_points_from_2d('r')
    
    fig, (left, right) = plt.subplots(1, 2)
    left.imshow(img_l)
    right.imshow(img_r)
    plt.show()