import sys
import cv2

def read_cam():
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        while True:
            ret_val, img = cap.read();
            cv2.waitKey(10)
    else:
     print("camera open failed")

if __name__ == '__main__':
    read_cam()
