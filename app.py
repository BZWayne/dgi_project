import time
import cv2

fps = 30
frame_width = 640
frame_height = 480
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
cap.set(cv2.CAP_PROP_FPS, fps)

gst_str_rtp = "v4l2src ! jpegenc ! rtpjpegpay ! udpsink host=172.20.10.2 port=5200"

if cap.isOpened() is not True:
    print("Cannot open camera. Exiting.")
    quit()
out = cv2.VideoWriter(gst_str_rtp, 0, fps, (frame_width, frame_height), True)
while True:
    ret, frame = cap.read()
    if ret is True:

        frame = cv2.flip(frame, 1)
        out.write(frame)
    else:
        print("Camera error.")
        time.sleep(10)

cap.release()


