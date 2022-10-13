from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from mylib.mailer import Mailer
from mylib import config, thread
import time, schedule, csv
import numpy as np
import argparse, imutils
import time, dlib, cv2, datetime
from itertools import zip_longest
import sys

# print OPENCV information to check if the current installation support Gstreamer
# if no Gstreamer module is aviable remove opencv and reinstall using pip install opencv-contrib-python
cv2info = cv2.getBuildInformation()
if 'GStreamer:                   NO' in cv2info:
	print(cv2info)
	print("GStreamer is not support in this opencv installation.")
	exit()

#python3 main.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel

# gst-launch-1.0 udpsrc address=0.0.0.0 port=5004 ! application/x-rtp,media=video,encoding-name=H264 ! queue ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink

t0 = time.time()

def run():

	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--prototxt", required=False,
		help="path to Caffe 'deploy' prototxt file")
	ap.add_argument("-m", "--model", required=True,
		help="path to Caffe pre-trained model")
	ap.add_argument("-i", "--input", type=str,
		help="path to optional input video file")
	ap.add_argument("-o", "--output", type=str,
		help="path to optional output video file")
	# confidence default 0.4
	ap.add_argument("-c", "--confidence", type=float, default=0.4,
		help="minimum probability to filter weak detections")
	ap.add_argument("-s", "--skip-frames", type=int, default=30,
		help="# of skip frames between detections")
	args = vars(ap.parse_args())

	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]

	# load our serialized model from disk
	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

	print("[INFO] Starting the live stream..")
	# vs = cv2.VideoCapture(0)
	
	# camset='v4l2src device=/dev/video0 ! video/x-raw,width=640,height=360,framerate=52/1 ! nvvidconv flip-method=0 ! video/x-raw(memory:NVMM), format=I420, width=640, height=360 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! queue ! appsink drop=1'

	gst_str_rtp = "appsrc ! video/x-raw,format=BGR ! queue ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv !\
	video/x-raw(memory:NVMM),format=NV12,width=640,height=360,framerate=52/1 ! nvv4l2h264enc insert-sps-pps=1  \
		insert-vui=1 idrinterval=30bitrate=1000000 EnableTwopassCBR=1  ! h264parse ! rtph264pay ! udpsink host=127.0.0.1 port=5004 auto-multicast=0"
	
	# NOTE (FROM MATTIA): 
	# 1. you where using the autovideosink, this wiil create a sink to create a video, woverve the vdeo is not sent back to opencv
	#    appsink and appsrc are instead used to send frame in the program memory enabling th program to load and use them
	# 2. opencv expect BGR image so I tell gstreamer to convert (videoparse) the camera input to bgr instad of YUY2
	# Your(old) camset -> camset = 'v4l2src device=/dev/video0 ! video/x-raw, format=YUY2, width=640, height=480, pixel-aspect-ratio=1/1, framerate=30/1 ! videoconvert ! appsink'
	camset = 'v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink'

	vs = cv2.VideoCapture(camset, cv2.CAP_GSTREAMER)

	frame_width = vs.get(cv2.CAP_PROP_FRAME_WIDTH)
	frame_height = vs.get(cv2.CAP_PROP_FRAME_HEIGHT)
	fps = vs.get(cv2.CAP_PROP_FPS)

	print('Src opened, %dx%d @ %d fps' % (frame_width, frame_height, fps))

	if not vs.isOpened():
		print("Pipleine failed to rollout")
		exit()

	show = True
	# out = cv2.VideoWriter(gst_str_rtp, cv2.CAP_GSTREAMER, 0, fps, (frame_width, frame_height), show)
	time.sleep(2.0)

	writer = None

	W = None
	H = None

	ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
	trackers = []
	trackableObjects = {}

	totalFrames = 0
	totalDown = 0
	totalUp = 0
	x = []
	empty=[]
	empty1=[]

	# start the frames per second throughput estimator
	fps = FPS().start()

	if config.Thread:
		vs = thread.ThreadingClass(config.url)

	while True:

		ret, frame = vs.read()
		if not frame:
			# Added check to wait for a frame from gStreamer
			continue
		
		print("recive frames")

		if frame is not None:  # add this line
			(height, width) = frame.shape[:2] 
			# print(height)

		if cv2.waitKey(1) == 27:
			exit(0)

		if args["input"] is not None and frame is None:
			break

		scale_percent = 60 # percent of original size
		width = int(frame.shape[1] * scale_percent / 100)
		height = int(frame.shape[0] * scale_percent / 100)
		dim = (width, height)


		frame = cv2.resize(frame, dim)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		

		# if the frame dimensions are empty, set them
		if W is None or H is None:
			(H, W) = frame.shape[:2]

		# if we are supposed to be writing a video to disk, initialize
		# the writer
		if args["output"] is not None and writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,
				(W, H), True)

		# initialize the current status along with our list of bounding
		# box rectangles returned by either (1) our object detector or
		# (2) the correlation trackers
		status = "Waiting"
		rects = []

		if totalFrames % args["skip_frames"] == 0:
			status = "Detecting"
			trackers = []
			blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
			net.setInput(blob)
			detections = net.forward()

			# loop over the detections
			for i in np.arange(0, detections.shape[2]):
				confidence = detections[0, 0, i, 2]
				if confidence > args["confidence"]:
					idx = int(detections[0, 0, i, 1])

					if CLASSES[idx] != "person":
						continue

					box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
					(startX, startY, endX, endY) = box.astype("int")


					tracker = dlib.correlation_tracker()
					rect = dlib.rectangle(startX, startY, endX, endY)
					tracker.start_track(rgb, rect)

					trackers.append(tracker)

		else:
			for tracker in trackers:
				status = "Tracking"

				tracker.update(rgb)
				pos = tracker.get_position()

				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())

				rects.append((startX, startY, endX, endY))

		# draw a horizontal line in the center of the frame -- once an
		# object crosses this line we will determine whether they were
		# moving 'up' or 'down'
		cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
		cv2.putText(frame, "-Prediction border - Entrance-", (10, H - ((i * 20) + 200)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


		objects = ct.update(rects)

		for (objectID, centroid) in objects.items():
			to = trackableObjects.get(objectID, None)

			if to is None:
				to = TrackableObject(objectID, centroid)

			else:

				y = [c[1] for c in to.centroids]
				direction = centroid[1] - np.mean(y)
				to.centroids.append(centroid)

				if not to.counted:

					if direction < 0 and centroid[1] < H // 2:
						totalUp += 1
						empty.append(totalUp)
						to.counted = True


					elif direction > 0 and centroid[1] > H // 2:
						totalDown += 1
						empty1.append(totalDown)
						x = []
						x.append(len(empty1)-len(empty))
						#print("Total people inside:", x)
						if sum(x) >= config.Threshold:
							cv2.putText(frame, "-ALERT: People limit exceeded-", (10, frame.shape[0] - 80),
								cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
							if config.ALERT:
								print("[INFO] Sending email alert..")
								Mailer().send(config.MAIL)
								print("[INFO] Alert sent")

						to.counted = True


			trackableObjects[objectID] = to

			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

		info = [
		("Exit", totalUp),
		("Enter", totalDown),
		("Status", status),
		]

		info2 = [
		("Total people inside", x),
		]

		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

		for (i, (k, v)) in enumerate(info2):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (265, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

		if config.Log:
			datetimee = [datetime.datetime.now()]
			d = [datetimee, empty1, empty, x]
			export_data = zip_longest(*d, fillvalue = '')

			with open('Log.csv', 'w', newline='') as myfile:
				wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
				wr.writerow(("End Time", "In", "Out", "Total Inside"))
				wr.writerows(export_data)


		# out.write(frame)
		cv2.imshow("Sender", frame)
		key = cv2.waitKey(1) & 0xFF

		if key == ord("q"):
			break

		totalFrames += 1
		fps.update()

		if config.Timer:
			t1 = time.time()
			num_seconds=(t1-t0)
			if num_seconds > 28800:
				break

	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	cv2.destroyAllWindows()


if config.Scheduler:
	schedule.every().day.at("9:00").do(run)
	while 1:
		schedule.run_pending()

else:
	run()
