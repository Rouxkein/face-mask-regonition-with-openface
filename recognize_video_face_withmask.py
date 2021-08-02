# import libraries
import pickle
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import os
from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from mylib.mailer import Mailer
from imutils.video import FileVideoStream
from mylib import config, thread
import time, schedule, csv
import numpy as np
import argparse, imutils
import time, dlib, cv2, datetime
import serial
import playsound
import urllib.request
import http
q=0
total=0
t0 = time.time()
prototxt = "mobilenet_ssd/MobileNetSSD_deploy.prototxt"
model = "mobilenet_ssd/MobileNetSSD_deploy.caffemodel"
defaultSkipFrames = 60
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
				default="face_detector",
				help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
				default="mask_detector.model",
				help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
				help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
								"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
# vs = FileVideoStream('name of video').start()
vs =VideoStream(url=0).start()

# load serialized face detector
print("Loading Face Detector...")
protoPath = "face_detection_model/deploy.prototxt"
modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load serialized face embedding model
print("Loading Face Recognizer...")
embedder = cv2.dnn.readNetFromTorch("nn4.small1.v1.t7")

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
# url = "http://:4747/video"

# start the FPS throughput estimator
fps = FPS().start()
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		   "sofa", "train", "tvmonitor"]
net = cv2.dnn.readNetFromCaffe(prototxt, model)
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0
x = []
text = []
empty = []
empty1 = []
name = []
label = []

# start the frames per second throughput estimator
fps = FPS().start()

if config.Thread:
	vs = thread.ThreadingClass(config.url)


def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
								 (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidp = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidp > 0.9:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)


# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
	frame = vs.read()

	# resize the frame to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
	frame = imutils.resize(frame, width=640)
	(h, w) = frame.shape[:2]
	status = "Waiting"
	rects = []
	if totalFrames % defaultSkipFrames == 0:
		# set the status and initialize our new set of object trackers
		status = "Detecting"
		trackers = []

		# convert the frame to a blob and pass the blob through the
		# network and obtain the detections
		blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
		net.setInput(blob)
		detections = net.forward()

		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated
			# with the prediction
			confiden = detections[0, 0, i, 2]

			# filter out weak detections by requiring a minimum
			# confidence
			if confiden > 0.5:
				# extract the index of the class label from the
				# detections list
				idx = int(detections[0, 0, i, 1])

				# if the class label is not a person, ignore it
				if CLASSES[idx] != "person":
					continue

				# compute the (x, y)-coordinates of the bounding box
				# for the object
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# construct a dlib rectangle object from the bounding
				# box coordinates and then start the dlib correlation
				# tracker
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				tracker.start_track(frame, rect)

				# add the tracker to our list of trackers so we can
				# utilize it during skip frames
				trackers.append(tracker)

	# otherwise, we should utilize our object *trackers* rather than
	# object *detectors* to obtain a higher frame processing throughput
	else:
		# loop over the trackers
		for tracker in trackers:
			# set the status of our system to be 'tracking' rather
			# than 'waiting' or 'detecting'
			status = "Tracking"

			# update the tracker and grab the updated position
			tracker.update(frame)
			pos = tracker.get_position()

			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			# add the bounding box coordinates to the rectangles list
			rects.append((startX, startY, endX, endY))

	# draw a horizontal line in the center of the frame -- once an
	# object crosses this line we will determine whether they were
	# moving 'up' or 'down'
	cv2.line(frame, (0, int(h // (1.15))), (w, int(h // (1.15))), (255, 0, 0), 3)
	# cv2.putText(frame, "-on and in -", (10, h - ((i * 20) + 200)),
	# 			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

	# use the centroid tracker to associate the (1) old object
	# centroids with (2) the newly computed object centroids
	objects = ct.update(rects)

	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		# check to see if a trackable object exists for the current
		# object ID
		to = trackableObjects.get(objectID, None)

		# if there is no existing trackable object, create one
		if to is None:
			to = TrackableObject(objectID, centroid)

		# otherwise, there is a trackable object so we can utilize it
		# to determine direction
		else:
			# the difference between the y-coordinate of the *current*
			# centroid and the mean of *previous* centroids will tell
			# us in which direction the object is moving (negative for
			# 'up' and positive for 'down')
			y = [c[1] for c in to.centroids]
			direction = centroid[1] - np.mean(y)
			to.centroids.append(centroid)

			# check to see if the object has been counted or not
			if not to.counted:
				# if the direction is negative (indicating the object
				# is moving up) AND the centroid is above the center
				# line, count the object
				if direction < 0 and centroid[1] < h // 2:
					totalUp += 1
					empty.append(totalUp)
					to.counted = True

				# if the direction is positive (indicating the object
				# is moving down) AND the centroid is below the
				# center line, count the object
				elif direction > 0 and centroid[1] > h // 2:
					totalDown += 1
					empty1.append(totalDown)
					# print(empty1[-1])
					x = []
					# compute the sum of total people inside
					x.append(len(empty1) - len(empty))
					# print("Total people inside:", x)
					# if the people limit exceeds over threshold, send an email alert
					if sum(x) >= config.Threshold:
						cv2.putText(frame, "-ALERT: People limit exceeded-", (10, frame.shape[0] - 80),
									cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
						if config.ALERT:
							print("[INFO] Sending email alert..")
							Mailer().send(config.MAIL)
							print("[INFO] Alert sent")

					to.counted = True

		# store the trackable object in our dictionary
		trackableObjects[objectID] = to

		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)
	# cv2.rectangle(frame, (startX+50, startY+50), (endX+50, endY+50),
	# 			  (0, 0, 255), 2)

	# construct a tuple of information we will be displaying on the
	info = [
		("Exit", totalUp),
		("Enter", totalDown),
		("Status", status),
	]

	info2 = [
		("Total people inside", x),
	]

	# Display the output
	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, h - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

	for (i, (k, v)) in enumerate(info2):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (265, h - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
	# show the output frame
	# cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
	# key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	# if key == ord("q"):
	# 	break

	# increment the total number of frames processed thus far and
	# then update the FPS counter
	totalFrames += 1
	fps.update()

	if config.Timer:
		# Automatic timer to stop the live stream. Set to 8 hours (28800s).
		t1 = time.time()
		num_seconds = (t1 - t0)
		if num_seconds > 28800:
			break
	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		# if label == "No Mask":
		# notification.notify(
		# 	title="***No Mask Detected***",
		# 	message="Wear Mask to stay safe! ",
		# 	app_icon="images/1.ico",  # ico file should be downloaded
		# 	timeout=1
		# )

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		y = startY - 30 if startY - 30 > 30 else startY + 30
		cv2.putText(frame, label, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY),
					  (0, 0, 255), 2)

	# Alarm when "No Mask" detected
	# if mask < withoutMask:
	# 	path = os.path.abspath("Alarm.wav")
	# 	playsound(path)

	# apply OpenCV's deep learning-based face detector to localize faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the prediction
		confidenc = detections[0, 0, i, 2]
		# filter out weak detections
		if confidenc > 0.75:
			# compute the (x, y)-coordinates of the bounding box for the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
											 (96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# perform classification to recognize the face
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]
			# draw the bounding box of the face along with the associated probability
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			if name == "unknown":
				cv2.putText(frame, text, (startX, y),
							cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
						  (0, 0, 255), 2)
			roi=frame[startY:endY, startX:endX]
			p=os.path.sep.join(["C://New folder (4)//face-recognition-using-deep-learning-master","{}.png".format(str(total).zfill(5))])
			cv2.imwrite(p,roi)
			cv2.putText(frame, text, (startX, y),
						cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
			print(q)
		# -----------------------------------------------#
	# input = attendancetoday.get()
	with open("x"+ ".txt", "a", newline='') as f:
		datetimee = datetime.datetime.now()
		wr = csv.writer(f, quoting=csv.QUOTE_ALL)
		# wr.writerow(("End Time", "In", "Out", "Total Inside", "name", "label"))
		print((datetimee.strftime('%Y-%m-%d %H:%M:%S'), empty1, empty, x, name, label), file=f)
	# update the FPS counter
	fps.update()

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# stop the timer and display FPS information
fps.stop()
print("Elasped time: {:.2f}".format(fps.elapsed()))
print("Approx. FPS: {:.2f}".format(fps.fps()))
# cleanup
cv2.destroyAllWindows()
vs.stop()