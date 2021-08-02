from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from mylib.centroidtracker import CentroidTracker
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from imutils.video import FileVideoStream
import os.path
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils import paths
from imutils.video import FPS
from mylib.mailer import Mailer
from mylib import config, thread
import time, schedule, csv
import numpy as np
import argparse, imutils
import time, dlib, cv2, datetime
import playsound
from PIL import Image,ImageTk
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
from plyer import notification
import tkinter.messagebox
import tkinter as tk
import datetime
import imutils
from imutils.video import VideoStream
import cv2
import os
import time
from playsound import playsound
#######################################################################################
def facereg():
    #using this code to tranfer data to adruino
    # base = "http://192.168.10.100/"
    #
    # def transfer(my_url):  # use to send and receive data
    #     try:
    #         n = urllib.request.urlopen(base + my_url).read()
    #         n = n.decode("utf-8")
    #         return n
    #     except http.client.HTTPException as e:
    #         return e
    total=0
    t0 = time.time()
    prototxt = "mobilenet_ssd/MobileNetSSD_deploy.prototxt"
    model = "mobilenet_ssd/MobileNetSSD_deploy.caffemodel"
    defaultSkipFrames = 30
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
    # vs = VideoStream(src=0).start()
    vs = FileVideoStream('IMG_6200.MOV').start()
    # load serialized face detector
    print("Loading Face Detector...")
    protoPath = "face_detection_model/deploy.prototxt"
    modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    # load serialized face embedding model
    print("Loading Face Recognizer...")
    embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")
    # load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
    le = pickle.loads(open("output/le.pickle", "rb").read())
    # initialize the video stream, then allow the camera sensor to warm up
    # url = "http://192.168.10.101:4747/video"
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
    #start the frames per second throughput estimator
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
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > args["confidence"]:
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
        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        if totalFrames % defaultSkipFrames == 0:
            # set the status and initialize our new set of object trackers
            status = "Detecting"
            trackers = []
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
            net.setInput(blob)
            detections = net.forward()
            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated
                # with the prediction
                confidence = detections[0, 0, i, 2]
                # filter out weak detections by requiring a minimum
                # confidence
                if confidence > 0.7:
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
        cv2.line(frame, (0, int(h // (2))), (w, int(h // (2))), (255, 0, 0), 3)
        # cv2.putText(frame, "-on and in -", (10, h - ((i * 20) + 200)),
        # 			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
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
                    if direction < 0 and centroid[1] < h // 2:
                        totalUp += 1
                        empty.append(totalUp)
                        to.counted = True
                    # if the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object
                    elif direction > 0 and centroid[1] > h // 2:
                        totalDown+=1
                        empty1.append(totalDown)
                        # print(empty1[-1])
                        x = []
                        # compute the sum of total people inside
                        x.append(len(empty1) - len(empty))
                        to.counted = True

            # store the trackable object in our dictionary
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
            #     notification.notify(
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
        # if (mask < withoutMask):
        # 	path = os.path.abspath("Alarm.wav")
        # 	playsound(path)
        # apply OpenCV's deep learning-based face detector to localize faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections
            if confidence > 0.8:
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
                # set condition ot tranfer rule to esp8266
                # if name == "name forder you just create ":
                # 	q += 1
                # 	if q==15:
                # 		df = transfer("45")
                # 		q=0
                # if name =="name forder you just create":
                # 	df = transfer("45")
                if name == "unknown":
                    cv2.putText(frame, text, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                roi = frame[startY:endY, startX:endX]
                p = os.path.sep.join(["C://New folder (4)//face-recognition-using-deep-learning-master//report",
                                      "{}.png".format(str(total).zfill(5))])
                cv2.imwrite(p, roi)
                cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        if(label=="No Mask"):
            path = os.path.abspath("Alarm.wav")
            playsound(path)
        input=attendancetoday.get()
        with open(input+".txt", "a", newline='') as f:
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
def sendemail():
    email = youremail.get()
    password = passwordd.get()
    send_to_email = emailaddress.get()
    subject = 'This is the subject'
    message = 'This is my message'
    # input+ txt
    input = attendancetoday.get()
    file_location = input+".txt"
    msg = MIMEMultipart()
    msg['From'] = email
    msg['To'] = send_to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))
    # Setup the attachment
    filename = os.path.basename(file_location)
    attachment = open(file_location, "rb")
    part = MIMEBase('application', 'octet-stream')
    part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
    # Attach the attachment to the MIMEMultipart object
    msg.attach(part)
    server = smtplib.SMTP('imap.gmail.com', 587)
    server.starttls()
    server.login(email, password)
    text = msg.as_string()
    server.sendmail(email, send_to_email, text)
    server.quit()
def TakeImages():
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    print("[INFO] starting video stream...")
    # input your ip of your cam here
    # url = "http://192.168. .:4747/video"
    vs = VideoStream(0).start()
    time.sleep(2.0)
    total = 0
    file = os.path.join(path.get(), folder.get())
    while True:
        frame = vs.read()
        orig = frame.copy()
        frame = imutils.resize(frame, width=640, height=480)
        rects = detector.detectMultiScale(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
            minNeighbors=5, minSize=(30, 30))
        for (x, y, W, H) in rects:
            # cv2.rectangle(frame,(x,y),(x+W,y+H),(0,255,0),2)
            # roi = frame[y:y + H, x:x + W]
            key = cv2.waitKey(1) & 0xFF
            # if key==ord("k"):
            #     t=time.time()+3
            # if time.time() < time.time() + 15:
            #     p = os.path.sep.join([file, "{}.png".format(str(total).zfill(5))])
            #     cv2.imwrite(p, frame)
            #     total += 1
            if key == ord("p"):
                p = os.path.sep.join([file, "{}.png".format(str(total).zfill(5))])
                cv2.imwrite(p, frame)
                total += 1
            if key == ord("z"):
                break
            cv2.imshow("Frame", frame)
    print("face images stored".format(total))
    print("cleaning up...")
    vs.stop()
    cv2.destroyAllWindows()
def trainning():
    # load serialized face detector
    print("Loading Face Detector...")
    protoPath = "face_detection_model/deploy.prototxt"
    modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    # load serialized face embedding model
    print("Loading Face Recognizer...")
    embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")
    # grab the paths to the input images in our dataset
    print("Quantifying Faces...")
    imagePaths = list(paths.list_images("dataset"))
    # initialize our lists of extracted facial embeddings and corresponding people names
    knownEmbeddings = []
    knownNames = []
    # initialize the total number of faces processed
    total = 0
    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        if (i % 50 == 0):
            print("Processing image {}/{}".format(i, len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]
        # load the image, resize it to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=640)
        (h, w) = image.shape[:2]
        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)
        # apply OpenCV's deep learning-based face detector to localize faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()
        # ensure at least one face was found
        if len(detections) > 0:
            # we're making the assumption that each image has only ONE face, so find the bounding box with the largest probability
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]
            # ensure that the detection with the largest probability also means our minimum probability test (thus helping filter out weak detections)
            if confidence > 0.8:
                # compute the (x, y)-coordinates of the bounding box for the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # extract the face ROI and grab the ROI dimensions
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
              # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue
                # construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                # add the name of the person + corresponding face embedding to their respective lists
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1
    # dump the facial embeddings + names to disk
    print("[INFO] serializing {} encodings...".format(total))
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open("output/embeddings.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()
    # load the face embeddings
    print("[INFO] loading face embeddings...")
    data = pickle.loads(open("output/embeddings.pickle", "rb").read())
    # encode the labels
    print("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])
    # train the model used to accept the 128-d embeddings of the face and
    # then produce the actual face recognition
    print("[INFO] training model...")
    ## this is SVM kernel algroithm
    recognizer = SVC(C=1, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)
    # write the actual face recognition model to disk
    f = open("output/recognizer.pickle", "wb")
    f.write(pickle.dumps(recognizer))
    f.close()
    # write the label encoder to disk
    f = open("output/le.pickle", "wb")
    f.write(pickle.dumps(le))
    f.close()
def create_file():
      # Receiving user's file_path selection
    path_ = 'C://...//...//dataset'
    path.set(path_)
    print("folder_name: ", folder.get())
    print("path_name: ", path.get())
    dirs = os.path.join(path.get(), folder.get())
    if not os.path.exists(dirs):
        os.makedirs(dirs)
        tkinter.messagebox.showinfo('Tips:','Folder name created successfully!')
    else:
        tkinter.messagebox.showerror('Tips','The folder name exists, please change it')
######################################## USED STUFFS ############################################
ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
day, month, year = date.split("-")
mont = {'01': 'January',
        '02': 'February',
        '03': 'March',
        '04': 'April',
        '05': 'May',
        '06': 'June',
        '07': 'July',
        '08': 'August',
        '09': 'September',
        '10': 'October',
        '11': 'November',
        '12': 'December'
        }
######################################## GUI FRONT-END ###########################################
window = tk.Tk()
window.geometry("400x240")
window.resizable(True, False)
window.title("Attendance System")
# window.configure(background='#e01635')
load =Image.open('ffa.png')
render=ImageTk.PhotoImage(load)
img=tk.Label(window,image=render)
img.place(x=0,y=0)
folder = tk.StringVar()
path = tk.StringVar()
emailaddress=tk.StringVar()
youremail=tk.StringVar()
passwordd=tk.StringVar()
attendancetoday=tk.StringVar()
tk.Label(window,text =   " Folder name:  ",width=12 ,height=1,fg="black").place(x=20, y= 10)
tk.Entry(window,textvariable = folder).place(x=120, y= 10)
tk.Label(window,text = " Your email : ",width=12 ,height=1,fg="black").place(x=130, y= 70)
tk.Entry(window,textvariable=youremail).place(x=220, y= 70)
tk.Label(window,text = " password : ",width=12 ,height=1,fg="black").place(x=130, y= 100)
tk.Entry(window,show="*",textvariable=passwordd).place(x=220, y= 100)
tk.Label(window,text = " emailaddress : ",width=12 ,height=1,fg="black").place(x=130, y= 130)
tk.Entry(window,textvariable=emailaddress).place(x=220, y= 130)
tk.Label(window,text = " namedata : ",width=12 ,height=1,fg="black").place(x=130, y= 160)
tk.Entry(window,textvariable=attendancetoday).place(x=220, y= 160)
tk.Button(window, text = "   Create       ", command = create_file,width=12  ,height=1, activebackground = "white" ,font=('times', 10, ' bold ')).place(x=265, y= 10)
tk.Button(window, text = "  CropFace     ", command =TakeImages,width=12  ,height=1, activebackground = "white" ,font=('times', 10, ' bold ')).place(x=20, y= 40)
tk.Button(window, text = "    Train      ", command =trainning,width=12  ,height=1, activebackground = "white" ,font=('times', 10, ' bold ')).place(x=20, y= 100)
tk.Button(window, text = " Face_Regonition ", command =facereg,width=12  ,height=1, activebackground = "white" ,font=('times', 10, ' bold ')).place(x=20, y= 130)
tk.Button(window, text = " Sent_Data ", command =sendemail,width=12  ,height=1, activebackground = "white" ,font=('times', 10, ' bold ')).place(x=20, y=160)
window.mainloop()
####################################################################################################
