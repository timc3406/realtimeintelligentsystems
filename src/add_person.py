from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np

import imutils
import pickle
import time
import cv2
import os
import csv


def create_csv(args):
    #just making sure there's a name for the file
    if not args.get("name"):
        print("NEEDS A NAME (-n JOHNDOE)")
        exit()

    person_name = args["name"]
    protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
    modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

    recognizer = pickle.loads(open(args["recognizer"], "rb").read())

    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    fps = FPS().start()

    # loop over frames from the video file stream
    frame_num = 0
    output = []

    while frame_num < 200:
        frame = vs.read()

        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        detector.setInput(imageBlob)
        detections = detector.forward()

        ## Instructions
        if frame_num < 50:
            cv2.putText(frame, "Look forward", (25, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        elif frame_num < 100:
            cv2.putText(frame, "Look left", (25, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        elif frame_num < 150:
            cv2.putText(frame, "Look right", (25, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Look up", (25, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > args["confidence"]:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                if fW < 20 or fH < 20:
                    continue

                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                output += vec.tolist()

        # update the FPS counter
        fps.update()

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        #increase framenum
        frame_num += 1

    #output file as csv
    with open('face_csv/{}.csv'.format(person_name), 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_NONE, escapechar=' ')
        for i in output:
            wr.writerow(i)

    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()
    vs.stop()
