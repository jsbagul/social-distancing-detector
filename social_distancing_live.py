from flask import Flask, render_template, Response
from configs import config
from configs.detection import detect_people
from scipy.spatial import distance as dist 
import numpy as np
import argparse
import imutils
import cv2
import os
from playsound import playsound

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="", help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="", help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1, help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# load the COCO class labels the YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# load the YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if GPU is to be used or not
if config.USE_GPU:
    # set CUDA s the preferable backend and target
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the "output" layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
# open input video if available else webcam stream


app = Flask(__name__)
app.config["CACHE_TYPE"] = "null"


@app.route('/')
def camera():
    return render_template('camera.html')


def get_frame():
    #vs = cv2.VideoCapture('pedestrians.mp4')
    writer = None
    vs = cv2.VideoCapture(0)  # this makes a web cam object
    while True:
        
        # read the next frame from the input video
        (grabbed, frame) = vs.read()

        # if the frame was not grabbed, then that's the end fo the stream 
        if not grabbed:
            break

        # resize the frame and then detect people (only people) in it
        frame = imutils.resize(frame, width=1000)
        results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))

        # initialize the set of indexes that violate the minimum social distance
        violate = set()

        # ensure there are at least two people detections (required in order to compute the
        # the pairwise distance maps)
        if len(results) >= 2:
            # extract all centroids from the results and compute the Euclidean distances
            # between all pairs of the centroids
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            # loop over the upper triangular of the distance matrix
            for i in range(0, D.shape[0]):
                for j in range(i+1, D.shape[1]):
                    # check to see if the distance between any two centroid pairs is less
                    # than the configured number of pixels
                    if D[i, j] < config.MIN_DISTANCE:
                        # update the violation set with the indexes of the centroid pairs
                        violate.add(i)
                        violate.add(j)
        
        # loop over the results
        for (i, (prob, bbox, centroid)) in enumerate(results):
            # extract teh bounding box and centroid coordinates, then initialize the color of the annotation
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            # if the index pair exists within the violation set, then update the color
            if i in violate:
                color = (0, 0, 255)

            # draw (1) a bounding box around the person and (2) the centroid coordinates of the person
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)

        # draw the total number of social distancing violations on the output frame
        text = "Social Distancing Violations: {}".format(len(violate))
        cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        if len(violate)>4:
             playsound('warning.mp3')
        

        # check to see if the output frame should be displayed to the screen
        if 0 > 0:
            # show the output frame
            cv2.imshow("Output", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the 'q' key is pressed, break from the loop
            if key == ord("q"):
                break
        
        # if an output video file path has been supplied and the video writer ahs not been 
        # initialized, do so now
        if 'output.avi' != "" and writer is None:
            # initialize the video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(' ', fourcc, 25, (frame.shape[1], frame.shape[0]), True)

        # if the video writer is not None, write the frame to the output video file
        if writer is not None:
            print("[INFO] writing stream to output")
            writer.write(frame)
        imgencode = cv2.imencode('.jpg', frame)[1]
        stringData = imgencode.tobytes()
            
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n' + stringData + b'\r\n')


@app.route('/video_stream')
def video_stream():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='192.168.43.105', debug=True, threaded=True)
