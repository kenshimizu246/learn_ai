import cv2
import numpy as np
import time

conf = 0.5

# Load coco.names and allocate color respectively.
classes = open('coco.names').read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

# Give the configuration and weight files for the model and load the network.
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# Set up Video Camera
cap = cv2.VideoCapture(0)

while True:
    # read picture from camera
    ret, img = cap.read()

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    r = blob[0, 0, :, :]

    # set blob
    net.setInput(blob)
    t0 = time.time()
    outputs = net.forward(ln)
    t = time.time()

    boxes = []
    confidences = []
    classIDs = []
    H, W = img.shape[:2]

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if(confidence > conf):
                x, y, w, h = detection[:4] * np.array([W, H, W, H])
                p0 = int(x - w//2), int(y - h//2)
                p1 = int(x + w//2), int(y + h//2)
                boxes.append([*p0, int(w), int(h)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    """
    NMS Non-Maximum Suppression:
    Given all scored regions in an image, we apply a greedy non-maximum suppression (for each class independently)
    that rejects a region if it has an intersection-over-union (IoU) overlap with a higher scoring selected region
    larger than a learned threshold.
    """
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf, conf-0.1)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # show image in the window
    cv2.imshow('Escape Key will close this window', img)
    key = cv2.waitKey(1) # wait 1 sec for preventing broken window.
    if key == 27: # break if escape key
        break

cv2.destroyAllWindows()

