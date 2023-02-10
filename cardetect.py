import cv2
from tracker import *

video = cv2.VideoCapture("car.mp4")
tracker = EuclideanDistTracker()
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=15)
#偵測
while video.isOpened():
    ret, frame = video.read()
    #height, width, _ = frame.shape
    #print(height , width)
    cv2.rectangle(frame, (200, 720), (1200, 300), (0, 0, 25 , 5), 3)
    cv2.putText(frame, "car detect:", (20, 90), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
    roi = frame[300:720, 200:1200]
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detect = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5500:
            #cv2.drawContours(frame, [cnt], -1, (0,255,0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x+w, y+h),(0, 255, 0), 3)
            #print(x, y, w, h)
            detect.append([x, y, w, h])
#標記
    carids = tracker.update(detect)
    print(carids)
    for carid in carids:
        x, y, w, h, id = carid
        cv2.putText(roi, str(id+1), (x, y-15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.rectangle(frame, (200, 720), (1200, 300), (0, 0, 255), 3)
        cv2.putText(frame, "car detect:"+str(id+1), (20, 90), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
        print("car detect:"+ str(id+1))
    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("mask", mask)
    key = cv2.waitKey(30)
    if key == 27:
        break

video.release()

#https://pyimagesearch.com/2018/08/13/opencv-people-counter/