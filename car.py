import cv2
from tracker import *

# 導入影片
video = cv2.VideoCapture("car3.mp4")
# 從固定畫面中偵測目標
video_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=55)
# 追蹤物件
tracker = EuclideanDistTracker()

while True:
    # 車輛偵測
    ret, frame = video.read()  # 讀取每FPS的畫面
    # 做雙邊車流的ROI
    roi1 = frame[850:1200, 1000:1900]
    roi2 = frame[850:1200, 0:900]
    #在影片中畫出ROI的範圍
    cv2.rectangle(frame, (1000, 1200), (1900, 850), (0, 0, 25, 5), 3)
    cv2.rectangle(frame, (0, 1200), (800, 850), (0, 0, 25, 5), 3)
    # 對ROI影像做mask
    mask1 = video_detector.apply(roi1)
    mask2 = video_detector.apply(roi2)
    # 刪除陰影後標出contours
    _, mask1 = cv2.threshold(mask1, 254, 255, cv2.THRESH_BINARY)
    contours1, _ = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, mask2 = cv2.threshold(mask2, 254, 255, cv2.THRESH_BINARY)
    contours2, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 建立偵測陣列
    detect1 = []
    detect2 = []
    for cnt1 in contours1:
        # 計算區域清除小噪點
        area1 = cv2.contourArea(cnt1)
        if area1 > 5200:
            # 偵測所有區域內噪點值>5200的影像
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x1, y1, w1, h1 = cv2.boundingRect(cnt1)  # 取得偵測到的物體值
            #cv2.rectangle(roi1, (x, y), (x + w, y + h), (0, 255, 0), 3)  # 對物體畫框
            #print(x, y, w, h) # 查看偵測到的座標大小
            detect1.append([x1, y1, w1, h1])  # 把偵測到的物體值存入detect
    # print(detect) #檢查detect

    for cnt2 in contours2:
        area2 = cv2.contourArea(cnt2)
        if area2 > 5200:
            #cv2.drawContours(frame, [cnt], -1, (0,255,0), 2)
            x2, y2, w2, h2 = cv2.boundingRect(cnt2)
            #cv2.rectangle(roi2, (x, y), (x + w, y + h), (0, 255, 0), 3)
            detect2.append([x2, y2, w2, h2])

#標記
    # 將detect的資料用tracker給ID並更新到carids以便等等進行標記
    carids1 = tracker.update(detect1)
    print(carids1)
    carids2 = tracker.update(detect2)
    #print(carids2)
    for carid1 in carids1:
        x1, y1, w1, h1, id1 = carid1
        cv2.putText(roi1, str(id1), (x1, y1 - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)  # 在車輛左上角計數
        cv2.rectangle(roi1, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 3)  # 畫出ROI內偵測到車
        cv2.rectangle(frame, (1000, 1200), (1900, 850), (0, 0, 255), 3)  # 畫出偵測範圍
        cv2.putText(frame, "car detect:" + str(id1), (1500, 90), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)  # 在畫面左上角計數

    for carid2 in carids2:
        x2, y2, w2, h2, id2 = carid2
        cv2.putText(roi2, str(id2), (x2, y2-15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi2, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), 3)
        cv2.rectangle(frame, (0, 1200), (800, 850), (0, 0, 255), 3)
        cv2.putText(frame, "car detect:"+str(id2), (20, 90), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
    #做出標籤
    cv2.putText(frame, "car detect:", (1500, 90), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
    cv2.putText(frame, "car detect:", (20, 90), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
    #cv2.imshow("ROI", roi1)
    #cv2.imshow("ROI", roi2)
    #cv2.imshow("mask", mask1)
    #cv2.imshow("mask", mask2)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(30)
    if key == 27:
        break
video.release()

