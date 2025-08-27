import cv2
import numpy as np
import cv2 as cv
img = cv.imread('goal.png')
img=cv.resize(img,(img.shape[1]//2,img.shape[0]//2))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)[1]

blur = cv2.GaussianBlur(thresh, (5, 5), 0)
edges = cv2.Canny(blur, 50,150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


max_area = 0
max_contour = None
for contour in contours:
    epsilon = 0.08 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        area = w * h
        if area > max_area:
            max_area = area
            max_contour = approx

if max_contour is not None:
    M = cv2.moments(max_contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    print(cX,cY)
    cv.circle(img, (cX,cY), 5, (0, 0, 255), -1)


# cv.imshow('thresh', thresh)
cv.imshow('img',img)
cv.waitKey(0)