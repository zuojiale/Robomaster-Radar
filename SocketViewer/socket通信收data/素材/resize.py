import cv2 as cv

img = cv.imread("xj.png")
img = cv.resize(img,(40,40))

cv.imwrite("xj1.png",img)

