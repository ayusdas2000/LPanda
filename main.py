import cv2
import numpy as np
import pytesseract
import imutils
pytesseract.pytesseract.tesseract_cmd = " C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# read image
image = cv2.imread("C:\\Users\\Ayus Das\\Desktop\\testing.png")
#resizing
image = imutils.resize(image, width=500)
#displaying the original image
cv2.imshow('Original Image', image)
cv2.waitKey(0)
#converting the original image to the gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('1-grayscale image', gray)
cv2.waitKey(0)
#Noise removal with bilateral gray filter(removing noise while preserving edges)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow('2-bilateral image', gray)
cv2.waitKey(0)
#find edges based on the grayscale image
edged = cv2.Canny(gray, 170, 200)
cv2.imshow("3 - canny edges", edged)
cv2.waitKey(0)
#finding edges based on contours
cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#create copy of original image to draw all contours
img1 = image.copy()
cv2.drawContours(img1, cnts, -1, (0,255,0), 3)
cv2.imshow("4 - all Contours", img1)
cv2.waitKey(0)
#sort contours based on the area of the contours
cnts = sorted(cnts, key=cv2.contourArea,reverse=True)[:30]
NumberPlateCnt = None #we currently have no Number plate detected
#top 30 contours
img2 = image.copy()
img2 = cv2.drawContours(img2, cnts, -1,(0,255,0), 3)
cv2.imshow("5 - Top 30 contours", img2)
cv2.waitKey(0)
#Loop over our contours
count = 0
idx = 7
for c in cnts:
    peri1 = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri1, True)
    print("approx ", approx)
    if len(approx)==4:#select the contour with 4 contours
        NumberPlateCnt = approx
        x,y,w,h = cv2.boundingRect(c)
        new_img = image[y:y+h,x:x+w]
        cv2.imwrite('Cropped-Image-Text/'+str(idx)+'.png',new_img)
        idx+=1
        break


#Drawing the selected contour on the original number plate
print(NumberPlateCnt)
cv2.drawContours(image, [NumberPlateCnt], -1, (0, 255, 0), 3)
cv2.imshow('6- Final image with number plate detected', image)
cv2.waitKey(0)

Cropped_img_loc = 'Cropped-Image-Text/7.png'
numberimg = cv2.imread(Cropped_img_loc)
cv2.imshow("Cropped image", numberimg)


text = pytesseract.image_to_string(Cropped_img_loc, lang='eng')
print("The number is "+text)
cv2.waitKey(0)

cv2.destroyAllWindows()
