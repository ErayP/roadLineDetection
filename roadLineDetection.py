import cv2
import numpy as np

def region_of_interest(image , vertices):
    mask = np.zeros_like(image)
    
    match_mask_color = 255
    cv2.fillPoly(mask,vertices, match_mask_color)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

def drawLine(img,lines):
    image = np.copy(img)
    blank_image=np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(blank_image,(x1,y1),(x2,y2),(255,255,255),thickness=2)
    image = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return image

def process(image):
    height, width = image.shape[0], image.shape[1]
    region_of_interest_vertices = [(0, height), (width/2, height/2), (width, height)]
    imgGRY = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cannyIMG = cv2.Canny(imgGRY, 220, 250)
    cropped_img = region_of_interest(cannyIMG, np.array([region_of_interest_vertices], np.int32))
    
    lines = cv2.HoughLinesP(cropped_img, rho=2, theta=np.pi/180, threshold=220, lines=np.array([]), minLineLength=100, maxLineGap=5)
    
    if lines is not None:
        imgWithLine = drawLine(image, lines)
    else:
        imgWithLine = image  # Eğer çizgi bulunamazsa orijinal resmi döndür.
    
    return imgWithLine

cap = cv2.VideoCapture("video2.mp4")

while True:
    success, img = cap.read()
    

    if success:
        img = process(img)
        cv2.imshow("img", img)
    else:
        break 
    
    k = cv2.waitKey(33) & 0xFF
    if k == 27:  
        break

cap.release()
cv2.destroyAllWindows()
