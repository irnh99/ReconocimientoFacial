import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

leyeCascade = cv2.CascadeClassifier('ojoI.xml')
reyeCascade = cv2.CascadeClassifier('ojoD.xml')
mouthCascade = cv2.CascadeClassifier('Mouth.xml')
noseCascade = cv2.CascadeClassifier('Nariz.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=2,
        minNeighbors=1,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    
    font = cv2.FONT_HERSHEY_SIMPLEX


    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        #ojo izquierdo
        ojoI = leyeCascade.detectMultiScale(roi_gray)
        for (x2,y2,w2,h2) in ojoI:
            cv2.rectangle(roi_color,(x2,y2),(x2+w2,y2+h2),(0,255,0),2)
            cv2.putText(roi_color,'Ojo izquierdo',(x2-w2/2,y2-h2/2), font, 0.5, (0,255,0), 2) 
            break
        #ojo derecho
        ojoD = reyeCascade.detectMultiScale(roi_gray)
        for (x2,y2,w2,h2) in ojoD:
            cv2.rectangle(roi_color,(x2,y2),(x2+w2,y2+h2),(0,215,0),2)
            cv2.putText(roi_color,'Ojo Derecho',(x2-w2/2,y2-h2/2), font, 0.5, (0,215,0), 2) 
            break
        #boca
        mouth = mouthCascade.detectMultiScale(roi_gray)
        for (x2,y2,w2,h2) in mouth:
            cv2.rectangle(roi_color,(x2,y2),(x2+w2,y2+h2),(0,0,255),2)
            cv2.putText(roi_color,'boca',(x2-w2/2,y2-h2/2), font, 0.5, (0,0,255), 2) 
            break
        #Nariz
        nose = noseCascade.detectMultiScale(roi_gray)
        for (x2,y2,w2,h2) in nose:
            cv2.rectangle(roi_color,(x2,y2),(x2+w2,y2+h2),(0,255,255),2)
            cv2.putText(roi_color,'nariz',(x2-w2/2,y2-h2/2), font, 0.5, (0,255,255), 2) 
            break
        break

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
