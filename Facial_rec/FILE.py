import cv2

faceCascade = cv2.CascadeClassifier(
    'haarcascades_cuda/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascades_cuda/haarcascade_eye.xml')

# print(faceCascade)
#faceCascade = cv2.CascadeClassifier('opencv/haarcascades_cuda/smile.xml')

img = cv2.imread('images/data/cerh_2.png')


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray, 1.1, 4)
eyes = eyeCascade.detectMultiScale(gray, 1.1, 4)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 5)
    cv2.putText(img, "face", (x, y), cv2.FONT_HERSHEY_COMPLEX,
                0.5, (255, 100, 255), 1)
for (x, y, w, h) in eyes:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 5)
    cv2.putText(img, "eye", (x, y), cv2.FONT_HERSHEY_COMPLEX,
                0.5, (255, 0, 255), 1)
    #roi_gray = gray[y:y+h, x:x+w]
    #roi_color = img[y:y+h, x:x+w]

cv2.imshow('redult', img)
cv2.waitKey(0)
