import numpy as np
import os
import cv2
from cv2 import face
import re

subjects = ['emilia', "kit", "cersei"]


def detect_face(img):
    faceCascade = cv2.CascadeClassifier(
        'haarcascades_cuda/haarcascade_frontalface_default.xml')
    # convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]


def prep_data(folder_path):
    dirs = os.listdir(folder_path)
    faces = []
    labels = []
    for dir_name in dirs:
        # print(type(dir_name))
        num = re.sub(r'[a-z,A-Z]+', '', dir_name)
        label = int(num[1])
        labels.append(label)
        image = cv2.imread(folder_path+"/"+dir_name)

        cv2.imshow('training...', image)
        cv2.waitKey(50)
        face, rect = detect_face(image)
        faces.append(face)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels


faces, labels = prep_data("images/data")

print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

# labels=[0]*len(faces)
print(labels)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

print(os.listdir('images/kit'))


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def predict(test_img):
    global face_recognizer
    img = test_img.copy()
    face, rect = detect_face(img)
    label = face_recognizer.predict(face)
    print(label)
    # if label[1] > 50:
    label_text = subjects[label[0]]
    # else:
    #label_text = subjects[1]
    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1]-5)
    return img


test_img1 = cv2.imread("images/test2.png")
test_img2 = cv2.imread("images/test.png")
test_img3 = cv2.imread("images/test3.png")
#print(test_img1, test_img2)
# perform a prediction
predicted_img2 = predict(test_img2)
predicted_img1 = predict(test_img1)
predicted_img3 = predict(test_img3)
print("Prediction complete")

# display both images
cv2.imshow(subjects[0], predicted_img1)
cv2.imshow(subjects[1], predicted_img2)
cv2.imshow(subjects[2], predicted_img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
