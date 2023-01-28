import cv2

face_cascade = cv2.CascadeClassifier('trained_models/haarcascade_frontalface_default.xml')


def run_cascade_classifier():

    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()
        if not ret:
            print("No frame received")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # rectangle
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cap.release()
    cap.destroyAllwindows()