import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def run_mp_classifier():
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()

            # converted into rgb format
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = holistic.process(image)
            # print(results.face_landmarks)

            # converted back into bgr
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # FOR FACE CONNECTIONS
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)

            # FOR right HAND CONNECTIONS
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # For left hand connections
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # # FOR POSE CONNECTIONS
            # mp_drawing.draw_landmarks(image,results.drawing_utils , mp_holistic.drawing_utils)

            cv2.imshow('Holistic Model Detection ', image)

            if cv2.waitKey(10) & 0xff == ord('q'):
                break
        cap.release()
        cap.destroyAllWindows()
