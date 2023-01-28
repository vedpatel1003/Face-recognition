from cascade import run_cascade_classifier
from run_mp import run_mp_classifier

CASCADE_CLASSIFIER = False
MEDIAPIPE = True


if __name__ == "__main__":

    if CASCADE_CLASSIFIER:
        run_cascade_classifier()
    elif MEDIAPIPE:
        run_mp_classifier()


