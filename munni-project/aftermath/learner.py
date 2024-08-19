import cv2
import math
import numpy as np

eye_casc_path = "data/haarcascade_eye.xml"
face_casc_path = "data/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_casc_path)
eye_cascade = cv2.CascadeClassifier(eye_casc_path)
video_capture = cv2.VideoCapture(0)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
font = cv2.FONT_HERSHEY_SIMPLEX
max_sample_count = 50


def learn_user(id):
    capture_faces = []
    face_ids = []
    sample_count = 0

    while(True):
        # Capture frame at a time
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Detect only one face at a time
        if len(faces) > 1:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            height, width = frame.shape[:2]
            cv2.putText(frame, "There are more than one face.", (int(width)/4, int(height)/2), font, 0.8,
                        (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('Capturing...', frame)
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_gray = gray[y:y+h, x:x+w]
            face_color = frame[y:y+h, x:x+w]

            # Detect eyes for each face
            eyes = eye_cascade.detectMultiScale(face_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.circle(face_color, (ex+ew/2, ey+eh/2),
                           int(math.sqrt(ew**2+eh**2))/2, (255, 255, 128), 2)
                if len(eyes) == 2:
                    ex1, ey1, ew1, eh1 = eyes[0]
                    ex2, ey2, ew2, eh2 = eyes[1]
                    cv2.line(face_color, (ex1+ew1/2, ey1+eh1/2),
                             (ex2+ew2/2, ey2+eh2/2), (128, 128, 0), 2)

                    # Capture face
                    sample_count += 1
                    faceNp = np.array(gray[y:y+h, x:x+w], 'uint8')
                    capture_faces.append(faceNp)
                    face_ids.append(int(id))
                    display = "capture "+str(sample_count)+" samples"
                    cv2.putText(frame, display, (x, y), font, 0.7,
                                (255, 255, 255), 1, cv2.LINE_AA)

        # Show modified frame
        cv2.imshow("Capturing...", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if(sample_count >= max_sample_count):
            ret, frame = video_capture.read()
            display = "Queued for training..."
            cv2.putText(frame, display, (x, y), font, 0.7,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Capturing...", frame)
            cv2.waitKey(5)
            break
    video_capture.release()
    cv2.destroyAllWindows()

    if len(capture_faces) >= max_sample_count and len(face_ids) == len(capture_faces):
        face_recognizer.train(capture_faces, np.array(face_ids))
        face_recognizer.save('data/trained_user-'+str(id)+'.yml')
        return True
    return False
