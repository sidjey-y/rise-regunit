import cv2
import numpy as np
import mediapipe as mp
import face_recognition


def detect_extract(image):
    if image is None:
        raise ValueError("[ERROR] image is None.")

    mp_fd = mp.solutions.face_detection
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_embeddings = []

    with mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.6) as detector:
        results = detector.process(image_rgb)

        if not results.detections:
            print("[INFO] No face detected.")
            return []

        for det in results.detections:
            box = det.location_data.relative_bounding_box
            h, w, _ = image.shape
            x1, y1 = int(box.xmin * w), int(box.ymin * h)
            x2 = x1 + int(box.width * w)
            y2 = y1 + int(box.height * h)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face_crop = image_rgb[y1:y2, x1:x2]
            face_crop = cv2.resize(face_crop, (150, 150))

            encoding = face_recognition.face_encodings(face_crop)
            if encoding:
                face_embeddings.append(encoding[0])
            else:
                print("[No encoding found.")

    return face_embeddings


