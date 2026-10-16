import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

camera = cv2.VideoCapture(0)

print("Press S to save your face.")
print("Press Q to quit.")

while True:
    ret, frame = camera.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    faces = app.get(frame)

    if faces:
        # For now, just use the first detected face
        face = faces[0]

        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Press S to enroll",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Employee Enrollment", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):

        if not faces:
            print("No face detected.")
            continue

        embedding = faces[0].normed_embedding

        np.save("raafay_embedding.npy", embedding)

        print("Raafay enrolled successfully.")

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()