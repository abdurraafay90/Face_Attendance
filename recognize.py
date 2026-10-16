import cv2
import numpy as np
from insightface.app import FaceAnalysis


# Load stored employee embedding
stored_embedding = np.load("raafay_embedding.npy")


# Load InsightFace
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))


camera = cv2.VideoCapture(0)


# We'll tune this later
THRESHOLD = 0.5


while True:

    ret, frame = camera.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)


    faces = app.get(frame)


    for face in faces:

        live_embedding = face.normed_embedding


        # Cosine similarity
        similarity = np.dot(
            stored_embedding,
            live_embedding
        )


        if similarity > THRESHOLD:
            name = "Raafay"
        else:
            name = "Unknown"


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
            f"{name} | {similarity:.2f}",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    cv2.imshow(
        "AI Receptionist Recognition",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()