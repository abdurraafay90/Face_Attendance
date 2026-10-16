import cv2
import numpy as np
from collections import deque
from insightface.app import FaceAnalysis


NAME = "Raafay"

THRESHOLD = 0.50

WINDOW_SIZE = 7

REQUIRED_MATCHES = 5


# Load stored template
stored_embedding = np.load(
    "raafay_embedding.npy"
)


# Load InsightFace
app = FaceAnalysis(name="buffalo_l")

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)


camera = cv2.VideoCapture(0)


# Keeps results from recent frames
history = deque(
    maxlen=WINDOW_SIZE
)


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


        # Current frame match
        is_match = similarity > THRESHOLD


        history.append(
            is_match
        )


        # Count matches in recent frames
        matches = sum(history)


        # Stable identity decision
        if (
            len(history) == WINDOW_SIZE
            and matches >= REQUIRED_MATCHES
        ):

            name = NAME
            color = (0, 255, 0)

        else:

            name = "Unknown"
            color = (0, 0, 255)


        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )


        cv2.putText(
            frame,
            f"{name} | sim: {similarity:.2f}",
            (x1, max(30, y1 - 35)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )


        cv2.putText(
            frame,
            f"Votes: {matches}/{len(history)}",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )


    cv2.imshow(
        "Stable Face Recognition",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()