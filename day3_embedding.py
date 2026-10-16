import cv2
from insightface.app import FaceAnalysis


# Load InsightFace model
app = FaceAnalysis(
    name="buffalo_l"
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)


# Open webcam
camera = cv2.VideoCapture(0)


while True:

    ret, frame = camera.read()

    if not ret:
        break


    # Mirror view so movement feels natural
    frame = cv2.flip(frame, 1)


    # Detect faces + generate embeddings
    faces = app.get(frame)


    for face in faces:

        # 512-dimensional face embedding
        embedding = face.embedding

        print("Embedding size:", embedding.shape)
        print("First 10 values:", embedding[:10])


        # Bounding box
        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox


        # Draw rectangle around face
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # Optional label
        cv2.putText(
            frame,
            "Face Detected",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # Show webcam feed
    cv2.imshow(
        "InsightFace Embedding Demo",
        frame
    )


    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()