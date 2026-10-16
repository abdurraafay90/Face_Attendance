import cv2
import numpy as np
from insightface.app import FaceAnalysis


NAME = "raafay"
NUM_SAMPLES = 20


# Load InsightFace
app = FaceAnalysis(name="buffalo_l")
app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)


camera = cv2.VideoCapture(0)

embeddings = []

print("MULTI-FRAME EMPLOYEE ENROLLMENT")
print("--------------------------------")
print("Move your head slowly:")
print("Front -> Left -> Right -> Slightly Up -> Slightly Down")
print()
print("Press S to start capturing.")
print("Press Q to quit.")

capturing = False


while True:

    ret, frame = camera.read()

    if not ret:
        break


    # Mirror camera
    frame = cv2.flip(frame, 1)


    # Detect faces
    faces = app.get(frame)


    # For enrollment we only want ONE person
    if len(faces) == 1:

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


        if capturing:

            embedding = face.normed_embedding

            embeddings.append(embedding)

            print(
                f"Captured sample "
                f"{len(embeddings)}/{NUM_SAMPLES}"
            )


            # Finished enrollment
            if len(embeddings) >= NUM_SAMPLES:

                # Average all embeddings
                template = np.mean(
                    embeddings,
                    axis=0
                )


                # Normalize again
                template = template / np.linalg.norm(template)


                filename = f"{NAME}_embedding.npy"

                np.save(
                    filename,
                    template
                )


                print()
                print("Enrollment complete.")
                print(f"Saved: {filename}")


                capturing = False


    elif len(faces) > 1:

        cv2.putText(
            frame,
            "Only one person allowed",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # Progress text
    cv2.putText(
        frame,
        f"Samples: {len(embeddings)}/{NUM_SAMPLES}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )


    if not capturing:

        cv2.putText(
            frame,
            "Press S to enroll",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


    cv2.imshow(
        "Multi-Frame Enrollment",
        frame
    )


    key = cv2.waitKey(80) & 0xFF


    if key == ord("s"):

        embeddings = []
        capturing = True

        print()
        print("Capturing started...")


    elif key == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()