import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize Face Detector with the downloaded task model
base_options = python.BaseOptions(
    model_asset_path="blaze_face_short_range.task"
)
options = vision.FaceDetectorOptions(
    base_options=base_options, min_detection_confidence=0.5
)
detector = vision.FaceDetector.create_from_options(options)

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Flip horizontally for natural mirror view
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB and wrap into a MediaPipe Image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    results = detector.detect(mp_image)

    if results.detections:
        for detection in results.detections:
            box = detection.bounding_box
            x, y = box.origin_x, box.origin_y
            w, h = box.width, box.height

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            score = detection.categories[0].score
            cv2.putText(
                frame,
                f"Face {score:.2f}",
                (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()