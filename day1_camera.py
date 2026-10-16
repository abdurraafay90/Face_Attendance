import cv2

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()

    if not ret:
        break

    # Flip horizontally for selfie/mirror view
    frame = cv2.flip(frame, 1)

    cv2.rectangle(frame, (200, 100), (400, 300), (0, 255, 0), 2)
    cv2.putText(
        frame,
        "Scanning...",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.imshow("AI Receptionist Camera", frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()