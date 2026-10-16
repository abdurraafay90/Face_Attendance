import argparse
import os
import sys
import time
import cv2
import numpy as np
from insightface.app import FaceAnalysis

NUM_SAMPLES = 20

POSE_GUIDANCE = [
    (0, 5, "Look directly at the camera"),
    (5, 10, "Turn your head slightly LEFT"),
    (10, 15, "Turn your head slightly RIGHT"),
    (15, 20, "Tilt your head slightly UP / DOWN"),
]


def get_current_guidance(sample_count):
    for start, end, guide in POSE_GUIDANCE:
        if start <= sample_count < end:
            return guide
    return "Finalizing enrollment..."


def sanitize_filename(name):
    clean = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).rstrip()
    return clean.replace(" ", "_").lower()


def main():
    parser = argparse.ArgumentParser(description="Multi-Angle Employee Face Enrollment Tool")
    parser.add_argument("--name", type=str, help="Employee full name (e.g. 'John Doe')")
    args = parser.parse_args()

    name = args.name
    if not name:
        name = input("Enter Employee Name to Enroll: ").strip()

    while not name:
        name = input("Name cannot be empty. Enter Employee Name: ").strip()

    file_slug = sanitize_filename(name)
    output_filename = f"{file_slug}_embedding.npy"

    print("\n" + "=" * 55)
    print(f"  ENROLLING EMPLOYEE: {name.upper()}")
    print(f"  Target File: {output_filename}")
    print("=" * 55)
    print("Instructions:")
    print("1. Stand in front of the camera with good lighting.")
    print("2. Press 'S' to start capturing.")
    print("3. Follow the on-screen head movement guidance.")
    print("4. Press 'Q' at any time to cancel.\n")

    # Load InsightFace
    print("Loading InsightFace model (buffalo_l)...")
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("[ERROR] Could not open webcam!")
        return

    embeddings = []
    capturing = False

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        faces = app.get(frame)

        # Draw HUD header
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 24), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        cv2.putText(
            frame,
            f"ENROLLMENT: {name.upper()}",
            (16, 36),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            (0, 255, 200),
            2,
            cv2.LINE_AA,
        )

        status_text = f"Samples: {len(embeddings)}/{NUM_SAMPLES}"
        (tw, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(
            frame,
            status_text,
            (w - tw - 20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Single person validation
        if len(faces) == 1:
            face = faces[0]
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            box_color = (0, 255, 120) if capturing else (255, 200, 50)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            if capturing:
                embeddings.append(face.normed_embedding)
                current_guide = get_current_guidance(len(embeddings))

                # Guide overlay
                cv2.putText(
                    frame,
                    f"> {current_guide}",
                    (x1, max(30, y1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Progress bar
                pct = len(embeddings) / NUM_SAMPLES
                bar_w = 400
                bx = (w - bar_w) // 2
                by = h - 60
                cv2.rectangle(frame, (bx, by), (bx + bar_w, by + 18), (60, 60, 60), 2)
                cv2.rectangle(frame, (bx + 2, by + 2), (bx + int(bar_w * pct) - 2, by + 16), (0, 255, 120), -1)

                if len(embeddings) >= NUM_SAMPLES:
                    # Average and normalize
                    template = np.mean(embeddings, axis=0)
                    template = template / np.linalg.norm(template)

                    np.save(output_filename, template)
                    print(f"\n[SUCCESS] Successfully enrolled {name}!")
                    print(f"[SAVED] Saved template to: {output_filename}")
                    print("[INFO] Press 'R' in attendance_system.py to reload the database.\n")
                    break

            else:
                cv2.putText(
                    frame,
                    "Face Ready - Press 'S' to Start Capture",
                    (x1, max(30, y1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        elif len(faces) > 1:
            cv2.putText(
                frame,
                "Warning: Only ONE person must be in view",
                (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame,
                "No face detected. Please face the camera.",
                (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Employee Face Enrollment", frame)

        key = cv2.waitKey(60) & 0xFF
        if key == ord("s") and len(faces) == 1 and not capturing:
            capturing = True
            embeddings = []
            print("[INFO] Capturing started. Please slowly turn your head as prompted...")
        elif key == ord("q"):
            print("[INFO] Enrollment cancelled by user.")
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
