import glob
import os
import time
import cv2
import numpy as np
from insightface.app import FaceAnalysis

from gesture_detector import GestureDetector
from attendance_logger import AttendanceLogger


def load_employee_database(base_dir="."):
    """
    Finds all '*_embedding.npy' files and loads them into a dictionary:
    { "Raafay": np.ndarray(512,), ... }
    """
    database = {}
    pattern = os.path.join(base_dir, "*_embedding.npy")
    for file_path in glob.glob(pattern):
        filename = os.path.basename(file_path)
        raw_name = filename.replace("_embedding.npy", "")
        # Capitalize name nicely (e.g. 'raafay' -> 'Raafay')
        employee_name = raw_name.replace("_", " ").title()
        embedding = np.load(file_path)
        # Ensure unit normalization
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        database[employee_name] = embedding

    print(f"Loaded {len(database)} employee profile(s): {list(database.keys())}")
    return database


def draw_hud_header(frame, employee_count):
    """Draws a translucent top bar with instructions and status."""
    h, w, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 24), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Title & Instructions
    cv2.putText(
        frame,
        "AI ATTENDANCE KIOSK",
        (16, 34),
        cv2.FONT_HERSHEY_DUPLEX,
        0.75,
        (0, 255, 200),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "[ Peace Sign 2 = Check-In ]   [ Open Palm 5 = Check-Out ]",
        (260, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # Right side: enrolled count
    db_text = f"Enrolled: {employee_count}"
    (tw, _), _ = cv2.getTextSize(db_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(
        frame,
        db_text,
        (w - tw - 16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (160, 200, 160),
        1,
        cv2.LINE_AA,
    )


def draw_progress_bar(frame, progress_ratio, action_name, employee_name):
    """Draws an animated hold-to-confirm progress bar."""
    h, w, _ = frame.shape
    bar_w = 420
    bar_h = 24
    bx = (w - bar_w) // 2
    by = h - 90

    # Background card
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx - 12, by - 32), (bx + bar_w + 12, by + bar_h + 12), (18, 18, 22), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    verb = "CHECKING IN" if action_name == "CHECK_IN" else "CHECKING OUT"
    color = (0, 255, 120) if action_name == "CHECK_IN" else (255, 180, 0)

    # Status text
    pct = int(progress_ratio * 100)
    msg = f"HOLD STILL: {verb} {employee_name.upper()} ({pct}%)"
    cv2.putText(frame, msg, (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Outer bar frame
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (80, 80, 80), 2)
    # Filled progress
    filled_w = int(bar_w * progress_ratio)
    if filled_w > 0:
        cv2.rectangle(frame, (bx + 2, by + 2), (bx + filled_w - 2, by + bar_h - 2), color, -1)


def draw_toast_notification(frame, message, is_success=True, action="CHECK_IN"):
    """Draws an eye-catching popup banner when an action succeeds or hits cooldown."""
    h, w, _ = frame.shape
    banner_w = min(560, w - 40)
    banner_h = 60
    bx = (w - banner_w) // 2
    by = 75

    if is_success:
        bg_color = (15, 120, 40) if action == "CHECK_IN" else (160, 90, 10)
        border_color = (0, 255, 120) if action == "CHECK_IN" else (255, 200, 50)
    else:
        bg_color = (20, 40, 140)
        border_color = (50, 100, 255)

    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx + banner_w, by + banner_h), bg_color, -1)
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
    cv2.rectangle(frame, (bx, by), (bx + banner_w, by + banner_h), border_color, 2)

    (tw, th), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
    tx = bx + (banner_w - tw) // 2
    ty = by + (banner_h + th) // 2
    cv2.putText(frame, message, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def draw_recent_logs_footer(frame, recent_logs):
    """Draws a footer strip showing the most recent attendance event."""
    h, w, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 35), (w, h), (15, 15, 18), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    if recent_logs:
        last = recent_logs[0]
        action_color = (0, 255, 120) if last.get("Action") == "CHECK_IN" else (255, 180, 0)
        text = f"Recent Punch: {last.get('Employee_Name')} | {last.get('Action')} at {last.get('Time')}"
    else:
        text = "Ready for check-in / check-out. Stand in front and show gesture."
        action_color = (180, 180, 180)

    cv2.putText(frame, text, (16, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, action_color, 1, cv2.LINE_AA)


def main():
    print("==================================================")
    print("      AI GESTURE ATTENDANCE SYSTEM STARTING       ")
    print("==================================================")

    # 1. Load Face Database
    employees = load_employee_database(".")
    if not employees:
        print("[WARNING] No '*_embedding.npy' files found! Please run enroll_multiframe.py first.")

    # 2. Initialize Models
    print("Initializing InsightFace (buffalo_l)...")
    face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    print("Initializing Gesture Detector...")
    gesture_detector = GestureDetector(model_path="gesture_recognizer.task")

    logger = AttendanceLogger(log_file="attendance.csv", cooldown_seconds=30)

    # 3. Open Camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("[ERROR] Could not open camera 0!")
        return

    # Recognition & Trigger Configuration
    FACE_SIMILARITY_THRESHOLD = 0.50
    REQUIRED_HOLD_FRAMES = 20  # ~0.8s at 25 FPS
    
    # State tracking
    current_action = None
    current_employee = None
    hold_counter = 0

    # Toast banner notification state
    toast_message = ""
    toast_expiry = 0
    toast_is_success = True
    toast_action = "CHECK_IN"

    print("\n[READY] Camera live. Press 'Q' to quit.\n")

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # Mirror camera feed for natural interaction
        frame = cv2.flip(frame, 1)

        # -------------------------------------------------------------
        # Step A: Face Detection & Recognition
        # -------------------------------------------------------------
        detected_faces = face_app.get(frame)
        active_recognized_employee = None
        best_similarity = 0.0

        for face in detected_faces:
            live_embedding = face.normed_embedding
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # Find best matching employee from database
            matched_name = "Unknown"
            max_sim = 0.0

            for emp_name, emp_embedding in employees.items():
                sim = float(np.dot(emp_embedding, live_embedding))
                if sim > max_sim:
                    max_sim = sim
                    if sim > FACE_SIMILARITY_THRESHOLD:
                        matched_name = emp_name

            # Draw Face Bounding Box
            if matched_name != "Unknown":
                color = (0, 255, 120)  # Green for recognized staff
                active_recognized_employee = matched_name
                best_similarity = max_sim
            else:
                color = (0, 0, 255)    # Red for unknown visitor

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{matched_name} ({max_sim:.2f})"
            cv2.putText(
                frame,
                label,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )

        # -------------------------------------------------------------
        # Step B: Hand Gesture Detection
        # -------------------------------------------------------------
        gestures = gesture_detector.detect(frame)
        active_gesture_action = None

        for g in gestures:
            gesture_detector.draw_hand(frame, g)
            if g["action"] in ["CHECK_IN", "CHECK_OUT"]:
                active_gesture_action = g["action"]

        # -------------------------------------------------------------
        # Step C: Hold-to-Confirm State Machine
        # -------------------------------------------------------------
        # Both a valid recognized employee AND an action gesture must be present
        if active_recognized_employee and active_gesture_action:
            if (current_employee == active_recognized_employee and 
                current_action == active_gesture_action):
                hold_counter += 1
            else:
                # Started new gesture/person combo
                current_employee = active_recognized_employee
                current_action = active_gesture_action
                hold_counter = 1

            # Draw on-screen progress indicator
            progress = min(1.0, hold_counter / REQUIRED_HOLD_FRAMES)
            draw_progress_bar(frame, progress, current_action, current_employee)

            # Trigger condition reached!
            if hold_counter >= REQUIRED_HOLD_FRAMES:
                success, msg = logger.log(
                    current_employee,
                    current_action,
                    confidence=best_similarity
                )
                toast_message = msg
                toast_is_success = success
                toast_action = current_action
                toast_expiry = time.time() + 3.5

                # Reset hold counter after firing
                hold_counter = 0
                current_action = None
                current_employee = None
        else:
            # Gesture dropped or face moved away: smoothly reset
            if hold_counter > 0:
                hold_counter = max(0, hold_counter - 2)

        # -------------------------------------------------------------
        # Step D: Draw HUD Overlays & Notifications
        # -------------------------------------------------------------
        draw_hud_header(frame, len(employees))

        # Show popup notification if active
        if time.time() < toast_expiry and toast_message:
            draw_toast_notification(
                frame,
                toast_message,
                is_success=toast_is_success,
                action=toast_action
            )

        # Draw recent punch footer
        recent_logs = logger.get_recent_logs(limit=1)
        draw_recent_logs_footer(frame, recent_logs)

        # Show window
        cv2.imshow("AI Office Attendance Machine", frame)

        # Key controls: 'q' to quit, 'r' to reload database
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            employees = load_employee_database(".")
            toast_message = "Employee database reloaded!"
            toast_is_success = True
            toast_action = "CHECK_IN"
            toast_expiry = time.time() + 2.0

    camera.release()
    cv2.destroyAllWindows()
    print("\nAI Attendance System shutdown cleanly.")


if __name__ == "__main__":
    main()
