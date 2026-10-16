import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Hand landmark connection topology
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                 # Palm base
]


class GestureDetector:
    def __init__(self, model_path="gesture_recognizer.task", min_confidence=0.5):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

    def _rule_based_check(self, landmarks, frame_h, frame_w):
        """
        Fallback rule-based gesture classifier using landmark geometry.
        Returns: ("CHECK_IN" | "CHECK_OUT" | None, confidence)
        """
        # Pixel coordinates
        pts = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in landmarks]
        
        # Check finger extension by comparing tip y to pip y
        # In image coords, y increases downwards, so extended finger has tip_y < pip_y
        index_ext = pts[8][1] < pts[6][1]
        middle_ext = pts[12][1] < pts[10][1]
        ring_ext = pts[16][1] < pts[14][1]
        pinky_ext = pts[20][1] < pts[18][1]

        # Peace sign (Victory): Index & Middle extended, Ring & Pinky folded
        if index_ext and middle_ext and not ring_ext and not pinky_ext:
            return "CHECK_IN", 0.85

        # Open palm: All 4 fingers clearly extended
        if index_ext and middle_ext and ring_ext and pinky_ext:
            return "CHECK_OUT", 0.85

        return None, 0.0

    def detect(self, frame_bgr):
        """
        Detects hand gestures in the given BGR frame.
        Returns: list of dicts: [
            {
                "action": "CHECK_IN" | "CHECK_OUT" | "OTHER",
                "label": raw_label_string,
                "confidence": float,
                "landmarks": list of normalized landmarks,
                "center": (x, y)
            }
        ]
        """
        h, w, _ = frame_bgr.shape
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.recognizer.recognize(mp_image)
        detected_gestures = []

        if not result.hand_landmarks:
            return detected_gestures

        for i, landmarks in enumerate(result.hand_landmarks):
            action = None
            label = "Unknown"
            score = 0.0

            # 1. Check MediaPipe neural gesture classification
            if result.gestures and i < len(result.gestures) and result.gestures[i]:
                top_gesture = result.gestures[i][0]
                label = top_gesture.category_name
                score = top_gesture.score

                if label == "Victory":
                    action = "CHECK_IN"
                elif label == "Open_Palm":
                    action = "CHECK_OUT"

            # 2. Hybrid fallback: verify with landmark geometry if neural result is ambiguous
            if action is None:
                rule_action, rule_score = self._rule_based_check(landmarks, h, w)
                if rule_action:
                    action = rule_action
                    label = "Peace (Rule)" if rule_action == "CHECK_IN" else "Open Palm (Rule)"
                    score = rule_score

            # Compute approximate hand center (wrist + middle mcp average)
            center_x = int(((landmarks[0].x + landmarks[9].x) / 2.0) * w)
            center_y = int(((landmarks[0].y + landmarks[9].y) / 2.0) * h)

            detected_gestures.append({
                "action": action,
                "label": label,
                "confidence": score,
                "landmarks": landmarks,
                "center": (center_x, center_y),
            })

        return detected_gestures

    def draw_hand(self, frame, gesture_info):
        """
        Draws stylized hand skeleton and gesture status badge onto the frame.
        """
        h, w, _ = frame.shape
        landmarks = gesture_info["landmarks"]
        action = gesture_info["action"]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        # Accent colors based on detected action
        if action == "CHECK_IN":
            bone_color = (0, 255, 128)  # Bright Green
            joint_color = (255, 255, 255)
        elif action == "CHECK_OUT":
            bone_color = (255, 180, 0)  # Bright Cyan / Blue
            joint_color = (255, 255, 255)
        else:
            bone_color = (180, 180, 180)  # Subtle Gray
            joint_color = (200, 200, 200)

        # Draw bones
        for p1_idx, p2_idx in HAND_CONNECTIONS:
            cv2.line(frame, pts[p1_idx], pts[p2_idx], bone_color, 2, cv2.LINE_AA)

        # Draw joints
        for pt in pts:
            cv2.circle(frame, pt, 4, joint_color, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 5, bone_color, 1, cv2.LINE_AA)

        # Gesture label badge above the hand
        cx, cy = gesture_info["center"]
        badge_y = max(25, cy - 60)

        if action == "CHECK_IN":
            text = f"Check-In [Peace 2] ({gesture_info['confidence']:.2f})"
            bg_color = (0, 180, 0)
        elif action == "CHECK_OUT":
            text = f"Check-Out [Palm 5] ({gesture_info['confidence']:.2f})"
            bg_color = (200, 100, 0)
        else:
            text = f"{gesture_info['label']}"
            bg_color = (60, 60, 60)

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        bx1, by1 = max(0, cx - tw // 2 - 8), max(0, badge_y - th - 8)
        bx2, by2 = min(w, cx + tw // 2 + 8), min(h, badge_y + 6)

        # Draw badge background pill
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), bg_color, -1)
        cv2.putText(
            frame,
            text,
            (bx1 + 8, by2 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
