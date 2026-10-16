import os
import csv
import time
import threading
from datetime import datetime

# Windows audio feedback fallback
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


class AttendanceLogger:
    def __init__(self, log_file="attendance.csv", cooldown_seconds=30):
        self.log_file = log_file
        self.cooldown_seconds = cooldown_seconds
        # Maps (employee_name, action) -> last_logged_timestamp
        self.last_actions = {}
        self._ensure_log_file()

    def _ensure_log_file(self):
        """Creates the CSV file with header if it doesn't already exist."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Date", "Time", "Employee_Name", "Action", "Confidence"])

    def get_cooldown_remaining(self, employee_name, action):
        """Returns the number of seconds remaining before this employee can perform this action again."""
        key = (employee_name.lower(), action)
        last_time = self.last_actions.get(key, 0)
        elapsed = time.time() - last_time
        remaining = self.cooldown_seconds - elapsed
        return max(0, int(remaining))

    def _play_chime_async(self, action):
        """Plays non-blocking audio feedback in a background thread."""
        if not HAS_WINSOUND:
            return

        def sound_worker():
            try:
                if action == "CHECK_IN":
                    # Upbeat ascending chime
                    winsound.Beep(880, 120)
                    winsound.Beep(1320, 200)
                else:
                    # Warm descending chime
                    winsound.Beep(1046, 120)
                    winsound.Beep(659, 220)
            except Exception:
                pass

        threading.Thread(target=sound_worker, daemon=True).start()

    def log(self, employee_name, action, confidence=1.0):
        """
        Logs a check-in or check-out event if cooldown allows.
        Returns: (success: bool, message: str)
        """
        remaining = self.get_cooldown_remaining(employee_name, action)
        if remaining > 0:
            return False, f"Cooldown: wait {remaining}s"

        now = datetime.now()
        timestamp_str = now.isoformat()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%I:%M:%S %p")

        with open(self.log_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp_str, date_str, time_str, employee_name, action, f"{confidence:.2f}"])

        self.last_actions[(employee_name.lower(), action)] = time.time()
        self._play_chime_async(action)

        verb = "Checked In" if action == "CHECK_IN" else "Checked Out"
        return True, f"{employee_name} {verb} at {time_str}"

    def get_recent_logs(self, limit=5):
        """Returns the most recent attendance records."""
        if not os.path.exists(self.log_file):
            return []

        records = []
        try:
            with open(self.log_file, mode="r", encoding="utf-8") as f:
                reader = list(csv.DictReader(f))
                for row in reversed(reader[-limit:]):
                    records.append(row)
        except Exception:
            pass

        return records
