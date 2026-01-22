import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------- MediaPipe Setup ----------
model_path = "face_landmarker.task"

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)

landmarker = vision.FaceLandmarker.create_from_options(options)

# ---------- Camera Setup ----------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # ---------- Grayscale (Sprint 1 Task) ----------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---------- FPS Calculation ----------
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # ---------- MediaPipe Face Detection ----------
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = landmarker.detect(mp_image)

    # ---------- Draw Face Landmarks ----------
    if result.face_landmarks:
        h, w = frame.shape[:2]
        for landmark in result.face_landmarks[0]:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # ---------- Overlays ----------
    h, w = frame.shape[:2]
    cv2.putText(
        frame,
        f"{w}x{h}",
        (10, 60),
        cv2.FONT_HERSHEY_TRIPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Camera + Face Detection + FPS", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
