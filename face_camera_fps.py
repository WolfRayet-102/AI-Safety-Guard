import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

# =========================
# Utility Functions
# =========================
def euclidean(p1, p2):
    return math.dist(p1, p2)

def compute_ear(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


# =========================
# MediaPipe Setup
# =========================
model_path = "face_landmarker.task"

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)

landmarker = vision.FaceLandmarker.create_from_options(options)

# =========================
# Eye Landmark Indices
# =========================

# Dense contours (visualisation)
LEFT_EYE_IDX = [
    33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145 ,
    144, 163, 7, 472, 471, 470, 469, 468, 130,247, 30, 
    29, 27, 28, 56, 190, 243, 112, 26 , 22, 23, 24, 110, 25
]

RIGHT_EYE_IDX = [
    362,382,381,380,374,373,390, 249, 263, 466, 388,387,386,385, 
    384, 398, 476, 477,474, 475, 473, 463, 414, 286,258,257, 
    259, 260,467, 359, 255, 339, 254, 253, 252,256,341
]


# Minimal EAR points (6-point standard)
LEFT_EAR_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EAR_IDX = [362, 385, 387, 263, 373, 380]

# =========================
# Camera Setup
# =========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = time.time()

# =========================
# Main Loop
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # FPS calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = landmarker.detect(mp_image)

    if result.face_landmarks:
        h, w = frame.shape[:2]
        landmarks = result.face_landmarks[0]

        # ---------- Draw Dense Eye Contours ----------
        for idx in LEFT_EYE_IDX:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        for idx in RIGHT_EYE_IDX:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        # ---------- EAR Calculation ----------
        left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EAR_IDX]
        right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EAR_IDX]

        left_ear = compute_ear(left_eye)
        right_ear = compute_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        cv2.putText(
            frame,
            f"EAR: {avg_ear:.2f}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # ---------- Overlays ----------
    h, w = frame.shape[:2]
    cv2.putText(frame, f"{w}x{h}", (10, 60),
                cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 0, 0), 2)

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Eye Landmarks + EAR", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
