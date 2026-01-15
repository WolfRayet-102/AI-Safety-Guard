import time
import cv2 # type: ignore

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)     

    current_time= time.time()
    fps = 1/(current_time-prev_time)
    prev_time = current_time

    h, w = frame.shape[:2]
    cv2.putText(gray, f"{w}x{h}", (10,60),
                cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255,0,0), 2)

    cv2.putText(gray, f"FPS: {fps:.2f}", (10,30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Camera Test", gray)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
