import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),         # thumb
    (0,5),(5,6),(6,7),(7,8),         # index
    (5,9),(9,10),(10,11),(11,12),    # middle
    (9,13),(13,14),(14,15),(15,16),  # ring
    (13,17),(17,18),(18,19),(19,20), # pinky
    (0,17)                           # palm
]

def draw_landmarks(image, detection_result):
    h, w, _ = image.shape
    for hand_landmarks in detection_result.hand_landmarks:
        # Draw connections
        for start, end in HAND_CONNECTIONS:
            x0 = int(hand_landmarks[start].x * w)
            y0 = int(hand_landmarks[start].y * h)
            x1 = int(hand_landmarks[end].x * w)
            y1 = int(hand_landmarks[end].y * h)
            cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
        # Draw joints
        for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
    return image

webcam_capture = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = webcam_capture.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        results = landmarker.detect(mp_image)

        if results.hand_landmarks:
            frame = draw_landmarks(frame, results)

        cv2.imshow("Handtracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

webcam_capture.release()
cv2.destroyAllWindows()