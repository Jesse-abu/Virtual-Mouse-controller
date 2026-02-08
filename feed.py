import cv2
import mediapipe as mp
from monitor import movement, window_size
import math

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing_utils = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=3)

if not cap.isOpened():
    print('Could not access camera')

print('Press q to exit')

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)
    wth, hgt = window_size()
    frame = cv2.resize(frame, (wth, hgt), interpolation=cv2.INTER_AREA)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        lmlist = []
        xlist = []
        ylist = []

        for handLMS in results.multi_hand_landmarks:

            for id, lm in enumerate(results.multi_hand_landmarks[0].landmark):
                h, w, _ = frame.shape
                xc, yc = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, xc, yc])
                xlist.append(xc)
                ylist.append(yc)

            x1, y1 = lmlist[4][1], lmlist[4][2]
            run, rise = lmlist[8][1], lmlist[8][2]

            click_distance = math.hypot((run - x1), (rise - y1))

            cv2.circle(frame, (run, rise), 20, (255, 0, 255), cv2.FILLED)
            cv2.putText(frame, str(int(click_distance)), (10, 120), cv2.FONT_HERSHEY_COMPLEX, 5, (0, 255, 0), 6 )
            mp_drawing_utils.draw_landmarks(
                frame,
                handLMS,
                mp_hands.HAND_CONNECTIONS,
            )

            movement(run, rise, click=True if click_distance < 100 else False)

    cv2.imshow('window', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()