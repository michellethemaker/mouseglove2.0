import cv2
import mediapipe as mp
import rerun as rr
import numpy as np
import pyautogui

# initialise mp stuff
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  # used to draw landmarks on image

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
screen_width, screen_height = pyautogui.size()

# start rerun. open separate term and run 'rerun' to open rerun gui
rr.init("mouseapp")
rr.connect()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # flip n convert frame to RGB (MediaPipe works with RGB images)
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame with MediaPipe Hands
    results = hands.process(frame_rgb)
    
     # If hands are detected
    if results.multi_hand_landmarks:
        for idx, landmarks in enumerate(results.multi_hand_landmarks):
            # draw hand landmarks on  frame
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # get index finger, conv to screen coords
            index_finger_tip = landmarks.landmark[8] 
            x = int(index_finger_tip.x * screen_width)
            y = int(index_finger_tip.y * screen_height)
            pyautogui.moveTo(x, y)
            
            # extract keypoints for logging (3D coordinates: x, y, z)
            keypoints = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
            hand_label = results.multi_handedness[idx].classification[0].label
            text= f'hand: {hand_label}'
            position = (10,10)
            font = cv2.FONT_HERSHEY_COMPLEX
            font_scale = 0.5
            color = (50,200,50)
            thickness = 1
            bottom_left_origin = False
            cv2.putText(frame, text, position, font, font_scale, color, thickness, bottomLeftOrigin=bottom_left_origin)
            # log keypoints with ReRun using Points3D loggable type (if u wanna use rerun)
            rr.log("hand/{}/keypoints".format(idx), rr.Points3D(positions=keypoints))

    # disp frame
    cv2.imshow('movement', frame)
    
    # exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()