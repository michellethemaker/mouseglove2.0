import cv2
import mediapipe as mp
# import rerun as rr
import numpy as np
import pyautogui
import math

class HandAngles:
    def __init__(self):
        self.landmark_names = {
            'thumb_joint': 1,
            'thumb_tip': 4,
            'index_joint': 5,
            'index_tip': 8,
            'pinkie_tip': 20,
            'pinkie_mid': 19,
            'pinkie_mid2': 18,
            'pinkie_joint': 17
        }

    def landmarkXY(self, hand_landmarks, target_name):
        target_idx = self.landmark_names[target_name]
        return hand_landmarks.landmark[target_idx]
    
    # calc angle btwn 2 pts and an origin 
    # input: hand landmark object from mp, pt1 name, pt2 name, origin name
    # returns: angle btwn the 3 pts(degrees)
    # issues:
    def calc_angle(self, hand_landmarks, pt1_name, pt2_name, origin_name):
        joint = self.landmarkXY(hand_landmarks, origin_name)
        tip1 = self.landmarkXY(hand_landmarks, pt1_name)
        tip2 = self.landmarkXY(hand_landmarks, pt2_name)

        vector_tip1 = np.array([tip1.x - joint.x, tip1.y - joint.y])
        vector_tip2 = np.array([tip2.x - joint.x, tip2.y - joint.y])
        dot_product = np.dot(vector_tip1, vector_tip2)
        norm_tip1 = np.linalg.norm(vector_tip1)
        norm_tip2 = np.linalg.norm(vector_tip2)
        angle_rad = np.arccos(np.clip(dot_product / (norm_tip1 * norm_tip2), -1.0, 1.0))
        angle_deg = math.degrees(angle_rad)
        if angle_deg >=180: # mirror case
            angle_deg = 180 - angle_deg

        return angle_deg
    
    # calc displacement btwn 2 landmarks
    # input: hand landmark object from mp, landmark1 name, landmark2 name
    # returns: raw displacement value
    # issues:
    def calc_displmt(self, hand_landmarks, pt1_name, pt2_name):
        pt1 = self.landmarkXY(hand_landmarks, pt1_name)
        pt2 = self.landmarkXY(hand_landmarks, pt2_name)
        displmt = math.sqrt(math.pow((pt1.x - pt2.x),2) + math.pow((pt1.y - pt2.y),2) )
        # print(f'displmt: {displmt}')
        return displmt
    
# initialise mp stuff
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  # used to draw landmarks on image

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
screen_width, screen_height = pyautogui.size()

# start rerun. open separate term and run 'rerun' to open rerun gui
# rr.init("mouseapp")
# rr.connect()

hand_angles = HandAngles()
smoothing_factor = 0.3 # higher=faster but less smooth
prev_x, prev_y = 0, 0

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
            angle_thumb = hand_angles.calc_angle(landmarks, pt1_name='thumb_tip', pt2_name='index_tip', origin_name='thumb_joint')

            x1, y1 = int(hand_angles.landmarkXY(landmarks, "index_tip").x * screen_width), int(hand_angles.landmarkXY(landmarks, "index_tip").y * screen_height)
            x_smooth = int(prev_x + smoothing_factor * (x1 - prev_x))
            y_smooth = int(prev_y + smoothing_factor * (y1 - prev_y))
            
            text = f'Angle: {angle_thumb:.2f}°'
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            pyautogui.moveTo(x_smooth, y_smooth)
            prev_x, prev_y = x_smooth, y_smooth

            if angle_thumb >= 40:
                angle_pinkie = hand_angles.calc_angle(landmarks, pt1_name='pinkie_tip', pt2_name='pinkie_joint', origin_name='pinkie_mid2')
                text = f'Angle: {angle_pinkie:.2f}°'
                cv2.putText(frame, text, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 150), 1)
                if angle_pinkie >=160: # extended pinkie
                    pyautogui.rightClick()
                    text = "RIGHT CLICK"
                else:
                    pyautogui.click()
                    text = "CLICK"
                cv2.putText(frame, text, (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
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
            # rr.log("hand/{}/keypoints".format(idx), rr.Points3D(positions=keypoints))

    # disp frame
    cv2.imshow('movement', frame)
    
    # exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()