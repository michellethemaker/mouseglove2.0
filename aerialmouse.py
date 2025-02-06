import cv2
import mediapipe as mp
# import rerun as rr
import numpy as np
import pyautogui
import math
import threading
import queue
import time

# ISSUES: fps still sucks for left hand

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

    # retrieve landmark's x/y/z coords based on its name
    # input: hand landmark object from mp, landmark name
    # returns: array of x,y,z coords corresponding to landmark pt's name
    # issues:
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
screen_height = 1.4 * screen_height
screen_width = 1.4 * screen_width # scaled so i dont need to move so much. TODO: add semaphore to change active frame area

# start rerun. open separate term and run 'rerun' to open rerun gui
# rr.init("mouseapp")
# rr.connect()

hand_angles = HandAngles()
smoothing_factor = 0.5 # higher=faster but less smooth
prev_x_L, prev_y_L = 0, 0
maxcount_L, maxcount_R = 3, 50
count_L, count_R = 0, 0
righthand_quit = False

queue_frames = queue.Queue() # main queue for camera frames
queue_calcs = queue.Queue() # side queue for calculations
queue_pos = queue.Queue() # side queue for cursor position
stop_event = threading.Event()

def thread_calculations():
    while not stop_event.is_set():
        try: 
            landmarks, hand_label = queue_frames.get(timeout=0.1) # timeout prevents blocking, i.e. hanging
            if landmarks is None:
                break
            if hand_label == "Left":
                angle_thumb_left = hand_angles.calc_angle(landmarks, pt1_name='thumb_tip', pt2_name='index_tip', origin_name='thumb_joint')
                angle_pinkie_left = hand_angles.calc_angle(landmarks, pt1_name='pinkie_tip', pt2_name='pinkie_joint', origin_name='pinkie_mid2')
                x1, y1 = int(hand_angles.landmarkXY(landmarks, "index_tip").x * screen_width), int(hand_angles.landmarkXY(landmarks, "index_tip").y * screen_height)
                x_smooth = int(prev_x_L + smoothing_factor * (x1 - prev_x_L))
                y_smooth = int(prev_y_L + smoothing_factor * (y1 - prev_y_L))
                queue_calcs.put(("Left", angle_thumb_left, angle_pinkie_left))
                queue_pos.put((x_smooth,y_smooth))
                # x1, y1 = int(hand_angles.landmarkXY(landmarks, "index_tip").x * screen_width), int(hand_angles.landmarkXY(landmarks, "index_tip").y * screen_height)
                # print(f'x: {x1}||y: {y1}')
                # queue_pos.put((x1,y1))

            elif hand_label == "Right":
                # TODO: hand recognition model here
                angle_thumb_right = hand_angles.calc_angle(landmarks, pt1_name='thumb_tip', pt2_name='index_tip', origin_name='thumb_joint')
                angle_pinkie_right = hand_angles.calc_angle(landmarks, pt1_name='pinkie_tip', pt2_name='pinkie_joint', origin_name='pinkie_mid2')
                queue_calcs.put(("Right", angle_thumb_right, angle_pinkie_right))
        except queue.Empty:
            continue # timeout, continue checking if stop_event is set
        except Exception as e:
            print(f'error in thread_calculations: {e}')

# start calculation thread
thread_calcs = threading.Thread(target=thread_calculations)
thread_calcs.start()

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
            hand_label = results.multi_handedness[idx].classification[0].label
            # draw hand landmarks on  frame
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            #send landmarks to calc thread
            queue_frames.put((landmarks, hand_label)) 
            
            try:
                if not queue_calcs.empty():
                    hand_label_received, angle_thumb, angle_pinkie = queue_calcs.get()
                    
                    if hand_label_received == "Left":
                        if not queue_pos.empty():
                            x1, y1 = queue_pos.get()
                            prev_x_L, prev_y_L = x1, y1
                            pyautogui.moveTo(x1, y1)
                        text = f"Left Angle: {angle_thumb:.2f} deg"
                        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                        if angle_thumb >= 40:
                            if not count_L <= maxcount_L:
                                if angle_pinkie >= 160:
                                    pyautogui.rightClick()
                                    time.sleep(0.2) # visual rep that i clicked
                                    cv2.putText(frame, "RIGHT CLICK", (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)    
                                    
                                else:
                                    pyautogui.click()
                                    time.sleep(0.2) # visual rep that i clicked
                                    cv2.putText(frame, "CLICK", (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                            else:
                                    count_L += 1
                                    # print(f'countL: {count_L}')
                        else:
                            count_L = 0
                            # print(f'0countL: {count_L}')
                    elif hand_label_received == "Right":
                        text = f"Right Angle: {angle_thumb:.2f} deg"
                        cv2.putText(frame, text, (150, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        if angle_thumb >= 40:
                            if not count_R <= maxcount_R:
                                righthand_quit = True
                                cv2.putText(frame, "STAP", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (25, 10, 255), 1)
                            else:
                                count_R+=1
                                # print(f'countR: {count_R}')
                        else:
                            count_R = 0 # reset back to 0 if thumb not maintained at extended position
                            # print(f'0countR: {count_R}')

            except Exception as e:
                print(f'error: {e}')
                continue
            # extract keypoints for logging (3D coordinates: x, y, z)
            # keypoints = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
            # log keypoints with ReRun using Points3D loggable type (if u wanna use rerun)
            # rr.log("hand/{}/keypoints".format(idx), rr.Points3D(positions=keypoints))

    # disp frame
    cv2.imshow('movement', frame)
    
    # exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q') or righthand_quit == True:
        stop_event.set()
        queue_calcs.put(None)
        thread_calcs.join()
        print("===               ===\n ===threads ended===\n===               ===")
        break

cap.release()
cv2.destroyAllWindows()