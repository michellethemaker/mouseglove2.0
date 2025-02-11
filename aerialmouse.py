import cv2
import mediapipe as mp
# import rerun as rr
import numpy as np
import pyautogui
import math
import threading
import queue
import time

# ISSUES: 
# fps still sucks for left hand: FIXED. pyautogui's MoveTo function has a hidden pause parameter. disabling it fixed alles.
# change scaling to the outer boundary of each hand/both hands. more consistent that way.
# script's getting long, move functions out when ready
# add: left/right index touching to en/disable pyautogui. need to rearrange threads for this

class HandAngles:
    def __init__(self):
        self.landmark_names = {
            'wrist': 0,

            'thumb_tip': 4,
            'thumb_mid': 3,
            'thumb_mid2': 2,
            'thumb_knuckle': 1,

            'index_tip': 8,
            'index_mid': 7,
            'index_mid2': 6,
            'index_knuckle': 5,
            
            'middle_tip': 12,
            'middle_mid': 11,
            'middle_mid2': 10,
            'middle_knuckle': 9,
            
            'ring_tip': 16,
            'ring_mid': 15,
            'ring_mid2': 14,
            'ring_knuckle': 13,
            
            'pinkie_tip': 20,
            'pinkie_mid': 19,
            'pinkie_mid2': 18,
            'pinkie_knuckle': 17
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
        origin = self.landmarkXY(hand_landmarks, origin_name)
        tip1 = self.landmarkXY(hand_landmarks, pt1_name)
        tip2 = self.landmarkXY(hand_landmarks, pt2_name)

        vector_tip1 = np.array([tip1.x - origin.x, tip1.y - origin.y])
        vector_tip2 = np.array([tip2.x - origin.x, tip2.y - origin.y])
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
    
def all_keypts_visible(landmarks):
    for landmark in landmarks:
        if landmark.x <= 0 or landmark.y <= 0:
            return False
    return True

def calc_avg_pos(positions):
    if len(positions) == 0:
        return (0, 0)
    avg_x = np.mean([pos[0] for pos in positions])
    avg_y = np.mean([pos[1] for pos in positions])
    avg_x_thumb = np.mean([pos[1] for pos in positions])
    avg_y_thumb = np.mean([pos[1] for pos in positions])
    return avg_x, avg_y, avg_x_thumb, avg_y_thumb

# initialise mp stuff
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  # used to draw landmarks on image

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
screen_width, screen_height = pyautogui.size()
screen_height = 1.5 * screen_height
screen_width = 1.5 * screen_width # scaled up so i dont need to move so much. TODO: add semaphore to change active frame area

# start rerun. open separate term and run 'rerun' to open rerun gui
# rr.init("mouseapp")
# rr.connect()

hand_angles = HandAngles()
smoothing_factor = 0.3 # higher=faster but less smooth
prev_x_L, prev_x_thumb_L, prev_y_L, prev_y_thumb_L = 0, 0, 0, 0
prev_x_R, prev_x_thumb_R, prev_y_R, prev_y_thumb_R = 0, 0, 0, 0
count_L, count_R, count_BOTH, count_missingL, count_missingR, count_clickies = 0, 0, 0, 0, 0, 0
maxcount_L, maxcount_R, maxcount_BOTH, maxcount_missingL, maxcount_missingR, maxcount_clickies = 30, 50, 40, 10, 10, 50
righthand_quit = False
enable_clickies = 1

xy_L, xy_R = [], [] # for 2 hand detection

queue_frames = queue.Queue() # main queue for camera frames
queue_calcs = queue.Queue() # side queue for calculations
queue_pos_L = queue.Queue() # side queue for left cursor position
queue_pos_R = queue.Queue() # side queue for right cursor position
stop_event = threading.Event()
enable_clickies_lock = threading.Lock()

def thread_calculations():
    while not stop_event.is_set():
        try: 
            landmarks, hand_label = queue_frames.get(timeout=0.1) # timeout prevents blocking, i.e. hanging
            if landmarks is None:
                break
            if hand_label == "Left":
                angle_thumb_left = hand_angles.calc_angle(landmarks, pt1_name='thumb_tip', pt2_name='index_tip', origin_name='thumb_knuckle')
                angle_pinkie_left = hand_angles.calc_angle(landmarks, pt1_name='pinkie_tip', pt2_name='pinkie_knuckle', origin_name='pinkie_mid2')
                
                disp_index_middle_tip = hand_angles.calc_displmt(landmarks, pt1_name='index_tip', pt2_name='middle_tip')
                disp_index_middle_mid2 = hand_angles.calc_displmt(landmarks, pt1_name='index_mid', pt2_name='middle_mid')
                
                disp_index = hand_angles.calc_displmt(landmarks, pt1_name='index_tip', pt2_name='index_knuckle')
                disp_index_wrist = hand_angles.calc_displmt(landmarks, pt1_name='index_knuckle', pt2_name='wrist')
                
                diff_displacement_index_middle_scaled = abs((disp_index_middle_tip - disp_index_middle_mid2)/disp_index_wrist)
                diff_displacement_index_scaled = abs((disp_index/disp_index_wrist))
                x1, y1 = int(hand_angles.landmarkXY(landmarks, "index_tip").x * screen_width), int(hand_angles.landmarkXY(landmarks, "index_tip").y * screen_height)
                x_smooth = int(prev_x_L + smoothing_factor * (x1 - prev_x_L))
                y_smooth = int(prev_y_L + smoothing_factor * (y1 - prev_y_L))
                x1_thumb_, y1_thumb_ = int(hand_angles.landmarkXY(landmarks, "thumb_tip").x * screen_width), int(hand_angles.landmarkXY(landmarks, "thumb_tip").y * screen_height)
                x_thumb_smooth = int(prev_x_thumb_L + smoothing_factor * (x1_thumb_ - prev_x_thumb_L))
                y_thumb_smooth = int(prev_y_thumb_L + smoothing_factor * (y1_thumb_ - prev_y_thumb_L))
                queue_calcs.put(("Left", diff_displacement_index_middle_scaled, diff_displacement_index_scaled, angle_thumb_left, angle_pinkie_left))
                queue_pos_L.put(("Left", x_smooth,y_smooth, x_thumb_smooth, y_thumb_smooth))
                # x1, y1 = int(hand_angles.landmarkXY(landmarks, "index_tip").x * screen_width), int(hand_angles.landmarkXY(landmarks, "index_tip").y * screen_height)
                # print(f'x: {x1}||y: {y1}')
                # queue_pos.put((x1,y1))

            elif hand_label == "Right":
                # TODO: hand recognition model here
                angle_thumb_right = hand_angles.calc_angle(landmarks, pt1_name='thumb_tip', pt2_name='index_tip', origin_name='thumb_knuckle')
                angle_pinkie_right = hand_angles.calc_angle(landmarks, pt1_name='pinkie_tip', pt2_name='pinkie_knuckle', origin_name='pinkie_mid2')
                x1, y1 = int(hand_angles.landmarkXY(landmarks, "index_tip").x * screen_width), int(hand_angles.landmarkXY(landmarks, "index_tip").y * screen_height)
                x_smooth = int(prev_x_R + smoothing_factor * (x1 - prev_x_R))
                y_smooth = int(prev_y_R + smoothing_factor * (y1 - prev_y_R))
                x1_thumb_, y1_thumb_ = int(hand_angles.landmarkXY(landmarks, "thumb_tip").x * screen_width), int(hand_angles.landmarkXY(landmarks, "thumb_tip").y * screen_height)
                x_thumb_smooth = int(prev_x_thumb_R + smoothing_factor * (x1_thumb_ - prev_x_thumb_R))
                y_thumb_smooth = int(prev_y_thumb_R + smoothing_factor * (y1_thumb_ - prev_y_thumb_R))
                queue_calcs.put(("Right", 0,0, angle_thumb_right, angle_pinkie_right))
                queue_pos_R.put(("Right", x_smooth,y_smooth, x_thumb_smooth, y_thumb_smooth))
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
            if all_keypts_visible(landmarks.landmark):
                try:
                    if not queue_calcs.empty():
                        hand_label_rxvd, displacement_index_middle, displacement_index, angle_thumb, angle_pinkie = queue_calcs.get()
                        
                        if hand_label_rxvd == "Left":
                            if not queue_pos_L.empty():
                                if displacement_index_middle < 0.1:
                                    if displacement_index > 0.5:
                                        pyautogui.scroll(10, _pause=False) # split by 0.5
                                    else:
                                        pyautogui.scroll(-10, _pause=False)
                                hand_label_rxvd_pos, x1, y1, x_thumb_1, y_thumb_1 = queue_pos_L.get()
                                if hand_label_rxvd_pos == "Left":
                                    count_missingL = 0
                                    count_missingR += 1
                                    prev_x_L, prev_y_L = x1, y1
                                    prev_x_thumb_L, prev_y_thumb_L = x_thumb_1, y_thumb_1
                                    pyautogui.moveTo(x1, y1, _pause=False) # IMPT: pause=false or script locks up while mouse is moving
                                    xy_L.append((x1, y1, x_thumb_1, y_thumb_1))
                                    if len(xy_L) > maxcount_BOTH:
                                        xy_L.pop(0)
                                        # print("LEFT MAXED")
                                    if count_missingR >= maxcount_missingR:
                                        xy_R.clear()
                                        count_missingR = 0

                            text = f"Left Angle: {angle_thumb:.2f} deg"
                            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                            if angle_thumb >= 40:
                                if not count_L <= maxcount_L:
                                    if angle_pinkie >= 160:
                                        # if enable_clickies == True:
                                        pyautogui.rightClick()
                                        time.sleep(0.2) # visual rep that i clicked
                                        cv2.putText(frame, "RIGHT CLICK", (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)    
                                        
                                    else:
                                        # if enable_clickies == True:
                                        pyautogui.click()
                                        time.sleep(0.2) # visual rep that i clicked
                                        cv2.putText(frame, "CLICK", (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                                else:
                                        count_L += 1 # window for left thumb extended 
                                        # print(f'countL: {count_L}')
                            else:
                                count_L = 0
                                # print(f'0countL: {count_L}')
                        elif hand_label_rxvd == "Right":
                            if not queue_pos_R.empty():
                                hand_label_rxvd_pos, x1, y1, x_thumb_1, y_thumb_1 = queue_pos_R.get()
                                if hand_label_rxvd_pos == "Right":
                                    count_missingR = 0
                                    count_missingL += 1
                                    prev_x_R, prev_y_R = x1, y1
                                    prev_x__thumb_R, prev_y__thumb_R = x_thumb_1, y_thumb_1
                                    xy_R.append((x1, y1, x_thumb_1, y_thumb_1))
                                    if len(xy_R) > maxcount_BOTH:
                                        xy_R.pop(0)
                                        # print("RIGHT MAXED")
                                    if count_missingL >= maxcount_missingL:
                                        xy_L.clear()
                                        count_missingL = 0
                                        # print("CLEAR LEFT")
                            
                            # if enable_clickies == True:
                            text = f"Right Angle: {angle_thumb:.2f} deg"
                            cv2.putText(frame, text, (150, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                            if angle_thumb >= 40:
                                if not count_R <= maxcount_R:
                                    righthand_quit = True
                                    cv2.putText(frame, "STAP", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (25, 10, 255), 1)
                                else:
                                    count_R+=1 # window for right thumb extended 
                                    # print(f'countR: {count_R}')
                            else:
                                count_R = 0 # reset back to 0 if thumb not maintained at extended position
                            # print(f'0countR: {count_R}')
                        # print(f'LEFT: {len(xy_L)}| RIGHT: {len(xy_R)} | {maxcount_BOTH}')
                        # if len(xy_L) ==maxcount_BOTH and len(xy_R) ==maxcount_BOTH: # TODO: put this in a thread too
                        #         left_avg = calc_avg_pos(xy_L)
                        #         right_avg = calc_avg_pos(xy_R)
                        #         displmt = math.sqrt(math.pow((left_avg[0] - right_avg[0]),2) + math.pow((left_avg[1] - right_avg[1]),2) )
                        #         displmt_thumb = math.sqrt(math.pow((left_avg[2] - right_avg[2]),2) + math.pow((left_avg[3] - right_avg[3]),2) )
                        #         # print(f'magic: {displmt/displmt_thumb * 0.5}')    
                        #         if (displmt/displmt_thumb * 0.5) < 0.9:
                        #             count_clickies+=1
                        #             if count_clickies >= maxcount_clickies:
                        #                 with enable_clickies_lock:
                        #                     # enable_clickies^=True # TODO: add the check if enable_clickies is on before clicking
                        #                     print(f'clickies: {enable_clickies}')
                        #                     # time.sleep(0.2)
                        #                     count_clickies = 0

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