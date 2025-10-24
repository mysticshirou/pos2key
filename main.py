# https://mediapipe.readthedocs.io/en/latest/solutions/hands.html

import cv2
import mediapipe as mp
import time
import math
import numpy as np

class HandController:
    def __init__(self, width=600, height=500, fist_threshold=0.07, thres_x=0.1, thres_y=0.1):
        self.cam_width = width
        self.cam_height = height
        self.control_mode = False
        self.fist_threshold = fist_threshold
        self.THRES_X = thres_x
        self.THRES_Y = thres_y
        self.prev_center = None

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)

        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        self.hand = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.finger_tips = [4, 8, 12, 16, 20] # thumb, index, middle, ring, pinky
        self.finger_mcps = [2, 5, 9, 13, 17]
        self.finger_ip = [3, 7, 11, 15, 19]
        self.wrist = 0
    
    def distance(self, x1, x2, y1, y2):
        return math.sqrt((x1-x2)**2+(y1-y2)**2)

    def check_fist(self, landmark):
        """return True when making a fist"""
        folded = 0
        for tip, mcp in zip(self.finger_ip[1:], self.finger_mcps[1:]):
            if landmark[tip].y > landmark[mcp].y:
                folded+=1

        closed_thumb = (
            abs(landmark[self.finger_ip[0]].x - landmark[self.finger_ip[1]].x) < self.fist_threshold and 
            abs(landmark[self.finger_ip[0]].y - landmark[self.finger_ip[1]].y) < self.fist_threshold
            )

        return folded==4 #and closed_thumb
    
    def find_center_palm(self, landmark, frame):
        mcps_x = [landmark[mcp].x for mcp in self.finger_mcps[2:]]
        mcps_y = [landmark[mcp].y for mcp in self.finger_mcps[2:]]
        wrist = landmark[self.wrist]

        cx = (np.sum(mcps_x) + wrist.x) / (len(mcps_x) + 1)
        cy = (np.sum(mcps_y) + wrist.y) / (len(mcps_y) + 1)

        cv2.circle(frame, (int(cx*self.cam_width), int(cy*self.cam_height)), 20, (255,0,255), -1)
        return cx, cy
    
    def draw_info(self, frame, text):
        cv2.putText(frame, text, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def run(self):
        while True:
            success, frame = self.cap.read() # Frames in BGR
            if not success:
                break

            flipped_frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
            result = self.hand.process(rgb_frame)

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                # print(hand_landmarks)
                self.mp_drawing.draw_landmarks(
                    flipped_frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS, 
                    self.drawing_styles.get_default_hand_landmarks_style(),
                    self.drawing_styles.get_default_hand_connections_style()
                    )

                control_mode = self.check_fist(hand_landmarks.landmark)
                mode_text = "Fist (Control Mode)" if control_mode else "Open (Idle)"
                self.draw_info(flipped_frame, mode_text)

                if control_mode:
                    cx, cy = self.find_center_palm(hand_landmarks.landmark, flipped_frame)
                    if self.prev_center is not None:
                        dx = cx - self.prev_center[0]
                        dy = cy - self.prev_center[1]

                        if abs(dx) > self.THRES_X:
                            if dx > 0:
                                self.move_right()
                            else:
                                self.move_left()
                        
                        if abs(dy) > self.THRES_Y:
                            if dy > 0:
                                self.slide()
                            else:
                                self.jump()

                    self.prev_center = (cx, cy)

            cv2.imshow("Hand Capture", flipped_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def jump(self):
        print("JUMP!")
        pass

    def slide(self):
        print("SLIDE!")
        pass

    def move_left(self):
        print("GO LEFT!")
        pass

    def move_right(self):
        print("GO RIGHT!")
        pass



if __name__ == "__main__":
    controller = HandController()
    controller.run()