#Author: Daniyar Boztayev
#A simple program to open websites with hand gestures.

import cv2
import numpy as np
import pyautogui
import webbrowser
import time

background = None
hand = None
frames_elapsed = 0
CALIBRATION_TIME = 30
BG_WEIGHT = 1
OBJ_THRESHOLD = 32
gesture_list = []

class HandData:
    def __init__(self, top, bottom, left, right, centerX):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.centerX = centerX
        self.fingers = 0
        self.gestureList = gesture_list
        self.isInFrame = False
        self.last_finger_count = None
        self.finger_count_start_time = 0

    def update(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

def write_on_image(frame, hand, frames_elapsed, CALIBRATION_TIME, region_left, region_top, region_right, region_bottom):
    text = "Searching..."

    if frames_elapsed < CALIBRATION_TIME:
        text = "Calibrating..."

    elif hand is None or not hand.isInFrame:
        text = "No hand detected"

    else:
        if hand.fingers == 0:
            text = "Rock"
        elif hand.fingers == 1:
            text = "Pointing"
        elif hand.fingers == 2:
            text = "Scissors"

    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (region_left, region_top), (region_right, region_bottom), (255, 255, 255), 2)

def get_region(frame, region_top, region_bottom, region_left, region_right):
    region = frame[region_top:region_bottom, region_left:region_right]
    region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    region = cv2.GaussianBlur(region, (5, 5), 0)
    return region

def get_average(region):
    global background
    if background is None:
        background = region.copy().astype("float")
        return
    cv2.accumulateWeighted(region, background, BG_WEIGHT)

def segment(region):
    global hand
    diff = cv2.absdiff(background.astype(np.uint8), region)
    thresholded_region = cv2.threshold(diff, OBJ_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresholded_region.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        if hand is not None:
            hand.isInFrame = False
        return None
    else:
        if hand is not None:
            hand.isInFrame = True
        segmented_region = max(contours, key=cv2.contourArea)
        return (thresholded_region, segmented_region)

def get_hand_data(thresholded_image, segmented_image):
    global hand, frames_elapsed

    convexHull = cv2.convexHull(segmented_image)
    top = tuple(convexHull[convexHull[:, :, 1].argmin()][0])
    bottom = tuple(convexHull[convexHull[:, :, 1].argmax()][0])
    left = tuple(convexHull[convexHull[:, :, 0].argmin()][0])
    right = tuple(convexHull[convexHull[:, :, 0].argmax()][0])

    centerX = int((left[0] + right[0]) / 2)

    if hand is None:
        hand = HandData(top, bottom, left, right, centerX)
    else:
        hand.update(top, bottom, left, right)

    hand.fingers = count_fingers(thresholded_image)

    hand.gestureList.append(hand.fingers)
    if frames_elapsed % 12 == 0:
        hand.fingers = most_frequent(hand.gestureList)
        hand.gestureList.clear()

        finger_consistency(hand, hand.fingers)

def count_fingers(thresholded_image):
    line_height = int(hand.top[1] + (0.2 * (hand.bottom[1] - hand.top[1])))

    # Create a copy of the thresholded image to draw the line
    image_with_line = thresholded_image.copy()
    cv2.line(image_with_line, (thresholded_image.shape[1], line_height), (0, line_height), 255, 1)

    # Create the line mask
    line_mask = np.zeros(thresholded_image.shape, dtype=np.uint8)
    cv2.line(line_mask, (thresholded_image.shape[1], line_height), (0, line_height), 255, 1)

    # Perform bitwise AND to get the line on the thresholded image
    line = cv2.bitwise_and(thresholded_image, thresholded_image, mask=line_mask)

    contours, _ = cv2.findContours(line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    fingers = 0

    for curr in contours:
        width = len(curr)
        if 5 < width < 3 * abs(hand.right[0] - hand.left[0]) / 4:
            fingers += 1
        
    return fingers


def most_frequent(input_list):
    dict = {}
    count = 0
    most_freq = 0

    for item in reversed(input_list):
        dict[item] = dict.get(item, 0) + 1
        if dict[item] >= count:
            count, most_freq = dict[item], item

    return most_freq

def finger_consistency(hand, fingers, duration = 2):
    current_time = time.time()
    
    if hand.last_finger_count != fingers:
        hand.last_finger_count = fingers
        hand.finger_count_start_time = current_time
    elif current_time - hand.finger_count_start_time >= duration:
        perform_task(fingers)
        hand.finger_count_start_time = current_time
        hand.last_finger_count = None


def perform_task(fingers):

    if fingers == 0:
        time.sleep(0.2)
        print("Opening Canvas...")
        webbrowser.open("https://canvas.txstate.edu/")
        time.sleep(3)
        print("available for the next task!")

    elif fingers == 1:
        time.sleep(0.2)  # Pointing gesture
        print("Performing task for pointing gesture: Opening ChatGPT")
        webbrowser.open("https://chatgpt.com/")
        time.sleep(3)
        print("available for the next task!")

    elif fingers == 2:
        time.sleep(0.2)
        print("Opening Outlook...")
        webbrowser.open("https://outlook.office.com/mail/")
        time.sleep(3)
        print("available for the next task!")

def main():
    global frames_elapsed

    region_top = 0
    region_bottom = int(2 * 480 / 3)
    region_left = int(640 / 2)
    region_right = 640

    # Open a connection to the camera
    cap = cv2.VideoCapture(0)  # 0 is the default camera

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video stream")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Could not read frame")
            break

        frame_rgb = frame #cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(frame_rgb, (640, 480))
        resized_frame = cv2.flip(resized_frame, 1)
        region = get_region(resized_frame, region_top, region_bottom, region_left, region_right)
        
        if frames_elapsed < CALIBRATION_TIME:
            get_average(region)
        else:
            region_pair = segment(region)
            if region_pair is not None:
                thresholded_region, segmented_region = region_pair
                cv2.drawContours(region, [segmented_region], -1, (255, 255, 255))
                get_hand_data(thresholded_region, segmented_region)
                cv2.imshow("Segmented Image", thresholded_region)

        write_on_image(resized_frame, hand, frames_elapsed, CALIBRATION_TIME, region_left, region_top, region_right, region_bottom)

        # Display the resulting frame
        cv2.imshow('Video Stream', resized_frame)
        frames_elapsed += 1

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
