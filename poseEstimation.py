import cv2
import mediapipe as mp
import time
import os
############ Initialize mp_drawing and mp_pose:
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

############  For static images:
#The with statement creates a context in which the mp_pose.Pose object is used.
#The with statement is used to ensure that the mp_pose.Pose object is properly released and its resources are freed after it 
#is no longer needed
##static_image_mode=True indicates that the input is a static image.
##model_complexity=2 specifies the complexity of the pose detection model. The possible values are 0, 1, or 2, with 2 being the most complex and accurate.
##detection_confidence=0.5 sets the minimum confidence score for detecting a pose.
## The mediapipe.solutions.pose.Pose module within Mediapipe provides a pre-trained machine learning model (CNN) for detecting
## and estimating human pose in images or videos. The module predict the 2D keypoint locations of the human body
## The CNN model then predicts the 2D keypoint locations, which are returned as a list of ‘Landmark‘ objects, 
## each representing a single keypoint on the body (  x and y coordinates of a single keypoint and 
## associated LandmarkVisibility score that indicates the confidence of the model in detecting that keypoint.).
with mp_pose.Pose(static_image_mode=True,  model_complexity=2,  min_detection_confidence=0.5) as pose:
# mediapipe.solutions.pose.Pose
#upper_body_only=True
#model_complexity default is 1
    #The image is loaded using cv2.imread() function.
    image = cv2.imread('4.jpg')
    #The shape of the image is obtained using the shape property.
    image_height, image_width, _  = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # Draw pose landmarks on the image.
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imwrite(r'4.png', annotated_image)

############  For webcam input:
#cap = cv2.VideoCapture(0)
############ For Video input:
cap = cv2.VideoCapture("1.mp4")
'''
The prevTime variable is used to calculate the time elapsed between successive frames in a video stream or webcam input. 
This is used to calculate the frame rate, which is displayed in the output video stream.

The prevTime variable is used to calculate the time elapsed between successive frames in a video stream or webcam input. 
This is used to calculate the frame rate, which is displayed in the output video stream.
The prevTime variable is initialized to 0 before the start of the while loop, and then updated to the current time at the 
end of each iteration of the loop
'''
prevTime = 0

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Because The input image is passed as an argument (not reference) to the process() method.
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference, by telling Python that the image should not be modified. 
    # This can help reduce memory usage and speed up the execution of the code
    image.flags.writeable = False
    
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    '''
    The currTime variable is set to the current time using the time.time() function, and the frame rate is calculated
     as the reciprocal of the time elapsed between the current frame and the previous frame
    '''
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    '''
    This calculation assumes that the time elapsed between successive frames is roughly constant, which may not
    always be the case. However, it provides a reasonable estimate of the frame rate for most purposes.
    The frame rate is then displayed on the output video stream or webcam feed using the cv2.putText() function
    '''
    cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
    '''
    This function adds a text string to the image or video frame, specifying the current frame rate. 
    The int() function is used to convert the frame rate to an integer, and the (20, 70) tuple specifies 
    the position of the text string on the image.
    '''
    #displays an image in a window with the given window name
    cv2.imshow('Pose Estimation tuto using mediapipe.solutions.pose.Pose model', image)
    #check if the user has pressed the ESC key, and if so, it breaks out of the main loop and exits the program.
    if cv2.waitKey(5) & 0xFF == 27:
      break
#
cap.release()
