import cv2
import numpy as np
from numpy import linalg as LA
from Utils import resize


EYES_LMS_NUMS = [33, 133, 160, 144, 158, 153, 362, 263, 385, 380, 387, 373]
LEFT_IRIS_NUM = 468
RIGHT_IRIS_NUM = 473

class EyeDetector:

    def __init__(self, show_processing: bool = False):
        self.show_processing = show_processing

    @staticmethod
    def _calc_EAR_eye(eye_pts):
        ear_eye = (LA.norm(eye_pts[2] - eye_pts[3]) + LA.norm(
            eye_pts[4] - eye_pts[5])) / (2 * LA.norm(eye_pts[0] - eye_pts[1]))
        return ear_eye
    
    def show_eye_keypoints(self, color_frame, landmarks, frame_size):
        cv2.circle(color_frame, (landmarks[LEFT_IRIS_NUM, :2] * frame_size).astype(np.uint32),
                   3, (255, 255, 255), cv2.FILLED)
        cv2.circle(color_frame, (landmarks[RIGHT_IRIS_NUM, :2] * frame_size).astype(np.uint32),
                   3, (255, 255, 255), cv2.FILLED)

        for n in EYES_LMS_NUMS:
            x = int(landmarks[n, 0] * frame_size[0])
            y = int(landmarks[n, 1] * frame_size[1])
            cv2.circle(color_frame, (x, y), 1, (0, 0, 255), -1)
        return

    def get_EAR(self, frame, landmarks):
        # numpy array for storing the keypoints positions of the left and right eyes
        eye_pts_l = np.zeros(shape=(6, 2))
        eye_pts_r = eye_pts_l.copy()

        # get the face mesh keypoints
        for i in range(len(EYES_LMS_NUMS)//2):
            # array of x,y coordinates for the left eye reference point
            eye_pts_l[i] = landmarks[EYES_LMS_NUMS[i], :2]
            # array of x,y coordinates for the right eye reference point
            eye_pts_r[i] = landmarks[EYES_LMS_NUMS[i+6], :2]

        ear_left = self._calc_EAR_eye(eye_pts_l)  # computing the left eye EAR score
        ear_right = self._calc_EAR_eye(eye_pts_r)  # computing the right eye EAR score

        # computing the average EAR score
        ear_avg = (ear_left + ear_right) / 2

        return ear_avg
    
    @staticmethod
    def _calc_1eye_score(landmarks, eye_lms_nums, eye_iris_num, frame_size, frame):
        iris = landmarks[eye_iris_num, :2]

        eye_x_min = landmarks[eye_lms_nums, 0].min()
        eye_y_min = landmarks[eye_lms_nums, 1].min()
        eye_x_max = landmarks[eye_lms_nums, 0].max()
        eye_y_max = landmarks[eye_lms_nums, 1].max()
        
        eye_center = np.array(((eye_x_min+eye_x_max)/2,
                                    (eye_y_min+eye_y_max)/2))
        
        eye_gaze_score = LA.norm(iris - eye_center) / eye_center[0]
        
        eye_x_min_frame = int(eye_x_min * frame_size[0])
        eye_y_min_frame = int(eye_y_min * frame_size[1])
        eye_x_max_frame = int(eye_x_max * frame_size[0])
        eye_y_max_frame = int(eye_y_max * frame_size[1])

        eye = frame[eye_y_min_frame:eye_y_max_frame,
                            eye_x_min_frame:eye_x_max_frame]

        return eye_gaze_score, eye

    def get_Gaze_Score(self, frame, landmarks, frame_size):

        left_gaze_score, left_eye = self._calc_1eye_score(
            landmarks, EYES_LMS_NUMS[:6], LEFT_IRIS_NUM, frame_size, frame)
        right_gaze_score, right_eye = self._calc_1eye_score(
            landmarks, EYES_LMS_NUMS[6:], RIGHT_IRIS_NUM, frame_size, frame)

        # if show_processing is True, shows the eyes ROI, eye center, pupil center and line distance

        # computes the average gaze score for the 2 eyes
        avg_gaze_score = (left_gaze_score + right_gaze_score) / 2

        if self.show_processing and (left_eye is not None) and (right_eye is not None):
            left_eye = resize(left_eye, 1000)
            right_eye = resize(right_eye, 1000)
            cv2.imshow("left eye", left_eye)
            cv2.imshow("right eye", right_eye)
        
        return avg_gaze_score
