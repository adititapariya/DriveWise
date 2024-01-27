import time
class AttentionScorer:

    def __init__(self, t_now, ear_thresh, gaze_thresh, perclos_thresh=0.2, roll_thresh=60,
                 pitch_thresh=20, yaw_thresh=30, ear_time_thresh=4.0, gaze_time_thresh=2.,
                 pose_time_thresh=4.0, verbose=False):

        self.ear_thresh = ear_thresh
        self.gaze_thresh = gaze_thresh
        self.perclos_thresh = perclos_thresh
        self.roll_thresh = roll_thresh
        self.pitch_thresh = pitch_thresh
        self.yaw_thresh = yaw_thresh
        self.ear_time_thresh = ear_time_thresh
        self.gaze_time_thresh = gaze_time_thresh
        self.pose_time_thresh = pose_time_thresh
        self.verbose = verbose

        self.perclos_time_period = 60
        
        self.last_time_eye_opened = t_now
        self.last_time_looked_ahead = t_now
        self.last_time_attended = t_now
        self.closure_time = 0
        self.not_look_ahead_time = 0
        self.distracted_time = 0

        self.prev_time = t_now
        self.eye_closure_counter = 0


    def eval_scores(self, t_now, ear_score, gaze_score, head_roll, head_pitch, head_yaw):
        # instantiating state of attention variables
        asleep = False
        looking_away = False
        distracted = False

        if self.closure_time >= self.ear_time_thresh:  # check if the ear cumulative counter surpassed the threshold
            asleep = True

        if self.not_look_ahead_time >= self.gaze_time_thresh:  # check if the gaze cumulative counter surpassed the threshold
            looking_away = True

        if self.distracted_time >= self.pose_time_thresh:  # check if the pose cumulative counter surpassed the threshold
            distracted = True

        if (ear_score is not None) and (ear_score <= self.ear_thresh):
            self.closure_time = t_now - self.last_time_eye_opened
        elif ear_score is None or (ear_score is not None and ear_score > self.ear_thresh):
            self.last_time_eye_opened = t_now
            self.closure_time = 0.

        if (gaze_score is not None) and (gaze_score > self.gaze_thresh):
            self.not_look_ahead_time = t_now - self.last_time_looked_ahead
        elif gaze_score is None or (gaze_score is not None and gaze_score <= self.gaze_thresh):
            self.last_time_looked_ahead = t_now
            self.not_look_ahead_time = 0.

        if ((head_roll is not None and abs(head_roll) > self.roll_thresh) or (
                head_pitch is not None and abs(head_pitch) > self.pitch_thresh) or (
                head_yaw is not None and abs(head_yaw) > self.yaw_thresh)):
            self.distracted_time = t_now - self.last_time_attended
        elif head_roll is None or head_pitch is None or head_yaw is None or (
            (abs(head_roll) <= self.roll_thresh) and (abs(head_pitch) <= self.pitch_thresh) and (
                abs(head_yaw) <= self.yaw_thresh)):
            self.last_time_attended = t_now
            self.distracted_time = 0.

        if self.verbose:  # print additional info if verbose is True
            print(
                f"ear counter:{self.ear_counter}/{self.ear_act_thresh}\ngaze counter:{self.gaze_counter}/{self.gaze_act_thresh}\npose counter:{self.pose_counter}/{self.pose_act_thresh}")
            print(
                f"eye closed:{asleep}\tlooking away:{looking_away}\tdistracted:{distracted}")

        return asleep, looking_away, distracted

    def get_PERCLOS(self, t_now, fps, ear_score):

        delta = t_now - self.prev_time  # set delta timer
        tired = False  # set default value for the tired state of the driver

        all_frames_numbers_in_perclos_duration = int(self.perclos_time_period * fps)

        # if the ear_score is lower or equal than the threshold, increase the eye_closure_counter
        if (ear_score is not None) and (ear_score <= self.ear_thresh):
            self.eye_closure_counter += 1

        # compute the PERCLOS over a given time period
        perclos_score = (self.eye_closure_counter) / all_frames_numbers_in_perclos_duration

        if perclos_score >= self.perclos_thresh:  # if the PERCLOS score is higher than a threshold, tired = True
            tired = True

        if delta >= self.perclos_time_period:  # at every end of the given time period, reset the counter and the timer
            self.eye_closure_counter = 0
            self.prev_time = t_now

        return tired, perclos_score
